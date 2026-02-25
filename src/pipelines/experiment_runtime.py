from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.config.spec import ToyConfig
from src.models.diffusion import DiffusionSchedule, GuidedReverseSampler
from src.models.gmm import IsotropicGMM
from src.models.tdp import ClosedFormTDP
from src.planners import Planner, ReverseDiffusionPlanner, TDPPlanner
from src.scoring import RewardScoreModel
from src.tasks.reward import ToyReward


@dataclass
class Runtime:
    cfg: DictConfig
    device: torch.device
    dtype: torch.dtype
    gmm: IsotropicGMM
    reward: ToyReward
    guidance_scale: float
    runners: Dict[str, Planner]
    selected_models: List[str] = field(default_factory=list)


@dataclass
class SharedRuntime:
    device: torch.device
    dtype: torch.dtype
    gmm: IsotropicGMM
    reward: ToyReward
    schedule: DiffusionSchedule
    reward_fn: Callable[[torch.Tensor], torch.Tensor]
    guidance_scale: float
    clip_norm: float | None


@dataclass
class ModelRequest:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


ModelFactory = Callable[[str, Mapping[str, Any], SharedRuntime, DictConfig], Planner]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(s: str) -> torch.device:
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_timezone(name: str):
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def make_outdir(root: str, exp_name: str, timezone_name: str = "UTC") -> Path:
    now = datetime.now(_resolve_timezone(timezone_name))
    out_dir = Path(root) / exp_name / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_cfg(dotlist: List[str], extra_defaults: Dict | None = None) -> DictConfig:
    base_struct = OmegaConf.structured(ToyConfig())
    base_plain = OmegaConf.create(OmegaConf.to_container(base_struct, resolve=True))
    merged = base_plain
    if extra_defaults:
        merged = OmegaConf.merge(merged, OmegaConf.create(extra_defaults))
    merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(dotlist))
    return merged


def _sanitize_models(raw_models: Sequence[str | Mapping[str, Any]]) -> List[ModelRequest]:
    out: List[ModelRequest] = []
    for raw in raw_models:
        if isinstance(raw, str):
            out.append(ModelRequest(name=str(raw), params={}))
            continue

        if not isinstance(raw, Mapping):
            raise ValueError(f"Model spec must be str or mapping, got type={type(raw)}")

        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError(f"Model mapping must include non-empty `name`: {raw}")

        enabled = raw.get("enabled", True)
        if not bool(enabled):
            continue

        params = raw.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            raise ValueError(f"`params` must be a mapping for model `{name}`")
        out.append(ModelRequest(name=name, params=dict(params)))

    names = [m.name for m in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"Duplicate model(s): {dup}")
    return out


def parse_model_list(raw_models: Sequence[str | Mapping[str, Any]]) -> List[str]:
    reqs = _sanitize_models(raw_models)
    bad = [r.name for r in reqs if r.name not in MODEL_FACTORIES]
    if bad:
        raise ValueError(f"Unsupported models: {bad}. Supported={supported_models()}")
    return [r.name for r in reqs]


def supported_models() -> List[str]:
    return list(MODEL_FACTORIES.keys())


def default_model_name() -> str:
    if "guided" in MODEL_FACTORIES:
        return "guided"
    names = supported_models()
    if not names:
        raise ValueError("No registered models available")
    return names[0]


def _resolve_goal_vector(cfg: DictConfig, action_dim: int, default_goal_scalar: float) -> List[float]:
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")

    goal = cfg.reward.get("goal", None)
    if goal is not None:
        goal_list = [float(v) for v in goal]
        if len(goal_list) != action_dim:
            raise ValueError(f"reward.goal length must be action_dim={action_dim}, got {len(goal_list)}")
        return goal_list

    gx = cfg.reward.get("goal_x", None)
    gy = cfg.reward.get("goal_y", None)

    vals = [float(default_goal_scalar) for _ in range(action_dim)]
    if gx is not None:
        vals[0] = float(gx)
    if action_dim >= 2 and gy is not None:
        vals[1] = float(gy)
    return vals


def _build_shared(cfg: DictConfig) -> SharedRuntime:
    set_seed(int(cfg.eval.seed))
    device = resolve_device(str(cfg.eval.device))
    dtype = torch.float32

    horizon = int(cfg.traj.horizon_T)
    action_dim = int(cfg.traj.action_dim)
    flat_dim = horizon * action_dim
    mean_mag = float(cfg.traj.base_action_mean)

    mu_pos = torch.full((flat_dim,), mean_mag, dtype=dtype)
    mu_neg = torch.full((flat_dim,), -mean_mag, dtype=dtype)
    mu = torch.stack([mu_pos, mu_neg], dim=0)
    pi_pos = float(getattr(cfg.traj, "pi_pos", 0.5))
    pi_pos = max(0.0, min(1.0, pi_pos))
    pi = torch.tensor([pi_pos, 1.0 - pi_pos], dtype=dtype)
    gmm = IsotropicGMM(pi=pi, mu=mu, sigma0_sq=float(cfg.traj.sigma0_sq))

    goal = _resolve_goal_vector(cfg, action_dim=action_dim, default_goal_scalar=mean_mag * horizon)
    reward = ToyReward(
        horizon_T=horizon,
        action_dim=action_dim,
        base_action_mean=mean_mag,
        w_neg=float(cfg.reward.w_neg),
        w_pos=float(cfg.reward.w_pos),
        offset=float(cfg.reward.offset),
        state_var=float(cfg.reward.state_var),
        goal=goal,
    )
    score_model = RewardScoreModel(reward=reward)

    schedule = DiffusionSchedule.make_linear(
        n_steps=int(cfg.diffusion.n_steps),
        beta_start=float(cfg.diffusion.beta_start),
        beta_end=float(cfg.diffusion.beta_end),
        device=device,
        dtype=dtype,
    )

    guidance_scale = float(cfg.guidance.scale) if bool(cfg.guidance.enabled) else 0.0
    clip_norm = cfg.guidance.get("clip_norm", None)
    clip_norm = None if clip_norm is None else float(clip_norm)

    return SharedRuntime(
        device=device,
        dtype=dtype,
        gmm=gmm,
        reward=reward,
        schedule=schedule,
        reward_fn=score_model.score,
        guidance_scale=guidance_scale,
        clip_norm=clip_norm,
    )


def _factory_nonguided(name: str, params: Mapping[str, Any], shared: SharedRuntime, cfg: DictConfig) -> Planner:
    sampler = GuidedReverseSampler(
        gmm=shared.gmm,
        schedule=shared.schedule,
        reward_fn=shared.reward_fn,
        guidance_scale=float(params.get("guidance_scale", shared.guidance_scale)),
        clip_norm=params.get("clip_norm", shared.clip_norm),
    )
    return ReverseDiffusionPlanner(
        name=name,
        sampler=sampler,
        guided=False,
        reward_fn=shared.reward_fn,
        n_candidates=int(params.get("n_candidates", cfg.reverse.n_candidates)),
    )


def _factory_guided(name: str, params: Mapping[str, Any], shared: SharedRuntime, cfg: DictConfig) -> Planner:
    sampler = GuidedReverseSampler(
        gmm=shared.gmm,
        schedule=shared.schedule,
        reward_fn=shared.reward_fn,
        guidance_scale=float(params.get("guidance_scale", shared.guidance_scale)),
        clip_norm=params.get("clip_norm", shared.clip_norm),
    )
    return ReverseDiffusionPlanner(
        name=name,
        sampler=sampler,
        guided=True,
        reward_fn=shared.reward_fn,
        n_candidates=int(params.get("n_candidates", cfg.reverse.n_candidates)),
    )


def _factory_tdp(name: str, params: Mapping[str, Any], shared: SharedRuntime, cfg: DictConfig) -> Planner:
    n_roots = int(params.get("n_roots", cfg.tdp.n_roots))
    renoise_frac = float(params.get("renoise_frac", cfg.tdp.renoise_frac))
    pg = bool(params.get("pg", cfg.tdp.pg))
    pg_scale = float(params.get("pg_scale", cfg.tdp.pg_scale))

    tdp = ClosedFormTDP(
        gmm=shared.gmm,
        schedule=shared.schedule,
        reward_fn=shared.reward_fn,
        horizon_T=int(cfg.traj.horizon_T),
        action_dim=int(cfg.traj.action_dim),
        renoise_frac=renoise_frac,
        pg=pg,
        pg_scale=pg_scale,
    )
    return TDPPlanner(
        name=name,
        tdp=tdp,
        n_roots=n_roots,
    )


MODEL_FACTORIES: MutableMapping[str, ModelFactory] = {
    "non-guided": _factory_nonguided,
    "guided": _factory_guided,
    "tdp": _factory_tdp,
}


def register_model_factory(name: str, factory: ModelFactory):
    MODEL_FACTORIES[name] = factory


def _resolve_model_requests(cfg: DictConfig, models: Sequence[str | Mapping[str, Any]] | None) -> List[ModelRequest]:
    if models is not None:
        reqs = _sanitize_models(models)
    else:
        cfg_models = cfg.eval.get("models", [])
        if cfg_models:
            reqs = _sanitize_models(cfg_models)
        else:
            reqs = _sanitize_models(supported_models())

    names = [r.name for r in reqs]
    bad = [n for n in names if n not in MODEL_FACTORIES]
    if bad:
        raise ValueError(f"Unsupported models: {bad}. Supported={supported_models()}")
    if not reqs:
        raise ValueError("No enabled models selected")
    return reqs


def build_runtime(cfg: DictConfig, models: Sequence[str | Mapping[str, Any]] | None = None) -> Runtime:
    shared = _build_shared(cfg)
    reqs = _resolve_model_requests(cfg, models=models)

    runners: Dict[str, Planner] = {}
    for req in reqs:
        runners[req.name] = MODEL_FACTORIES[req.name](req.name, req.params, shared, cfg)

    return Runtime(
        cfg=cfg,
        device=shared.device,
        dtype=shared.dtype,
        gmm=shared.gmm,
        reward=shared.reward,
        guidance_scale=shared.guidance_scale,
        runners=runners,
        selected_models=[r.name for r in reqs],
    )


def sample_model(runtime: Runtime, model_name: str, n: int, batch_size: int) -> torch.Tensor:
    if model_name not in runtime.runners:
        raise ValueError(f"Model `{model_name}` was not built. Built={runtime.selected_models}")
    return runtime.runners[model_name].sample(
        n=n,
        batch_size=batch_size,
        device=runtime.device,
        dtype=runtime.dtype,
    )
