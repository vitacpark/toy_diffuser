from __future__ import annotations

import random
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from kuka_planner_simple.config.spec import KukaConfig
from kuka_planner_simple.planners.registry import MODEL_REGISTRY, SharedRuntime, supported_models
from kuka_planner_simple.tasks import make_task_adapter, supported_task_names


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PARAM_KEYS = {
    "batch_size",
    "pg",
    "pg_scale",
    "guide_step",
    "selection_mode",
    "sample_ub",
    "sample_jump_size",
    "diffusion_step",
}
PLANNER_ALLOWED_MODES = {
    "diffusion": {"direct"},
    "tdp": {"direct", "subtree"},
}


@dataclass
class Runtime:
    cfg: DictConfig
    device: torch.device
    dtype: torch.dtype
    out_dir: Path
    selected_models: Dict[str, DictConfig]
    shared: SharedRuntime


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    v = str(value).strip().lower()
    if v == "cpu":
        return torch.device("cpu")
    if v == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if v.startswith("cuda") and torch.cuda.is_available():
        return torch.device(v)
    return torch.device("cpu")


def _resolve_timezone(name: str):
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def make_outdir(root: str, exp_name: str, timezone_name: str = "UTC") -> Path:
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path
    now = datetime.now(_resolve_timezone(timezone_name))
    out_dir = root_path / exp_name / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _extract_task_from_dotlist(dotlist: Sequence[str]):
    filtered: List[str] = []
    task_name = "conditional_stack"
    for item in dotlist:
        if item.startswith("task="):
            task_name = item.split("=", 1)[1]
        else:
            filtered.append(item)
    return filtered, task_name


def _apply_task_model_defaults(cfg: DictConfig):
    task_module = str(cfg.task.get("diffusion_module", "")).strip()
    if not task_module:
        return

    for model in cfg.models:
        if not str(model.get("diffusion_module", "")).strip():
            model.diffusion_module = task_module


def load_cfg(dotlist: Sequence[str]) -> DictConfig:
    dotlist, task_name = _extract_task_from_dotlist(dotlist)

    base_struct = OmegaConf.structured(KukaConfig())
    cfg = OmegaConf.create(OmegaConf.to_container(base_struct, resolve=True))

    base_yaml = PROJECT_ROOT / "configs" / "base.yaml"
    task_yaml = PROJECT_ROOT / "configs" / "tasks" / f"{task_name}.yaml"
    model_yaml = PROJECT_ROOT / "configs" / "models" / "default_compare.yaml"

    if base_yaml.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(base_yaml))
    if task_yaml.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(task_yaml))
    else:
        raise FileNotFoundError(f"Task config not found: {task_yaml}")
    if model_yaml.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(model_yaml))
    else:
        raise FileNotFoundError(f"Model config not found: {model_yaml}")

    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(dotlist)))

    if str(cfg.task.name) != task_name:
        cfg.task.name = task_name

    _apply_task_model_defaults(cfg)
    _validate_cfg(cfg)
    return cfg


def _validate_cfg(cfg: DictConfig):
    task_name = str(cfg.task.name)
    task_names = set(supported_task_names())
    if task_name not in task_names:
        raise ValueError(f"Unsupported task `{task_name}`. Supported={sorted(task_names)}")

    tdp_cfg = cfg.get("tdp", {})
    tdp_keys = set(tdp_cfg.keys()) if OmegaConf.is_config(tdp_cfg) else set()
    deprecated_tdp_keys = sorted({"use_tree", "use_sub_tree"} & tdp_keys)
    if deprecated_tdp_keys:
        raise ValueError(
            f"Deprecated tdp keys found: {deprecated_tdp_keys}. "
            "Use `models[].planner_params.selection_mode` instead."
        )

    for model in cfg.models:
        planner = str(model.planner)
        if planner not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported planner `{planner}` in model `{model.name}`. Supported={supported_models()}"
            )

        diffusion_module = str(model.get("diffusion_module", "")).strip()
        if not diffusion_module:
            raise ValueError(
                f"Model `{model.name}` is missing `diffusion_module`. "
                "Set it in model config or `task.diffusion_module`."
            )

        overrides = model.get("overrides", {})
        has_overrides = (OmegaConf.is_config(overrides) and len(overrides) > 0) or (
            isinstance(overrides, dict) and bool(overrides)
        )
        if has_overrides:
            raise ValueError(
                f"Model `{model.name}` uses deprecated `overrides`. "
                "Move all options into `planner_params`."
            )

        params = model.get("planner_params", {})
        if OmegaConf.is_config(params):
            param_keys = set(params.keys())
        elif isinstance(params, dict):
            param_keys = set(params.keys())
        else:
            param_keys = set()
        unknown = sorted(param_keys - PLANNER_PARAM_KEYS)
        if unknown:
            raise ValueError(
                f"Model `{model.name}` has unknown planner_params keys: {unknown}. "
                f"Supported keys={sorted(PLANNER_PARAM_KEYS)}"
            )
        if "pg" not in param_keys:
            raise ValueError(
                f"Model `{model.name}` missing required planner_params.pg (set true/false explicitly)."
            )

        if planner in PLANNER_ALLOWED_MODES:
            mode = params.get("selection_mode", None) if (OmegaConf.is_config(params) or isinstance(params, dict)) else None
            if mode is None:
                raise ValueError(
                    f"Model `{model.name}` missing required planner_params.selection_mode for planner `{planner}`."
                )
            mode_s = str(mode).strip().lower()
            allowed = PLANNER_ALLOWED_MODES[planner]
            if mode_s not in {"direct", "subtree"}:
                raise ValueError(
                    f"Model `{model.name}` has invalid selection_mode `{mode}`. "
                    "Supported=['direct','subtree']"
                )
            if mode_s not in allowed:
                raise ValueError(
                    f"Model `{model.name}` selection_mode `{mode_s}` is not allowed for planner `{planner}`. "
                    f"Allowed={sorted(allowed)}"
                )


def _check_runtime_dependencies():
    required = ["pybullet", "gym", "easydict", "imageio"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise ModuleNotFoundError(
            "Missing runtime dependency packages for evaluation runtime: "
            f"{missing}. Install them in your environment before running evaluation."
        )


def _select_models(cfg: DictConfig) -> Dict[str, DictConfig]:
    defined = {str(m.name): m for m in cfg.models if bool(m.enabled)}
    requested = list(cfg.eval.models) if cfg.eval.models else list(defined.keys())

    selected: Dict[str, DictConfig] = {}
    for name in requested:
        n = str(name)
        if n not in defined:
            raise ValueError(
                f"Model `{n}` not found in cfg.models enabled set={list(defined.keys())}."
            )
        selected[n] = defined[n]

    if not selected:
        raise ValueError("No enabled models selected.")
    return selected


def build_runtime(cfg: DictConfig) -> Runtime:
    _apply_task_model_defaults(cfg)
    _validate_cfg(cfg)
    _check_runtime_dependencies()

    from denoising_diffusion_pytorch import Trainer
    from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
    from denoising_diffusion_pytorch.temporal_attention import TemporalUnet
    from diffusion.models.mlp import TimeConditionedMLP

    set_seed(int(cfg.runtime.seed))
    device = resolve_device(str(cfg.runtime.device))
    dtype = torch.float32

    out_dir = make_outdir(
        root=str(cfg.output.root),
        exp_name=str(cfg.output.experiment_name),
        timezone_name=str(cfg.output.timezone),
    )

    task_adapter = make_task_adapter(
        task_name=str(cfg.task.name),
        device=device,
        horizon=int(cfg.task.horizon),
    )

    dataset = KukaDataset(int(cfg.task.horizon))
    task_adapter.bind_dataset(dataset)

    selected_models = _select_models(cfg)
    shared = SharedRuntime(
        cfg=cfg,
        device=device,
        dtype=dtype,
        dataset=dataset,
        obs_dim=int(dataset.obs_dim),
        trainer_cls=Trainer,
        temporal_unet_cls=TemporalUnet,
        guide_cls=TimeConditionedMLP,
        task_adapter=task_adapter,
        artifacts_root=out_dir / "artifacts",
        project_root=PROJECT_ROOT,
    )

    return Runtime(
        cfg=cfg,
        device=device,
        dtype=dtype,
        out_dir=out_dir,
        selected_models=selected_models,
        shared=shared,
    )
