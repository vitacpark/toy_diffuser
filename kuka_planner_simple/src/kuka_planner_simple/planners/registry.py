from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .base import EpisodeResult, PlannerRunner
from .stage_planner import StagePlanner
from kuka_planner_simple.tasks.adapters import to_np


@dataclass
class SharedRuntime:
    cfg: DictConfig
    device: torch.device
    dtype: torch.dtype
    dataset: Any
    obs_dim: int
    trainer_cls: Any
    temporal_unet_cls: Any
    guide_cls: Any
    task_adapter: Any
    artifacts_root: Path
    project_root: Path
    trainer_cache: Dict[Tuple[Any, ...], Any] = field(default_factory=dict)
    guide_cache: Dict[Tuple[Any, ...], Any] = field(default_factory=dict)


ModelFactory = Callable[[DictConfig, SharedRuntime], PlannerRunner]


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    return root / p


class KukaRunner:
    def __init__(
        self,
        model_cfg: DictConfig,
        shared: SharedRuntime,
        *,
        allowed_modes: set[str],
    ):
        self.model_cfg = model_cfg
        self.shared = shared
        self.name = str(model_cfg.name)
        self.planner = str(model_cfg.planner)

        self.env = self.shared.task_adapter.make_env(save_render=bool(self.shared.cfg.eval.save_render))
        self.dataset = self.shared.dataset
        self.obs_dim = int(self.shared.obs_dim)

        self.trainer = self._get_or_create_trainer()
        self.guide = self._get_or_create_guide()
        self.stage_planner = StagePlanner(
            model_cfg=self.model_cfg,
            shared=self.shared,
            trainer=self.trainer,
            guide=self.guide,
            obs_dim=self.obs_dim,
            allowed_modes=set(allowed_modes),
        )

    def _trainer_key(self) -> Tuple[Any, ...]:
        return (
            str(self.model_cfg.diffusion_module),
            str(self.model_cfg.diffusion_log_dir),
            int(self.model_cfg.diffusion_epoch),
            int(self.shared.cfg.task.horizon),
            int(self.shared.cfg.task.diffusion_steps),
            int(self.obs_dim),
        )

    def _guide_key(self) -> Tuple[Any, ...]:
        arch = self.model_cfg.guide_arch
        hidden_dims = tuple(int(x) for x in arch.hidden_dims)
        return (
            str(self.model_cfg.guide_ckpt),
            int(arch.time_dim),
            int(arch.input_dim),
            hidden_dims,
            int(arch.output_dim),
        )

    def _get_or_create_trainer(self):
        key = self._trainer_key()
        cached = self.shared.trainer_cache.get(key)
        if cached is not None:
            return cached

        unet = self.shared.temporal_unet_cls(
            horizon=int(self.shared.cfg.task.horizon),
            transition_dim=self.obs_dim,
            cond_dim=int(self.shared.cfg.task.horizon),
            dim=128,
            dim_mults=(1, 2, 4, 8),
        ).to(self.shared.device)

        diffusion_mod = importlib.import_module(str(self.model_cfg.diffusion_module))
        diffusion_cls = getattr(diffusion_mod, "GaussianDiffusion")
        diffusion = diffusion_cls(
            unet,
            channels=2,
            image_size=(int(self.shared.cfg.task.horizon), self.obs_dim),
            timesteps=int(self.shared.cfg.task.diffusion_steps),
            loss_type="l1",
        ).to(self.shared.device)

        results_folder = _resolve_path(self.shared.project_root, str(self.model_cfg.diffusion_log_dir))
        trainer = self.shared.trainer_cls(
            diffusion,
            self.dataset,
            self.env,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=700000,
            gradient_accumulate_every=2,
            ema_decay=0.995,
            fp16=False,
            results_folder=str(results_folder),
        )
        model_path = results_folder / f"model-{int(self.model_cfg.diffusion_epoch)}.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Diffusion checkpoint not found: {model_path}. "
                "Set `models[].diffusion_log_dir` and `models[].diffusion_epoch` correctly."
            )
        trainer.load(int(self.model_cfg.diffusion_epoch))
        trainer.ema_model.eval()

        self.shared.trainer_cache[key] = trainer
        return trainer

    def _get_or_create_guide(self):
        key = self._guide_key()
        cached = self.shared.guide_cache.get(key)
        if cached is not None:
            return cached

        arch = self.model_cfg.guide_arch
        guide = self.shared.guide_cls(
            time_dim=int(arch.time_dim),
            input_dim=int(arch.input_dim),
            hidden_dims=list(arch.hidden_dims),
            output_dim=int(arch.output_dim),
        ).to(self.shared.device)

        guide_ckpt = _resolve_path(self.shared.project_root, str(self.model_cfg.guide_ckpt))
        if not guide_ckpt.exists():
            raise FileNotFoundError(
                f"Guide checkpoint not found: {guide_ckpt}. "
                "Set `models[].guide_ckpt` correctly."
            )
        ckpt = torch.load(str(guide_ckpt), map_location=self.shared.device)

        state_dict = ckpt
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]

        guide.load_state_dict(state_dict)
        guide.eval()

        self.shared.guide_cache[key] = guide
        return guide

    def run_episode(self, episode_idx: int, seed: int) -> EpisodeResult:
        params = self.stage_planner.resolve_params()
        save_traj = bool(self.shared.cfg.eval.save_trajectories)
        save_render = bool(self.shared.cfg.eval.save_render)

        start_time = time.perf_counter()

        state = self.shared.task_adapter.reset_env(self.env, seed=seed)
        samples = self.shared.task_adapter.normalize_state(np.asarray(state))
        conditions = [(0, self.obs_dim, samples)]

        rewards = 0.0
        frames = [] if save_render else None
        total_samples = [] if save_traj else None

        for _ in range(self.shared.task_adapter.num_stages()):
            ctx = self.shared.task_adapter.get_stage_context(self.env)

            selected = self.stage_planner.plan_stage(conditions=conditions, ctx=ctx, params=params)
            selected_unorm = self.shared.task_adapter.unnormalize_samples(selected)
            selected_np = to_np(selected_unorm.squeeze(0).squeeze(0))

            next_state, state_seq, frame_seq, reward = self.shared.task_adapter.execute(
                selected_np,
                self.env,
                save_render,
            )

            if save_traj and total_samples is not None:
                total_samples.extend(state_seq)
            if save_render and frames is not None:
                frames.extend(frame_seq)
            rewards += float(reward)

            samples = self.shared.task_adapter.normalize_state(np.asarray(next_state))
            conditions = [(0, self.obs_dim, samples)]
            self.shared.task_adapter.advance_after_stage(self.env)

        elapsed = float(time.perf_counter() - start_time)
        success_threshold = float(getattr(self.env, "ref_max_score", 1.0))
        success = bool(rewards >= success_threshold - 1e-6)

        episode_dir = (
            self.shared.artifacts_root / str(self.shared.cfg.task.name) / self.name / f"episode_{episode_idx:04d}"
        )
        episode_dir.mkdir(parents=True, exist_ok=True)

        rollout_path = episode_dir / "rollout.json"
        model_cfg_payload = OmegaConf.to_container(self.model_cfg, resolve=True)
        with open(rollout_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": str(self.shared.cfg.task.name),
                    "model": self.name,
                    "planner": self.planner,
                    "planning_mode": str(params.selection_mode),
                    "resolved_params": params.to_dict(),
                    "model_config": model_cfg_payload,
                    "score": float(rewards),
                    "success": success,
                    "elapsed_sec": elapsed,
                    "seed": int(seed),
                },
                f,
                indent=2,
                sort_keys=True,
            )

        traj_path = None
        if save_traj and total_samples is not None:
            traj_path = episode_dir / "trajectory.npy"
            np.save(traj_path, np.asarray(total_samples))

        video_path = None
        if save_render and frames:
            import imageio

            video_path = episode_dir / "video.mp4"
            writer = imageio.get_writer(str(video_path))
            for frame in frames:
                writer.append_data(frame)
            writer.close()

        artifacts = {
            "episode_dir": str(episode_dir),
            "rollout": str(rollout_path),
        }
        if traj_path is not None:
            artifacts["trajectory"] = str(traj_path)
        if video_path is not None:
            artifacts["video"] = str(video_path)

        return EpisodeResult(
            reward=float(rewards),
            success=success,
            elapsed_sec=elapsed,
            artifacts=artifacts,
        )


PLANNER_SPECS: Dict[str, set[str]] = {
    "diffusion": {"direct"},
    "tdp": {"direct", "subtree"},
}


def _build_factory(allowed_modes: set[str]) -> ModelFactory:
    def _factory(model_cfg: DictConfig, shared: SharedRuntime) -> PlannerRunner:
        return KukaRunner(
            model_cfg=model_cfg,
            shared=shared,
            allowed_modes=allowed_modes,
        )

    return _factory


MODEL_REGISTRY: MutableMapping[str, ModelFactory] = {
    name: _build_factory(allowed_modes=modes) for name, modes in PLANNER_SPECS.items()
}


def register_model_factory(name: str, factory: ModelFactory):
    MODEL_REGISTRY[name] = factory


def supported_models():
    return list(MODEL_REGISTRY.keys())
