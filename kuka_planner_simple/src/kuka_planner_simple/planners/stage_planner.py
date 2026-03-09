from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

@dataclass
class ResolvedPlannerParams:
    batch_size: int
    pg: bool
    pg_scale: float
    guide_step: int
    selection_mode: str
    sample_ub: int
    sample_jump_size: int
    diffusion_step: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StagePlanner:
    def __init__(
        self,
        *,
        model_cfg,
        shared,
        trainer,
        guide,
        obs_dim: int,
        allowed_modes: set[str],
    ):
        self.model_cfg = model_cfg
        self.shared = shared
        self.trainer = trainer
        self.guide = guide
        self.obs_dim = int(obs_dim)
        self.allowed_modes = set(allowed_modes)

    def _to_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if OmegaConf.is_config(value):
            raw = OmegaConf.to_container(value, resolve=True)
            return dict(raw) if isinstance(raw, dict) else {}
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def _clean_none(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in value.items() if v is not None}

    def _model_options(self) -> Dict[str, Any]:
        return self._clean_none(self._to_dict(self.model_cfg.get("planner_params", {})))

    def _resolve_selection_mode(self, opts: Dict[str, Any]) -> str:
        mode_s = str(opts.get("selection_mode", "")).strip().lower()
        if not mode_s:
            raise ValueError(
                "Missing required `planner_params.selection_mode`. "
                f"Allowed={sorted(self.allowed_modes)} for planner `{self.model_cfg.planner}`."
            )
        if mode_s not in self.allowed_modes:
            raise ValueError(
                f"selection_mode `{mode_s}` is not allowed for planner `{self.model_cfg.planner}`. "
                f"Allowed={sorted(self.allowed_modes)}"
            )
        return mode_s

    def resolve_params(self) -> ResolvedPlannerParams:
        opts = self._model_options()
        if "pg" not in opts:
            raise ValueError(
                f"Model `{self.model_cfg.name}` is missing required `planner_params.pg` "
                "(set true/false explicitly)."
            )

        task_name = str(self.shared.cfg.task.name)
        pg_defaults = dict(self.shared.cfg.pg.defaults_by_task)
        task_pg_default = float(pg_defaults.get(task_name, 0.5))

        pg_enabled = bool(opts["pg"])

        if "pg_scale" in opts:
            pg_scale = float(opts["pg_scale"])
        elif self.shared.cfg.pg.get("scale", None) is not None:
            pg_scale = float(self.shared.cfg.pg.scale)
        else:
            pg_scale = task_pg_default

        return ResolvedPlannerParams(
            batch_size=int(opts.get("batch_size", int(self.shared.cfg.eval.batch_size))),
            pg=pg_enabled,
            pg_scale=pg_scale,
            guide_step=int(opts.get("guide_step", int(self.shared.cfg.pg.guide_step))),
            selection_mode=self._resolve_selection_mode(opts),
            sample_ub=int(opts.get("sample_ub", int(self.shared.cfg.tdp.sample_ub))),
            sample_jump_size=int(opts.get("sample_jump_size", int(self.shared.cfg.tdp.sample_jump_size))),
            diffusion_step=int(opts.get("diffusion_step", int(self.shared.cfg.tdp.diffusion_step))),
        )

    def _guided_candidates(self, conditions, ctx, params: ResolvedPlannerParams) -> torch.Tensor:
        return self.shared.task_adapter.guided_sample(
            self.trainer.ema_model,
            self.guide,
            int(params.batch_size),
            conditions,
            ctx,
            pg=bool(params.pg),
            pg_scale=float(params.pg_scale),
            guide_step=int(params.guide_step),
        )

    def _main_values(self, samples: torch.Tensor, ctx) -> np.ndarray:
        return self.shared.task_adapter.compute_values(samples, self.shared.dataset, ctx)

    def _best_candidate(self, samples: torch.Tensor, main_values: np.ndarray) -> torch.Tensor:
        best_idx = self.shared.task_adapter.choose_main(main_values)
        return samples[best_idx][None]

    def _best_subtree_candidate(
        self,
        *,
        samples: torch.Tensor,
        sub_samples: torch.Tensor,
        main_values: np.ndarray,
        sub_values: np.ndarray,
    ) -> torch.Tensor:
        best_idx, is_sub_best = self.shared.task_adapter.choose_subtree(main_values, sub_values)
        selected = sub_samples if is_sub_best else samples
        return selected[best_idx][None]

    def plan_stage(self, *, conditions, ctx, params: ResolvedPlannerParams) -> torch.Tensor:
        samples = self._guided_candidates(conditions=conditions, ctx=ctx, params=params)
        main_values = self._main_values(samples=samples, ctx=ctx)

        if params.selection_mode == "subtree":
            sub_plan_index = self.shared.task_adapter.make_sub_plan_index(
                batch_size=int(params.batch_size),
                sample_jump_size=int(params.sample_jump_size),
                sample_ub=int(params.sample_ub),
            )
            sub_idx_list = sub_plan_index.tolist()
            sub_conditions = [
                [(i, self.obs_dim, samples[j, i, :]) for i in range(int(sub_idx_list[j]))]
                for j in range(len(sub_idx_list))
            ]
            sub_samples = self.shared.task_adapter.fast_guided_sample(
                self.trainer.ema_model,
                self.guide,
                int(params.batch_size),
                sub_conditions,
                samples,
                ctx,
                diffusion_step=int(params.diffusion_step),
            )
            sub_values = self._main_values(samples=sub_samples, ctx=ctx)
            return self._best_subtree_candidate(
                samples=samples,
                sub_samples=sub_samples,
                main_values=main_values,
                sub_values=sub_values,
            )

        return self._best_candidate(samples=samples, main_values=main_values)
