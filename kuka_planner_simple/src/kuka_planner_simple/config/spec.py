from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RuntimeSpec:
    device: str = "auto"  # auto|cpu|cuda|cuda:0
    seed: int = 128


@dataclass
class TaskSpec:
    name: str = "conditional_stack"  # conditional_stack|pick2put|pnwp
    horizon: int = 128
    diffusion_steps: int = 1000
    diffusion_module: str = "diffusion.denoising_diffusion_pytorch"
    env_variant: str = "default"
    value_rule: str = "default"


@dataclass
class EvalSpec:
    episodes: int = 100
    batch_size: int = 32
    save_render: bool = False
    save_trajectories: bool = True
    models: List[str] = field(default_factory=lambda: ["diffusion", "tdp"])


@dataclass
class TDPSpec:
    sample_ub: int = 52
    sample_jump_size: int = 4
    diffusion_step: int = 100


@dataclass
class PGSpec:
    enabled: bool = False
    scale: Optional[float] = None
    guide_step: int = 1
    defaults_by_task: Dict[str, float] = field(
        default_factory=lambda: {
            "conditional_stack": 0.25,
            "pick2put": 0.5,
            "pnwp": 0.5,
        }
    )


@dataclass
class GuideArchSpec:
    time_dim: int = 128
    input_dim: int = 39
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
    output_dim: int = 12


@dataclass
class PlannerParamsSpec:
    # Common planner params
    batch_size: Optional[int] = None
    pg: Optional[bool] = None
    pg_scale: Optional[float] = None
    guide_step: Optional[int] = None

    # Selection strategy
    selection_mode: Optional[str] = None  # direct|subtree

    # Subtree params
    sample_ub: Optional[int] = None
    sample_jump_size: Optional[int] = None
    diffusion_step: Optional[int] = None


@dataclass
class ModelSpec:
    name: str = "diffusion"
    enabled: bool = True
    planner: str = "diffusion"  # diffusion|tdp
    diffusion_module: Optional[str] = None
    diffusion_log_dir: str = "assets/checkpoints/multiple_cube_kuka_temporal_convnew_real2_128"
    diffusion_epoch: int = 650
    guide_ckpt: str = "assets/checkpoints/kuka_cube_stack_classifier_new3/value_0.99/state_80.pt"
    guide_arch: GuideArchSpec = field(default_factory=GuideArchSpec)
    planner_params: PlannerParamsSpec = field(default_factory=PlannerParamsSpec)


@dataclass
class OutputSpec:
    root: str = "outputs"
    experiment_name: str = "run_kuka_eval"
    timezone: str = "Asia/Seoul"


@dataclass
class KukaConfig:
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    task: TaskSpec = field(default_factory=TaskSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    tdp: TDPSpec = field(default_factory=TDPSpec)
    pg: PGSpec = field(default_factory=PGSpec)
    models: List[ModelSpec] = field(default_factory=lambda: [
        ModelSpec(
            name="diffusion",
            planner="diffusion",
            enabled=True,
            planner_params=PlannerParamsSpec(selection_mode="direct", pg=False),
        ),
        ModelSpec(
            name="tdp",
            planner="tdp",
            enabled=True,
            planner_params=PlannerParamsSpec(selection_mode="subtree", pg=True),
        ),
    ])
    output: OutputSpec = field(default_factory=OutputSpec)
