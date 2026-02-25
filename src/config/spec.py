from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional



@dataclass
class TrajectorySpec:
    """Trajectory tau_0 is a_{1:T} flattened to R^{T*action_dim}."""
    horizon_T: int = 32
    action_dim: int = 2
    # Base component means: (+/- base_action_mean) repeated for each timestep and each action dim
    base_action_mean: float = 0.1
    # Base covariance Sigma_0 = (sigma0^2) I
    sigma0_sq: float = 1.0 / 40.0
    pi_pos: float = 0.5



@dataclass
class RewardSpec:
    """
    r(s_t, a_t) = w_neg N(s_t; m_neg, state_var I) + w_pos N(s_t; m_pos, state_var I) + offset
    where m_pos = ( +base_action_mean*T, +base_action_mean*T )
          m_neg = ( -base_action_mean*T, -base_action_mean*T )
    """
    w_neg: float = 0.2
    w_pos: float = 0.7
    offset: float = 0.1
    state_var: float = 1.0 / 4.0  # covariance = state_var * I (2D)

    # Optional goal position. If None, defaults to (base_action_mean*T, base_action_mean*T) in run scripts.
    goal_x: Optional[float] = None
    goal_y: Optional[float] = None
    # Optional full goal vector. If set, this takes precedence over goal_x/goal_y.
    goal: Optional[List[float]] = None


@dataclass
class DiffusionSpec:
    n_steps: int = 256
    beta_start: float = 1e-4
    beta_end: float = 2e-2


@dataclass
class GuidanceSpec:
    enabled: bool = True
    scale: float = 10.0
    # Gradient clipping (L2 norm per-sample). Set None to disable.
    clip_norm: Optional[float] = 1.0


@dataclass
class ReverseSelectSpec:
    # Number of candidates sampled per rollout for non-guided/guided models.
    n_candidates: int = 1


@dataclass
class EvalSpec:
    seed: int = 0
    device: str = "auto"
    # Sample sizes
    n_base: int = 50_000
    n_guided: int = 50_000
    # Bootstrap (resampling of already-computed f, logw), cheap and stable.
    bootstrap_reps: int = 200
    # Batch size for diffusion sampling
    batch_size: int = 10_000
    # Which statistics to report
    f_list: List[str] = field(default_factory=lambda: ["final_x", "final_y", "pos_indicator", "R"])
    # Models to run in multi-model evaluation.
    models: List[str] = field(default_factory=list)


@dataclass
class TDPSpec:
    # Number of parent particles (B)
    n_roots: int = 10
    # Fixed fraction of diffusion steps to re-noise (0~1).
    renoise_frac: float = 0.15
    # Particle guidance on parent expansion only.
    pg: bool = True
    # Strength multiplier for particle-guidance shift.
    pg_scale: float = 1.0


@dataclass
class DriftLiteSpec:
    n_particles: int = 512
    rollout_batch: int = 0  # <=0 means auto from eval.batch_size // n_particles
    batch_particles: int = 0  # <=0 means auto from min(eval.batch_size, rollout_batch*n_particles)
    ess_threshold_ratio: float = 0.5
    resample: bool = True
    dt_mode: str = "auto"  # auto | fixed
    dt: float = 0.0
    ctrl_scale: float = 1.0
    basis: List[str] = field(default_factory=lambda: ["grad_r", "score"])
    proxy_gamma: float = 1.0
    divergence_mode: str = "grad_r_hutch"  # none | grad_r_hutch | all_hutch
    hutch_samples: int = 1
    reward_path: str = "constant"  # constant | linear
    reward_scale: float = 1.0
    reg_lambda: float = 1e-4
    output_mode: str = "best"  # best | resampled | weighted
    save_diagnostics: bool = False
    seed_offset: int = 3000


@dataclass
class OutputSpec:
    root: str = "outputs"
    timezone: str = "Asia/Seoul"
    save_base_max: int = 50_000
    save_chain_max: int = 20_000
    save_plots_max_points: int = 50_000
    save_diagnostics: bool = False


@dataclass
class ToyConfig:
    traj: TrajectorySpec = field(default_factory=TrajectorySpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    diffusion: DiffusionSpec = field(default_factory=DiffusionSpec)
    guidance: GuidanceSpec = field(default_factory=GuidanceSpec)
    reverse: ReverseSelectSpec = field(default_factory=ReverseSelectSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    tdp: TDPSpec = field(default_factory=TDPSpec)
    driftlite: DriftLiteSpec = field(default_factory=DriftLiteSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
