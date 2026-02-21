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
class EvalSpec:
    seed: int = 0
    device: str = "auto"
    # Sample sizes
    n_base: int = 500_000
    n_guided: int = 200_000
    # Bootstrap (resampling of already-computed f, logw), cheap and stable.
    bootstrap_reps: int = 200
    # Batch size for diffusion sampling
    batch_size: int = 1024
    # Which statistics to report
    f_list: List[str] = field(default_factory=lambda: ["final_x", "final_y", "pos_indicator", "R"])


@dataclass
class ToyConfig:
    traj: TrajectorySpec = field(default_factory=TrajectorySpec)
    reward: RewardSpec = field(default_factory=RewardSpec)
    diffusion: DiffusionSpec = field(default_factory=DiffusionSpec)
    guidance: GuidanceSpec = field(default_factory=GuidanceSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
