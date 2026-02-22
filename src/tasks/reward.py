from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


def normal_pdf_isotropic(x: torch.Tensor, mean: torch.Tensor, var: float) -> torch.Tensor:
    """
    Isotropic multivariate normal density N(x; mean, var*I).
    Shapes:
      x:    [B,T,2]
      mean: [1,1,2] (broadcastable)
    Returns:
      pdf:  [B,T]
    """
    diff = x - mean
    d2 = (diff * diff).sum(dim=-1)
    d = x.shape[-1]
    log_norm = -0.5 * d * math.log(2.0 * math.pi * var)
    return torch.exp(log_norm - 0.5 * d2 / var)


@dataclass
class ToyReward:
    horizon_T: int
    action_dim: int
    base_action_mean: float
    w_neg: float
    w_pos: float
    offset: float
    state_var: float

    # goal position in action_dim-space
    goal: Sequence[float]

    def __post_init__(self):
        if len(self.goal) != self.action_dim:
            raise ValueError(f"len(goal) must equal action_dim, got {len(self.goal)} vs {self.action_dim}")

    @property
    def goal_x(self) -> float:
        return float(self.goal[0])

    @property
    def goal_y(self) -> float:
        return float(self.goal[1]) if self.action_dim >= 2 else float("nan")

    def actions_to_states(self, tau0: torch.Tensor) -> torch.Tensor:
        """
        Deterministic dynamics: s_t = sum_{i<=t} a_i, s_0=0.
        tau0: [B, D] with D=T*action_dim, or [B,T,action_dim]
        returns: [B,T,action_dim]
        """
        if tau0.dim() == 2:
            B, D = tau0.shape
            assert D == self.horizon_T * self.action_dim
            a = tau0.view(B, self.horizon_T, self.action_dim)
        else:
            a = tau0
            assert a.shape[1] == self.horizon_T and a.shape[2] == self.action_dim
        return torch.cumsum(a, dim=1)

    def per_t_reward(self, tau0: torch.Tensor) -> torch.Tensor:
        s = self.actions_to_states(tau0)
        m_pos = torch.tensor(list(self.goal), device=s.device, dtype=s.dtype)
        m_neg = -m_pos

        pdf_neg = normal_pdf_isotropic(s, m_neg.view(1, 1, self.action_dim), var=self.state_var)
        pdf_pos = normal_pdf_isotropic(s, m_pos.view(1, 1, self.action_dim), var=self.state_var)
        return self.w_neg * pdf_neg + self.w_pos * pdf_pos + self.offset

    def total_reward(self, tau0: torch.Tensor) -> torch.Tensor:
        return self.per_t_reward(tau0).sum(dim=1)

    def final_state(self, tau0: torch.Tensor) -> torch.Tensor:
        s = self.actions_to_states(tau0)
        return s[:, -1, :]
