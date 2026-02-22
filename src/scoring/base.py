from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.tasks.reward import ToyReward


class ScoreModel(Protocol):
    """Minimal scoring interface for planner guidance and evaluation."""

    def score(self, tau0: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class RewardScoreModel:
    reward: ToyReward

    def score(self, tau0: torch.Tensor) -> torch.Tensor:
        return self.reward.total_reward(tau0)
