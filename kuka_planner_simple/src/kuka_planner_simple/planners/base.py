from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass
class EpisodeResult:
    reward: float
    success: bool
    elapsed_sec: float
    artifacts: Dict[str, Any] = field(default_factory=dict)


class PlannerRunner(Protocol):
    name: str

    def run_episode(self, episode_idx: int, seed: int) -> EpisodeResult:
        ...
