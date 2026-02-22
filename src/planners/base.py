from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.models.diffusion import GuidedReverseSampler
from src.models.tdp import ClosedFormTDP


class Planner(Protocol):
    """Common sampling interface for pluggable planners."""

    name: str

    def sample(
        self,
        n: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ...


@dataclass
class ReverseDiffusionPlanner:
    """Adapter that exposes GuidedReverseSampler through the Planner interface."""

    name: str
    sampler: GuidedReverseSampler
    guided: bool

    def sample(
        self,
        n: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.sampler.sample(
            n=n,
            batch_size=batch_size,
            guided=self.guided,
            device=device,
            dtype=dtype,
        )


@dataclass
class TDPPlanner:
    """Tree-guided planner wrapper for ClosedFormTDP."""

    name: str
    tdp: ClosedFormTDP
    n_roots: int
    topk_final: int
    guided: bool = True

    def sample(
        self,
        n: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        outs = []
        remaining = n
        while remaining > 0:
            # Batch TDP rollouts so we avoid Python-looping one elite at a time.
            rollout_chunk = max(1, min(remaining, int(batch_size) // max(1, int(self.n_roots))))
            tau0 = self.tdp.sample_many(
                n_rollouts=rollout_chunk,
                n_roots=self.n_roots,
                topk_final=self.topk_final,
                guided=self.guided,
                device=device,
                dtype=dtype,
            )
            take = min(remaining, tau0.shape[0])
            outs.append(tau0[:take])
            remaining -= take
        return torch.cat(outs, dim=0)
