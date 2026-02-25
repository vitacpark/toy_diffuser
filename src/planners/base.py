from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Protocol

import torch

from src.models.diffusion import GuidedReverseSampler
from src.models.driftlitelite import DriftLiteLiteSampler
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
    reward_fn: Callable[[torch.Tensor], torch.Tensor]
    n_candidates: int = 1

    def sample(
        self,
        n: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        n_candidates = int(self.n_candidates)
        if n_candidates <= 0:
            raise ValueError(f"n_candidates must be positive, got {n_candidates}")

        if n_candidates == 1:
            return self.sampler.sample(
                n=n,
                batch_size=batch_size,
                guided=self.guided,
                device=device,
                dtype=dtype,
            )

        outs = []
        remaining = n
        while remaining > 0:
            # Each rollout keeps exactly one best candidate (top-1).
            rollout_chunk = max(1, min(remaining, int(batch_size) // max(1, n_candidates)))
            n_cand = rollout_chunk * n_candidates
            tau = self.sampler.sample(
                n=n_cand,
                batch_size=max(1, min(int(batch_size), n_cand)),
                guided=self.guided,
                device=device,
                dtype=dtype,
            )
            dim = tau.shape[1]
            tau = tau.view(rollout_chunk, n_candidates, dim)
            with torch.no_grad():
                score = self.reward_fn(tau.reshape(rollout_chunk * n_candidates, dim)).view(rollout_chunk, n_candidates)
            best_idx = torch.argmax(score, dim=1)
            gather_idx = best_idx.view(rollout_chunk, 1, 1).expand(rollout_chunk, 1, dim)
            kept = tau.gather(dim=1, index=gather_idx).reshape(rollout_chunk, dim)

            take = min(remaining, kept.shape[0])
            outs.append(kept[:take])
            remaining -= take

        return torch.cat(outs, dim=0)


@dataclass
class TDPPlanner:
    """Tree-guided planner wrapper for ClosedFormTDP."""

    name: str
    tdp: ClosedFormTDP
    n_roots: int

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
                device=device,
                dtype=dtype,
            )
            take = min(remaining, tau0.shape[0])
            outs.append(tau0[:take])
            remaining -= take
        return torch.cat(outs, dim=0)


@dataclass
class DriftLiteLitePlanner:
    """Planner wrapper for DriftLiteLiteSampler."""

    name: str
    sampler: DriftLiteLiteSampler
    n_particles: int
    batch_particles: int | None = None
    rollout_batch: int | None = None
    return_diagnostics: bool = False
    last_diagnostics: Dict[str, torch.Tensor] | None = None

    def sample(
        self,
        n: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.rollout_batch is not None and int(self.rollout_batch) > 0:
            batch_samples = min(int(n), int(self.rollout_batch))
        else:
            batch_samples = max(1, min(int(n), int(batch_size) // max(1, int(self.n_particles))))
        chunk = self.batch_particles
        if chunk is None or int(chunk) <= 0:
            chunk = min(int(batch_size), int(self.n_particles) * int(batch_samples))
        chunk = max(1, int(chunk))
        out = self.sampler.sample(
            n_samples=n,
            n_particles=int(self.n_particles),
            batch_particles=chunk,
            batch_samples=int(batch_samples),
            device=device,
            dtype=dtype,
            return_diagnostics=bool(self.return_diagnostics),
        )
        if self.return_diagnostics:
            tau0, diag = out
            self.last_diagnostics = {
                key: (val.detach().cpu() if isinstance(val, torch.Tensor) else torch.as_tensor(val).cpu())
                for key, val in diag.items()
            }
            return tau0
        self.last_diagnostics = None
        return out
