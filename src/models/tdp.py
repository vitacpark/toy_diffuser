from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .diffusion import DiffusionSchedule
from .gmm import IsotropicGMM


@dataclass
class ClosedFormTDP:
    """
    Parent-child mutate-select planner:
      1) sample B parents (tau0)
      2) for each parent, extract one random timestep index u ~ Uniform[0, horizon_T)
      3) re-noise with fixed depth = renoise_frac * total diffusion steps
      4) reverse denoise while overwriting extracted timestep each step
      5) evaluate 2B candidates (parents + children), keep top-k
    """

    gmm: IsotropicGMM
    schedule: DiffusionSchedule
    reward_fn: Callable[[torch.Tensor], torch.Tensor]
    horizon_T: int = 32
    action_dim: int = 2
    renoise_frac: float = 0.15

    def estimate_transitions(self, n_roots: int) -> int:
        k = max(1, int(round(self.renoise_frac * self.schedule.n_steps)))
        return int(n_roots) * (self.schedule.n_steps + k)

    # Backward-compatible alias for earlier naming.
    def transitions_per_tree(self, n_roots: int) -> int:
        return self.estimate_transitions(n_roots)

    def sample(
        self,
        n_roots: int,
        topk_final: int,
        guided: bool,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self.sample_many(
            n_rollouts=1,
            n_roots=n_roots,
            topk_final=topk_final,
            guided=guided,
            device=device,
            dtype=dtype,
        )

    def sample_many(
        self,
        n_rollouts: int,
        n_roots: int,
        topk_final: int,
        guided: bool,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        _ = guided  # not used in current TDP variant
        if n_rollouts <= 0:
            raise ValueError(f"n_rollouts must be positive, got {n_rollouts}")
        if n_roots <= 0:
            raise ValueError(f"n_roots must be positive, got {n_roots}")

        total_parents = int(n_rollouts) * int(n_roots)
        parents = self._sample_parents(n_roots=total_parents, device=device, dtype=dtype)
        children = self._sample_children_from_parents(parents=parents, device=device, dtype=dtype)

        dim = parents.shape[1]
        parents = parents.view(n_rollouts, n_roots, dim)
        children = children.view(n_rollouts, n_roots, dim)
        candidates = torch.cat([parents, children], dim=1)  # [R, 2B, D]

        flat = candidates.reshape(n_rollouts * (2 * n_roots), dim)
        scores = self.reward_fn(flat).detach().view(n_rollouts, 2 * n_roots)

        k = min(max(1, int(topk_final)), 2 * n_roots)
        keep_idx = torch.topk(scores, k=k, dim=1, largest=True).indices  # [R, k]
        gather_idx = keep_idx.unsqueeze(-1).expand(n_rollouts, k, dim)
        chosen = candidates.gather(dim=1, index=gather_idx)  # [R, k, D]
        return chosen.reshape(n_rollouts * k, dim)

    def _reverse_step(self, tau_t: torch.Tensor, t_idx: int) -> torch.Tensor:
        beta_t, alpha_t, alpha_bar_t, alpha_bar_prev = self.schedule.scalars(t_idx)
        gamma, mu_k, var = self.gmm.reverse_kernel_params(tau_t, alpha_t, alpha_bar_t, alpha_bar_prev, beta_t)
        bsz, _, dim = mu_k.shape
        k_idx = torch.multinomial(gamma, num_samples=1).squeeze(1)
        gather_idx = k_idx.view(bsz, 1, 1).expand(bsz, 1, dim)
        mu = mu_k.gather(dim=1, index=gather_idx).squeeze(1)
        return mu + torch.sqrt(var) * torch.randn_like(mu)

    def _sample_parents(self, n_roots: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tau_t = torch.randn(n_roots, self.gmm.D, device=device, dtype=dtype)
        for t_idx in range(self.schedule.n_steps, 0, -1):
            tau_t = self._reverse_step(tau_t=tau_t, t_idx=t_idx)
        return tau_t

    def _sample_children_from_parents(self, parents: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n_parent = parents.shape[0]
        dim = parents.shape[1]
        if dim != self.horizon_T * self.action_dim:
            raise ValueError(
                f"Parent dim mismatch: got {dim}, expected horizon_T*action_dim={self.horizon_T*self.action_dim}"
            )

        if not 0.0 <= self.renoise_frac <= 1.0:
            raise ValueError(f"tdp.renoise_frac must be in [0,1], got {self.renoise_frac}")
        k = max(1, int(round(self.renoise_frac * self.schedule.n_steps)))
        k = min(k, self.schedule.n_steps)

        # Extract index u ~ Uniform[0, horizon_T), then map to action block in flattened tau.
        u = torch.randint(low=0, high=self.horizon_T, size=(n_parent,), device=device)
        base = (u * self.action_dim).unsqueeze(1)  # [B,1]
        offsets = torch.arange(self.action_dim, device=device).unsqueeze(0)  # [1,A]
        idx = base + offsets  # [B,A]

        mask = torch.zeros(n_parent, dim, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=idx, src=torch.ones_like(idx, dtype=torch.bool, device=device))

        eps_ref = torch.randn_like(parents, dtype=dtype, device=device)
        tau_t = self._forward_noise(parent=parents, t_idx=k, eps_ref=eps_ref)

        for t_idx in range(k, 0, -1):
            tau_t = self._reverse_step(tau_t=tau_t, t_idx=t_idx)
            known_prev = self._forward_noise(parent=parents, t_idx=t_idx - 1, eps_ref=eps_ref)
            tau_t = torch.where(mask, known_prev, tau_t)

        return tau_t

    def _forward_noise(self, parent: torch.Tensor, t_idx: int, eps_ref: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.schedule.alpha_bars[t_idx]
        return torch.sqrt(alpha_bar_t) * parent + torch.sqrt(1.0 - alpha_bar_t) * eps_ref
