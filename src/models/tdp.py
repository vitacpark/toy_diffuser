from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

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
      5) evaluate 2B candidates (parents + children), keep top-1
    """

    gmm: IsotropicGMM
    schedule: DiffusionSchedule
    reward_fn: Callable[[torch.Tensor], torch.Tensor]
    horizon_T: int = 32
    action_dim: int = 2
    renoise_frac: float = 0.15
    guidance_scale: float = 10.0
    clip_norm: Optional[float] = 1.0
    pg: bool = False
    pg_scale: float = 1.0

    def estimate_transitions(self, n_roots: int) -> int:
        k = max(1, int(round(self.renoise_frac * self.schedule.n_steps)))
        return int(n_roots) * (self.schedule.n_steps + k)

    # Backward-compatible alias for earlier naming.
    def transitions_per_tree(self, n_roots: int) -> int:
        return self.estimate_transitions(n_roots)

    def sample(
        self,
        n_roots: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self.sample_many(
            n_rollouts=1,
            n_roots=n_roots,
            device=device,
            dtype=dtype,
        )

    def sample_many(
        self,
        n_rollouts: int,
        n_roots: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
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

        keep_idx = torch.topk(scores, k=1, dim=1, largest=True).indices  # [R, 1]
        gather_idx = keep_idx.unsqueeze(-1).expand(n_rollouts, 1, dim)
        chosen = candidates.gather(dim=1, index=gather_idx)  # [R, 1, D]
        return chosen.reshape(n_rollouts, dim)

    def sample_many_with_trace(
        self,
        n_rollouts: int,
        n_roots: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        max_frames: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Debug sampler that records parent/child denoising traces."""
        if n_rollouts <= 0:
            raise ValueError(f"n_rollouts must be positive, got {n_rollouts}")
        if n_roots <= 0:
            raise ValueError(f"n_roots must be positive, got {n_roots}")

        total_parents = int(n_rollouts) * int(n_roots)
        dim = self.gmm.D

        parent_trace = []
        tau_t = torch.randn(total_parents, dim, device=device, dtype=dtype)
        parent_trace.append(tau_t.detach().clone())  # x_T
        for t_idx in range(self.schedule.n_steps, 0, -1):
            tau_t = self._reverse_step(tau_t=tau_t, t_idx=t_idx, pg=bool(self.pg))
            parent_trace.append(tau_t.detach().clone())  # x_{t-1}
        parents = tau_t

        if dim != self.horizon_T * self.action_dim:
            raise ValueError(
                f"Parent dim mismatch: got {dim}, expected horizon_T*action_dim={self.horizon_T*self.action_dim}"
            )
        if not 0.0 <= self.renoise_frac <= 1.0:
            raise ValueError(f"tdp.renoise_frac must be in [0,1], got {self.renoise_frac}")

        k = max(1, int(round(self.renoise_frac * self.schedule.n_steps)))
        k = min(k, self.schedule.n_steps)

        u = torch.randint(low=0, high=self.horizon_T, size=(total_parents,), device=device)
        base = (u * self.action_dim).unsqueeze(1)
        offsets = torch.arange(self.action_dim, device=device).unsqueeze(0)
        idx = base + offsets

        mask = torch.zeros(total_parents, dim, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=idx, src=torch.ones_like(idx, dtype=torch.bool, device=device))

        eps_ref = torch.randn_like(parents, dtype=dtype, device=device)
        tau_t = self._forward_noise(parent=parents, t_idx=k, eps_ref=eps_ref)
        child_renoise_trace = [tau_t.detach().clone()]  # x_k
        child_denoise_trace = []
        for t_idx in range(k, 0, -1):
            tau_t = self._reverse_step(tau_t=tau_t, t_idx=t_idx, pg=False)
            known_prev = self._forward_noise(parent=parents, t_idx=t_idx - 1, eps_ref=eps_ref)
            tau_t = torch.where(mask, known_prev, tau_t)
            child_denoise_trace.append(tau_t.detach().clone())  # x_{t-1}
        children = tau_t

        parents_r = parents.view(n_rollouts, n_roots, dim)
        children_r = children.view(n_rollouts, n_roots, dim)
        candidates = torch.cat([parents_r, children_r], dim=1)  # [R, 2B, D]
        flat = candidates.reshape(n_rollouts * (2 * n_roots), dim)
        scores = self.reward_fn(flat).detach().view(n_rollouts, 2 * n_roots)
        keep_idx = torch.topk(scores, k=1, dim=1, largest=True).indices
        gather_idx = keep_idx.unsqueeze(-1).expand(n_rollouts, 1, dim)
        chosen = candidates.gather(dim=1, index=gather_idx).reshape(n_rollouts, dim)

        def _stack_and_trim(frames):
            stacked = torch.stack(frames, dim=0)
            if int(max_frames) > 0 and stacked.shape[0] > int(max_frames):
                pick = torch.linspace(0, stacked.shape[0] - 1, int(max_frames), device=stacked.device)
                pick = torch.round(pick).to(torch.long).unique(sorted=True)
                stacked = stacked.index_select(dim=0, index=pick)
            return stacked

        out: Dict[str, torch.Tensor] = {
            "chosen": chosen,
            "parents_final": parents,
            "children_final": children,
            "u": u.detach().clone(),
            "mask": mask.detach().clone(),
            "k_renoise": torch.tensor(k, device=device, dtype=torch.int64),
            "parent_trace": _stack_and_trim(parent_trace),
            "child_renoise_trace": _stack_and_trim(child_renoise_trace),
            "child_denoise_trace": _stack_and_trim(child_denoise_trace),
        }
        return out

    def _reverse_step(self, tau_t: torch.Tensor, t_idx: int, pg: bool = False) -> torch.Tensor:
        beta_t, alpha_t, alpha_bar_t, alpha_bar_prev = self.schedule.scalars(t_idx)
        gamma, mu_k, var = self.gmm.reverse_kernel_params(tau_t, alpha_t, alpha_bar_t, alpha_bar_prev, beta_t)

        if pg and tau_t.shape[0] > 2:
            # Particle-guidance repulsion term (kernel gradient) adapted from TDP diffusion pg branch.
            diff = tau_t.unsqueeze(1) - tau_t.unsqueeze(0)  # [B, B, D]
            eye = torch.eye(diff.shape[0], device=tau_t.device, dtype=torch.bool)
            diff = diff[~eye].reshape(diff.shape[0], diff.shape[0] - 1, diff.shape[-1])  # [B, B-1, D]

            distance = torch.linalg.vector_norm(diff, ord=2, dim=-1, keepdim=True)  # [B, B-1, 1]
            denom = max(1e-6, math.log(float(diff.shape[0] - 1)))
            h_t = (distance.median(dim=1, keepdim=True).values ** 2) / denom
            h_t = h_t.clamp_min(1e-12)
            weights = torch.exp(-((distance ** 2) / h_t))
            pg_grad = (2.0 * weights * diff / h_t) * (var.view(-1, 1, 1) * float(self.pg_scale))
            pg_grad = pg_grad.sum(dim=1)  # [B, D]
            mu_k = mu_k + pg_grad.unsqueeze(1)

        if self.guidance_scale != 0.0:
            with torch.enable_grad():
                tau_req = tau_t.detach().requires_grad_(True)
                mu0_hat = self.gmm.posterior_mean_tau0(tau_req, alpha_bar_t)
                j_val = self.reward_fn(mu0_hat)
                grad = torch.autograd.grad(j_val.sum(), tau_req, retain_graph=False, create_graph=False)[0]

            if self.clip_norm is not None and self.clip_norm > 0:
                gnorm = torch.linalg.vector_norm(grad, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
                scale = (self.clip_norm / gnorm).clamp_max(1.0)
                grad = grad * scale

            shift = (self.guidance_scale * var) * grad
            mu_k = mu_k + shift.unsqueeze(1)

        bsz, _, dim = mu_k.shape
        k_idx = torch.multinomial(gamma, num_samples=1).squeeze(1)
        gather_idx = k_idx.view(bsz, 1, 1).expand(bsz, 1, dim)
        mu = mu_k.gather(dim=1, index=gather_idx).squeeze(1)
        return mu + torch.sqrt(var) * torch.randn_like(mu)

    def _sample_parents(self, n_roots: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tau_t = torch.randn(n_roots, self.gmm.D, device=device, dtype=dtype)
        for t_idx in range(self.schedule.n_steps, 0, -1):
            tau_t = self._reverse_step(tau_t=tau_t, t_idx=t_idx, pg=bool(self.pg))
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
            tau_t = self._reverse_step(tau_t=tau_t, t_idx=t_idx, pg=False)
            known_prev = self._forward_noise(parent=parents, t_idx=t_idx - 1, eps_ref=eps_ref)
            tau_t = torch.where(mask, known_prev, tau_t)

        return tau_t

    def _forward_noise(self, parent: torch.Tensor, t_idx: int, eps_ref: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.schedule.alpha_bars[t_idx]
        return torch.sqrt(alpha_bar_t) * parent + torch.sqrt(1.0 - alpha_bar_t) * eps_ref
