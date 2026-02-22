from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .gmm import IsotropicGMM


@dataclass
class DiffusionSchedule:
    """
    DDPM schedule scalars stored as tensors.
    Indexing:
      t_idx in [1..n_steps]
      arrays length n_steps+1, with alpha_bars[0]=1
    """
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @property
    def n_steps(self) -> int:
        return self.betas.shape[0] - 1

    @staticmethod
    def make_linear(n_steps: int, beta_start: float, beta_end: float, device: torch.device, dtype: torch.dtype) -> "DiffusionSchedule":
        betas = torch.zeros(n_steps + 1, device=device, dtype=dtype)
        betas[1:] = torch.linspace(beta_start, beta_end, n_steps, device=device, dtype=dtype)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars[0] = 1.0
        return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)

    def scalars(self, t_idx: int):
        beta_t = self.betas[t_idx]
        alpha_t = self.alphas[t_idx]
        alpha_bar_t = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx - 1]
        return beta_t, alpha_t, alpha_bar_t, alpha_bar_prev


@dataclass
class GuidedReverseSampler:
    """
    Exact reverse sampling for isotropic base-GMM, plus optional guidance.

    Guidance (per step):
      - compute mu0_hat(tau_t) = E[tau0 | tau_t]  (closed form)
      - compute J(mu0_hat) (here J = total reward R)
      - compute grad_{tau_t} J(mu0_hat(tau_t)) via autograd
      - shift each mixture component mean by: shift = guidance_scale * var_t * grad

    This matches the "J(mu) 기준" requirement: J is evaluated at mu0_hat.
    """
    gmm: IsotropicGMM
    schedule: DiffusionSchedule
    reward_fn: callable
    guidance_scale: float = 0.0
    clip_norm: Optional[float] = None

    def sample(
        self,
        n: int,
        batch_size: int = 1024,
        guided: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outs = []
        remaining = n
        while remaining > 0:
            b = min(batch_size, remaining)
            outs.append(self._sample_batch(b, guided=guided, device=device, dtype=dtype))
            remaining -= b
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def _sample_batch(self, b: int, guided: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        alpha_bar_T = self.schedule.alpha_bars[self.schedule.n_steps]
        tau_t = torch.randn(b, self.gmm.D, device=device, dtype=dtype)

        for t_idx in range(self.schedule.n_steps, 0, -1):
            tau_t = self._one_step(tau_t, t_idx=t_idx, guided=guided)
        return tau_t

    def _one_step(self, tau_t: torch.Tensor, t_idx: int, guided: bool) -> torch.Tensor:
        beta_t, alpha_t, alpha_bar_t, alpha_bar_prev = self.schedule.scalars(t_idx)
        gamma, mu_k, var = self.gmm.reverse_kernel_params(tau_t, alpha_t, alpha_bar_t, alpha_bar_prev, beta_t)

        if guided and self.guidance_scale != 0.0:
            # _sample_batch runs under torch.no_grad(); re-enable gradients locally.
            with torch.enable_grad():
                tau_req = tau_t.detach().requires_grad_(True)
                mu0_hat = self.gmm.posterior_mean_tau0(tau_req, alpha_bar_t)
                J = self.reward_fn(mu0_hat)  # [B]
                grad = torch.autograd.grad(J.sum(), tau_req, retain_graph=False, create_graph=False)[0]

            if self.clip_norm is not None and self.clip_norm > 0:
                gnorm = torch.linalg.vector_norm(grad, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
                scale = (self.clip_norm / gnorm).clamp_max(1.0)
                grad = grad * scale

            shift = (self.guidance_scale * var) * grad  # [B,D]
            mu_k = mu_k + shift.unsqueeze(1)

        with torch.no_grad():
            k = torch.multinomial(gamma, num_samples=1).squeeze(1)  # [B]
            B, K, D = mu_k.shape
            idx = k.view(B, 1, 1).expand(B, 1, D)
            mu = mu_k.gather(dim=1, index=idx).squeeze(1)
            eps = torch.randn_like(mu)
            return mu + torch.sqrt(var) * eps
