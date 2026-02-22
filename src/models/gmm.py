from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


def _log_normal_diag(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    Log pdf of N(mean, var*I) evaluated at x.
    Shapes:
      x:    [B, D]
      mean: [K, D] or [B, K, D]
      var:  scalar tensor or [K] or [B,K] (broadcastable)
    Returns:
      logp: [B, K] if mean is [K,D] or [B,K,D] respectively.
    """
    if mean.dim() == 2:
        mean_b = mean.unsqueeze(0)  # [1,K,D]
    else:
        mean_b = mean
    x_b = x.unsqueeze(1)  # [B,1,D]
    diff2 = (x_b - mean_b).pow(2).sum(dim=-1)  # [B,K]
    D = x.shape[-1]
    log_det = D * torch.log(var)  # [B,K] or scalar
    return -0.5 * (diff2 / var + log_det + D * math.log(2.0 * math.pi))


@dataclass
class IsotropicGMM:
    """
    Base distribution p(tau0) = sum_k pi_k N(mu_k, sigma0_sq I).
    Implements diffusion posteriors in closed form (isotropic).
    """
    pi: torch.Tensor  # [K]
    mu: torch.Tensor  # [K, D]
    sigma0_sq: float

    def __post_init__(self):
        assert self.pi.dim() == 1
        assert self.mu.dim() == 2
        assert self.pi.shape[0] == self.mu.shape[0]
        self.K = self.pi.shape[0]
        self.D = self.mu.shape[1]

    def sample_tau0(self, n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        pi = self.pi.to(device=device, dtype=dtype)
        mu = self.mu.to(device=device, dtype=dtype)
        k = torch.distributions.Categorical(probs=pi).sample((n,))
        means = mu[k]
        eps = torch.randn(n, self.D, device=device, dtype=dtype)
        return means + math.sqrt(self.sigma0_sq) * eps

    def sample_tau_t_marginal(
        self, n: int, t_idx: int, alpha_bar_t: torch.Tensor, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Sample tau_t from q(tau_t) induced by tau0~p and q(tau_t|tau0).
        Component: N(sqrt(alpha_bar_t) mu_k, c_t I),
          c_t = alpha_bar_t*sigma0_sq + (1-alpha_bar_t).
        """
        assert alpha_bar_t.ndim == 0
        pi = self.pi.to(device=device, dtype=dtype)
        mu0 = self.mu.to(device=device, dtype=dtype)
        k = torch.distributions.Categorical(probs=pi).sample((n,))
        m = torch.sqrt(alpha_bar_t) * mu0[k]
        c_t = alpha_bar_t * self.sigma0_sq + (1.0 - alpha_bar_t)
        eps = torch.randn(n, self.D, device=device, dtype=dtype)
        return m + torch.sqrt(c_t) * eps

    def responsibilities(self, tau_t: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        device = tau_t.device
        dtype = tau_t.dtype
        pi = self.pi.to(device=device, dtype=dtype)
        mu0 = self.mu.to(device=device, dtype=dtype)

        c_t = alpha_bar_t * self.sigma0_sq + (1.0 - alpha_bar_t)  # scalar
        m_kt = torch.sqrt(alpha_bar_t) * mu0  # [K,D]
        logp = _log_normal_diag(tau_t, m_kt, c_t) + torch.log(pi.clamp_min(1e-30)).unsqueeze(0)
        logZ = torch.logsumexp(logp, dim=1, keepdim=True)
        return torch.exp(logp - logZ)

    def posterior_tau0_given_taut(self, tau_t: torch.Tensor, alpha_bar_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        q(tau0 | tau_t) = sum_k gamma_k(tau_t) N(m0|t^k(tau_t), s0t I)
        Returns:
          gamma: [B,K]
          m0:    [B,K,D]
          s0t:   scalar tensor
        """
        device = tau_t.device
        dtype = tau_t.dtype
        mu0 = self.mu.to(device=device, dtype=dtype)

        gamma = self.responsibilities(tau_t, alpha_bar_t)

        one_minus = (1.0 - alpha_bar_t).clamp_min(1e-12)
        inv = (1.0 / self.sigma0_sq) + (alpha_bar_t / one_minus)
        s0t = (1.0 / inv)

        term1 = (1.0 / self.sigma0_sq) * mu0  # [K,D]
        term2 = (torch.sqrt(alpha_bar_t) / one_minus) * tau_t  # [B,D]
        m0 = s0t * (term1.unsqueeze(0) + term2.unsqueeze(1))
        return gamma, m0, s0t

    def posterior_mean_tau0(self, tau_t: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        gamma, m0, _ = self.posterior_tau0_given_taut(tau_t, alpha_bar_t)
        return (gamma.unsqueeze(-1) * m0).sum(dim=1)

    def reverse_kernel_params(
        self,
        tau_t: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        alpha_bar_prev: torch.Tensor,
        beta_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        q(tau_{t-1} | tau_t) = sum_k gamma_k(tau_t) N(mu_k(tau_t), var I)
        where:
          mu_k = A_t * m0|t^k + B_t * tau_t
          var  = tilde_beta_t + A_t^2 * s0t
        All scalars are 0-dim tensors.
        """
        gamma, m0, s0t = self.posterior_tau0_given_taut(tau_t, alpha_bar_t)

        one_minus = (1.0 - alpha_bar_t).clamp_min(1e-12)
        A_t = (torch.sqrt(alpha_bar_prev) * beta_t) / one_minus
        B_t = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / one_minus

        mu_k = A_t * m0 + B_t * tau_t.unsqueeze(1)

        tilde_beta_t = ((1.0 - alpha_bar_prev) / one_minus) * beta_t
        var = tilde_beta_t + (A_t * A_t) * s0t
        return gamma, mu_k, var
