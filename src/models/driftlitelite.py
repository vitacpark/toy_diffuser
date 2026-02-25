from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import torch

from .diffusion import DiffusionSchedule
from .gmm import IsotropicGMM


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    return 1.0 / (weights.pow(2).sum().clamp_min(1e-12))


def systematic_resample_particles(
    particles: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = particles.shape[0]
    cdf = torch.cumsum(weights, dim=0)
    cdf = torch.clamp(cdf, max=1.0)

    u0 = torch.rand(1, device=particles.device, dtype=weights.dtype) / float(n)
    u = u0 + (torch.arange(n, device=particles.device, dtype=weights.dtype) / float(n))
    idx = torch.searchsorted(cdf, u, right=False).clamp(max=n - 1)

    out_particles = particles[idx]
    out_weights = torch.full_like(weights, 1.0 / float(n))
    out_logw = torch.full_like(weights, -math.log(float(n)))
    return out_particles, out_weights, out_logw, idx


def hutchinson_divergence(
    x: torch.Tensor,
    vecfield: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    n_hutch = max(1, int(n_samples))
    out = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(n_hutch):
        z = torch.randn_like(x)
        dot = (vecfield * z).sum()
        grad = torch.autograd.grad(dot, x, retain_graph=(i + 1 < n_hutch), create_graph=False)[0]
        out = out + (grad * z).sum(dim=1)
    return out / float(n_hutch)


@dataclass
class VCGController:
    basis_names: List[str]
    divergence_mode: str = "grad_r_hutch"  # none | grad_r_hutch | all_hutch
    hutch_samples: int = 1
    reg_lambda: float = 1e-4

    def requires_higher_order(self) -> bool:
        if self.divergence_mode == "grad_r_hutch":
            return "grad_r" in self.basis_names
        return self.divergence_mode == "all_hutch"

    def compute_divergence_terms(
        self,
        x: torch.Tensor,
        basis_map: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        n = x.shape[0]
        m = len(self.basis_names)
        div = torch.zeros(n, m, device=x.device, dtype=x.dtype)

        if self.divergence_mode == "none":
            return div

        if self.divergence_mode == "grad_r_hutch":
            for i, name in enumerate(self.basis_names):
                if name == "grad_r":
                    div[:, i] = hutchinson_divergence(x, basis_map[name], self.hutch_samples)
            return div

        if self.divergence_mode == "all_hutch":
            for i, name in enumerate(self.basis_names):
                div[:, i] = hutchinson_divergence(x, basis_map[name], self.hutch_samples)
            return div

        raise ValueError(f"Unsupported divergence_mode: {self.divergence_mode}")

    def solve_theta(self, H: torch.Tensor, g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        m = H.shape[1]
        A = H.transpose(0, 1) @ (w.unsqueeze(1) * H)
        A = A + self.reg_lambda * torch.eye(m, device=H.device, dtype=H.dtype)
        c = -(H.transpose(0, 1) @ (w * g))

        try:
            theta = torch.linalg.solve(A, c)
        except RuntimeError:
            theta = torch.linalg.pinv(A) @ c
        return theta

    def solve_theta_batched(self, H: torch.Tensor, g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # H: [R, N, M], g/w: [R, N]
        R, _, M = H.shape
        weighted_H = w.unsqueeze(-1) * H
        A = torch.matmul(H.transpose(1, 2), weighted_H)  # [R, M, M]
        eye = torch.eye(M, device=H.device, dtype=H.dtype).unsqueeze(0).expand(R, M, M)
        A = A + self.reg_lambda * eye
        c = -torch.matmul(H.transpose(1, 2), (w * g).unsqueeze(-1))  # [R, M, 1]
        try:
            theta = torch.linalg.solve(A, c).squeeze(-1)
        except RuntimeError:
            theta = torch.matmul(torch.linalg.pinv(A), c).squeeze(-1)
        return theta


@dataclass
class DriftLiteLiteSampler:
    gmm: IsotropicGMM
    schedule: DiffusionSchedule
    reward_fn: Callable[[torch.Tensor], torch.Tensor]

    ess_threshold_ratio: float = 0.5
    resample: bool = True
    dt_mode: str = "auto"  # auto | fixed
    dt: float = 0.0
    ctrl_scale: float = 1.0
    basis: List[str] | None = None
    proxy_gamma: float = 1.0
    divergence_mode: str = "grad_r_hutch"
    hutch_samples: int = 1
    reward_path: str = "constant"  # constant | linear
    reward_scale: float = 1.0
    reg_lambda: float = 1e-4
    output_mode: str = "best"  # best | resampled | weighted
    seed: int | None = None

    def sample(
        self,
        n_samples: int,
        n_particles: int,
        batch_particles: int | None = None,
        batch_samples: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if n_particles <= 0:
            raise ValueError(f"n_particles must be positive, got {n_particles}")
        if self.hutch_samples <= 0:
            raise ValueError(f"hutch_samples must be positive, got {self.hutch_samples}")
        if self.output_mode not in {"best", "resampled", "weighted"}:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")
        if self.reward_path not in {"constant", "linear"}:
            raise ValueError(f"Unsupported reward_path: {self.reward_path}")

        if self.seed is not None:
            torch.manual_seed(int(self.seed))

        run_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        basis_names = list(self.basis) if self.basis is not None else ["grad_r", "score"]
        self._validate_basis(basis_names)

        controller = VCGController(
            basis_names=basis_names,
            divergence_mode=self.divergence_mode,
            hutch_samples=self.hutch_samples,
            reg_lambda=self.reg_lambda,
        )

        dt = self._resolve_dt()
        chunk = n_particles if (batch_particles is None or int(batch_particles) <= 0) else min(n_particles, int(batch_particles))

        if self.output_mode == "best":
            rollout_batch = n_samples if (batch_samples is None or int(batch_samples) <= 0) else int(batch_samples)
            rollout_batch = max(1, min(n_samples, rollout_batch))

            selected: list[torch.Tensor] = []
            diag_list: list[Dict[str, torch.Tensor]] = []
            diag_rollout_counts: list[int] = []
            best_rewards: list[float] = []
            run_seconds_chunks: list[float] = []
            remaining = n_samples
            done = 0

            while remaining > 0:
                b_rollouts = min(remaining, rollout_batch)
                x, _, diag, one_run_seconds = self._run_particle_system(
                    n_particles=n_particles,
                    n_rollouts=b_rollouts,
                    controller=controller,
                    basis_names=basis_names,
                    batch_particles=chunk,
                    dt=dt,
                    device=run_device,
                    dtype=dtype,
                    collect_diagnostics=return_diagnostics,
                )
                with torch.no_grad():
                    flat = x.reshape(b_rollouts * n_particles, self.gmm.D)
                    r = self.reward_fn(flat).view(b_rollouts, n_particles)
                    best_idx = torch.argmax(r, dim=1)
                    gather = best_idx.view(b_rollouts, 1, 1).expand(b_rollouts, 1, self.gmm.D)
                    best_tau = x.gather(dim=1, index=gather).squeeze(1)
                selected.append(best_tau)
                run_seconds_chunks.append(float(one_run_seconds))
                if return_diagnostics and diag is not None:
                    diag_list.append(diag)
                    diag_rollout_counts.append(int(b_rollouts))
                    best_rewards.extend(r.gather(dim=1, index=best_idx.unsqueeze(1)).squeeze(1).detach().cpu().tolist())

                done += b_rollouts
                remaining -= b_rollouts

            tau0 = torch.cat(selected, dim=0) if selected else torch.empty(0, self.gmm.D, device=run_device, dtype=dtype)
            if not return_diagnostics:
                return tau0
            return tau0, self._aggregate_best_mode_diagnostics(
                diag_list=diag_list,
                diag_rollout_counts=diag_rollout_counts,
                best_rewards=best_rewards,
                run_seconds_chunks=run_seconds_chunks,
            )

        x, w, diag, one_run_seconds = self._run_particle_system(
            n_particles=n_particles,
            n_rollouts=1,
            controller=controller,
            basis_names=basis_names,
            batch_particles=chunk,
            dt=dt,
            device=run_device,
            dtype=dtype,
            collect_diagnostics=return_diagnostics,
        )

        x_pool = x[0]
        w_pool = w[0]
        if self.output_mode == "weighted":
            idx = torch.multinomial(w_pool, num_samples=n_samples, replacement=True)
            tau0 = x_pool[idx]
        else:
            # resampled mode: treat final particles as empirical equally-weighted pool.
            if n_samples <= n_particles:
                perm = torch.randperm(n_particles, device=run_device)
                idx = perm[:n_samples]
            else:
                idx = torch.randint(0, n_particles, size=(n_samples,), device=run_device)
            tau0 = x_pool[idx]

        if not return_diagnostics:
            return tau0

        if diag is None:
            raise RuntimeError("Internal error: diagnostics requested but missing.")
        diag["run_seconds"] = torch.tensor(float(one_run_seconds), dtype=torch.float32)
        return tau0, diag

    def _run_particle_system(
        self,
        n_particles: int,
        n_rollouts: int,
        controller: VCGController,
        basis_names: Sequence[str],
        batch_particles: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype,
        collect_diagnostics: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor] | None, float]:
        run_start = time.perf_counter()
        x = torch.randn(n_rollouts, n_particles, self.gmm.D, device=device, dtype=dtype)
        logw = torch.full((n_rollouts, n_particles), -math.log(float(n_particles)), device=device, dtype=dtype)
        w = torch.softmax(logw, dim=1)

        ess_trace: list[float] = []
        theta_norm_trace: list[float] = []
        resample_steps: list[int] = []
        resample_events = 0
        logw_mean_trace: list[float] = []
        logw_std_trace: list[float] = []
        logw_min_trace: list[float] = []
        logw_max_trace: list[float] = []

        for t_idx in range(self.schedule.n_steps, 0, -1):
            control, phi, theta_norm = self._compute_control_and_potential_batched(
                x=x,
                w=w,
                t_idx=t_idx,
                controller=controller,
                basis_names=basis_names,
                batch_particles=batch_particles,
            )

            beta_t, alpha_t, alpha_bar_t, alpha_bar_prev = self.schedule.scalars(t_idx)
            flat_x = x.reshape(n_rollouts * n_particles, self.gmm.D)
            gamma, mu_k, var = self.gmm.reverse_kernel_params(flat_x, alpha_t, alpha_bar_t, alpha_bar_prev, beta_t)
            mu_k = mu_k + (self.ctrl_scale * dt) * control.reshape(n_rollouts * n_particles, self.gmm.D).unsqueeze(1)

            bsz, _, dim = mu_k.shape
            k_idx = torch.multinomial(gamma, num_samples=1).squeeze(1)
            gather_idx = k_idx.view(bsz, 1, 1).expand(bsz, 1, dim)
            mu = mu_k.gather(dim=1, index=gather_idx).squeeze(1)
            x_prev = (mu + torch.sqrt(var) * torch.randn_like(mu)).view(n_rollouts, n_particles, self.gmm.D)

            logw = logw + phi * dt
            w = torch.softmax(logw, dim=1)

            ess_vec = 1.0 / w.pow(2).sum(dim=1).clamp_min(1e-12)
            ess = float(ess_vec.mean().item())
            did_resample = False
            if self.resample:
                need = ess_vec < (self.ess_threshold_ratio * float(n_particles))
                if torch.any(need):
                    idxs = torch.nonzero(need, as_tuple=False).flatten()
                    resample_events += int(idxs.numel())
                    for ridx in idxs.tolist():
                        x_i, w_i, logw_i, _ = systematic_resample_particles(x_prev[ridx], w[ridx])
                        x_prev[ridx] = x_i
                        w[ridx] = w_i
                        logw[ridx] = logw_i
                    did_resample = True

            x = x_prev

            if collect_diagnostics:
                ess_trace.append(ess)
                theta_norm_trace.append(float(theta_norm))
                logw_mean_trace.append(float(logw.mean().item()))
                logw_std_trace.append(float(logw.std(unbiased=False).item()))
                logw_min_trace.append(float(logw.min().item()))
                logw_max_trace.append(float(logw.max().item()))
                if did_resample:
                    resample_steps.append(t_idx)
        elapsed_run = time.perf_counter() - run_start
        if not collect_diagnostics:
            return x, w, None, elapsed_run

        final_ess = 1.0 / w.pow(2).sum(dim=1).clamp_min(1e-12)
        diagnostics = {
            "ess_trace": torch.tensor(ess_trace, dtype=torch.float32),
            "theta_trace_norm": torch.tensor(theta_norm_trace, dtype=torch.float32),
            "resample_steps": torch.tensor(resample_steps, dtype=torch.int64),
            "logw_mean_trace": torch.tensor(logw_mean_trace, dtype=torch.float32),
            "logw_std_trace": torch.tensor(logw_std_trace, dtype=torch.float32),
            "logw_min_trace": torch.tensor(logw_min_trace, dtype=torch.float32),
            "logw_max_trace": torch.tensor(logw_max_trace, dtype=torch.float32),
            "final_ess": torch.tensor(float(final_ess.mean().item()), dtype=torch.float32),
            "final_ess_min": torch.tensor(float(final_ess.min().item()), dtype=torch.float32),
            "resample_count": torch.tensor(int(resample_events), dtype=torch.int64),
            "run_seconds": torch.tensor(float(elapsed_run), dtype=torch.float32),
            "rollout_count": torch.tensor(int(n_rollouts), dtype=torch.int64),
        }
        return x, w, diagnostics, elapsed_run

    @staticmethod
    def _aggregate_best_mode_diagnostics(
        diag_list: Sequence[Dict[str, torch.Tensor]],
        diag_rollout_counts: Sequence[int],
        best_rewards: Sequence[float],
        run_seconds_chunks: Sequence[float],
    ) -> Dict[str, torch.Tensor]:
        if len(diag_list) == 0:
            rewards = torch.tensor(best_rewards, dtype=torch.float32)
            runtime = torch.tensor(run_seconds_chunks, dtype=torch.float32)
            empty = torch.empty(0, dtype=torch.float32)
            return {
                "ess_trace": empty,
                "theta_trace_norm": empty,
                "resample_steps": torch.empty(0, dtype=torch.int64),
                "logw_mean_trace": empty,
                "logw_std_trace": empty,
                "logw_min_trace": empty,
                "logw_max_trace": empty,
                "final_ess": torch.tensor(float("nan"), dtype=torch.float32),
                "resample_count": torch.tensor(0, dtype=torch.int64),
                "rollout_count": torch.tensor(int(sum(diag_rollout_counts)), dtype=torch.int64),
                "best_reward_mean": torch.tensor(float(rewards.mean().item()) if rewards.numel() else float("nan"), dtype=torch.float32),
                "best_reward_std": torch.tensor(float(rewards.std(unbiased=False).item()) if rewards.numel() else float("nan"), dtype=torch.float32),
                "run_seconds_mean": torch.tensor(
                    float(runtime.sum().item() / max(1, int(sum(diag_rollout_counts)))) if runtime.numel() else float("nan"),
                    dtype=torch.float32,
                ),
                "run_seconds_total": torch.tensor(float(runtime.sum().item()) if runtime.numel() else float("nan"), dtype=torch.float32),
            }

        weights = torch.tensor(diag_rollout_counts, dtype=torch.float32)
        weight_sum = weights.sum().clamp_min(1.0)

        def _weighted_mean_stack(key: str) -> torch.Tensor:
            vals = torch.stack([d[key].to(torch.float32) for d in diag_list], dim=0)
            w = weights.view(-1, *([1] * (vals.dim() - 1)))
            return (vals * w).sum(dim=0) / weight_sum

        steps = [d["resample_steps"].to(torch.int64) for d in diag_list if d["resample_steps"].numel() > 0]
        if steps:
            resample_steps = torch.unique(torch.cat(steps, dim=0), sorted=True)
        else:
            resample_steps = torch.empty(0, dtype=torch.int64)

        rewards = torch.tensor(best_rewards, dtype=torch.float32)
        runtime = torch.tensor(run_seconds_chunks, dtype=torch.float32)
        rollout_total = int(sum(diag_rollout_counts))
        return {
            "ess_trace": _weighted_mean_stack("ess_trace"),
            "theta_trace_norm": _weighted_mean_stack("theta_trace_norm"),
            "resample_steps": resample_steps,
            "logw_mean_trace": _weighted_mean_stack("logw_mean_trace"),
            "logw_std_trace": _weighted_mean_stack("logw_std_trace"),
            "logw_min_trace": _weighted_mean_stack("logw_min_trace"),
            "logw_max_trace": _weighted_mean_stack("logw_max_trace"),
            "final_ess": _weighted_mean_stack("final_ess"),
            "final_ess_min": torch.stack([d["final_ess_min"].to(torch.float32) for d in diag_list], dim=0).min(),
            "resample_count": torch.stack([d["resample_count"].to(torch.int64) for d in diag_list], dim=0).sum(),
            "rollout_count": torch.tensor(rollout_total, dtype=torch.int64),
            "best_reward_mean": torch.tensor(float(rewards.mean().item()) if rewards.numel() else float("nan"), dtype=torch.float32),
            "best_reward_std": torch.tensor(float(rewards.std(unbiased=False).item()) if rewards.numel() else float("nan"), dtype=torch.float32),
            "run_seconds_mean": torch.tensor(float(runtime.sum().item() / max(1, rollout_total)), dtype=torch.float32),
            "run_seconds_total": torch.tensor(float(runtime.sum().item()) if runtime.numel() else float("nan"), dtype=torch.float32),
        }

    def _compute_control_and_potential_batched(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        t_idx: int,
        controller: VCGController,
        basis_names: Sequence[str],
        batch_particles: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        n_rollouts, n_particles, dim = x.shape
        flat_x = x.reshape(n_rollouts * n_particles, dim)
        S_parts = []
        H_parts = []
        r_parts = []

        n = flat_x.shape[0]
        for start in range(0, n, batch_particles):
            end = min(n, start + batch_particles)
            s_chunk, h_chunk, r_chunk = self._compute_chunk_terms(
                x_chunk=flat_x[start:end],
                t_idx=t_idx,
                controller=controller,
                basis_names=basis_names,
            )
            S_parts.append(s_chunk)
            H_parts.append(h_chunk)
            r_parts.append(r_chunk)

        m = len(basis_names)
        S = torch.cat(S_parts, dim=0).view(n_rollouts, n_particles, m, dim)
        H = torch.cat(H_parts, dim=0).view(n_rollouts, n_particles, m)
        r = torch.cat(r_parts, dim=0).view(n_rollouts, n_particles)

        g_raw = self._compute_reward_potential(r=r, t_idx=t_idx)
        g = g_raw - torch.sum(w * g_raw, dim=1, keepdim=True)

        theta = controller.solve_theta_batched(H=H, g=g, w=w)
        control = (S * theta[:, None, :, None]).sum(dim=2)
        h_b = (H * theta[:, None, :]).sum(dim=2)
        phi = g + h_b
        theta_norm = float(torch.linalg.vector_norm(theta, ord=2, dim=1).mean().item())
        return control.detach(), phi.detach(), theta_norm

    def _compute_chunk_terms(
        self,
        x_chunk: torch.Tensor,
        t_idx: int,
        controller: VCGController,
        basis_names: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_bar_t = self.schedule.alpha_bars[t_idx]

        with torch.enable_grad():
            x_req = x_chunk.detach().requires_grad_(True)
            mu0_hat = self.gmm.posterior_mean_tau0(x_req, alpha_bar_t)
            r = self.reward_fn(mu0_hat)
            grad_r = torch.autograd.grad(
                r.sum(),
                x_req,
                create_graph=controller.requires_higher_order(),
                retain_graph=controller.requires_higher_order(),
            )[0]
            score_hat = self._score_hat(x_req, mu0_hat, alpha_bar_t)

            basis_map: Dict[str, torch.Tensor] = {
                "grad_r": grad_r,
                "score": score_hat,
                "x": x_req,
            }
            S = torch.stack([basis_map[name] for name in basis_names], dim=1)

            grad_logq_proxy = self.proxy_gamma * score_hat + grad_r
            div = controller.compute_divergence_terms(x_req, basis_map)
            H = (grad_logq_proxy.unsqueeze(1) * S).sum(dim=2) + div

        return S.detach(), H.detach(), r.detach()

    def _score_hat(self, x_t: torch.Tensor, mu0_hat: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        one_minus = (1.0 - alpha_bar_t).clamp_min(1e-12)
        return -(x_t - torch.sqrt(alpha_bar_t) * mu0_hat) / one_minus

    def _compute_reward_potential(self, r: torch.Tensor, t_idx: int) -> torch.Tensor:
        if self.reward_path == "constant":
            return self.reward_scale * r
        if self.reward_path == "linear":
            s_t = self._reward_path_scale(t_idx)
            s_prev = self._reward_path_scale(t_idx - 1)
            return (s_t - s_prev) * self.reward_scale * r
        raise ValueError(f"Unsupported reward_path: {self.reward_path}")

    def _reward_path_scale(self, t_idx: int) -> float:
        if self.reward_path == "constant":
            return 1.0
        # linear schedule over reverse index
        return float(t_idx) / float(self.schedule.n_steps)

    def _resolve_dt(self) -> float:
        if self.dt_mode == "auto":
            return 1.0 / float(self.schedule.n_steps)
        if self.dt_mode == "fixed":
            if self.dt <= 0:
                raise ValueError(f"driftlite.dt must be > 0 when dt_mode=fixed, got {self.dt}")
            return float(self.dt)
        raise ValueError(f"Unsupported dt_mode: {self.dt_mode}")

    @staticmethod
    def _validate_basis(basis_names: Sequence[str]):
        allowed = {"grad_r", "score", "x"}
        bad = [b for b in basis_names if b not in allowed]
        if bad:
            raise ValueError(f"Unsupported basis entries: {bad}. Supported={sorted(allowed)}")
        if len(basis_names) == 0:
            raise ValueError("basis must not be empty")
