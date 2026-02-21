from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .reward import ToyReward


def _f_map(reward: ToyReward, tau0: torch.Tensor, f_list: List[str]) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        final = reward.final_state(tau0)
        R = reward.total_reward(tau0)

        T = reward.horizon_T
        gx = getattr(reward, "goal_x", None)
        gy = getattr(reward, "goal_y", None)

        if gx is None or gy is None:
            # fallback to legacy default
            gx = reward.base_action_mean * T
            gy = reward.base_action_mean * T
        else:
            # ensure python floats (in case something passes numpy/torch scalar)
            gx = float(gx)
            gy = float(gy)

        pos = torch.tensor([gx,gy], device=tau0.device, dtype=tau0.dtype)
        neg = -pos
        dist_pos = torch.linalg.vector_norm(final - pos, dim=1)
        dist_neg = torch.linalg.vector_norm(final - neg, dim=1)
        pos_ind = (dist_pos < dist_neg).to(tau0.dtype)

        feats = {
            "final_x": final[:, 0].detach().cpu().numpy(),
            "final_y": final[:, 1].detach().cpu().numpy(),
            "pos_indicator": pos_ind.detach().cpu().numpy(),
            "R": R.detach().cpu().numpy(),
        }
        return {k: feats[k] for k in f_list}


def _snis_estimate(f: np.ndarray, logw: np.ndarray) -> float:
    lw = logw - np.max(logw)
    w = np.exp(lw)
    w = w / (np.sum(w) + 1e-12)
    return float(np.sum(w * f))


def _bootstrap_snis(f: np.ndarray, logw: np.ndarray, reps: int, rng: np.random.Generator) -> Tuple[float, float]:
    n = f.shape[0]
    est = _snis_estimate(f, logw)
    if reps <= 0:
        return est, float("nan")
    boots = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        idx = rng.integers(0, n, size=n)
        boots[b] = _snis_estimate(f[idx], logw[idx])
    return est, float(np.std(boots, ddof=1))


def _bootstrap_mean(x: np.ndarray, reps: int, rng: np.random.Generator) -> Tuple[float, float]:
    est = float(np.mean(x))
    if reps <= 0:
        return est, float("nan")
    n = x.shape[0]
    boots = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(x[idx]))
    return est, float(np.std(boots, ddof=1))


@torch.no_grad()
def run_eval(
    reward: ToyReward,
    base_samples: torch.Tensor,
    guided_samples: torch.Tensor,
    f_list: List[str],
    bootstrap_reps: int,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)

    # Tilted: SNIS using logw = R (since h=exp(R))
    logw = reward.total_reward(base_samples).detach().cpu().numpy()
    f_base = _f_map(reward, base_samples, f_list)
    f_guided = _f_map(reward, guided_samples, f_list)

    out = {
        "meta": {
            "seed": int(seed),
            "n_base": int(base_samples.shape[0]),
            "n_guided": int(guided_samples.shape[0]),
            "bootstrap_reps": int(bootstrap_reps),
        },
        "tilted_snis": {},
        "guided_mean": {},
        "diff_tilted_minus_guided": {},
    }

    for k in f_list:
        est_t, se_t = _bootstrap_snis(f_base[k], logw, bootstrap_reps, rng)
        est_g, se_g = _bootstrap_mean(f_guided[k], bootstrap_reps, rng)
        out["tilted_snis"][k] = {"mean": est_t, "bootstrap_se": se_t}
        out["guided_mean"][k] = {"mean": est_g, "bootstrap_se": se_g}
        out["diff_tilted_minus_guided"][k] = {
            "mean": est_t - est_g,
            "approx_se": float(np.sqrt(se_t**2 + se_g**2)) if np.isfinite(se_t) and np.isfinite(se_g) else float("nan"),
        }

    return out
