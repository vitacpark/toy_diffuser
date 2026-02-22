from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from src.tasks.reward import ToyReward


FeatureExtractor = Callable[[Dict[str, torch.Tensor]], np.ndarray]


def _feature_final_x(ctx: Dict[str, torch.Tensor]) -> np.ndarray:
    return ctx["final"][:, 0].detach().cpu().numpy()


def _feature_final_y(ctx: Dict[str, torch.Tensor]) -> np.ndarray:
    if ctx["final"].shape[1] < 2:
        raise ValueError("Feature `final_y` requires action_dim >= 2")
    return ctx["final"][:, 1].detach().cpu().numpy()


def _feature_pos_indicator(ctx: Dict[str, torch.Tensor]) -> np.ndarray:
    return ctx["pos_ind"].detach().cpu().numpy()


def _feature_reward(ctx: Dict[str, torch.Tensor]) -> np.ndarray:
    return ctx["reward"].detach().cpu().numpy()


FEATURE_REGISTRY: Dict[str, FeatureExtractor] = {
    "final_x": _feature_final_x,
    "final_y": _feature_final_y,
    "pos_indicator": _feature_pos_indicator,
    "R": _feature_reward,
}


def register_feature(name: str, extractor: FeatureExtractor):
    FEATURE_REGISTRY[name] = extractor


def _f_map(reward: ToyReward, tau0: torch.Tensor, f_list: List[str]) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        final = reward.final_state(tau0)
        R = reward.total_reward(tau0)
        pos = torch.tensor(list(reward.goal), device=tau0.device, dtype=tau0.dtype)
        neg = -pos
        dist_pos = torch.linalg.vector_norm(final - pos, dim=1)
        dist_neg = torch.linalg.vector_norm(final - neg, dim=1)
        pos_ind = (dist_pos < dist_neg).to(tau0.dtype)
        ctx = {
            "final": final,
            "reward": R,
            "pos_ind": pos_ind,
        }

        out: Dict[str, np.ndarray] = {}
        bad = [k for k in f_list if k not in FEATURE_REGISTRY]
        if bad:
            raise ValueError(f"Unsupported eval feature(s): {bad}. Supported={sorted(FEATURE_REGISTRY.keys())}")
        for key in f_list:
            out[key] = FEATURE_REGISTRY[key](ctx)
        return out


def _snis_estimate(f: np.ndarray, logw: np.ndarray) -> float:
    lw = logw - np.max(logw)
    w = np.exp(lw)
    w = w / (np.sum(w) + 1e-12)
    return float(np.sum(w * f))


def _snis_diagnostics(logw: np.ndarray) -> Dict[str, float]:
    lw = logw - np.max(logw)
    w = np.exp(lw)
    w = w / (np.sum(w) + 1e-12)
    ess = float(1.0 / (np.sum(w * w) + 1e-12))
    n = float(w.shape[0])
    top1 = float(np.max(w))
    k = min(10, w.shape[0])
    topk = float(np.sum(np.partition(w, -k)[-k:]))
    return {
        "n_base": n,
        "ess": ess,
        "ess_ratio": ess / max(n, 1.0),
        "max_weight": top1,
        "top10_weight_mass": topk,
    }


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
    reward_scale: float = 1.0,
) -> Dict:
    rng = np.random.default_rng(seed)

    # Tilted: SNIS using logw = reward_scale * R (h=exp(reward_scale * R))
    logw = (reward_scale * reward.total_reward(base_samples)).detach().cpu().numpy()
    snis_diag = _snis_diagnostics(logw)
    f_base = _f_map(reward, base_samples, f_list)
    f_guided = _f_map(reward, guided_samples, f_list)

    out = {
        "meta": {
            "seed": int(seed),
            "n_base": int(base_samples.shape[0]),
            "n_guided": int(guided_samples.shape[0]),
            "bootstrap_reps": int(bootstrap_reps),
            "reward_scale": float(reward_scale),
            "snis_diagnostics": snis_diag,
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
