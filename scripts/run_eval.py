from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

import matplotlib.pyplot as plt

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spec import ToyConfig
from src.gmm import IsotropicGMM
from src.reward import ToyReward
from src.plan import DiffusionSchedule, GuidedReverseSampler
from src.eval import run_eval

console = Console()
KST = timezone(timedelta(hours=9))

# -----------------------------
# Saving policy (edit here if you want)
# -----------------------------
SAVE_BASE_MAX = 50_000          # base samples to save at most
SAVE_CHAIN_MAX = 20_000         # (non-guided / guided) samples to save at most
SAVE_PLOTS_MAX_POINTS = 50_000  # scatter points to plot at most


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(s: str) -> torch.device:
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_outdir(root: str, exp_name: str) -> Path:
    now = datetime.now(KST)
    out_dir = Path(root) / exp_name / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_cfg_from_dotlist(dotlist: list[str]) -> ToyConfig:
    """
    Keep using ToyConfig (structured) so your existing overrides still work:
      python scripts/run_eval.py traj.horizon_T=64 guidance.scale=15 eval.n_base=200000 ...
    """
    base = OmegaConf.structured(ToyConfig())
    override = OmegaConf.from_dotlist(dotlist)
    merged = OmegaConf.merge(base, override)
    # resolve any interpolation; then re-structure
    merged = OmegaConf.to_container(merged, resolve=True)
    return OmegaConf.structured(merged)  # type: ignore


def _save_tensor(path: Path, x: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x.detach().cpu(), path)


def _save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _plot_and_save(out_plots: Path, sT_ng: np.ndarray, sT_g: np.ndarray, R_ng: np.ndarray, R_g: np.ndarray,
                   target_pos: np.ndarray, target_neg: np.ndarray):
    out_plots.mkdir(parents=True, exist_ok=True)

    # Downsample for plotting
    def ds(a: np.ndarray, nmax: int):
        if a.shape[0] <= nmax:
            return a
        idx = np.random.choice(a.shape[0], size=nmax, replace=False)
        return a[idx]

    sT_ng_p = ds(sT_ng, SAVE_PLOTS_MAX_POINTS)
    sT_g_p = ds(sT_g, SAVE_PLOTS_MAX_POINTS)

    # 1) Final state scatter
    plt.figure(figsize=(7, 7))
    plt.scatter(sT_ng_p[:, 0], sT_ng_p[:, 1], s=4, alpha=0.25, label="non-guided")
    plt.scatter(sT_g_p[:, 0], sT_g_p[:, 1], s=4, alpha=0.25, label="guided")
    plt.scatter([target_pos[0]], [target_pos[1]], marker="x", s=120, label="+ target")
    plt.scatter([target_neg[0]], [target_neg[1]], marker="x", s=120, label="- target")
    plt.title("Final states $s_T$ scatter")
    plt.xlabel("$s_{T,x}$")
    plt.ylabel("$s_{T,y}$")
    plt.axis("equal")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_plots / "final_states_scatter.png", dpi=200)
    plt.close()

    # 2) Reward histogram
    plt.figure(figsize=(8, 4))
    plt.hist(R_ng, bins=80, alpha=0.5, label="non-guided")
    plt.hist(R_g, bins=80, alpha=0.5, label="guided")
    plt.title("Total reward $R(\\tau_0)$ histogram")
    plt.xlabel("$R$")
    plt.ylabel("count")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_plots / "reward_hist.png", dpi=200)
    plt.close()


def main():
    cfg = _load_cfg_from_dotlist(sys.argv[1:])

    # outputs/run_eval/YYYY-MM-DD/HH-MM-SS/
    out_dir = _make_outdir(root="outputs", exp_name="run_eval")
    out_data = out_dir / "data"
    out_plots = out_dir / "plots"
    out_data.mkdir(exist_ok=True)
    out_plots.mkdir(exist_ok=True)

    # Save config snapshot
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    _set_seed(cfg.eval.seed)
    device = _resolve_device(cfg.eval.device)
    dtype = torch.float32

    T = cfg.traj.horizon_T
    D = T * cfg.traj.action_dim

    # Base GMM
    m = cfg.traj.base_action_mean
    mu_pos = torch.full((D,), m, dtype=dtype)
    mu_neg = torch.full((D,), -m, dtype=dtype)
    mu = torch.stack([mu_pos, mu_neg], dim=0)
    pi_pos = float(getattr(cfg.traj, "pi_pos", 0.5))
    pi_pos = max(0.0, min(1.0, pi_pos))
    pi = torch.tensor([pi_pos, 1.0 - pi_pos], dtype=dtype)  

    gmm = IsotropicGMM(pi=pi, mu=mu, sigma0_sq=cfg.traj.sigma0_sq)

    # Goal default: (0.1T,0.1T) unless reward.goal_x/y override exists
    goal_x = getattr(cfg.reward, "goal_x", None)
    goal_y = getattr(cfg.reward, "goal_y", None)
    if goal_x is None or goal_y is None:
        default_goal = cfg.traj.base_action_mean * T
        goal_x = default_goal if goal_x is None else goal_x
        goal_y = default_goal if goal_y is None else goal_y
    goal_x = float(goal_x)
    goal_y = float(goal_y)

    reward = ToyReward(
        horizon_T=T,
        base_action_mean=cfg.traj.base_action_mean,
        w_neg=cfg.reward.w_neg,
        w_pos=cfg.reward.w_pos,
        offset=cfg.reward.offset,
        state_var=cfg.reward.state_var,
        goal_x=goal_x,
        goal_y=goal_y,
    )

    schedule = DiffusionSchedule.make_linear(
        n_steps=cfg.diffusion.n_steps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        device=device,
        dtype=dtype,
    )

    guidance_scale = cfg.guidance.scale if cfg.guidance.enabled else 0.0
    sampler = GuidedReverseSampler(
        gmm=gmm,
        schedule=schedule,
        reward_fn=reward.total_reward,
        guidance_scale=float(guidance_scale),
        clip_norm=cfg.guidance.clip_norm,
    )

    console.print(f"[bold]Output:[/bold] {out_dir}")
    console.log(f"Device={device} | steps={cfg.diffusion.n_steps} | T={T} | guidance.scale={guidance_scale}")

    # Sample base ~ p(tau0)
    console.log(f"Sampling base tau0: n={cfg.eval.n_base} ...")
    base = gmm.sample_tau0(cfg.eval.n_base, device=device, dtype=dtype)

    # Sample non-guided / guided
    console.log(f"Sampling non-guided tau0: n={cfg.eval.n_guided} ...")
    nonguided = sampler.sample(
        n=cfg.eval.n_guided,
        batch_size=cfg.eval.batch_size,
        guided=False,
        device=device,
        dtype=dtype,
    )

    console.log(f"Sampling guided tau0: n={cfg.eval.n_guided} ...")
    guided = sampler.sample(
        n=cfg.eval.n_guided,
        batch_size=cfg.eval.batch_size,
        guided=True,
        device=device,
        dtype=dtype,
    )

    # Evaluate: (tilted vs nonguided) and (tilted vs guided)
    console.log("Evaluating generator-matching moments ...")
    results_guided = run_eval(
        reward=reward,
        base_samples=base,
        guided_samples=guided,
        f_list=cfg.eval.f_list,
        bootstrap_reps=cfg.eval.bootstrap_reps,
        seed=cfg.eval.seed + 1234,
        reward_scale=float(guidance_scale),
    )
    results_nonguided = run_eval(
        reward=reward,
        base_samples=base,
        guided_samples=nonguided,
        f_list=cfg.eval.f_list,
        bootstrap_reps=cfg.eval.bootstrap_reps,
        seed=cfg.eval.seed + 2345,
        reward_scale=float(guidance_scale),
    )

    snis_diag = results_guided["meta"].get("snis_diagnostics", {})
    if snis_diag:
        console.log(
            "SNIS diagnostics | "
            f"ESS={snis_diag.get('ess', float('nan')):.3f}/{int(snis_diag.get('n_base', 0))} "
            f"({snis_diag.get('ess_ratio', float('nan')):.6f}) | "
            f"max_w={snis_diag.get('max_weight', float('nan')):.6f} | "
            f"top10_mass={snis_diag.get('top10_weight_mass', float('nan')):.6f}"
        )

    # Print table (three-way)
    table = Table(title="Tilted vs Non-guided vs Guided")
    table.add_column("f", style="bold")
    table.add_column("tilted (SNIS) mean ± se")
    table.add_column("non-guided mean ± se")
    table.add_column("guided mean ± se")
    table.add_column("(non-guided - tilted) ± se")
    table.add_column("(guided - tilted) ± se")

    rows = []
    for k in cfg.eval.f_list:
        t = results_guided["tilted_snis"][k]
        ng = results_nonguided["guided_mean"][k]
        g = results_guided["guided_mean"][k]
        d_ng = results_nonguided["diff_tilted_minus_guided"][k]
        d_g = results_guided["diff_tilted_minus_guided"][k]

        table.add_row(
            k,
            f"{t['mean']:.6g} ± {t['bootstrap_se']:.3g}",
            f"{ng['mean']:.6g} ± {ng['bootstrap_se']:.3g}",
            f"{g['mean']:.6g} ± {g['bootstrap_se']:.3g}",
            f"{d_ng['mean']:.6g} ± {d_ng['approx_se']:.3g}",
            f"{d_g['mean']:.6g} ± {d_g['approx_se']:.3g}",
        )

        # for CSV
        rows.append({
            "f": k,
            "tilted_mean": float(t["mean"]),
            "tilted_se": float(t["bootstrap_se"]),
            "nonguided_mean": float(ng["mean"]),
            "nonguided_se": float(ng["bootstrap_se"]),
            "guided_mean": float(g["mean"]),
            "guided_se": float(g["bootstrap_se"]),
            "tilted_minus_nonguided": float(d_ng["mean"]),
            "tilted_minus_nonguided_se": float(d_ng["approx_se"]),
            "tilted_minus_guided": float(d_g["mean"]),
            "tilted_minus_guided_se": float(d_g["approx_se"]),
            "snis_ess": float(snis_diag.get("ess", float("nan"))),
            "snis_ess_ratio": float(snis_diag.get("ess_ratio", float("nan"))),
            "snis_max_weight": float(snis_diag.get("max_weight", float("nan"))),
            "snis_top10_mass": float(snis_diag.get("top10_weight_mass", float("nan"))),
        })

    console.print(table)

    # -----------------------------
    # Save results.json (top-level)
    # -----------------------------
    payload = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "results": {
            "nonguided_vs_tilted": results_nonguided,
            "guided_vs_tilted": results_guided,
        },
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # -----------------------------
    # Save CSV summary into data/
    # -----------------------------
    csv_path = out_data / "summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = list(rows[0].keys()) if rows else []
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # -----------------------------
    # Save sample tensors into data/
    # (downsample to keep disk reasonable)
    # -----------------------------
    nb = min(int(base.shape[0]), SAVE_BASE_MAX)
    ng = min(int(nonguided.shape[0]), SAVE_CHAIN_MAX)
    ng2 = min(int(guided.shape[0]), SAVE_CHAIN_MAX)

    #_save_tensor(out_data / f"base_tau0_{nb}.pt", base[:nb])
    #_save_tensor(out_data / f"nonguided_tau0_{ng}.pt", nonguided[:ng])
    #_save_tensor(out_data / f"guided_tau0_{ng2}.pt", guided[:ng2])

    # Also save final states / rewards for those saved subsets
    with torch.no_grad():
        sT_ng = reward.final_state(nonguided[:ng]).cpu().numpy()
        sT_g = reward.final_state(guided[:ng2]).cpu().numpy()
        R_ng = reward.total_reward(nonguided[:ng]).cpu().numpy()
        R_g = reward.total_reward(guided[:ng2]).cpu().numpy()

    _save_npz(
        out_data / "derived_saved_subsets.npz",
        sT_nonguided=sT_ng,
        sT_guided=sT_g,
        R_nonguided=R_ng,
        R_guided=R_g,
        goal_pos=np.array([goal_x, goal_y], dtype=np.float32),
        goal_neg=np.array([-goal_x, -goal_y], dtype=np.float32),
    )

    # -----------------------------
    # Save plots into plots/
    # -----------------------------
    target_pos = np.array([goal_x, goal_y], dtype=np.float32)
    target_neg = -target_pos
    _plot_and_save(out_plots, sT_ng, sT_g, R_ng, R_g, target_pos, target_neg)

    console.log(f"Saved: {out_dir / 'config.yaml'}")
    console.log(f"Saved: {out_dir / 'results.json'}")
    console.log(f"Saved: {csv_path}")
    console.log(f"Saved samples under: {out_data}")
    console.log(f"Saved plots under: {out_plots}")


if __name__ == "__main__":
    main()
