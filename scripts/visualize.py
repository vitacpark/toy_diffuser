from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from rich.console import Console

import matplotlib.pyplot as plt

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spec import ToyConfig
from src.gmm import IsotropicGMM
from src.reward import ToyReward
from src.plan import DiffusionSchedule, GuidedReverseSampler


console = Console()
KST = timezone(timedelta(hours=9))


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


def _build_cfg(dotlist: list[str]) -> DictConfig:
    """
    Build non-structured DictConfig (so we can add extra keys like viz.*).
    """
    # 1) Start from ToyConfig defaults, but convert to plain container (no struct constraints)
    base_toy_struct = OmegaConf.structured(ToyConfig())
    base_toy_plain = OmegaConf.create(OmegaConf.to_container(base_toy_struct, resolve=True))

    # 2) Add viz defaults
    base_viz = OmegaConf.create({"viz": {"n_traj": 5}})

    # 3) Apply overrides
    overrides = OmegaConf.from_dotlist(dotlist)

    # 4) Merge in plain mode
    cfg = OmegaConf.merge(base_toy_plain, base_viz, overrides)
    return cfg


def main():
    cfg = _build_cfg(sys.argv[1:])

    out_dir = _make_outdir(root="outputs", exp_name="visualize")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Save full resolved config (including viz)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    n_traj = int(cfg.viz.n_traj)

    _set_seed(int(cfg.eval.seed))
    device = _resolve_device(str(cfg.eval.device))
    dtype = torch.float32

    T = int(cfg.traj.horizon_T)
    D = T * int(cfg.traj.action_dim)

    # Base GMM
    m = float(cfg.traj.base_action_mean)
    mu_pos = torch.full((D,), m, dtype=dtype)
    mu_neg = torch.full((D,), -m, dtype=dtype)
    mu = torch.stack([mu_pos, mu_neg], dim=0)
    pi_pos = float(getattr(cfg.traj, "pi_pos", 0.5))
    pi_pos = max(0.0, min(1.0, pi_pos))
    pi = torch.tensor([pi_pos, 1.0 - pi_pos], dtype=dtype)  
    gmm = IsotropicGMM(pi=pi, mu=mu, sigma0_sq=float(cfg.traj.sigma0_sq))

    # Goal default: (0.1T,0.1T) unless reward.goal_x/y override exists
    goal_x = cfg.reward.get("goal_x", None)
    goal_y = cfg.reward.get("goal_y", None)
    if goal_x is None or goal_y is None:
        default_goal = float(cfg.traj.base_action_mean) * T
        goal_x = default_goal if goal_x is None else goal_x
        goal_y = default_goal if goal_y is None else goal_y
    goal_x = float(goal_x)
    goal_y = float(goal_y)

    reward = ToyReward(
        horizon_T=T,
        base_action_mean=float(cfg.traj.base_action_mean),
        w_neg=float(cfg.reward.w_neg),
        w_pos=float(cfg.reward.w_pos),
        offset=float(cfg.reward.offset),
        state_var=float(cfg.reward.state_var),
        goal_x=goal_x,
        goal_y=goal_y,
    )

    schedule = DiffusionSchedule.make_linear(
        n_steps=int(cfg.diffusion.n_steps),
        beta_start=float(cfg.diffusion.beta_start),
        beta_end=float(cfg.diffusion.beta_end),
        device=device,
        dtype=dtype,
    )

    guidance_scale = float(cfg.guidance.scale) if bool(cfg.guidance.enabled) else 0.0
    clip_norm = cfg.guidance.get("clip_norm", None)
    clip_norm = None if clip_norm is None else float(clip_norm)

    sampler = GuidedReverseSampler(
        gmm=gmm,
        schedule=schedule,
        reward_fn=reward.total_reward,
        guidance_scale=guidance_scale,
        clip_norm=clip_norm,
    )

    n = int(cfg.eval.n_guided)

    console.print(f"[bold]Output:[/bold] {out_dir}")
    console.log(f"Device={device} | steps={int(cfg.diffusion.n_steps)} | T={T} | guidance.scale={guidance_scale}")
    console.log(f"Sampling n={n} for non-guided and guided ...")

    tau_ng = sampler.sample(n=n, batch_size=int(cfg.eval.batch_size), guided=False, device=device, dtype=dtype)
    tau_g = sampler.sample(n=n, batch_size=int(cfg.eval.batch_size), guided=True, device=device, dtype=dtype)

    with torch.no_grad():
        sT_ng = reward.final_state(tau_ng).cpu().numpy()
        sT_g = reward.final_state(tau_g).cpu().numpy()
        R_ng = reward.total_reward(tau_ng).cpu().numpy()
        R_g = reward.total_reward(tau_g).cpu().numpy()

    target_pos = np.array([goal_x, goal_y])
    target_neg = -target_pos

    # 1) Final state scatter
    plt.figure(figsize=(7, 7))
    plt.scatter(sT_ng[:, 0], sT_ng[:, 1], s=4, alpha=0.25, label="non-guided")
    plt.scatter(sT_g[:, 0], sT_g[:, 1], s=4, alpha=0.25, label="guided")
    plt.scatter([target_pos[0]], [target_pos[1]], marker="x", s=120, label="+ target")
    plt.scatter([target_neg[0]], [target_neg[1]], marker="x", s=120, label="- target")
    plt.title("Final states $s_T$ scatter")
    plt.xlabel("$s_{T,x}$")
    plt.ylabel("$s_{T,y}$")
    plt.axis("equal")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_dir / "final_states_scatter.png", dpi=200)
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
    plt.savefig(plot_dir / "reward_hist.png", dpi=200)
    plt.close()

    # 3) Trajectory paths in XY plane (points + lines), t=0..T
    k = min(n_traj, n)
    idx = np.random.choice(n, size=k, replace=False)

    with torch.no_grad():
        s_path_g = reward.actions_to_states(tau_g[idx]).cpu().numpy()  # [k,T,2]
        s0 = np.zeros((k, 1, 2), dtype=s_path_g.dtype)
        s_path_g = np.concatenate([s0, s_path_g], axis=1)  # [k,T+1,2]

    plt.figure(figsize=(7, 7))
    for i in range(k):
        x = s_path_g[i, :, 0]
        y = s_path_g[i, :, 1]
        plt.plot(x, y, marker="o", markersize=3, linewidth=1, alpha=0.8)
    plt.scatter([0], [0], s=80, label="start")
    plt.scatter([target_pos[0]], [target_pos[1]], marker="x", s=120, label="+ target")
    plt.scatter([target_neg[0]], [target_neg[1]], marker="x", s=120, label="- target")
    plt.title(f"Guided trajectories (k={k}): path of $s_t$ from t=0..T")
    plt.xlabel("$s_x$")
    plt.ylabel("$s_y$")
    plt.axis("equal")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_dir / "traj_paths_guided.png", dpi=200)
    plt.close()

    console.log(f"Saved config: {out_dir / 'config.yaml'}")
    console.log(f"Saved plots: {plot_dir}")
    console.log("Files: final_states_scatter.png, reward_hist.png, traj_paths_guided.png")


if __name__ == "__main__":
    main()
