from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from rich.console import Console

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipelines.experiment_runtime import (
    build_runtime,
    default_model_name,
    load_cfg,
    make_outdir,
    parse_model_list,
    sample_model,
    supported_models,
)

console = Console()


def main():
    default_models = supported_models()
    default_path_model = default_model_name()
    cfg = load_cfg(
        sys.argv[1:],
        extra_defaults={
            "viz": {
                "n_traj": 5,
                "models": default_models,
                "path_model": default_path_model,
            }
        },
    )
    model_specs = list(cfg.viz.models)
    models = parse_model_list(model_specs)
    path_model = str(cfg.viz.path_model)
    parse_model_list([path_model])
    if path_model not in models:
        model_specs = [*model_specs, path_model]
        models = parse_model_list(model_specs)

    out_dir = make_outdir(
        root=str(cfg.output.root),
        exp_name=Path(__file__).stem,
        timezone_name=str(cfg.output.timezone),
    )
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    runtime = build_runtime(cfg, models=model_specs)

    n = int(cfg.eval.n_guided)
    batch_size = int(cfg.eval.batch_size)

    console.print(f"[bold]Output:[/bold] {out_dir}")
    console.log(
        f"Device={runtime.device} | steps={int(cfg.diffusion.n_steps)} | T={int(cfg.traj.horizon_T)} | models={models}"
    )

    sampled = {}
    for model_name in models:
        console.log(f"Sampling {model_name}: n={n} ...")
        tau = sample_model(runtime, model_name=model_name, n=n, batch_size=batch_size)
        with np.errstate(all="ignore"):
            sampled[model_name] = {
                "tau": tau,
                "sT": runtime.reward.final_state(tau).cpu().numpy(),
                "R": runtime.reward.total_reward(tau).cpu().numpy(),
            }

    target_pos = np.array([runtime.reward.goal_x, runtime.reward.goal_y], dtype=np.float32)
    target_neg = -target_pos

    # 1) Final state scatter (all selected models)
    plt.figure(figsize=(7, 7))
    for model_name in models:
        sT = sampled[model_name]["sT"]
        plt.scatter(sT[:, 0], sT[:, 1], s=4, alpha=0.22, label=model_name)
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

    # 2) Reward histogram (all selected models)
    plt.figure(figsize=(8, 4))
    for model_name in models:
        plt.hist(sampled[model_name]["R"], bins=80, alpha=0.45, label=model_name)
    plt.title("Total reward $R(\\tau_0)$ histogram")
    plt.xlabel("$R$")
    plt.ylabel("count")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_dir / "reward_hist.png", dpi=200)
    plt.close()

    # 3) Trajectory paths for one selected model
    tau_path = sampled[path_model]["tau"]
    k = min(int(cfg.viz.n_traj), n)
    idx = np.random.choice(n, size=k, replace=False)
    s_path = runtime.reward.actions_to_states(tau_path[idx]).cpu().numpy()  # [k,T,2]
    s0 = np.zeros((k, 1, 2), dtype=s_path.dtype)
    s_path = np.concatenate([s0, s_path], axis=1)  # [k,T+1,2]

    plt.figure(figsize=(7, 7))
    for i in range(k):
        x = s_path[i, :, 0]
        y = s_path[i, :, 1]
        plt.plot(x, y, marker="o", markersize=3, linewidth=1, alpha=0.85)
    plt.scatter([0], [0], s=80, label="start")
    plt.scatter([target_pos[0]], [target_pos[1]], marker="x", s=120, label="+ target")
    plt.scatter([target_neg[0]], [target_neg[1]], marker="x", s=120, label="- target")
    plt.title(f"{path_model} trajectories (k={k}): path of $s_t$ from t=0..T")
    plt.xlabel("$s_x$")
    plt.ylabel("$s_y$")
    plt.axis("equal")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_dir / "traj_paths.png", dpi=200)
    plt.close()

    console.log(f"Saved config: {out_dir / 'config.yaml'}")
    console.log(f"Saved plots: {plot_dir}")
    console.log("Files: final_states_scatter.png, reward_hist.png, traj_paths.png")


if __name__ == "__main__":
    main()
