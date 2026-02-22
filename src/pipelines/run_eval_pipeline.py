from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from src.evaluation.moment_eval import run_eval
from src.pipelines.experiment_runtime import (
    build_runtime,
    load_cfg,
    make_outdir,
    parse_model_list,
    sample_model,
    supported_models,
)


def _save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _safe_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_")


def _downsample_rows(a: np.ndarray, nmax: int) -> np.ndarray:
    if a.shape[0] <= nmax:
        return a
    idx = np.random.choice(a.shape[0], size=nmax, replace=False)
    return a[idx]


def _plot_and_save(
    out_plots: Path,
    sampled: Dict[str, Dict[str, np.ndarray]],
    target_pos: np.ndarray,
    target_neg: np.ndarray,
    max_points: int,
):
    out_plots.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    for model_name, values in sampled.items():
        sT = _downsample_rows(values["sT"], max_points)
        plt.scatter(sT[:, 0], sT[:, 1], s=4, alpha=0.22, label=model_name)
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

    plt.figure(figsize=(8, 4))
    for model_name, values in sampled.items():
        plt.hist(values["R"], bins=80, alpha=0.45, label=model_name)
    plt.title("Total reward $R(\\tau_0)$ histogram")
    plt.xlabel("$R$")
    plt.ylabel("count")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_plots / "reward_hist.png", dpi=200)
    plt.close()


def _build_rows_and_table(cfg: DictConfig, models: Iterable[str], results_by_model: Dict[str, Dict]):
    table = Table(title="Tilted vs Selected Models")
    table.add_column("model", style="bold")
    table.add_column("f")
    table.add_column("tilted mean ± se")
    table.add_column("model mean ± se")
    table.add_column("(model - tilted) ± se")

    rows = []
    for model_name in models:
        r = results_by_model[model_name]
        for key in cfg.eval.f_list:
            tilted = r["tilted_snis"][key]
            model = r["guided_mean"][key]
            diff = r["diff_tilted_minus_guided"][key]
            table.add_row(
                model_name,
                str(key),
                f"{tilted['mean']:.6g} ± {tilted['bootstrap_se']:.3g}",
                f"{model['mean']:.6g} ± {model['bootstrap_se']:.3g}",
                f"{-diff['mean']:.6g} ± {diff['approx_se']:.3g}",
            )
            rows.append(
                {
                    "model": model_name,
                    "f": str(key),
                    "tilted_mean": float(tilted["mean"]),
                    "tilted_se": float(tilted["bootstrap_se"]),
                    "model_mean": float(model["mean"]),
                    "model_se": float(model["bootstrap_se"]),
                    "model_minus_tilted": float(-diff["mean"]),
                    "model_minus_tilted_se": float(diff["approx_se"]),
                    "snis_ess": float(r["meta"]["snis_diagnostics"]["ess"]),
                    "snis_ess_ratio": float(r["meta"]["snis_diagnostics"]["ess_ratio"]),
                    "snis_max_weight": float(r["meta"]["snis_diagnostics"]["max_weight"]),
                    "snis_top10_mass": float(r["meta"]["snis_diagnostics"]["top10_weight_mass"]),
                }
            )
    return table, rows


def run_eval_pipeline(cfg: DictConfig, console: Console, exp_name: str = "run_eval") -> Path:
    model_specs: List[str] = list(cfg.eval.models) if cfg.eval.models else supported_models()
    models = parse_model_list(model_specs)

    out_dir = make_outdir(root=cfg.output.root, exp_name=exp_name, timezone_name=cfg.output.timezone)
    out_data = out_dir / "data"
    out_plots = out_dir / "plots"
    out_data.mkdir(exist_ok=True)
    out_plots.mkdir(exist_ok=True)

    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    runtime = build_runtime(cfg=cfg, models=model_specs)

    console.print(f"[bold]Output:[/bold] {out_dir}")
    console.log(
        f"Device={runtime.device} | steps={int(cfg.diffusion.n_steps)} | T={int(cfg.traj.horizon_T)} | models={models}"
    )

    console.log(f"Sampling base tau0: n={int(cfg.eval.n_base)} ...")
    base = runtime.gmm.sample_tau0(int(cfg.eval.n_base), device=runtime.device, dtype=runtime.dtype)

    results_by_model: Dict[str, Dict] = {}
    samples_by_model: Dict[str, torch.Tensor] = {}

    for model_name in models:
        console.log(f"Sampling {model_name} tau0: n={int(cfg.eval.n_guided)} ...")
        samples = sample_model(
            runtime,
            model_name=model_name,
            n=int(cfg.eval.n_guided),
            batch_size=int(cfg.eval.batch_size),
        )
        samples_by_model[model_name] = samples
        results_by_model[model_name] = run_eval(
            reward=runtime.reward,
            base_samples=base,
            guided_samples=samples,
            f_list=list(cfg.eval.f_list),
            bootstrap_reps=int(cfg.eval.bootstrap_reps),
            seed=int(cfg.eval.seed) + (1000 + len(results_by_model) * 97),
            reward_scale=float(runtime.guidance_scale),
        )

    table, rows = _build_rows_and_table(cfg=cfg, models=models, results_by_model=results_by_model)
    console.print(table)

    payload = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "results": results_by_model,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = out_data / "summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = list(rows[0].keys()) if rows else []
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    sampled_stats: Dict[str, Dict[str, np.ndarray]] = {}
    save_chain_max = int(cfg.output.save_chain_max)
    for model_name in models:
        take = min(int(samples_by_model[model_name].shape[0]), save_chain_max)
        tau = samples_by_model[model_name][:take]
        with torch.no_grad():
            sampled_stats[model_name] = {
                "sT": runtime.reward.final_state(tau).cpu().numpy(),
                "R": runtime.reward.total_reward(tau).cpu().numpy(),
            }

    save_base_max = int(cfg.output.save_base_max)
    base_take = min(int(base.shape[0]), save_base_max)
    base_saved = base[:base_take].detach().cpu().numpy()

    flat_arrays: Dict[str, np.ndarray] = {
        "base_tau0": base_saved,
        "goal_pos": np.array(list(runtime.reward.goal), dtype=np.float32),
        "goal_neg": np.array([-v for v in runtime.reward.goal], dtype=np.float32),
    }
    for model_name, values in sampled_stats.items():
        suffix = _safe_name(model_name)
        flat_arrays[f"sT_{suffix}"] = values["sT"]
        flat_arrays[f"R_{suffix}"] = values["R"]

    _save_npz(out_data / "derived_saved_subsets.npz", **flat_arrays)

    target_pos = np.array(list(runtime.reward.goal), dtype=np.float32)
    target_neg = -target_pos
    _plot_and_save(
        out_plots=out_plots,
        sampled=sampled_stats,
        target_pos=target_pos,
        target_neg=target_neg,
        max_points=int(cfg.output.save_plots_max_points),
    )

    console.log(f"Saved: {out_dir / 'config.yaml'}")
    console.log(f"Saved: {out_dir / 'results.json'}")
    console.log(f"Saved: {csv_path}")
    console.log(f"Saved samples under: {out_data}")
    console.log(f"Saved plots under: {out_plots}")
    return out_dir


def run_eval_from_cli(dotlist: list[str], console: Console | None = None) -> Path:
    run_console = console or Console()
    cfg = load_cfg(dotlist, extra_defaults={"eval": {"models": supported_models()}})
    return run_eval_pipeline(cfg=cfg, console=run_console)
