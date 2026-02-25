from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.evaluation.moment_eval import run_eval
from src.pipelines.experiment_runtime import (
    MODEL_FACTORIES,
    Runtime,
    build_runtime,
    load_cfg,
    make_outdir,
    parse_model_list,
)


def sample_model_batched_with_timing(
    runtime: Runtime,
    model_name: str,
    n: int,
    batch_size: int,
) -> tuple[torch.Tensor, Dict[str, object]]:
    if n <= 0:
        return (
            torch.empty(0, runtime.gmm.D, device=runtime.device, dtype=runtime.dtype),
            {
                "n_samples": 0,
                "batch_samples": 0,
                "n_batches": 0,
                "seconds_total": 0.0,
                "seconds_mean_batch": 0.0,
                "seconds_first_batch": 0.0,
                "samples_per_second": float("nan"),
            },
        )
    if model_name not in runtime.runners:
        raise ValueError(f"Model `{model_name}` was not built. Built={runtime.selected_models}")

    per_call = int(n) if batch_size <= 0 else min(int(n), int(batch_size))
    n_batches = int(math.ceil(float(n) / float(max(1, per_call))))
    outs: List[torch.Tensor] = []
    batch_seconds: List[float] = []
    remaining = int(n)

    with tqdm(total=int(n), desc=model_name, unit="sample", leave=False) as pbar:
        while remaining > 0:
            take = min(per_call, remaining)
            t0 = time.perf_counter()
            chunk = runtime.runners[model_name].sample(
                n=take,
                batch_size=batch_size,
                device=runtime.device,
                dtype=runtime.dtype,
            )
            elapsed = time.perf_counter() - t0
            outs.append(chunk)
            batch_seconds.append(float(elapsed))
            remaining -= int(take)
            pbar.update(int(take))

    samples = torch.cat(outs, dim=0)
    total_seconds = float(sum(batch_seconds))
    stats: Dict[str, object] = {
        "n_samples": int(n),
        "batch_samples": int(per_call),
        "n_batches": int(n_batches),
        "seconds_total": total_seconds,
        "seconds_mean_batch": float(total_seconds / max(1, n_batches)),
        "seconds_first_batch": float(batch_seconds[0]) if batch_seconds else 0.0,
        "samples_per_second": float(n / max(total_seconds, 1e-12)),
    }
    return samples, stats


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


def _diag_to_numpy(diag: Mapping[str, object]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, val in diag.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.detach().cpu().numpy()
        elif isinstance(val, np.ndarray):
            out[key] = val
        else:
            out[key] = np.asarray(val)
    return out


def _diag_to_json(diag_np: Mapping[str, np.ndarray]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, arr in diag_np.items():
        if arr.ndim == 0:
            scalar = arr.item()
            out[key] = float(scalar) if isinstance(scalar, (np.floating, float)) else int(scalar)
        else:
            out[key] = arr.tolist()
    return out


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
    table.add_column("f")
    table.add_column("model", style="bold")
    table.add_column("tilted mean ± se")
    table.add_column("model mean ± se")
    table.add_column("(model - tilted) ± se")

    model_list = list(models)
    rows = []
    for key in cfg.eval.f_list:
        for model_idx, model_name in enumerate(model_list):
            r = results_by_model[model_name]
            tilted = r["tilted_snis"][key]
            model = r["guided_mean"][key]
            diff = r["diff_tilted_minus_guided"][key]
            f_cell = str(key) if model_idx == 0 else ""
            tilted_cell = f"{tilted['mean']:.6g} ± {tilted['bootstrap_se']:.3g}" if model_idx == 0 else ""
            table.add_row(
                f_cell,
                model_name,
                tilted_cell,
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


def _build_timing_rows_and_table(
    models: Iterable[str],
    sample_batch_stats_by_model: Mapping[str, Mapping[str, object]],
):
    table = Table(title="Sampling Time by Model")
    table.add_column("model", style="bold")
    table.add_column("n_samples")
    table.add_column("total sec")
    table.add_column("samples/sec")
    table.add_column("n_batches")
    table.add_column("sec/batch")
    table.add_column("first batch sec")

    rows = []
    for model_name in models:
        stats = sample_batch_stats_by_model[model_name]
        n_samples = int(stats["n_samples"])
        total_sec = float(stats["seconds_total"])
        samples_per_sec = float(stats["samples_per_second"])
        n_batches = int(stats["n_batches"])
        sec_per_batch = float(stats["seconds_mean_batch"])
        first_batch_sec = float(stats["seconds_first_batch"])

        table.add_row(
            model_name,
            str(n_samples),
            f"{total_sec:.6g}",
            f"{samples_per_sec:.6g}",
            str(n_batches),
            f"{sec_per_batch:.6g}",
            f"{first_batch_sec:.6g}",
        )
        rows.append(
            {
                "model": model_name,
                "n_samples": n_samples,
                "seconds_total": total_sec,
                "samples_per_second": samples_per_sec,
                "n_batches": n_batches,
                "seconds_mean_batch": sec_per_batch,
                "seconds_first_batch": first_batch_sec,
                "batch_samples": int(stats["batch_samples"]),
            }
        )
    return table, rows


def run_eval_pipeline(cfg: DictConfig, console: Console, exp_name: str = "run_eval") -> Path:
    model_specs: List[str] = list(cfg.eval.models) if cfg.eval.models else list(MODEL_FACTORIES.keys())
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

    base = runtime.gmm.sample_tau0(int(cfg.eval.n_base), device=runtime.device, dtype=runtime.dtype)

    results_by_model: Dict[str, Dict] = {}
    samples_by_model: Dict[str, torch.Tensor] = {}
    diagnostics_by_model_np: Dict[str, Dict[str, np.ndarray]] = {}
    diagnostics_by_model_json: Dict[str, Dict[str, object]] = {}
    sample_seconds_by_model: Dict[str, float] = {}
    sample_batch_stats_by_model: Dict[str, Dict[str, object]] = {}

    for model_name in models:
        samples, sample_stats = sample_model_batched_with_timing(
            runtime,
            model_name=model_name,
            n=int(cfg.eval.n_guided),
            batch_size=int(cfg.eval.batch_size),
        )
        sample_seconds_by_model[model_name] = float(sample_stats["seconds_total"])
        sample_batch_stats_by_model[model_name] = sample_stats

        samples_by_model[model_name] = samples
        planner = runtime.runners[model_name]
        planner_diag = getattr(planner, "last_diagnostics", None)
        if isinstance(planner_diag, Mapping):
            diag_np = _diag_to_numpy(planner_diag)
            diagnostics_by_model_np[model_name] = diag_np
            diagnostics_by_model_json[model_name] = _diag_to_json(diag_np)
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
    timing_table, timing_rows = _build_timing_rows_and_table(
        models=models,
        sample_batch_stats_by_model=sample_batch_stats_by_model,
    )
    console.print(table)
    console.print(timing_table)

    payload = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "results": results_by_model,
        "sampling_seconds": sample_seconds_by_model,
        "sampling_batch_stats": sample_batch_stats_by_model,
    }
    if diagnostics_by_model_json:
        payload["driftlite_diagnostics"] = diagnostics_by_model_json
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = out_data / "summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = list(rows[0].keys()) if rows else []
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    timing_csv_path = out_data / "timing_summary.csv"
    with open(timing_csv_path, "w", encoding="utf-8") as f:
        timing_cols = list(timing_rows[0].keys()) if timing_rows else []
        f.write(",".join(timing_cols) + "\n")
        for row in timing_rows:
            f.write(",".join(str(row[c]) for c in timing_cols) + "\n")

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

    save_diag_file = bool(cfg.output.get("save_diagnostics", False))
    if save_diag_file and diagnostics_by_model_np:
        diag_arrays: Dict[str, np.ndarray] = {}
        for model_name, diag_np in diagnostics_by_model_np.items():
            prefix = _safe_name(model_name)
            for key, arr in diag_np.items():
                diag_arrays[f"{prefix}__{key}"] = arr
        _save_npz(out_data / "driftlite_diagnostics.npz", **diag_arrays)

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
    console.log(f"Saved: {timing_csv_path}")
    console.log(f"Saved samples under: {out_data}")
    if save_diag_file and diagnostics_by_model_np:
        console.log(f"Saved diagnostics: {out_data / 'driftlite_diagnostics.npz'}")
    console.log(f"Saved plots under: {out_plots}")
    return out_dir


def run_eval_from_cli(dotlist: list[str], console: Console | None = None) -> Path:
    run_console = console or Console()
    cfg = load_cfg(dotlist, extra_defaults={"eval": {"models": list(MODEL_FACTORIES.keys())}})
    return run_eval_pipeline(cfg=cfg, console=run_console)
