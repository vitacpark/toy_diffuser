from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.moment_eval import run_eval
from src.pipelines.experiment_runtime import (
    MODEL_FACTORIES,
    build_runtime,
    load_cfg,
    make_outdir,
    parse_model_list,
)
from src.pipelines.run_eval_pipeline import sample_model_batched_with_timing

console = Console()


def _diag_to_json(diag: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, val in diag.items():
        if isinstance(val, torch.Tensor):
            arr = val.detach().cpu().numpy()
        elif isinstance(val, np.ndarray):
            arr = val
        else:
            arr = np.asarray(val)
        if arr.ndim == 0:
            scalar = arr.item()
            out[key] = float(scalar) if isinstance(scalar, (np.floating, float)) else int(scalar)
        else:
            out[key] = arr.tolist()
    return out


def _save_diag_npz(path: Path, diag: dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, val in diag.items():
        if isinstance(val, torch.Tensor):
            arrays[key] = val.detach().cpu().numpy()
        elif isinstance(val, np.ndarray):
            arrays[key] = val
        else:
            arrays[key] = np.asarray(val)
    np.savez_compressed(path, **arrays)


def main():
    model_names = list(MODEL_FACTORIES.keys())
    if not model_names:
        raise ValueError("No registered models available")
    default_name = "guided" if "guided" in MODEL_FACTORIES else model_names[0]
    cfg = load_cfg(sys.argv[1:], extra_defaults={"model": {"name": default_name}})
    model_name = str(cfg.model.name)
    parse_model_list([model_name])

    out_dir = make_outdir(
        root=str(cfg.output.root),
        exp_name=Path(__file__).stem,
        timezone_name=str(cfg.output.timezone),
    )
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    runtime = build_runtime(cfg, models=[model_name])

    n_base = int(cfg.eval.n_base)
    n_model = int(cfg.eval.n_guided)
    batch_size = int(cfg.eval.batch_size)

    console.print(f"[bold]Output:[/bold] {out_dir}")
    console.log(
        f"Device={runtime.device} | steps={int(cfg.diffusion.n_steps)} | T={int(cfg.traj.horizon_T)} | "
        f"model={model_name}"
    )

    base = runtime.gmm.sample_tau0(n_base, device=runtime.device, dtype=runtime.dtype)
    samples, sample_stats = sample_model_batched_with_timing(
        runtime,
        model_name=model_name,
        n=n_model,
        batch_size=batch_size,
    )
    planner = runtime.runners[model_name]
    planner_diag = getattr(planner, "last_diagnostics", None)

    results = run_eval(
        reward=runtime.reward,
        base_samples=base,
        guided_samples=samples,
        f_list=list(cfg.eval.f_list),
        bootstrap_reps=int(cfg.eval.bootstrap_reps),
        seed=int(cfg.eval.seed) + 123,
        reward_scale=float(runtime.guidance_scale),
    )

    table = Table(title=f"Tilted vs {model_name}")
    table.add_column("f", style="bold")
    table.add_column("tilted mean ± se")
    table.add_column(f"{model_name} mean ± se")
    table.add_column(f"({model_name} - tilted) ± se")

    for k in cfg.eval.f_list:
        t = results["tilted_snis"][k]
        m = results["guided_mean"][k]
        d = results["diff_tilted_minus_guided"][k]
        table.add_row(
            str(k),
            f"{t['mean']:.6g} ± {t['bootstrap_se']:.3g}",
            f"{m['mean']:.6g} ± {m['bootstrap_se']:.3g}",
            f"{-d['mean']:.6g} ± {d['approx_se']:.3g}",
        )
    console.print(table)

    payload = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "model": model_name,
        "supported_models": model_names,
        "results": results,
        "sampling_seconds": {
            model_name: float(sample_stats["seconds_total"]),
        },
        "sampling_batch_stats": {
            model_name: sample_stats,
        },
    }
    if isinstance(planner_diag, dict):
        payload["driftlite_diagnostics"] = _diag_to_json(planner_diag)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    save_diag_file = bool(cfg.output.get("save_diagnostics", False))
    if save_diag_file and isinstance(planner_diag, dict):
        diag_path = out_dir / "data" / "driftlite_diagnostics.npz"
        _save_diag_npz(diag_path, planner_diag)
        console.log(f"Saved: {diag_path}")

    console.log(f"Saved: {out_dir / 'config.yaml'}")
    console.log(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
