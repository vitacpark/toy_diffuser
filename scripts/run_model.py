from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.moment_eval import run_eval
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
    cfg = load_cfg(sys.argv[1:], extra_defaults={"model": {"name": default_model_name()}})
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
    samples = sample_model(runtime, model_name=model_name, n=n_model, batch_size=batch_size)

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
        "supported_models": supported_models(),
        "results": results,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    console.log(f"Saved: {out_dir / 'config.yaml'}")
    console.log(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
