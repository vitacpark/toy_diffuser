from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

from rich.console import Console

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipelines.experiment_runtime import MODEL_FACTORIES, load_cfg
from src.pipelines.run_eval_pipeline import run_eval_pipeline


def _parse_args(argv: List[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run eval with matched candidate count K across models: "
            "non-guided/guided(n_candidates=K), tdp(n_roots=K), driftlite(n_particles=K)."
        )
    )
    parser.add_argument("--k", type=int, default=64, help="Matched candidate/particle count")
    parser.add_argument(
        "--models",
        type=str,
        default="non-guided,guided,tdp,driftlite",
        help="Comma-separated model list",
    )
    return parser.parse_known_args(argv)


def _build_model_specs(models: List[str], k: int) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for name in models:
        if name in {"non-guided", "guided"}:
            specs.append(
                {
                    "name": name,
                    "params": {
                        "n_candidates": int(k),
                    },
                }
            )
        elif name == "tdp":
            specs.append(
                {
                    "name": "tdp",
                    "params": {
                        "n_roots": int(k),
                    },
                }
            )
        elif name == "driftlite":
            specs.append(
                {
                    "name": "driftlite",
                    "params": {
                        "n_particles": int(k),
                        # Top-1 selection per rollout.
                        "output_mode": "best",
                    },
                }
            )
        else:
            raise ValueError(f"Unsupported model `{name}`")
    return specs


def main():
    console = Console()
    args, dotlist = _parse_args(sys.argv[1:])

    if args.k <= 0:
        raise ValueError(f"--k must be positive, got {args.k}")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No models selected")

    known = set(MODEL_FACTORIES.keys())
    bad = [m for m in models if m not in known]
    if bad:
        raise ValueError(f"Unsupported models: {bad}. Supported={sorted(known)}")

    model_specs = _build_model_specs(models=models, k=int(args.k))
    cfg = load_cfg(dotlist, extra_defaults={"eval": {"models": model_specs}})

    console.log(
        f"Matched compare: K={int(args.k)}, models={models}, "
        f"batch_size={int(cfg.eval.batch_size)}"
    )
    run_eval_pipeline(cfg=cfg, console=console, exp_name="run_eval_matched")


if __name__ == "__main__":
    main()
