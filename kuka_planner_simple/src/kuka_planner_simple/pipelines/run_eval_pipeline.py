from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from typing import Dict, List

import numpy as np
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from kuka_planner_simple.pipelines.experiment_runtime import Runtime, build_runtime, load_cfg
from kuka_planner_simple.planners.registry import MODEL_REGISTRY


def _std_err(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / math.sqrt(len(values)))


def _save_csv(path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(runtime: Runtime, episode_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in episode_rows:
        grouped[str(row["model"])].append(row)

    out = []
    for model_name in runtime.selected_models.keys():
        model_rows = grouped.get(model_name, [])
        rewards = [float(r["reward"]) for r in model_rows]
        successes = [1.0 if bool(r["success"]) else 0.0 for r in model_rows]
        elapsed = [float(r["elapsed_sec"]) for r in model_rows]

        out.append(
            {
                "task": str(runtime.cfg.task.name),
                "model": model_name,
                "episodes": len(model_rows),
                "mean_reward": float(np.mean(rewards) if rewards else 0.0),
                "std_err_reward": _std_err(rewards),
                "success_rate": float(np.mean(successes) if successes else 0.0),
                "time_per_episode": float(np.mean(elapsed) if elapsed else 0.0),
                "time_total": float(np.sum(elapsed) if elapsed else 0.0),
            }
        )
    return out


def _print_tables(console: Console, summary_rows: List[Dict], episode_rows: List[Dict]):
    summary = Table(title="KUKA Evaluation Summary")
    summary.add_column("task")
    summary.add_column("model", style="bold")
    summary.add_column("episodes")
    summary.add_column("mean_reward")
    summary.add_column("std_err")
    summary.add_column("success_rate")
    summary.add_column("time/ep(s)")
    summary.add_column("time_total(s)")

    for row in summary_rows:
        summary.add_row(
            str(row["task"]),
            str(row["model"]),
            str(row["episodes"]),
            f"{float(row['mean_reward']):.6g}",
            f"{float(row['std_err_reward']):.6g}",
            f"{float(row['success_rate']):.6g}",
            f"{float(row['time_per_episode']):.6g}",
            f"{float(row['time_total']):.6g}",
        )

    console.print(summary)

    per_ep = Table(title="Episode Metrics")
    per_ep.add_column("model", style="bold")
    per_ep.add_column("episode")
    per_ep.add_column("seed")
    per_ep.add_column("reward")
    per_ep.add_column("success")
    per_ep.add_column("elapsed(s)")

    for row in episode_rows:
        per_ep.add_row(
            str(row["model"]),
            str(row["episode"]),
            str(row["seed"]),
            f"{float(row['reward']):.6g}",
            str(bool(row["success"])),
            f"{float(row['elapsed_sec']):.6g}",
        )
    console.print(per_ep)


def run_eval_pipeline(cfg: DictConfig, console: Console) -> Runtime:
    runtime = build_runtime(cfg)

    OmegaConf.save(runtime.cfg, runtime.out_dir / "config.yaml", resolve=True)

    console.log(f"Output: {runtime.out_dir}")
    console.log(
        f"Task={runtime.cfg.task.name} | Device={runtime.device} | Models={list(runtime.selected_models.keys())}"
    )

    runners = {
        model_name: MODEL_REGISTRY[str(model_cfg.planner)](model_cfg, runtime.shared)
        for model_name, model_cfg in runtime.selected_models.items()
    }

    episode_rows: List[Dict] = []

    for model_name, runner in runners.items():
        for episode_idx in range(int(runtime.cfg.eval.episodes)):
            seed = int(runtime.cfg.runtime.seed) + int(episode_idx)
            result = runner.run_episode(episode_idx=episode_idx, seed=seed)
            row = {
                "task": str(runtime.cfg.task.name),
                "model": model_name,
                "episode": int(episode_idx),
                "seed": int(seed),
                "reward": float(result.reward),
                "success": bool(result.success),
                "elapsed_sec": float(result.elapsed_sec),
                "artifact_dir": str(result.artifacts.get("episode_dir", "")),
                "rollout": str(result.artifacts.get("rollout", "")),
                "trajectory": str(result.artifacts.get("trajectory", "")),
                "video": str(result.artifacts.get("video", "")),
            }
            episode_rows.append(row)
            console.log(
                f"[{model_name}] episode={episode_idx} seed={seed} reward={result.reward:.4f} success={result.success} time={result.elapsed_sec:.3f}s"
            )

    summary_rows = _build_summary(runtime, episode_rows)
    _print_tables(console, summary_rows=summary_rows, episode_rows=episode_rows)

    data_dir = runtime.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(data_dir / "summary.csv", summary_rows)
    _save_csv(data_dir / "episode_metrics.csv", episode_rows)

    result_payload = {
        "task": str(runtime.cfg.task.name),
        "models": list(runtime.selected_models.keys()),
        "summary": summary_rows,
        "episode_metrics": episode_rows,
        "config": OmegaConf.to_container(runtime.cfg, resolve=True),
    }

    with open(runtime.out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    console.log(f"Saved: {runtime.out_dir / 'config.yaml'}")
    console.log(f"Saved: {runtime.out_dir / 'results.json'}")
    console.log(f"Saved: {data_dir / 'summary.csv'}")
    console.log(f"Saved: {data_dir / 'episode_metrics.csv'}")
    return runtime


def run_eval_from_cli(dotlist: List[str], console: Console | None = None) -> Runtime:
    run_console = console or Console()
    cfg = load_cfg(dotlist)
    return run_eval_pipeline(cfg=cfg, console=run_console)
