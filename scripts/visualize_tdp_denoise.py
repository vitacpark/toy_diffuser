from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from omegaconf import OmegaConf
from rich.console import Console

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipelines.experiment_runtime import build_runtime, load_cfg, make_outdir


def _goal_xy(goal_vals) -> np.ndarray:
    arr = np.asarray(list(goal_vals), dtype=np.float32).reshape(-1)
    if arr.size >= 2:
        return arr[:2]
    if arr.size == 1:
        return np.array([arr[0], 0.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def _to_final_state_xy(trace: torch.Tensor, reward) -> np.ndarray:
    # trace: [F, B, D] -> [F, B, T+1, 2] via cumulative state projection
    f, b, d = trace.shape
    with torch.no_grad():
        s_path = reward.actions_to_states(trace.reshape(f * b, d)).view(f, b, int(reward.horizon_T), -1)
    s_path = s_path.detach().cpu().numpy()
    s0 = np.zeros((f, b, 1, s_path.shape[-1]), dtype=s_path.dtype)
    s_path = np.concatenate([s0, s_path], axis=2)
    if s_path.shape[-1] >= 2:
        return s_path[..., :2]
    pad = np.zeros((*s_path.shape[:-1], 2), dtype=s_path.dtype)
    pad[..., : s_path.shape[-1]] = s_path
    return pad


def main():
    console = Console()
    cfg = load_cfg(
        sys.argv[1:],
        extra_defaults={
            "viz_tdp": {
                "n_rollouts": 1,
                "max_frames": 120,
                "fps": 12,
                "format": "gif",  # gif | mp4
                "filename": "tdp_denoise",
            }
        },
    )

    n_roots = int(cfg.tdp.n_roots)
    n_rollouts = int(cfg.viz_tdp.n_rollouts)
    max_frames = int(cfg.viz_tdp.max_frames)
    fps = int(cfg.viz_tdp.fps)
    fmt = str(cfg.viz_tdp.format).lower()
    if fmt not in {"gif", "mp4"}:
        raise ValueError(f"viz_tdp.format must be gif or mp4, got {fmt}")

    out_dir = make_outdir(
        root=str(cfg.output.root),
        exp_name=Path(__file__).stem,
        timezone_name=str(cfg.output.timezone),
    )
    plot_dir = out_dir / "plots"
    data_dir = out_dir / "data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    runtime = build_runtime(
        cfg=cfg,
        models=[
            {
                "name": "tdp",
                "params": {
                    "n_roots": n_roots,
                },
            }
        ],
    )
    planner = runtime.runners["tdp"]
    if not hasattr(planner, "tdp"):
        raise TypeError("tdp runner does not expose internal tdp sampler")

    console.print(f"[bold]Output:[/bold] {out_dir}")
    console.log(
        f"Device={runtime.device} | steps={int(cfg.diffusion.n_steps)} | "
        f"T={int(cfg.traj.horizon_T)} | n_roots={n_roots} | n_rollouts={n_rollouts}"
    )

    trace = planner.tdp.sample_many_with_trace(
        n_rollouts=n_rollouts,
        n_roots=n_roots,
        device=runtime.device,
        dtype=runtime.dtype,
        max_frames=max_frames,
    )

    parent_trace = trace["parent_trace"]  # [Fp, B, D]
    child_renoise_trace = trace["child_renoise_trace"]  # [Fr, B, D]
    child_denoise_trace = trace["child_denoise_trace"]  # [Fc, B, D]

    parent_paths = _to_final_state_xy(parent_trace, runtime.reward)  # [F,B,T+1,2]
    child_renoise_paths = _to_final_state_xy(child_renoise_trace, runtime.reward)  # [F,B,T+1,2]
    child_denoise_paths = _to_final_state_xy(child_denoise_trace, runtime.reward)  # [F,B,T+1,2]

    np.savez_compressed(
        data_dir / "tdp_trace.npz",
        parent_trace=parent_trace.detach().cpu().numpy(),
        child_renoise_trace=child_renoise_trace.detach().cpu().numpy(),
        child_denoise_trace=child_denoise_trace.detach().cpu().numpy(),
        parent_paths=parent_paths,
        child_renoise_paths=child_renoise_paths,
        child_denoise_paths=child_denoise_paths,
        u=trace["u"].detach().cpu().numpy(),
        mask=trace["mask"].detach().cpu().numpy(),
        k_renoise=int(trace["k_renoise"].item()),
    )

    goal_pos = _goal_xy(runtime.reward.goal)
    goal_neg = -goal_pos
    all_pts = np.concatenate(
        [
            parent_paths.reshape(-1, 2),
            child_renoise_paths.reshape(-1, 2),
            child_denoise_paths.reshape(-1, 2),
            goal_pos.reshape(1, 2),
            goal_neg.reshape(1, 2),
        ],
        axis=0,
    )
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins) + 1e-6)
    xlim = (center[0] - radius, center[0] + radius)
    ylim = (center[1] - radius, center[1] + radius)

    f_parent = parent_paths.shape[0]
    f_renoise = child_renoise_paths.shape[0]
    f_child = child_denoise_paths.shape[0]
    n_frames = max(f_parent, f_renoise + f_child)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_parent, ax_child = axes
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("$s_{T,x}$")
        ax.set_ylabel("$s_{T,y}$")
        ax.scatter([goal_pos[0]], [goal_pos[1]], marker="x", s=90, label="+ target")
        ax.scatter([goal_neg[0]], [goal_neg[1]], marker="x", s=90, label="- target")
        ax.legend(loc="best")

    n_particles = parent_paths.shape[1]
    parent_lines = [ax_parent.plot([], [], linewidth=1.0, alpha=0.7)[0] for _ in range(n_particles)]
    child_lines = [ax_child.plot([], [], linewidth=1.0, alpha=0.7)[0] for _ in range(n_particles)]
    parent_heads = ax_parent.scatter([], [], s=16, alpha=0.75, label="parents")
    child_heads = ax_child.scatter([], [], s=16, alpha=0.75, label="children")

    def _set_offsets(scat, pts):
        if pts.shape[0] == 0:
            scat.set_offsets(np.zeros((0, 2), dtype=np.float32))
            return
        scat.set_offsets(pts)

    def _set_paths(lines, paths):
        for i, ln in enumerate(lines):
            xy = paths[i]
            ln.set_data(xy[:, 0], xy[:, 1])

    def init():
        for ln in parent_lines:
            ln.set_data([], [])
        for ln in child_lines:
            ln.set_data([], [])
        _set_offsets(parent_heads, np.zeros((0, 2), dtype=np.float32))
        _set_offsets(child_heads, np.zeros((0, 2), dtype=np.float32))
        ax_parent.set_title("Parent denoise")
        ax_child.set_title("Child re-noise + denoise")
        return [*parent_lines, *child_lines, parent_heads, child_heads]

    def update(frame_idx: int):
        p_idx = min(frame_idx, f_parent - 1)
        p_paths = parent_paths[p_idx]
        _set_paths(parent_lines, p_paths)
        _set_offsets(parent_heads, p_paths[:, -1, :])

        if frame_idx < f_renoise:
            c_paths = child_renoise_paths[frame_idx]
            child_title = f"Child re-noise ({frame_idx + 1}/{f_renoise})"
        else:
            d_idx = min(frame_idx - f_renoise, f_child - 1)
            c_paths = child_denoise_paths[d_idx]
            child_title = f"Child denoise ({d_idx + 1}/{f_child})"

        _set_paths(child_lines, c_paths)
        _set_offsets(child_heads, c_paths[:, -1, :])
        ax_parent.set_title(f"Parent denoise ({p_idx + 1}/{f_parent})")
        ax_child.set_title(child_title)
        return [*parent_lines, *child_lines, parent_heads, child_heads]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=int(1000 / max(1, fps)),
        blit=True,
        repeat=False,
    )

    out_name = f"{str(cfg.viz_tdp.filename)}.{fmt}"
    out_path = plot_dir / out_name
    if fmt == "gif":
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)

    console.log(f"Saved config: {out_dir / 'config.yaml'}")
    console.log(f"Saved trace: {data_dir / 'tdp_trace.npz'}")
    console.log(f"Saved animation: {out_path}")


if __name__ == "__main__":
    main()
