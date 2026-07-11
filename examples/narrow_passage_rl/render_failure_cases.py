#!/usr/bin/env python3
"""Render saved narrow-passage failure cases as paper-ready top-down videos."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


VIDEO_NAMES = {
    "collision": "failure_collision.mp4",
    "stuck": "failure_stuck.mp4",
    "timeout": "failure_timeout.mp4",
    "oscillation": "failure_oscillation.mp4",
    "false_feasible_wrong_entry": "failure_false_feasible.mp4",
    "turn_failure": "failure_turn.mp4",
}


def _load_cases(input_dir: Path):
    cases = []
    for path in sorted(input_dir.glob("*/*.json")):
        with path.open() as f:
            case = json.load(f)
        case["_path"] = str(path)
        cases.append(case)
    return cases


def _first_by_type(cases):
    selected = {}
    for case in cases:
        failure_type = case.get("failure_type")
        if failure_type and failure_type not in selected:
            selected[failure_type] = case
    return selected


def _width_at(s, width_profile):
    if not width_profile:
        return 1.0
    if len(width_profile) == 1:
        return float(width_profile[0][1])
    for idx in range(len(width_profile) - 1):
        s0, w0 = width_profile[idx]
        s1, w1 = width_profile[idx + 1]
        if s0 <= s <= s1:
            t = (s - s0) / max(s1 - s0, 1e-9)
            return float(w0 + (w1 - w0) * t)
    return float(width_profile[-1][1])


def _path_arcs(path):
    pts = [np.asarray(p, dtype=float) for p in path]
    arcs = [0.0]
    for idx in range(1, len(pts)):
        arcs.append(arcs[-1] + float(np.linalg.norm(pts[idx] - pts[idx - 1])))
    return arcs, pts


def _point_at_s(path, width_profile, s):
    arcs, pts = _path_arcs(path)
    seg_i = 0
    for idx in range(len(arcs) - 1):
        if arcs[idx] <= s <= arcs[idx + 1]:
            seg_i = idx
            break
    t = (s - arcs[seg_i]) / max(arcs[seg_i + 1] - arcs[seg_i], 1e-9)
    center = pts[seg_i] + t * (pts[seg_i + 1] - pts[seg_i])
    tang = pts[seg_i + 1] - pts[seg_i]
    tang = tang / max(float(np.linalg.norm(tang)), 1e-9)
    perp = np.asarray([-tang[1], tang[0]], dtype=float)
    width = _width_at(s, width_profile)
    return center, tang, perp, width


def _draw_v1_geometry(ax, geom):
    width = float(geom["width"])
    length = float(geom["length"])
    half_w = width * 0.5
    ax.plot([-half_w, -half_w], [0.0, length], color="#2f3a45", linewidth=4)
    ax.plot([half_w, half_w], [0.0, length], color="#2f3a45", linewidth=4)
    ax.fill_betweenx(
        [0.0, length],
        [-half_w, -half_w],
        [half_w, half_w],
        color="#edf1f2",
        alpha=0.8,
    )
    obstacle = geom.get("obstacle", {})
    if float(obstacle.get("radius", 0.0)) > 0.0:
        ax.add_patch(
            patches.Circle(
                (float(obstacle["x"]), float(obstacle["y"])),
                float(obstacle["radius"]),
                facecolor="#d9534f",
                edgecolor="#7b2118",
                alpha=0.9,
            )
        )
    goal = geom.get("goal", [0.0, length + 0.5])
    ax.scatter([goal[0]], [goal[1]], marker="*", s=160, color="#2e8b57", zorder=5)


def _draw_v2_geometry(ax, geom):
    path = geom["path"]
    width_profile = geom["width_profile"]
    arcs, _pts = _path_arcs(path)
    samples = np.linspace(0.0, arcs[-1], 220)
    left_wall = []
    right_wall = []
    for s in samples:
        center, _tang, perp, width = _point_at_s(path, width_profile, float(s))
        left_wall.append(center + perp * width * 0.5)
        right_wall.append(center - perp * width * 0.5)
    left_wall = np.asarray(left_wall)
    right_wall = np.asarray(right_wall)
    ax.fill(
        np.r_[left_wall[:, 0], right_wall[::-1, 0]],
        np.r_[left_wall[:, 1], right_wall[::-1, 1]],
        color="#edf1f2",
        alpha=0.85,
        zorder=0,
    )
    ax.plot(left_wall[:, 0], left_wall[:, 1], color="#2f3a45", linewidth=2)
    ax.plot(right_wall[:, 0], right_wall[:, 1], color="#2f3a45", linewidth=2)

    blocker = geom.get("blocker")
    if blocker is not None:
        center, _tang, perp, _width = _point_at_s(path, width_profile, float(blocker[0]))
        pos = center + perp * float(blocker[1])
        ax.add_patch(
            patches.Circle(
                pos,
                float(blocker[2]),
                facecolor="#d9534f",
                edgecolor="#7b2118",
                alpha=0.9,
                zorder=3,
            )
        )
    start = geom.get("start_world", path[0])
    goal = geom.get("goal_world", path[-1])
    ax.scatter([start[0]], [start[1]], marker="o", s=70, color="#3aa657", zorder=5)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=160, color="#2e8b57", zorder=5)


def _draw_case_frame(ax, case, frame_idx):
    geom = case["geometry"]
    traj = np.asarray(case.get("trajectory", []), dtype=float)
    steps = case.get("steps", [])
    if traj.size == 0:
        return
    frame_idx = min(frame_idx, len(traj) - 1)
    pose = traj[frame_idx]
    step_info = steps[min(frame_idx, len(steps) - 1)] if steps else {}
    reward_terms = step_info.get("reward_breakdown", {})

    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f8f8f4")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    if geom.get("env_version") == "v2":
        _draw_v2_geometry(ax, geom)
    else:
        _draw_v1_geometry(ax, geom)

    sofar = traj[: frame_idx + 1]
    ax.plot(sofar[:, 0], sofar[:, 1], color="#2b6cb0", linewidth=2.0, alpha=0.85)
    if case.get("failure_type") == "collision" and frame_idx == len(traj) - 1:
        ax.scatter([pose[0]], [pose[1]], marker="x", s=120, color="#c0392b", zorder=7)

    radius = float(geom.get("robot_radius", 0.18))
    ax.add_patch(
        patches.Circle(
            (pose[0], pose[1]),
            radius,
            facecolor="#f2b84b",
            edgecolor="#1f1f1f",
            linewidth=1.2,
            zorder=6,
        )
    )
    yaw = float(pose[2]) if len(pose) > 2 else 0.0
    ax.arrow(
        pose[0],
        pose[1],
        math.sin(yaw) * radius * 1.4,
        math.cos(yaw) * radius * 1.4,
        width=0.01,
        head_width=0.06,
        color="#1f1f1f",
        zorder=7,
    )

    reward_preview = ", ".join(
        f"{k.split('/')[-1]}={v:.2f}"
        for k, v in list(reward_terms.items())[:4]
    )
    text = (
        f"type: {case.get('failure_type')} | scene: {case.get('scene_type')}\n"
        f"episode: {case.get('episode_id')} seed: {case.get('seed')}\n"
        f"step: {frame_idx}/{len(traj)-1} mode: {step_info.get('mode', 'PPO')}\n"
        f"width: {case.get('passage_width', 0.0):.2f} "
        f"margin: {case.get('body_margin', 0.0):.3f} "
        f"min_clr: {case.get('min_clearance', 0.0):.3f}\n"
        f"{reward_preview}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88},
    )
    if traj.shape[0] > 0:
        pad = 0.8
        ax.set_xlim(float(np.min(traj[:, 0]) - pad), float(np.max(traj[:, 0]) + pad))
        ax.set_ylim(float(np.min(traj[:, 1]) - pad), float(np.max(traj[:, 1]) + pad))


def save_case_video(case, output_path: Path, fps: int, dpi: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traj = case.get("trajectory", [])
    if not traj:
        return None
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    def update(idx):
        _draw_case_frame(ax, case, idx)

    ani = animation.FuncAnimation(
        fig, update, frames=len(traj), interval=1000.0 / fps, repeat=False
    )
    try:
        ani.save(output_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
        final_path = output_path
    except Exception as exc:
        final_path = output_path.with_suffix(".gif")
        print(f"[warn] mp4 failed for {output_path.name}: {exc}; writing {final_path}")
        ani.save(final_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return final_path


def _reference_success_case(case):
    geom = case["geometry"]
    ref = json.loads(json.dumps(case))
    ref["failure_type"] = "fsm_success"
    ref["final_status"] = "success"
    ref["success"] = 1.0
    if geom.get("env_version") == "v2":
        path = geom["path"]
        width_profile = geom["width_profile"]
        arcs, _pts = _path_arcs(path)
        samples = np.linspace(0.0, arcs[-1], max(24, len(case.get("trajectory", []))))
        traj = []
        for s in samples:
            center, tang, _perp, _width = _point_at_s(path, width_profile, float(s))
            yaw = math.atan2(float(tang[0]), float(tang[1]))
            traj.append([float(center[0]), float(center[1]), yaw])
    else:
        length = float(geom["length"])
        samples = np.linspace(-0.8, length + 0.5, max(24, len(case.get("trajectory", []))))
        traj = [[0.0, float(y), 0.0] for y in samples]
    ref["trajectory"] = traj
    ref["steps"] = [
        {
            "mode": "FSM",
            "reward_breakdown": {},
        }
        for _ in traj
    ]
    return ref


def save_comparison_video(ppo_case, output_path: Path, fps: int, dpi: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fsm_case = _reference_success_case(ppo_case)
    n_frames = max(len(ppo_case.get("trajectory", [])), len(fsm_case["trajectory"]))
    if n_frames == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=dpi)

    def update(idx):
        ppo_idx = min(idx, len(ppo_case["trajectory"]) - 1)
        fsm_idx = min(idx, len(fsm_case["trajectory"]) - 1)
        _draw_case_frame(axes[0], ppo_case, ppo_idx)
        _draw_case_frame(axes[1], fsm_case, fsm_idx)
        axes[0].set_title("PPO local skill", fontsize=10)
        axes[1].set_title("FSM reference", fontsize=10)

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000.0 / fps, repeat=False
    )
    try:
        ani.save(output_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2200))
        final_path = output_path
    except Exception as exc:
        final_path = output_path.with_suffix(".gif")
        print(f"[warn] mp4 failed for {output_path.name}: {exc}; writing {final_path}")
        ani.save(final_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/narrow_passage_rl/failure_cases"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/narrow_passage_rl/videos"),
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()

    cases = _load_cases(args.input_dir)
    selected = _first_by_type(cases)
    if not selected:
        raise RuntimeError(f"No failure cases found under {args.input_dir}")

    for failure_type, filename in VIDEO_NAMES.items():
        case = selected.get(failure_type)
        if case is None:
            continue
        out = save_case_video(case, args.output_dir / filename, args.fps, args.dpi)
        print(f"[video] {failure_type}: {out}")

    comparison_case = selected.get("turn_failure") or selected.get("collision")
    if comparison_case is not None:
        out = save_comparison_video(
            comparison_case,
            args.output_dir / "success_fsm_vs_ppo.mp4",
            args.fps,
            args.dpi,
        )
        print(f"[video] success_fsm_vs_ppo: {out}")


if __name__ == "__main__":
    main()
