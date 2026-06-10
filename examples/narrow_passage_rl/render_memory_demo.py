#!/usr/bin/env python3

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DemoPassage:
    name: str
    center_x: float
    width: float
    length: float
    blocked: bool
    obstacle_y: float


@dataclass
class DemoState:
    x: float
    y: float
    yaw: float
    passage_idx: int
    mode: str
    attempt: int
    step: int
    status: str


def build_scene():
    return [
        DemoPassage("A", -0.72, 0.56, 3.2, False, 0.0),
        DemoPassage("B", 0.0, 0.50, 3.2, True, 1.55),
        DemoPassage("C", 0.72, 0.72, 3.2, False, 0.0),
    ]


def choose_passage(passages, failed_passages, use_memory):
    candidates = sorted(passages, key=lambda p: p.width)
    for passage in candidates:
        if use_memory and passage.name in failed_passages:
            continue
        return passages.index(passage)
    return passages.index(max(passages, key=lambda p: p.width))


def interpolate_pose(start, target, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    x = start[0] + (target[0] - start[0]) * alpha
    y = start[1] + (target[1] - start[1]) * alpha
    yaw = start[2] + (target[2] - start[2]) * alpha
    return x, y, yaw


def generate_demo_frames(args):
    passages = build_scene()
    failed_passages = set()
    frames = []
    start_pose = (0.0, -1.0, 0.0)
    step = 0

    for attempt in range(1, args.max_attempts + 1):
        passage_idx = choose_passage(passages, failed_passages, args.use_memory)
        passage = passages[passage_idx]
        approach_pose = (passage.center_x, -0.20, 0.0)
        enter_pose = (passage.center_x, 0.75, 0.0)
        exit_pose = (passage.center_x, passage.length + 0.35, 0.0)

        for i in range(args.approach_steps):
            pose = interpolate_pose(start_pose, approach_pose, (i + 1) / args.approach_steps)
            frames.append(
                snapshot(
                    pose,
                    passages,
                    failed_passages,
                    passage_idx,
                    "SELECT" if i < 3 else "APPROACH",
                    attempt,
                    step,
                    "running",
                    args,
                )
            )
            step += 1

        for i in range(args.align_steps):
            wobble = math.sin(i / max(1, args.align_steps - 1) * math.pi) * 0.10
            pose = (
                passage.center_x + wobble,
                -0.12 + 0.12 * (i + 1) / args.align_steps,
                -0.18 * (1.0 - (i + 1) / args.align_steps),
            )
            frames.append(
                snapshot(
                    pose,
                    passages,
                    failed_passages,
                    passage_idx,
                    "ALIGN",
                    attempt,
                    step,
                    "running",
                    args,
                )
            )
            step += 1

        traversal_target = enter_pose
        if passage.blocked:
            traversal_target = (passage.center_x, passage.obstacle_y - 0.12, 0.0)

        for i in range(args.traverse_steps):
            pose = interpolate_pose(approach_pose, traversal_target, (i + 1) / args.traverse_steps)
            status = "running"
            mode = "COMMIT"
            if passage.blocked and i >= args.traverse_steps - 2:
                status = "collision"
                mode = "FAILURE"
            frames.append(
                snapshot(
                    pose,
                    passages,
                    failed_passages,
                    passage_idx,
                    mode,
                    attempt,
                    step,
                    status,
                    args,
                )
            )
            step += 1

        if passage.blocked:
            failed_passages.add(passage.name)
            collision_pose = (passage.center_x, passage.obstacle_y - 0.12, 0.0)
            for i in range(args.recovery_steps):
                back = (i + 1) / args.recovery_steps
                pose = (
                    collision_pose[0],
                    collision_pose[1] - 0.75 * back,
                    0.0,
                )
                mode = "WRITE_MEMORY" if args.use_memory and i < 5 else "RECOVER"
                frames.append(
                    snapshot(
                        pose,
                        passages,
                        failed_passages,
                        passage_idx,
                        mode,
                        attempt,
                        step,
                        "memory_written" if args.use_memory else "retry_without_memory",
                        args,
                    )
                )
                step += 1
            start_pose = (0.0, -1.0, 0.0)
            if not args.use_memory and attempt >= args.max_attempts:
                break
            continue

        for i in range(args.exit_steps):
            pose = interpolate_pose(enter_pose, exit_pose, (i + 1) / args.exit_steps)
            frames.append(
                snapshot(
                    pose,
                    passages,
                    failed_passages,
                    passage_idx,
                    "PASS",
                    attempt,
                    step,
                    "success" if i == args.exit_steps - 1 else "running",
                    args,
                )
            )
            step += 1
        break

    return frames


def snapshot(
    pose,
    passages,
    failed_passages,
    passage_idx,
    mode,
    attempt,
    step,
    status,
    args,
):
    passage = passages[passage_idx]
    margin = passage.width * 0.5 - args.robot_radius
    risk = 0.15
    if passage.width < 0.58:
        risk = 0.55
    if passage.blocked:
        risk = 0.85
    if passage.name in failed_passages:
        risk = 1.0
    return {
        "pose": np.asarray(pose, dtype=np.float32),
        "passages": passages,
        "failed_passages": set(failed_passages),
        "active_passage": passage.name,
        "active_idx": passage_idx,
        "mode": mode,
        "attempt": attempt,
        "step": step,
        "status": status,
        "risk": risk,
        "clearance": max(0.0, margin),
        "use_memory": args.use_memory,
    }


def draw_frame(ax, frame, trajectory, args):
    import matplotlib.patches as patches

    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 4.0)
    ax.set_facecolor("#f7f7f3")
    ax.grid(True, color="#dddddd", linewidth=0.6)

    wall_color = "#26323d"
    active_color = "#2f73b8"
    failed_color = "#c8483a"
    safe_color = "#8ba6a9"

    for i, passage in enumerate(frame["passages"]):
        half = passage.width * 0.5
        left = passage.center_x - half
        right = passage.center_x + half
        color = active_color if i == frame["active_idx"] else safe_color
        if passage.name in frame["failed_passages"]:
            color = failed_color

        ax.plot([left, left], [0.0, passage.length], color=wall_color, linewidth=4)
        ax.plot([right, right], [0.0, passage.length], color=wall_color, linewidth=4)
        ax.plot([left, right], [passage.length, passage.length], color=color, linewidth=3)
        ax.text(
            passage.center_x,
            -0.18,
            f"{passage.name}\n{passage.width:.2f}m",
            ha="center",
            va="top",
            fontsize=9,
            color=color,
        )
        if passage.blocked:
            obstacle = patches.Circle(
                (passage.center_x, passage.obstacle_y),
                0.20,
                facecolor="#c8483a",
                edgecolor="#7b2118",
                alpha=0.88,
                zorder=3,
            )
            ax.add_patch(obstacle)

    ax.scatter([0.0], [3.75], marker="*", s=140, color="#298f46", zorder=4)

    if trajectory:
        xy = np.asarray(trajectory)
        ax.plot(xy[:, 0], xy[:, 1], color="#3777b8", linewidth=2.0, alpha=0.75)

    pose = frame["pose"]
    robot = patches.Circle(
        (pose[0], pose[1]),
        args.robot_radius,
        facecolor="#f2b84b",
        edgecolor="#1f1f1f",
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(robot)
    ax.arrow(
        pose[0],
        pose[1],
        math.sin(float(pose[2])) * args.robot_radius * 1.5,
        math.cos(float(pose[2])) * args.robot_radius * 1.5,
        width=0.014,
        head_width=0.065,
        head_length=0.08,
        color="#1f1f1f",
        zorder=6,
    )

    memory_text = ", ".join(sorted(frame["failed_passages"])) or "empty"
    label = "with memory" if frame["use_memory"] else "no memory"
    text = (
        f"policy: {label}\n"
        f"mode: {frame['mode']} | status: {frame['status']}\n"
        f"attempt: {frame['attempt']} | selected passage: {frame['active_passage']}\n"
        f"risk: {frame['risk']:.2f} | clearance: {frame['clearance']:.2f} m\n"
        f"failure memory: {memory_text}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88},
    )


def save_video(frames, args):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 8), dpi=args.dpi)
    trajectory = []

    def update(i):
        trajectory.append(frames[i]["pose"][:2].copy())
        draw_frame(ax, frames[i], trajectory, args)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000.0 / args.fps, repeat=False
    )
    if args.output.suffix.lower() == ".gif":
        ani.save(args.output, writer=animation.PillowWriter(fps=args.fps))
    else:
        ani.save(
            args.output,
            writer=animation.FFMpegWriter(fps=args.fps, bitrate=args.bitrate),
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-memory", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--robot-radius", type=float, default=0.18)
    parser.add_argument("--approach-steps", type=int, default=18)
    parser.add_argument("--align-steps", type=int, default=12)
    parser.add_argument("--traverse-steps", type=int, default=18)
    parser.add_argument("--recovery-steps", type=int, default=14)
    parser.add_argument("--exit-steps", type=int, default=18)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--bitrate", type=int, default=1800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("video_dir/memory_demo_with_memory.mp4"),
    )
    args = parser.parse_args()

    frames = generate_demo_frames(args)
    save_video(frames, args)
    print(f"frames: {len(frames)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
