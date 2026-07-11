#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
from eval_memory_fsm import choose_mode as choose_memory_mode
from eval_memory_fsm import restore_attempt_start
from eval_mode_fsm import align_action
from eval_rule_baseline import CenterlineRulePolicy
from failure_memory import FailureMemoryConfig, PassageFailureMemory
from procedural_env import ProceduralNarrowPassageEnv
from risk_estimator import GeometryRiskEstimator, RiskConfig
from train_sb3 import DIFFICULTY_CONFIGS


def require_ppo():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for learned policies. Activate the "
            "habitat conda env before running this script."
        ) from exc
    return PPO


def is_recovered_state(obs):
    return (
        abs(float(obs[11])) < 0.12
        and abs(float(obs[10])) < 0.25
        and min(float(obs[6]), float(obs[7])) > 0.14
        and float(obs[15]) < 0.25
    )


def snapshot(env, obs, mode, risk_info, step, info=None):
    info = info or {}
    return {
        "pose": env.pose.copy(),
        "params": env.params,
        "obs": np.asarray(obs).copy(),
        "mode": mode,
        "risk": float(risk_info.get("risk", 0.0)),
        "min_clearance": min(float(obs[6]), float(obs[7])),
        "step": step,
        "success": float(info.get("success", 0.0)),
        "collision": float(info.get("collision", 0.0)),
        "stuck": float(info.get("stuck", 0.0)),
    }


def rollout_rule(env, args):
    policy = CenterlineRulePolicy()
    obs, _ = env.reset(seed=args.seed)
    frames = []
    done = False
    step = 0
    while not done and step < args.max_steps:
        frames.append(snapshot(env, obs, "RULE", {"risk": 0.0}, step))
        action = policy.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    frames.append(snapshot(env, obs, "DONE", {"risk": 0.0}, step, info))
    return frames


def rollout_passage(env, args, passage_model):
    obs, _ = env.reset(seed=args.seed)
    frames = []
    done = False
    step = 0
    while not done and step < args.max_steps:
        frames.append(snapshot(env, obs, "COMMIT", {"risk": 0.0}, step))
        action, _ = passage_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    frames.append(snapshot(env, obs, "DONE", {"risk": 0.0}, step, info))
    return frames


def rollout_mode_fsm(env, args, passage_model, recovery_model):
    obs, _ = env.reset(seed=args.seed)
    estimator = GeometryRiskEstimator(RiskConfig())
    frames = []
    done = False
    step = 0
    info = {}
    while not done and step < args.max_steps:
        risk_info = estimator.score(obs)
        mode = estimator.mode(obs)
        frames.append(snapshot(env, obs, mode, risk_info, step))

        if mode == "REJECT":
            info = {"success": 0.0, "collision": 0.0, "stuck": float(obs[15] >= 0.7)}
            break
        if mode == "ALIGN":
            action = align_action(obs)
        elif mode == "RECOVER":
            action, _ = recovery_model.predict(obs, deterministic=True)
        else:
            action, _ = passage_model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, info = env.step(action)
        if terminated and info.get("collision", 0.0) > 0.5:
            estimator.mark_failure()
        done = terminated or truncated
        step += 1
    frames.append(snapshot(env, obs, "DONE", estimator.score(obs), step, info))
    return frames


def rollout_memory_fsm(env, args, passage_model, recovery_model):
    obs, _ = env.reset(seed=args.seed)
    start_pose = env.pose.copy()
    estimator = GeometryRiskEstimator(RiskConfig())
    memory = PassageFailureMemory(
        FailureMemoryConfig(
            trigger_count=args.memory_trigger_count,
            reject_count=args.memory_reject_count,
        )
    )
    frames = []
    done = False
    step = 0
    attempts = 1
    info = {}
    while not done and step < args.max_steps:
        mode, risk_info = choose_memory_mode(obs, estimator, memory, args)
        risk_info = dict(risk_info)
        risk_info["risk"] = max(risk_info["risk"], memory.count(obs) / 3.0)
        frames.append(snapshot(env, obs, mode, risk_info, step))

        if mode == "REJECT":
            info = {"success": 0.0, "collision": 0.0, "stuck": float(obs[15] >= 0.7)}
            break

        if mode == "RECOVER":
            recovered = False
            for _ in range(args.max_recover_steps):
                action, _ = recovery_model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                step += 1
                frames.append(snapshot(env, obs, "RECOVER", estimator.score(obs), step))
                if terminated and info.get("collision", 0.0) > 0.5:
                    estimator.mark_failure()
                    memory.add_failure(obs)
                    if attempts < args.max_attempts:
                        attempts += 1
                        obs = restore_attempt_start(env, start_pose, args)
                    else:
                        done = True
                    break
                if is_recovered_state(obs):
                    recovered = True
                    break
                if terminated or truncated:
                    done = True
                    break
            if not recovered and not done:
                estimator.mark_failure()
                memory.add_failure(obs)
            continue

        action = align_action(obs) if mode == "ALIGN" else passage_model.predict(obs, deterministic=True)[0]
        prev_obs = obs.copy()
        obs, _, terminated, truncated, info = env.step(action)
        if terminated and info.get("collision", 0.0) > 0.5:
            estimator.mark_failure()
            memory.add_failure(prev_obs)
            if attempts < args.max_attempts:
                attempts += 1
                obs = restore_attempt_start(env, start_pose, args)
                terminated = False
                truncated = False
        done = terminated or truncated
        step += 1
    frames.append(snapshot(env, obs, "DONE", estimator.score(obs), step, info))
    return frames


def draw_frame(ax, frame, trajectory, args):
    import matplotlib.patches as patches

    pose = frame["pose"]
    p = frame["params"]
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.4, p.length + 0.9)
    ax.set_facecolor("#f7f7f3")
    ax.grid(True, color="#dddddd", linewidth=0.6)

    wall_color = "#2f3a45"
    half_w = p.width * 0.5
    ax.plot([-half_w, -half_w], [0.0, p.length], color=wall_color, linewidth=5)
    ax.plot([half_w, half_w], [0.0, p.length], color=wall_color, linewidth=5)
    ax.plot([-half_w, half_w], [p.length, p.length], color="#b7c4cc", linewidth=2)
    ax.scatter([0.0], [p.length + 0.5], marker="*", s=140, color="#298f46", zorder=4)

    if p.obstacle_radius > 0.0:
        obstacle = patches.Circle(
            (p.obstacle_x, p.obstacle_y),
            p.obstacle_radius,
            facecolor="#c8483a",
            edgecolor="#7b2118",
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(obstacle)

    if trajectory:
        xy = np.asarray(trajectory)
        ax.plot(xy[:, 0], xy[:, 1], color="#3777b8", linewidth=2, alpha=0.75)

    robot = patches.Circle(
        (pose[0], pose[1]),
        args.robot_radius,
        facecolor="#f2b84b",
        edgecolor="#1f1f1f",
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(robot)
    dx = math.sin(float(pose[2])) * args.robot_radius * 1.5
    dy = math.cos(float(pose[2])) * args.robot_radius * 1.5
    ax.arrow(
        pose[0],
        pose[1],
        dx,
        dy,
        width=0.015,
        head_width=0.07,
        head_length=0.08,
        color="#1f1f1f",
        zorder=6,
    )

    status = "success" if frame["success"] > 0.5 else "collision" if frame["collision"] > 0.5 else "running"
    text = (
        f"policy: {args.policy}\n"
        f"mode: {frame['mode']} | status: {status}\n"
        f"step: {frame['step']} | risk: {frame['risk']:.2f}\n"
        f"min clearance: {frame['min_clearance']:.3f} m\n"
        f"width: {p.width:.2f} m | false feasible: {int(p.false_feasible)}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.86},
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
    suffix = args.output.suffix.lower()
    if suffix == ".gif":
        ani.save(args.output, writer=animation.PillowWriter(fps=args.fps))
    else:
        try:
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=args.bitrate)
            ani.save(args.output, writer=writer)
        except Exception as exc:
            fallback = args.output.with_suffix(".gif")
            print(f"[warn] mp4 failed for {args.output}: {exc}; writing {fallback}")
            ani.save(fallback, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["rule", "passage", "mode_fsm", "memory_fsm"],
        default="memory_fsm",
    )
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="hard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--bitrate", type=int, default=1800)
    parser.add_argument("--robot-radius", type=float, default=0.18)
    parser.add_argument(
        "--passage-model",
        type=Path,
        default=Path("data/narrow_passage_sb3_hard/ppo_narrow_passage.zip"),
    )
    parser.add_argument(
        "--risk-recovery-model",
        type=Path,
        default=Path("data/narrow_passage_risk_recovery/ppo_risk_recovery.zip"),
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-recover-steps", type=int, default=35)
    parser.add_argument("--memory-recovery-risk", type=float, default=0.45)
    parser.add_argument("--memory-trigger-count", type=int, default=1)
    parser.add_argument("--memory-reject-count", type=int, default=3)
    parser.add_argument("--clearance-low", type=float, default=0.06)
    parser.add_argument("--clearance-high", type=float, default=0.13)
    parser.add_argument("--lateral-low", type=float, default=0.24)
    parser.add_argument("--lateral-high", type=float, default=0.34)
    parser.add_argument("--heading-low", type=float, default=0.45)
    parser.add_argument("--heading-high", type=float, default=0.75)
    parser.add_argument("--high-risk-threshold", type=float, default=0.75)
    parser.add_argument("--medium-risk-threshold", type=float, default=0.50)
    parser.add_argument("--use-geometry-align", action="store_true")
    parser.add_argument("--use-medium-align", action="store_true")
    parser.add_argument("--reset-step-budget-on-retry", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("video_dir/narrow_passage_memory_fsm.mp4"),
    )
    args = parser.parse_args()

    env_config = dict(DIFFICULTY_CONFIGS[args.difficulty])
    env_config["robot_radius"] = args.robot_radius
    env = ProceduralNarrowPassageEnv(env_config)

    passage_model = None
    recovery_model = None
    if args.policy != "rule":
        PPO = require_ppo()
        passage_model = PPO.load(args.passage_model, device="cpu")
        if args.policy in {"mode_fsm", "memory_fsm"}:
            recovery_model = PPO.load(args.risk_recovery_model, device="cpu")

    if args.policy == "rule":
        frames = rollout_rule(env, args)
    elif args.policy == "passage":
        frames = rollout_passage(env, args, passage_model)
    elif args.policy == "mode_fsm":
        frames = rollout_mode_fsm(env, args, passage_model, recovery_model)
    else:
        frames = rollout_memory_fsm(env, args, passage_model, recovery_model)

    save_video(frames, args)
    print(f"frames: {len(frames)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
