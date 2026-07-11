#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from procedural_env import ProceduralNarrowPassageEnv
from rl_eval_utils import (
    finalize_episode_info,
    make_episode_trace,
    summarize_episode_stats,
    update_episode_trace,
    write_summary_csv,
)
from train_sb3 import DIFFICULTY_CONFIGS


def is_recovered_state(env):
    obs = env._obs()
    return (
        abs(obs[11]) < 0.12
        and abs(obs[10]) < 0.25
        and min(obs[6], obs[7]) > 0.12
        and obs[15] < 0.25
    )


def prepare_recovery_state(env):
    """Move the robot from a terminal collision pose to a recoverable pose."""
    p = env.params
    wall_limit = p.width * 0.5 - env.robot_radius - 0.02
    if wall_limit > 0.0:
        env.pose[0] = float(np.clip(env.pose[0], -wall_limit, wall_limit))
    env.pose[1] = max(-0.2, float(env.pose[1] - 0.12))
    env.collision = False
    env.stuck_steps = max(env.stuck_steps, 12)


def is_risk_state(obs, clearance_threshold, lateral_threshold, heading_threshold):
    min_clearance = min(obs[6], obs[7])
    return (
        min_clearance < clearance_threshold
        or abs(obs[11]) > lateral_threshold
        or abs(obs[10]) > heading_threshold
        or obs[15] > 0.5
    )


def apply_recovery_to_env(
    env, recovery_model, max_recovery_steps, min_recovery_steps, trace=None
):
    recovery_steps = 0
    for _ in range(max_recovery_steps):
        obs = env._obs()
        if recovery_steps >= min_recovery_steps and is_recovered_state(env):
            return True, recovery_steps
        action, _ = recovery_model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        env.collision = False
        obs, _, terminated, _, _ = env.step(action)
        if trace is not None:
            update_episode_trace(trace, obs)
        recovery_steps += 1
        if terminated and env.collision:
            return False, recovery_steps
        if terminated and not env.collision:
            return True, recovery_steps
    return is_recovered_state(env), recovery_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passage-model", type=Path, default=Path("data/narrow_passage_sb3_hard/ppo_narrow_passage.zip"))
    parser.add_argument("--recovery-model", type=Path, default=Path("data/narrow_passage_recovery/ppo_recovery.zip"))
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="hard")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-recovery-steps", type=int, default=40)
    parser.add_argument("--min-recovery-steps", type=int, default=3)
    parser.add_argument("--enable-risk-recovery", action="store_true")
    parser.add_argument(
        "--risk-recovery-model",
        type=Path,
        default=None,
        help="Use this model for pre-collision risk recovery if provided.",
    )
    parser.add_argument("--risk-clearance-threshold", type=float, default=0.10)
    parser.add_argument("--risk-lateral-threshold", type=float, default=0.30)
    parser.add_argument("--risk-heading-threshold", type=float, default=0.65)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional one-row summary CSV path for paper tables.",
    )
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = ProceduralNarrowPassageEnv(DIFFICULTY_CONFIGS[args.difficulty])
    passage_model = PPO.load(args.passage_model, device="cpu")
    recovery_model = PPO.load(args.recovery_model, device="cpu")
    risk_recovery_model = (
        PPO.load(args.risk_recovery_model, device="cpu")
        if args.risk_recovery_model is not None
        else recovery_model
    )

    stats = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        trace = make_episode_trace()
        update_episode_trace(trace, obs)
        done = False
        info = {}
        retries = 0
        recovery_successes = 0
        total_recovery_steps = 0
        collision_count = 0
        risk_triggers = 0
        while not done:
            action, _ = passage_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            update_episode_trace(trace, obs)
            should_recover = False
            triggered_by_collision = terminated and info.get("collision", 0.0) > 0.5
            triggered_by_risk = (
                args.enable_risk_recovery
                and
                not terminated
                and retries < args.max_retries
                and is_risk_state(
                    obs,
                    args.risk_clearance_threshold,
                    args.risk_lateral_threshold,
                    args.risk_heading_threshold,
                )
            )

            if triggered_by_collision or triggered_by_risk:
                should_recover = retries < args.max_retries

            if should_recover:
                if triggered_by_collision:
                    collision_count += 1
                    prepare_recovery_state(env)
                else:
                    risk_triggers += 1
                    env.stuck_steps = max(env.stuck_steps, 8)
                retries += 1
                active_recovery_model = (
                    risk_recovery_model if triggered_by_risk else recovery_model
                )
                recovered, rec_steps = apply_recovery_to_env(
                    env,
                    active_recovery_model,
                    args.max_recovery_steps,
                    args.min_recovery_steps,
                    trace,
                )
                total_recovery_steps += rec_steps
                if recovered:
                    recovery_successes += 1
                    obs = env._obs()
                    terminated = False
                    truncated = False
                    info = dict(info)
                    info["collision"] = 0.0
                else:
                    done = True
            else:
                done = terminated or truncated

        info = finalize_episode_info(info, trace)
        info["retries"] = retries
        info["recovery_successes"] = recovery_successes
        info["collision_count"] = collision_count
        info["risk_triggers"] = risk_triggers
        info["recovery_steps"] = total_recovery_steps
        stats.append(info)

    print(f"policy: passage_ppo_plus_recovery_ppo")
    print(f"difficulty: {args.difficulty}")
    print(f"episodes: {args.episodes}")
    summary = summarize_episode_stats(stats)
    print(f"success_rate: {np.mean([s['success'] for s in stats]):.3f}")
    print(f"final_collision_rate: {np.mean([s['collision'] for s in stats]):.3f}")
    print(f"stuck_rate: {np.mean([s['stuck'] for s in stats]):.3f}")
    print(f"near_collision_rate: {summary['near_collision_rate']:.3f}")
    print(f"avg_high_risk_fraction: {summary['avg_high_risk_fraction']:.3f}")
    print(f"false_feasible_success_rate: {summary.get('false_feasible_success_rate', 0.0):.3f}")
    print(f"false_feasible_collision_rate: {summary.get('false_feasible_collision_rate', 0.0):.3f}")
    print(f"narrow_success_rate: {summary.get('narrow_success_rate', 0.0):.3f}")
    print(f"narrow_collision_rate: {summary.get('narrow_collision_rate', 0.0):.3f}")
    print(f"avg_retries: {np.mean([s['retries'] for s in stats]):.3f}")
    print(f"avg_risk_triggers: {np.mean([s['risk_triggers'] for s in stats]):.3f}")
    print(f"avg_recovery_successes: {np.mean([s['recovery_successes'] for s in stats]):.3f}")
    print(f"avg_recovery_steps: {np.mean([s['recovery_steps'] for s in stats]):.1f}")
    print(f"avg_min_clearance: {np.mean([s['min_clearance'] for s in stats]):.3f}")
    if args.csv is not None:
        row = {
            "policy": "passage_ppo_plus_recovery_ppo",
            "difficulty": args.difficulty,
            "episodes": args.episodes,
            "max_retries": args.max_retries,
            "enable_risk_recovery": float(args.enable_risk_recovery),
            **summary,
            "avg_retries": round(float(np.mean([s["retries"] for s in stats])), 4),
            "avg_risk_triggers": round(
                float(np.mean([s["risk_triggers"] for s in stats])), 4
            ),
            "avg_recovery_successes": round(
                float(np.mean([s["recovery_successes"] for s in stats])), 4
            ),
            "avg_recovery_steps": round(
                float(np.mean([s["recovery_steps"] for s in stats])), 2
            ),
        }
        write_summary_csv(args.csv, [row])
        print(f"[write] {args.csv}")


if __name__ == "__main__":
    main()
