#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from failure_memory import FailureMemoryConfig, PassageFailureMemory
from procedural_env import ProceduralNarrowPassageEnv
from rl_eval_utils import (
    finalize_episode_info,
    make_episode_trace,
    summarize_episode_stats,
    update_episode_trace,
)
from risk_estimator import GeometryRiskEstimator, RiskConfig
from train_sb3 import DIFFICULTY_CONFIGS


def align_action(obs):
    heading_error = float(obs[10])
    lateral_offset = float(obs[11])
    clearance_left = float(obs[6])
    clearance_right = float(obs[7])
    wz = -0.9 * heading_error - 1.2 * lateral_offset
    wz += 0.5 * (clearance_right - clearance_left)
    return np.array([0.08, np.clip(wz, -0.6, 0.6)], dtype=np.float32)


def is_recovered_state(obs):
    return (
        abs(float(obs[11])) < 0.12
        and abs(float(obs[10])) < 0.25
        and min(float(obs[6]), float(obs[7])) > 0.14
        and float(obs[15]) < 0.25
    )


def choose_mode(obs, estimator, memory, args):
    risk_info = estimator.score(obs)
    risk = risk_info["risk"]
    if memory.should_reject(obs):
        return "REJECT", risk_info
    memory_active = (
        memory.should_recover(obs)
        or memory.total_failures >= args.memory_trigger_count
    )
    if memory_active and risk >= args.memory_recovery_risk:
        return "RECOVER", risk_info
    if args.use_geometry_align and risk >= args.high_risk_threshold:
        return "ALIGN", risk_info
    if args.use_medium_align and risk >= args.medium_risk_threshold:
        return "ALIGN", risk_info
    return "COMMIT", risk_info


def restore_attempt_start(env, start_pose, args):
    env.pose[:] = start_pose
    env.prev_action[:] = 0.0
    env.prev_dist = env._distance_to_goal()
    env.stuck_steps = 0
    env.collision = False
    env.step_count = 0 if args.reset_step_budget_on_retry else env.step_count
    return env._obs()


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="hard")
    parser.add_argument("--episodes", type=int, default=500)
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
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = ProceduralNarrowPassageEnv(DIFFICULTY_CONFIGS[args.difficulty])
    passage_model = PPO.load(args.passage_model, device="cpu")
    recovery_model = PPO.load(args.risk_recovery_model, device="cpu")

    risk_cfg = RiskConfig(
        clearance_low=args.clearance_low,
        clearance_high=args.clearance_high,
        lateral_low=args.lateral_low,
        lateral_high=args.lateral_high,
        heading_low=args.heading_low,
        heading_high=args.heading_high,
        high_risk_threshold=args.high_risk_threshold,
        medium_risk_threshold=args.medium_risk_threshold,
        reject_after_failures=args.memory_reject_count,
    )
    mem_cfg = FailureMemoryConfig(
        trigger_count=args.memory_trigger_count,
        reject_count=args.memory_reject_count,
    )

    stats = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        trace = make_episode_trace()
        update_episode_trace(trace, obs)
        start_pose = env.pose.copy()
        estimator = GeometryRiskEstimator(risk_cfg)
        memory = PassageFailureMemory(mem_cfg)
        done = False
        info = {}
        attempts = 1
        mode_counts = {"COMMIT": 0, "ALIGN": 0, "RECOVER": 0, "REJECT": 0}
        recovery_successes = 0
        recovery_failures = 0
        recovery_steps = 0
        memory_queries = 0
        memory_writes = 0
        max_risk = 0.0

        while not done:
            mode, risk_info = choose_mode(obs, estimator, memory, args)
            max_risk = max(max_risk, risk_info["risk"])
            mode_counts[mode] += 1

            if mode == "REJECT":
                info = {
                    "success": 0.0,
                    "collision": 0.0,
                    "stuck": float(obs[15] >= 0.7),
                    "min_clearance": min(float(obs[6]), float(obs[7])),
                    "passage_width": env.params.width,
                    "false_feasible": float(env.params.false_feasible),
                    "rejected": 1.0,
                }
                done = True
                continue

            if mode == "RECOVER":
                memory_queries += 1
                recovered = False
                for _ in range(args.max_recover_steps):
                    action, _ = recovery_model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    update_episode_trace(trace, obs)
                    recovery_steps += 1
                    if terminated and info.get("collision", 0.0) > 0.5:
                        estimator.mark_failure()
                        memory.add_failure(obs)
                        memory_writes += 1
                        recovery_failures += 1
                        if attempts < args.max_attempts:
                            attempts += 1
                            obs = restore_attempt_start(env, start_pose, args)
                            terminated = False
                            truncated = False
                            info = {}
                        else:
                            done = True
                        break
                    if is_recovered_state(obs):
                        recovered = True
                        recovery_successes += 1
                        break
                    if terminated or truncated:
                        done = True
                        break
                if done:
                    continue
                if not recovered:
                    estimator.mark_failure()
                    memory.add_failure(obs)
                    memory_writes += 1
                    recovery_failures += 1
                continue

            action = align_action(obs) if mode == "ALIGN" else passage_model.predict(obs, deterministic=True)[0]
            prev_obs = obs.copy()
            obs, _, terminated, truncated, info = env.step(action)
            update_episode_trace(trace, obs)
            if terminated and info.get("collision", 0.0) > 0.5:
                estimator.mark_failure()
                memory.add_failure(prev_obs)
                memory_writes += 1
                if attempts < args.max_attempts:
                    attempts += 1
                    obs = restore_attempt_start(env, start_pose, args)
                    terminated = False
                    truncated = False
                    info = {}
                else:
                    done = True
                    continue
            done = terminated or truncated

        info = finalize_episode_info(info, trace)
        info.setdefault("rejected", 0.0)
        info["attempts"] = attempts
        info["mode_commit"] = mode_counts["COMMIT"]
        info["mode_align"] = mode_counts["ALIGN"]
        info["mode_recover"] = mode_counts["RECOVER"]
        info["mode_reject"] = mode_counts["REJECT"]
        info["recovery_successes"] = recovery_successes
        info["recovery_failures"] = recovery_failures
        info["recovery_steps"] = recovery_steps
        info["memory_queries"] = memory_queries
        info["memory_writes"] = memory_writes
        info["memory_failed_states"] = memory.num_failed_states
        info["memory_total_failures"] = memory.total_failures
        info["max_risk"] = max_risk
        stats.append(info)

    print("policy: failure_memory_gated_fsm")
    print(f"difficulty: {args.difficulty}")
    print(f"episodes: {args.episodes}")
    summary = summarize_episode_stats(stats)
    print(f"success_rate: {np.mean([s['success'] for s in stats]):.3f}")
    print(f"collision_rate: {np.mean([s['collision'] for s in stats]):.3f}")
    print(f"stuck_rate: {np.mean([s['stuck'] for s in stats]):.3f}")
    print(f"near_collision_rate: {summary['near_collision_rate']:.3f}")
    print(f"avg_high_risk_fraction: {summary['avg_high_risk_fraction']:.3f}")
    print(f"false_feasible_success_rate: {summary.get('false_feasible_success_rate', 0.0):.3f}")
    print(f"false_feasible_collision_rate: {summary.get('false_feasible_collision_rate', 0.0):.3f}")
    print(f"narrow_success_rate: {summary.get('narrow_success_rate', 0.0):.3f}")
    print(f"narrow_collision_rate: {summary.get('narrow_collision_rate', 0.0):.3f}")
    print(f"reject_rate: {np.mean([s['rejected'] for s in stats]):.3f}")
    print(f"avg_min_clearance: {np.mean([s['min_clearance'] for s in stats]):.3f}")
    print(f"avg_recovery_successes: {np.mean([s['recovery_successes'] for s in stats]):.3f}")
    print(f"avg_recovery_failures: {np.mean([s['recovery_failures'] for s in stats]):.3f}")
    print(f"avg_recovery_steps: {np.mean([s['recovery_steps'] for s in stats]):.1f}")
    print(f"avg_attempts: {np.mean([s['attempts'] for s in stats]):.3f}")
    print(f"avg_commit_steps: {np.mean([s['mode_commit'] for s in stats]):.1f}")
    print(f"avg_align_steps: {np.mean([s['mode_align'] for s in stats]):.1f}")
    print(f"avg_recover_triggers: {np.mean([s['mode_recover'] for s in stats]):.3f}")
    print(f"avg_memory_queries: {np.mean([s['memory_queries'] for s in stats]):.3f}")
    print(f"avg_memory_writes: {np.mean([s['memory_writes'] for s in stats]):.3f}")
    print(f"avg_memory_failed_states: {np.mean([s['memory_failed_states'] for s in stats]):.3f}")
    print(f"avg_max_risk: {np.mean([s['max_risk'] for s in stats]):.3f}")


if __name__ == "__main__":
    main()
