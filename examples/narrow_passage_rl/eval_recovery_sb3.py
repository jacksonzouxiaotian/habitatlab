#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from recovery_env import ProceduralRecoveryEnv
from rl_eval_utils import write_summary_csv
from train_sb3 import DIFFICULTY_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/narrow_passage_recovery/ppo_recovery.zip"))
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="hard")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional one-row summary CSV path for paper tables.",
    )
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = ProceduralRecoveryEnv(DIFFICULTY_CONFIGS[args.difficulty])
    model = PPO.load(args.model, device="cpu")
    stats = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        info = {}
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        info = dict(info)
        info["steps"] = steps
        stats.append(info)

    summary = {
        "policy": "recovery_ppo",
        "difficulty": args.difficulty,
        "episodes": args.episodes,
        "recovery_success_rate": round(
            float(np.mean([s["recovery_success"] for s in stats])), 4
        ),
        "collision_rate": round(float(np.mean([s["collision"] for s in stats])), 4),
        "stuck_rate": round(float(np.mean([s["stuck"] for s in stats])), 4),
        "avg_min_clearance": round(
            float(np.mean([s["min_clearance"] for s in stats])), 4
        ),
        "avg_steps": round(float(np.mean([s["steps"] for s in stats])), 2),
    }

    print(f"policy: {summary['policy']}")
    print(f"difficulty: {args.difficulty}")
    print(f"episodes: {args.episodes}")
    print(f"recovery_success_rate: {summary['recovery_success_rate']:.3f}")
    print(f"collision_rate: {summary['collision_rate']:.3f}")
    print(f"stuck_rate: {summary['stuck_rate']:.3f}")
    print(f"avg_min_clearance: {summary['avg_min_clearance']:.3f}")
    print(f"avg_steps: {summary['avg_steps']:.1f}")
    if args.csv is not None:
        write_summary_csv(args.csv, [summary])
        print(f"[write] {args.csv}")


if __name__ == "__main__":
    main()
