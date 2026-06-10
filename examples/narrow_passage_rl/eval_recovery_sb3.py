#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from recovery_env import ProceduralRecoveryEnv
from train_sb3 import DIFFICULTY_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/narrow_passage_recovery/ppo_recovery.zip"))
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="hard")
    parser.add_argument("--episodes", type=int, default=500)
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

    print(f"policy: recovery_ppo")
    print(f"difficulty: {args.difficulty}")
    print(f"episodes: {args.episodes}")
    print(f"recovery_success_rate: {np.mean([s['recovery_success'] for s in stats]):.3f}")
    print(f"collision_rate: {np.mean([s['collision'] for s in stats]):.3f}")
    print(f"stuck_rate: {np.mean([s['stuck'] for s in stats]):.3f}")
    print(f"avg_min_clearance: {np.mean([s['min_clearance'] for s in stats]):.3f}")
    print(f"avg_steps: {np.mean([s['steps'] for s in stats]):.1f}")


if __name__ == "__main__":
    main()
