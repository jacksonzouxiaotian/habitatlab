#!/usr/bin/env python3

import argparse
from pathlib import Path

from obs_wrappers import ABLATION_MASKS, apply_ablation
from procedural_env import ProceduralNarrowPassageEnv
from rl_eval_utils import (
    finalize_episode_info,
    make_episode_trace,
    print_summary,
    summarize_episode_stats,
    update_episode_trace,
    write_summary_csv,
)
from train_sb3 import DIFFICULTY_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/narrow_passage_sb3/ppo_narrow_passage.zip"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="hard")
    parser.add_argument("--ablation", choices=ABLATION_MASKS, default="full")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = apply_ablation(
        ProceduralNarrowPassageEnv(DIFFICULTY_CONFIGS[args.difficulty]),
        args.ablation,
    )
    model = PPO.load(args.model, device="cpu")
    stats = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        trace = make_episode_trace()
        update_episode_trace(trace, obs)
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            update_episode_trace(trace, obs)
            done = terminated or truncated
        stats.append(finalize_episode_info(info, trace))

    summary = summarize_episode_stats(stats)
    summary.update(
        {
            "policy": "passage_ppo",
            "difficulty": args.difficulty,
            "ablation": args.ablation,
            "model": str(args.model),
        }
    )
    print_summary(summary)
    if args.csv is not None:
        write_summary_csv(args.csv, [summary])


if __name__ == "__main__":
    main()
