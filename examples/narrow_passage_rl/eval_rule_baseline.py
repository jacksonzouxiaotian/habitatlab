#!/usr/bin/env python3

import argparse

import numpy as np
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


class CenterlineRulePolicy:
    def __init__(self):
        self.recovery_steps = 0
        self.recovery_turn = 1.0

    def reset(self):
        self.recovery_steps = 0
        self.recovery_turn = 1.0

    def act(self, obs):
        clearance_left = obs[6]
        clearance_right = obs[7]
        heading_error = obs[10]
        lateral_offset = obs[11]
        collision = obs[16] > 0.5

        if collision:
            self.recovery_steps = 10
            self.recovery_turn = -1.0 if clearance_left < clearance_right else 1.0

        if self.recovery_steps > 0:
            self.recovery_steps -= 1
            return np.array([-0.12, 0.45 * self.recovery_turn], dtype=np.float32)

        center_cmd = -1.8 * lateral_offset
        align_cmd = -1.2 * heading_error
        clearance_cmd = 0.8 * (clearance_right - clearance_left)
        wz = np.clip(center_cmd + align_cmd + clearance_cmd, -0.8, 0.8)

        min_clearance = min(clearance_left, clearance_right)
        vx = 0.28
        if abs(heading_error) > 0.35 or abs(lateral_offset) > 0.25:
            vx = 0.12
        if min_clearance < 0.12:
            vx = 0.06
        return np.array([vx, wz], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="hard")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    env = ProceduralNarrowPassageEnv(DIFFICULTY_CONFIGS[args.difficulty])
    policy = CenterlineRulePolicy()
    stats = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        trace = make_episode_trace()
        update_episode_trace(trace, obs)
        policy.reset()
        done = False
        info = {}
        steps = 0
        while not done:
            action = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            update_episode_trace(trace, obs)
            done = terminated or truncated
            steps += 1
        info = finalize_episode_info(info, trace)
        info["steps"] = steps
        stats.append(info)

    summary = summarize_episode_stats(stats)
    summary.update(
        {
            "policy": "rule_centerline_recovery",
            "difficulty": args.difficulty,
            "avg_steps": round(float(np.mean([s["steps"] for s in stats])), 4),
        }
    )
    print_summary(summary)
    if args.csv is not None:
        write_summary_csv(args.csv, [summary])


if __name__ == "__main__":
    main()
