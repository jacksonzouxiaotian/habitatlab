#!/usr/bin/env python3
"""Collect Geometry-FSM expert trajectories for BC/DAgger baselines."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from eval_harder_benchmark import TurnCommitFSM
from procedural_env_v2 import HarderNarrowPassageEnv
from train_sb3_v2 import CTYPE_CONFIGS

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"


def collect(args):
    corridor_types = CTYPE_CONFIGS[args.ctypes]
    env_cfg = {"seed": args.seed}
    if corridor_types is not None:
        env_cfg["corridor_types"] = corridor_types
    env = HarderNarrowPassageEnv(env_cfg)

    obs_rows = []
    action_rows = []
    mode_rows = []
    episode_rows = []
    step_rows = []
    ctype_rows = []
    episode_summary = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        fsm = TurnCommitFSM(variant=args.variant)
        done = False
        steps = 0
        min_bm = float("inf")
        info = {}
        while not done and steps < args.max_steps:
            min_bm = min(min_bm, float(obs[9]))
            action, mode = fsm.step(obs)
            obs_rows.append(obs.astype(np.float32))
            action_rows.append(action.astype(np.float32))
            mode_rows.append(mode)
            episode_rows.append(ep)
            step_rows.append(steps)
            ctype_rows.append(env._params.ctype.value)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        episode_summary.append(
            {
                "episode": ep,
                "corridor_type": info.get("corridor_type", "unknown"),
                "success": float(info.get("success", 0.0)),
                "collision": float(info.get("collision", 0.0)),
                "steps": steps,
                "min_body_margin": min_bm,
            }
        )
        if ep < 5 or (ep + 1) % 25 == 0:
            sr = np.mean([r["success"] for r in episode_summary])
            print(f"[expert] ep={ep + 1}/{args.episodes} rolling_SR={sr:.3f} steps={steps}")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        obs=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.float32),
        modes=np.asarray(mode_rows),
        episode_ids=np.asarray(episode_rows, dtype=np.int32),
        step_ids=np.asarray(step_rows, dtype=np.int32),
        corridor_types=np.asarray(ctype_rows),
    )
    print(f"[write] {args.output_npz} transitions={len(obs_rows)}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_summary[0].keys()))
            writer.writeheader()
            writer.writerows(episode_summary)
        print(f"[write] {args.output_csv}")
    return episode_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctypes", choices=list(CTYPE_CONFIGS), default="full")
    ap.add_argument("--variant", default="full", choices=["full", "no_recovery", "no_alignment"])
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--output-npz", type=Path, default=RESULTS / "expert_fsm_v2.npz")
    ap.add_argument("--output-csv", type=Path, default=RESULTS / "expert_fsm_v2_episodes.csv")
    args = ap.parse_args()
    rows = collect(args)
    print(
        f"[summary] episodes={len(rows)} SR={np.mean([r['success'] for r in rows]):.3f} "
        f"collision={np.mean([r['collision'] for r in rows]):.3f}"
    )


if __name__ == "__main__":
    main()
