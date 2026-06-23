#!/usr/bin/env python3
"""Evaluate trained NarrowPassagePolicy on the Habitat val set.

Loads the final checkpoint (ckpt.49.pth) from data/narrow_passage_checkpoints/,
runs all 157 val episodes, and outputs a per-episode CSV in the same format
as eval_habitat_geometry_fsm.py — compatible with plot_delta_d_phase.py.

The trained policy:
  - Input: narrow_passage_features (19-dim) + narrow_passage_memory (4-dim) = 23
  - Output: 2D Gaussian action [lin_vel_norm, ang_vel_norm] in [-1, 1]
  - Architecture: Linear(23,256) → LayerNorm → ReLU → Linear(256,256) → GRU(256)

Run:
    conda run -n habitat python3 examples/narrow_passage_rl/eval_habitat_ppo_policy.py
    conda run -n habitat python3 examples/narrow_passage_rl/eval_habitat_ppo_policy.py \
        --ckpt data/narrow_passage_checkpoints/ckpt.49.pth --num-episodes -1
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from gym import spaces as gym_spaces

import habitat
from habitat_baselines.rl.ppo.narrow_passage_policy import NarrowPassagePolicy

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"
DEFAULT_CKPT = Path("data/narrow_passage_checkpoints/latest.pth")
HIDDEN_SIZE = 256
FEATURE_DIM = 19
MEMORY_DIM = 4


def make_obs_space():
    return gym_spaces.Dict({
        "narrow_passage_features": gym_spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(FEATURE_DIM,),
            dtype=np.float32,
        ),
        "narrow_passage_memory": gym_spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(MEMORY_DIM,),
            dtype=np.float32,
        ),
    })


def make_action_space():
    return gym_spaces.Box(
        low=-1.0, high=1.0, shape=(2,), dtype=np.float32
    )


def obs_to_tensors(obs: dict, device="cpu"):
    """Convert habitat observation dict to tensors with batch dim=1."""
    feats = obs.get("narrow_passage_features", np.zeros(FEATURE_DIM, dtype=np.float32))
    mem = obs.get("narrow_passage_memory", np.zeros(MEMORY_DIM, dtype=np.float32))
    return {
        "narrow_passage_features": torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0),
        "narrow_passage_memory": torch.tensor(mem, dtype=torch.float32, device=device).unsqueeze(0),
    }


def make_env(args):
    data_path = args.data_path.format(split=args.split)
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            f"habitat.dataset.data_path={data_path}",
            f"habitat.dataset.split={args.split}",
            "habitat.dataset.type=PointNav-v1",
            "habitat.simulator.habitat_sim_v0.allow_sliding=False",
        ],
    )
    return habitat.Env(config=config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data-path", default="data/datasets/narrow_passage/{split}/{split}.json.gz")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-episodes", type=int, default=-1, help="-1 = all episodes")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── load policy ───────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_config = ckpt["config"]

    # Use from_config so GaussianNet is built with the exact same hyper-params
    # (action_dist, clamp_std, etc.) that were used during training.
    actor_critic = NarrowPassagePolicy.from_config(
        train_config,
        observation_space=make_obs_space(),
        action_space=make_action_space(),
    ).to(device)
    actor_critic.eval()
    actor_critic.load_state_dict(ckpt["state_dict"])
    print(f"Loaded checkpoint: {args.ckpt}")

    # ── run evaluation ────────────────────────────────────────────────────
    all_stats = []

    with make_env(args) as env:
        total = env.number_of_episodes
        num_ep = total if args.num_episodes <= 0 else min(args.num_episodes, total)
        print(f"[habitat_ppo_policy] evaluating {num_ep}/{total} episodes (split={args.split})")

        for ep_idx in range(num_ep):
            obs = env.reset()
            episode = env.current_episode

            # Extract difficulty/body_margin from episode metadata
            info = episode.info if hasattr(episode, "info") and episode.info else {}
            body_margin = float(info.get("body_margin", float("nan")))
            difficulty = info.get("difficulty", "?")
            episode_id = str(episode.episode_id)
            scene_id = str(episode.scene_id).split("/")[-2]

            # Initial RNN state + masks (bool dtype required by RNN state encoder)
            rnn_hidden = torch.zeros(1, 1, HIDDEN_SIZE, device=device)
            masks = torch.zeros(1, 1, dtype=torch.bool, device=device)
            prev_actions = torch.zeros(1, 2, device=device)

            steps = 0
            any_collision = False
            min_bm = float("inf")
            near_collision = False
            done = False

            with torch.no_grad():
                while not done and steps < args.max_steps:
                    obs_t = obs_to_tensors(obs, device)
                    feats = obs.get("narrow_passage_features", np.zeros(FEATURE_DIM, dtype=np.float32))
                    bm = float(feats[9])  # body_margin at this step
                    if bm < min_bm:
                        min_bm = bm
                    if bm < 0.05:
                        near_collision = True
                    if float(feats[16]) > 0.5:  # collision_flag
                        any_collision = True

                    action_data = actor_critic.act(
                        obs_t, rnn_hidden, prev_actions, masks, deterministic=True
                    )
                    rnn_hidden = action_data.rnn_hidden_states
                    masks = torch.ones(1, 1, dtype=torch.bool, device=device)

                    act = action_data.actions[0]  # shape (2,)
                    lin_norm = float(torch.clamp(act[0], -1.0, 1.0))
                    ang_norm = float(torch.clamp(act[1], -1.0, 1.0))
                    prev_actions = action_data.actions

                    obs = env.step({
                        "action": "velocity_control",
                        "action_args": {
                            "linear_velocity": lin_norm,
                            "angular_velocity": ang_norm,
                        },
                    })
                    steps += 1
                    if env.episode_over:
                        done = True

            metrics = env.get_metrics()
            success = float(metrics.get("narrow_passage_success", 0.0))
            collision = float(any_collision)
            stuck = float(metrics.get("narrow_passage_stuck", 0.0))

            if not np.isfinite(min_bm):
                feats = obs.get("narrow_passage_features", np.zeros(FEATURE_DIM))
                min_bm = float(feats[9])

            stat = {
                "episode_id": episode_id,
                "scene_id": scene_id,
                "difficulty": difficulty,
                "body_margin": round(body_margin, 4),
                "steps": steps,
                "success": success,
                "collision": collision,
                "stuck": stuck,
                "rejected": 0.0,
                "near_collision": float(near_collision),
                "min_clearance": float(min_bm),
                "recover_triggers": 0,
                "memory_writes": 0,
            }
            all_stats.append(stat)

            print(
                f"  ep {ep_idx:3d}  {difficulty:6s}  bm={body_margin:+.3f}"
                f"  success={success:.0f}  collision={collision:.0f}"
                f"  stuck={stuck:.0f}  steps={steps:3d}"
            )

    # ── summary ────────────────────────────────────────────────────────────
    n = len(all_stats)
    sr = np.mean([s["success"] for s in all_stats])
    cr = np.mean([s["collision"] for s in all_stats])
    sr_str = np.mean([s["stuck"] for s in all_stats])
    ncr = np.mean([s["near_collision"] for s in all_stats])
    avg_bm = np.mean([s["min_clearance"] for s in all_stats])

    print(f"\n=== habitat_ppo_policy  |  {n} episodes  |  split={args.split} ===")
    print(f"  success_rate:        {sr:.3f}")
    print(f"  collision_rate:      {cr:.3f}")
    print(f"  stuck_rate:          {sr_str:.3f}")
    print(f"  near_collision_rate: {ncr:.3f}")
    print(f"  avg_min_clearance:   {avg_bm:.3f}")

    print()
    for diff in ["narrow", "normal", "wide"]:
        rows = [s for s in all_stats if s["difficulty"] == diff]
        if rows:
            d_sr = np.mean([r["success"] for r in rows])
            print(f"  {diff:6s}: SR={d_sr:.3f} ({sum(r['success']>0.5 for r in rows)}/{len(rows)})")

    print()
    print("# Copy into results/narrow_passage_rl/results_rl_summary.csv:")
    print(f"habitat_ppo_policy,{sr:.4f},{cr:.4f},{cr:.4f},{ncr:.4f},{avg_bm:.4f},,,,0.0000,0.0000")

    out_path = args.output_csv or (RESULTS / "habitat_ppo_policy_episodes.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_stats[0].keys()))
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
