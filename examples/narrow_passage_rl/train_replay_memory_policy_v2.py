#!/usr/bin/env python3
"""Train/evaluate a generic Replay Memory Policy baseline with SB3."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv
from replay_memory_wrapper import ReplayMemoryObservationWrapper
from train_sb3_v2 import CTYPE_CONFIGS

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"


def make_env(seed: int, cfg: dict):
    from stable_baselines3.common.monitor import Monitor

    env_cfg = dict(cfg)
    env_cfg["seed"] = seed
    return Monitor(ReplayMemoryObservationWrapper(HarderNarrowPassageEnv(env_cfg)))


def evaluate(model, episodes: int, seed: int, output_csv: Path):
    env = ReplayMemoryObservationWrapper(HarderNarrowPassageEnv({"seed": seed}))
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        min_bm = float("inf")
        info = {}
        while not done and steps < env.env.max_steps:
            raw_obs = env._last_obs
            if raw_obs is not None:
                min_bm = min(min_bm, float(raw_obs[9]))
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        rows.append(
            {
                "episode": ep,
                "method": "replay_memory_policy",
                "corridor_type": info.get("corridor_type", "unknown"),
                "success": float(info.get("success", 0.0)),
                "collision": float(info.get("collision", 0.0)),
                "steps": steps,
                "min_body_margin": min_bm,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {output_csv}")
    print(
        f"[eval] replay_memory_policy episodes={episodes} "
        f"SR={np.mean([r['success'] for r in rows]):.3f} "
        f"collision={np.mean([r['collision'] for r in rows]):.3f}"
    )
    return rows


def write_summary(rows, total_steps: int, path: Path):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["corridor_type"], []).append(row)
    fields = [
        "case",
        "train_steps",
        "success_rate",
        "collision_rate",
        "avg_min_clearance",
    ] + [f"sr_{k}" for k in sorted(grouped)]
    out = {
        "case": "v2_replay_memory_policy",
        "train_steps": total_steps,
        "success_rate": round(float(np.mean([r["success"] for r in rows])), 4),
        "collision_rate": round(float(np.mean([r["collision"] for r in rows])), 4),
        "avg_min_clearance": round(float(np.mean([r["min_body_margin"] for r in rows])), 4),
    }
    for ctype, vals in sorted(grouped.items()):
        out[f"sr_{ctype}"] = round(float(np.mean([r["success"] for r in vals])), 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(out)
    print(f"[write] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac", "td3"], default="ppo")
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctypes", choices=list(CTYPE_CONFIGS), default="full")
    ap.add_argument("--save-dir", type=Path, default=RESULTS / "checkpoints" / "replay_memory_policy_v2")
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--output-csv", type=Path, default=RESULTS / "replay_memory_policy_v2_eval.csv")
    ap.add_argument("--output-summary", type=Path, default=RESULTS / "replay_memory_policy_v2_summary.csv")
    args = ap.parse_args()

    corridor_types = CTYPE_CONFIGS[args.ctypes]
    env_cfg = {}
    if corridor_types is not None:
        env_cfg["corridor_types"] = corridor_types
    env = make_env(args.seed, env_cfg)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(args.save_dir / "tb"),
            seed=args.seed,
            device="cpu",
        )
    elif args.algo == "sac":
        from stable_baselines3 import SAC

        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=500_000,
            learning_starts=5_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            verbose=1,
            tensorboard_log=str(args.save_dir / "tb"),
            seed=args.seed,
            device="cpu",
        )
    else:
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise

        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[-1]),
            sigma=0.10 * np.ones(env.action_space.shape[-1]),
        )
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=500_000,
            learning_starts=5_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=str(args.save_dir / "tb"),
            seed=args.seed,
            device="cpu",
        )

    model.learn(total_timesteps=args.total_steps)
    ckpt = args.save_dir / f"{args.algo}_replay_memory_policy_v2"
    model.save(str(ckpt))
    print(f"[saved] {ckpt}.zip")

    if args.eval_episodes > 0:
        rows = evaluate(model, args.eval_episodes, 10000 + args.seed, args.output_csv)
        write_summary(rows, args.total_steps, args.output_summary)


if __name__ == "__main__":
    main()
