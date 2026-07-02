#!/usr/bin/env python3
"""Train/evaluate SB3-Contrib RecurrentPPO on HarderNarrowPassageEnv v2."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv
from fair_reward_wrapper import FairNarrowPassageRewardWrapper
from train_sb3_v2 import CTYPE_CONFIGS

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"


def maybe_wrap_reward(env, reward_mode: str):
    if reward_mode == "fair":
        return FairNarrowPassageRewardWrapper(env)
    return env


def make_env(seed: int, cfg: dict, reward_mode: str):
    from stable_baselines3.common.monitor import Monitor

    env_cfg = dict(cfg)
    env_cfg["seed"] = seed
    return Monitor(maybe_wrap_reward(HarderNarrowPassageEnv(env_cfg), reward_mode))


def evaluate(model, episodes: int, seed: int, output_csv: Path, reward_mode: str):
    env = maybe_wrap_reward(HarderNarrowPassageEnv({"seed": seed}), reward_mode)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        done = False
        steps = 0
        min_bm = float("inf")
        info = {}
        while not done and steps < env.max_steps:
            min_bm = min(min_bm, float(obs[9]))
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start[:] = False
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        rows.append(
            {
                "episode": ep,
                "method": "recurrent_ppo",
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
    sr = float(np.mean([r["success"] for r in rows]))
    col = float(np.mean([r["collision"] for r in rows]))
    print(f"[eval] recurrent_ppo episodes={episodes} SR={sr:.3f} collision={col:.3f}")
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
        "case": "v2_recurrent_ppo",
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
    ap.add_argument("--total-steps", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctypes", choices=list(CTYPE_CONFIGS), default="full")
    ap.add_argument("--save-dir", type=Path, default=RESULTS / "checkpoints" / "recurrent_ppo_v2")
    ap.add_argument("--load-model", type=Path, default=None)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--reward-mode", choices=["fair", "native"], default="fair",
                    help="fair clips open-space clearance reward and penalizes timeout")
    ap.add_argument("--output-csv", type=Path, default=RESULTS / "recurrent_ppo_v2_eval.csv")
    ap.add_argument("--output-summary", type=Path, default=RESULTS / "recurrent_ppo_v2_summary.csv")
    args = ap.parse_args()

    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        raise ImportError("Install sb3-contrib to run RecurrentPPO.") from exc

    args.save_dir.mkdir(parents=True, exist_ok=True)
    corridor_types = CTYPE_CONFIGS[args.ctypes]
    env_cfg = {}
    if corridor_types is not None:
        env_cfg["corridor_types"] = corridor_types
    env = make_env(args.seed, env_cfg, args.reward_mode)

    if args.load_model:
        model = RecurrentPPO.load(str(args.load_model), env=env, device="cpu", verbose=1)
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
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

    model.learn(total_timesteps=args.total_steps)
    ckpt = args.save_dir / "recurrent_ppo_narrow_passage_v2"
    model.save(str(ckpt))
    print(f"[saved] {ckpt}.zip")

    if args.eval_episodes > 0:
        rows = evaluate(
            model,
            args.eval_episodes,
            10000 + args.seed,
            args.output_csv,
            args.reward_mode,
        )
        write_summary(rows, args.total_steps, args.output_summary)


if __name__ == "__main__":
    main()
