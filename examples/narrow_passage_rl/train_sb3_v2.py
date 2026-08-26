#!/usr/bin/env python3
"""Train PPO/SAC/TD3 on HarderNarrowPassageEnv (v2) — fair RL baseline (Experiment ③④).

Key differences from train_sb3.py (v1):
  - Uses HarderNarrowPassageEnv (v2) instead of ProceduralNarrowPassageEnv (v1)
  - obs[10] = path-relative heading error (same as Habitat NarrowPassageGeometrySensor)
  - Includes all 7 corridor types (L-shaped, S-shaped, false_feasible, ...)
  - Supports PPO, SAC, and TD3 (pass --algo ppo|sac|td3)

The trained checkpoint can be directly evaluated on Habitat (no obs-format mismatch).

Usage
-----
    # PPO (③)
    python examples/narrow_passage_rl/train_sb3_v2.py --algo ppo \
        --total-steps 3000000 --save-dir data/narrow_passage_sb3_v2_ppo

    # SAC (④)
    python examples/narrow_passage_rl/train_sb3_v2.py --algo sac \
        --total-steps 2000000 --save-dir data/narrow_passage_sb3_v2_sac

    # TD3 (④)
    python examples/narrow_passage_rl/train_sb3_v2.py --algo td3 \
        --total-steps 2000000 --save-dir data/narrow_passage_sb3_v2_td3
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv
from fair_reward_wrapper import FairNarrowPassageRewardWrapper


# ── Corridor-type configurations ──────────────────────────────────────────────

CTYPE_CONFIGS = {
    "full": None,          # all 7 types (default probs)
    "no_ff": ["STRAIGHT", "L_SHAPED", "S_SHAPED",
               "NARROW_EXIT", "NARROW_ENTRY", "ASYMMETRIC"],
    "straight_only": ["STRAIGHT"],
    "hard": ["L_SHAPED", "S_SHAPED", "NARROW_EXIT", "NARROW_ENTRY", "ASYMMETRIC"],
}


def maybe_wrap_reward(env, reward_mode: str):
    if reward_mode == "fair":
        return FairNarrowPassageRewardWrapper(env)
    return env


def make_env(rank: int, cfg: dict, seed: int, reward_mode: str = "fair"):
    try:
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        raise
    env_cfg = dict(cfg)
    env_cfg["seed"] = seed + rank
    env = maybe_wrap_reward(HarderNarrowPassageEnv(env_cfg), reward_mode)
    return Monitor(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac", "td3"], default="ppo",
                    help="RL algorithm")
    ap.add_argument("--total-steps", type=int, default=3_000_000)
    ap.add_argument("--save-dir", type=Path, default=None,
                    help="Directory to save checkpoint (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=8,
                    help="Parallel envs (only PPO; SAC/TD3 use 1)")
    ap.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy",
                    help="PPO vectorization backend; subproc parallelizes environment stepping")
    ap.add_argument("--ctypes", choices=list(CTYPE_CONFIGS), default="full",
                    help="Corridor type set to train on")
    ap.add_argument("--reward-mode", choices=["fair", "native"], default="fair",
                    help="fair clips open-space clearance reward and penalizes timeout")
    ap.add_argument("--load-model", type=Path, default=None,
                    help="Resume from existing checkpoint")
    ap.add_argument("--eval-episodes", type=int, default=200,
                    help="Quick eval at the end; 0 = skip")
    ap.add_argument("--n-steps", type=int, default=1024,
                    help="PPO rollout steps per environment")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="PPO minibatch size")
    ap.add_argument("--n-epochs", type=int, default=10,
                    help="PPO optimization epochs per rollout")
    ap.add_argument("--device", default="cpu",
                    help="SB3 device, e.g. cpu, cuda, or auto")
    args = ap.parse_args()

    # Default save dir
    if args.save_dir is None:
        args.save_dir = Path(f"data/narrow_passage_sb3_v2_{args.algo}_{args.ctypes}")

    try:
        import stable_baselines3 as sb3
    except ImportError:
        raise ImportError("Install stable-baselines3: pip install stable-baselines3")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_sb3_v2] algo={args.algo} steps={args.total_steps} "
          f"ctypes={args.ctypes} save={args.save_dir}")

    corridor_types = CTYPE_CONFIGS[args.ctypes]
    env_cfg = {}
    if corridor_types is not None:
        env_cfg["corridor_types"] = corridor_types

    if args.algo == "ppo":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        n = args.num_envs
        env_fns = [
            lambda rank=i: make_env(rank, env_cfg, args.seed, args.reward_mode)
            for i in range(n)
        ]
        vec_env = (
            SubprocVecEnv(env_fns, start_method="fork")
            if args.vec_env == "subproc"
            else DummyVecEnv(env_fns)
        )

        if args.load_model:
            model = PPO.load(str(args.load_model), env=vec_env, device=args.device,
                             verbose=1)
        else:
            model = PPO(
                "MlpPolicy", vec_env,
                learning_rate=2.5e-4,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log=str(args.save_dir / "tb"),
                seed=args.seed,
                device=args.device,
            )

    elif args.algo == "sac":
        from stable_baselines3 import SAC

        single_env = make_env(0, env_cfg, args.seed, args.reward_mode)
        if args.load_model:
            model = SAC.load(str(args.load_model), env=single_env, device=args.device,
                             verbose=1)
        else:
            model = SAC(
                "MlpPolicy", single_env,
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
                device=args.device,
            )

    else:  # TD3
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise

        single_env = make_env(0, env_cfg, args.seed, args.reward_mode)
        action_noise = NormalActionNoise(
            mean=np.zeros(single_env.action_space.shape[-1]),
            sigma=0.10 * np.ones(single_env.action_space.shape[-1]),
        )
        if args.load_model:
            model = TD3.load(str(args.load_model), env=single_env, device=args.device,
                             verbose=1)
        else:
            model = TD3(
                "MlpPolicy", single_env,
                learning_rate=3e-4,
                buffer_size=500_000,
                learning_starts=5_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                train_freq=(1, "step"),
                gradient_steps=1,
                action_noise=action_noise,
                verbose=1,
                tensorboard_log=str(args.save_dir / "tb"),
                seed=args.seed,
                device=args.device,
            )

    from stable_baselines3.common.callbacks import CheckpointCallback
    ckpt_name = f"{args.algo}_narrow_passage_v2"
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 500_000 // max(1, args.num_envs if args.algo == "ppo" else 1)),
        save_path=str(args.save_dir),
        name_prefix=ckpt_name,
        verbose=1,
    )
    model.learn(total_timesteps=args.total_steps, callback=checkpoint_cb)
    model.save(str(args.save_dir / ckpt_name))
    print(f"[saved] {args.save_dir / ckpt_name}.zip")

    # Quick eval
    if args.eval_episodes > 0:
        print(f"\n[eval] {args.eval_episodes} episodes on v2 env (all types)...")
        eval_env = maybe_wrap_reward(HarderNarrowPassageEnv({}), args.reward_mode)
        successes = []
        by_type: dict = {}
        for ep in range(args.eval_episodes):
            obs, _ = eval_env.reset(seed=10000 + ep)
            done = False
            steps = 0
            while not done and steps < 300:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = eval_env.step(action)
                done = term or trunc
                steps += 1
            succ = float(info.get("success", 0.0))
            ctype = info.get("corridor_type", "?")
            successes.append(succ)
            by_type.setdefault(ctype, []).append(succ)

        sr = float(np.mean(successes))
        print(f"  overall SR = {sr:.3f}  ({sr*100:.1f}%)")
        for t, vals in sorted(by_type.items()):
            tsr = float(np.mean(vals))
            print(f"  {t:20s}: {tsr:.3f}  ({len(vals)} eps)")

        # Save quick eval CSV
        import csv
        eval_csv = args.save_dir / f"eval_{ckpt_name}.csv"
        with eval_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "ctype", "success"])
            writer.writeheader()
            eval_env2 = maybe_wrap_reward(HarderNarrowPassageEnv({}), args.reward_mode)
            for ep in range(args.eval_episodes):
                obs, _ = eval_env2.reset(seed=10000 + ep)
                done = False
                steps = 0
                while not done and steps < 300:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = eval_env2.step(action)
                    done = term or trunc
                    steps += 1
                writer.writerow({
                    "episode": ep,
                    "ctype": info.get("corridor_type", "?"),
                    "success": float(info.get("success", 0.0)),
                })
        print(f"[write] {eval_csv}")


if __name__ == "__main__":
    main()
