#!/usr/bin/env python3

import argparse
from pathlib import Path

from obs_wrappers import ABLATION_MASKS, apply_ablation
from procedural_env import ProceduralNarrowPassageEnv


DIFFICULTY_CONFIGS = {
    "easy": {
        "width_range": (0.85, 1.2),
        "yaw_range": (-0.35, 0.35),
        "start_x_range": (-0.18, 0.18),
        "false_feasible_prob": 0.0,
    },
    "medium": {
        "width_range": (0.65, 1.2),
        "yaw_range": (-0.55, 0.55),
        "start_x_range": (-0.28, 0.28),
        "false_feasible_prob": 0.05,
    },
    "hard": {
        "width_range": (0.45, 1.2),
        "yaw_range": (-0.75, 0.75),
        "start_x_range": (-0.35, 0.35),
        "false_feasible_prob": 0.15,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--save-dir", type=Path, default=Path("data/narrow_passage_sb3"))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="easy")
    parser.add_argument("--ablation", choices=ABLATION_MASKS, default="full")
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for this training script. Install it "
            "inside your habitat conda env, then rerun this script."
        ) from exc

    args.save_dir.mkdir(parents=True, exist_ok=True)

    def make_env(rank):
        env_config = dict(DIFFICULTY_CONFIGS[args.difficulty])
        env_config["seed"] = args.seed + rank
        env = ProceduralNarrowPassageEnv(env_config)
        return Monitor(apply_ablation(env, args.ablation))

    env = DummyVecEnv([lambda rank=i: make_env(rank) for i in range(args.num_envs)])
    if args.load_model is not None:
        model = PPO.load(args.load_model, env=env, device="cpu")
    else:
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
    model.learn(total_timesteps=args.total_steps)
    model.save(args.save_dir / "ppo_narrow_passage")


if __name__ == "__main__":
    main()
