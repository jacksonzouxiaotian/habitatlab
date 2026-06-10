#!/usr/bin/env python3

import argparse
from pathlib import Path

from risk_recovery_env import ProceduralRiskRecoveryEnv
from train_sb3 import DIFFICULTY_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=700_000)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("data/narrow_passage_risk_recovery"),
    )
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="hard")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    args.save_dir.mkdir(parents=True, exist_ok=True)

    def make_env(rank):
        env_config = dict(DIFFICULTY_CONFIGS[args.difficulty])
        env_config["seed"] = args.seed + rank
        return Monitor(ProceduralRiskRecoveryEnv(env_config))

    env = DummyVecEnv([lambda rank=i: make_env(rank) for i in range(args.num_envs)])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(args.save_dir / "tb"),
        seed=args.seed,
        device="cpu",
    )
    model.learn(total_timesteps=args.total_steps)
    model.save(args.save_dir / "ppo_risk_recovery")


if __name__ == "__main__":
    main()
