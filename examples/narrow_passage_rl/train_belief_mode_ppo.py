#!/usr/bin/env python3
"""Train DEGNAV-RL: PPO over belief-state high-level modes.

This is intentionally separate from ``train_sb3_v2.py``.  ``train_sb3_v2.py``
trains direct-control PPO/SAC/TD3 baselines that output continuous velocities.
This script trains the DEGNAV-RL variant: observation = feasibility belief,
action = one of COMMIT / EXPLORE / RECOVER / REJECT.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv
from narrow_passage.envs.belief_mode_env import (
    VALID_BELIEF_ABLATIONS,
    BeliefModeEnv,
    belief_observation_feature_names,
)
from eval_belief_mode_ppo import (
    CTYPE_CONFIGS,
    evaluate_model,
    print_summary,
    write_csv,
    write_markdown,
)


PPO_KWARGS = {
    "learning_rate": 2.5e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "clip_range": 0.2,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(seed)
    except ImportError:
        pass


def make_env(rank: int, env_cfg: dict[str, Any], seed: int, ablation: str):
    from stable_baselines3.common.monitor import Monitor

    def _init():
        cfg = dict(env_cfg)
        cfg["seed"] = int(seed + rank)
        env = BeliefModeEnv(HarderNarrowPassageEnv(cfg), ablation=ablation)
        return Monitor(env)

    return _init


def build_env_config(ctypes: str) -> dict[str, Any]:
    env_cfg: dict[str, Any] = {}
    corridor_types = CTYPE_CONFIGS[ctypes]
    if corridor_types is not None:
        env_cfg["corridor_types"] = corridor_types
    return env_cfg


def save_config(args, save_dir: Path, model_path: Path) -> Path:
    config = {
        "script": "train_belief_mode_ppo.py",
        "method": "DEGNAV-RL belief-guided mode-selection PPO",
        "model_path": str(model_path),
        "total_steps": args.total_steps,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "ctypes": args.ctypes,
        "reward_mode": args.reward_mode,
        "ablation": args.ablation,
        "device": args.device,
        "belief_features": belief_observation_feature_names(args.ablation),
        "mode_mapping": {
            "0": "COMMIT",
            "1": "EXPLORE",
            "2": "RECOVER",
            "3": "REJECT",
        },
        "ppo": dict(PPO_KWARGS),
    }
    out = save_dir / "belief_mode_ppo_config.json"
    out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"[write] {out}")
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument(
        "--save-dir",
        type=Path,
        default=Path("data/narrow_passage_degnav_rl_belief_mode_ppo"),
    )
    ap.add_argument("--ctypes", choices=list(CTYPE_CONFIGS), default="full")
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument(
        "--reward-mode",
        choices=["strict"],
        default="strict",
        help="DEGNAV-RL currently uses the strict BeliefModeEnv reward.",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--ablation",
        choices=list(VALID_BELIEF_ABLATIONS),
        default="full",
        help="Belief-state ablation for the learned high-level mode selector.",
    )
    ap.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Optional checkpoint for resume/evaluation.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError("Install stable-baselines3 to train DEGNAV-RL") from exc

    env_cfg = build_env_config(args.ctypes)
    vec_env = DummyVecEnv(
        [
            make_env(rank, env_cfg, args.seed, args.ablation)
            for rank in range(args.num_envs)
        ]
    )

    model_path = args.save_dir / "belief_mode_ppo.zip"
    if args.load_model is not None:
        model = PPO.load(str(args.load_model), env=vec_env, device=args.device, verbose=1)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=str(args.save_dir / "tb"),
            seed=args.seed,
            device=args.device,
            **PPO_KWARGS,
        )

    print(
        f"[train_belief_mode_ppo] steps={args.total_steps} "
        f"num_envs={args.num_envs} ctypes={args.ctypes} "
        f"ablation={args.ablation} save_dir={args.save_dir}"
    )
    if args.total_steps > 0:
        model.learn(total_timesteps=args.total_steps)
        model.save(str(model_path))
        print(f"[saved] {model_path}")
    elif args.load_model is not None:
        model_path = args.load_model
        print(f"[eval-only] using {model_path}")
    else:
        raise ValueError("--total-steps 0 requires --load-model")

    save_config(args, args.save_dir, model_path)

    if args.eval_episodes > 0:
        rows, summary = evaluate_model(
            model=model,
            episodes=args.eval_episodes,
            seed=args.seed + 10000,
            ctypes=args.ctypes,
            max_steps=400,
            deterministic=True,
            ablation=args.ablation,
        )
        print_summary(summary)
        write_csv(rows, args.save_dir / "belief_mode_ppo_quick_eval.csv")
        write_markdown(summary, args.save_dir / "belief_mode_ppo_quick_summary.md")


if __name__ == "__main__":
    main()
