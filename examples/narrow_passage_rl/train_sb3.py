#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from fair_reward_wrapper import FairNarrowPassageRewardWrapper
from obs_wrappers import ABLATION_MASKS, apply_ablation
from procedural_env import ProceduralNarrowPassageEnv
from procedural_env_v2 import HarderNarrowPassageEnv


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

CURRICULUM_STAGE_CONFIGS = {
    "straight_wide": {
        "name": "stage_1_straight_wide",
        "passage_width_range": (0.85, 1.20),
        "obstacle_noise": 0.0,
        "depth_noise": 0.0,
        "start_pose_jitter": 0.10,
        "heading_jitter": 0.20,
        "max_episode_steps": 220,
        "false_feasible_ratio": 0.0,
        "turn_ratio": 0.0,
        "corridor_types": ["straight"],
    },
    "straight_narrow": {
        "name": "stage_2_straight_narrow",
        "passage_width_range": (0.62, 0.90),
        "obstacle_noise": 0.0,
        "depth_noise": 0.0,
        "start_pose_jitter": 0.18,
        "heading_jitter": 0.35,
        "max_episode_steps": 260,
        "false_feasible_ratio": 0.0,
        "turn_ratio": 0.0,
        "corridor_types": ["straight", "narrow_entry", "narrow_exit"],
    },
    "asymmetric": {
        "name": "stage_3_asymmetric",
        "passage_width_range": (0.55, 0.85),
        "obstacle_noise": 0.01,
        "depth_noise": 0.005,
        "start_pose_jitter": 0.22,
        "heading_jitter": 0.45,
        "max_episode_steps": 300,
        "false_feasible_ratio": 0.0,
        "turn_ratio": 0.0,
        "corridor_types": ["asymmetric", "narrow_entry", "narrow_exit"],
    },
    "l_shaped": {
        "name": "stage_4_l_shaped",
        "passage_width_range": (0.55, 0.85),
        "obstacle_noise": 0.01,
        "depth_noise": 0.01,
        "start_pose_jitter": 0.25,
        "heading_jitter": 0.55,
        "max_episode_steps": 360,
        "false_feasible_ratio": 0.0,
        "turn_ratio": 1.0,
        "corridor_types": ["l_shaped"],
    },
    "s_shaped": {
        "name": "stage_5_s_shaped",
        "passage_width_range": (0.50, 0.82),
        "obstacle_noise": 0.015,
        "depth_noise": 0.015,
        "start_pose_jitter": 0.28,
        "heading_jitter": 0.65,
        "max_episode_steps": 420,
        "false_feasible_ratio": 0.0,
        "turn_ratio": 1.0,
        "corridor_types": ["s_shaped"],
    },
    "false_feasible": {
        "name": "stage_6_false_feasible",
        "passage_width_range": (0.50, 0.85),
        "obstacle_noise": 0.02,
        "depth_noise": 0.02,
        "start_pose_jitter": 0.30,
        "heading_jitter": 0.70,
        "max_episode_steps": 420,
        "false_feasible_ratio": 0.60,
        "turn_ratio": 0.30,
        "corridor_types": ["straight", "false_feasible", "l_shaped"],
    },
    "noise_delay": {
        "name": "stage_7_noise_delay",
        "passage_width_range": (0.45, 0.82),
        "obstacle_noise": 0.035,
        "depth_noise": 0.035,
        "start_pose_jitter": 0.35,
        "heading_jitter": 0.80,
        "max_episode_steps": 450,
        "false_feasible_ratio": 0.30,
        "turn_ratio": 0.50,
        "corridor_types": [
            "straight",
            "asymmetric",
            "l_shaped",
            "s_shaped",
            "false_feasible",
        ],
    },
}

_STAGE_ALIASES = {
    cfg["name"]: key for key, cfg in CURRICULUM_STAGE_CONFIGS.items()
}
_STAGE_ALIASES.update(
    {
        "stage_1": "straight_wide",
        "stage_2": "straight_narrow",
        "stage_3": "asymmetric",
        "stage_4": "l_shaped",
        "stage_5": "s_shaped",
        "stage_6": "false_feasible",
        "stage_7": "noise_delay",
    }
)


def _resolve_curriculum_stages(stage_arg: str) -> List[str]:
    stages = []
    for raw_name in stage_arg.split(","):
        name = raw_name.strip()
        if not name:
            continue
        key = _STAGE_ALIASES.get(name, name)
        if key not in CURRICULUM_STAGE_CONFIGS:
            valid = sorted(CURRICULUM_STAGE_CONFIGS) + sorted(_STAGE_ALIASES)
            raise ValueError(
                f"Unknown curriculum stage '{name}'. Valid stages: {valid}"
            )
        stages.append(key)
    if not stages:
        raise ValueError("At least one curriculum stage is required.")
    return stages


def _stage_to_env_config(stage_key: str, seed: int) -> Dict:
    stage = dict(CURRICULUM_STAGE_CONFIGS[stage_key])
    false_ratio = float(stage["false_feasible_ratio"])
    corridor_types = list(stage["corridor_types"])
    probs = None
    if "false_feasible" in corridor_types:
        non_false = [c for c in corridor_types if c != "false_feasible"]
        non_false_weight = max(0.0, 1.0 - false_ratio)
        probs = {"false_feasible": false_ratio}
        if non_false:
            for ctype in non_false:
                probs[ctype] = non_false_weight / float(len(non_false))
        else:
            probs["false_feasible"] = 1.0

    return {
        "seed": seed,
        "width_range": tuple(stage["passage_width_range"]),
        "corridor_types": corridor_types,
        "corridor_type_probs": probs,
        "obstacle_noise": float(stage["obstacle_noise"]),
        "depth_noise": float(stage["depth_noise"]),
        "start_pose_jitter": float(stage["start_pose_jitter"]),
        "yaw_noise": float(stage["heading_jitter"]),
        "max_steps": int(stage["max_episode_steps"]),
        "false_feasible_ratio": false_ratio,
        "turn_ratio": float(stage["turn_ratio"]),
        "stage_name": stage["name"],
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--save-dir", type=Path, default=Path("data/narrow_passage_sb3"))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="easy")
    parser.add_argument("--ablation", choices=ABLATION_MASKS, default="full")
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument(
        "--curriculum-stages",
        type=str,
        default=(
            "straight_wide,straight_narrow,asymmetric,l_shaped,s_shaped,"
            "false_feasible,noise_delay"
        ),
    )
    parser.add_argument("--stage-steps", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for this training script. Install it "
            "inside your habitat conda env, then rerun this script."
        ) from exc

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    train_log_path = args.save_dir / "curriculum_train_log.csv"
    eval_log_path = args.save_dir / "curriculum_eval_summary.csv"

    class TrainingStatsCallback(BaseCallback):
        def __init__(self, stage_name: str, csv_path: Path, window: int = 100):
            super().__init__()
            self.stage_name = stage_name
            self.csv_path = csv_path
            self.window = window
            self.rows = []
            self.episodes = 0
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.fieldnames = [
                "stage",
                "timesteps",
                "episodes",
                "success_rate",
                "collision_rate",
                "stuck_rate",
                "episode_length",
                "window",
            ]
            if not self.csv_path.exists():
                with self.csv_path.open("w", newline="") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for done, info in zip(dones, infos):
                if not done:
                    continue
                episode_info = info.get("episode", {})
                row = {
                    "success": float(info.get("success", 0.0)),
                    "collision": float(info.get("collision", 0.0)),
                    "stuck": float(info.get("stuck", 0.0)),
                    "episode_length": float(episode_info.get("l", 0.0)),
                }
                self.rows.append(row)
                self.rows = self.rows[-self.window :]
                self.episodes += 1
                summary = self._summary()
                with self.csv_path.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writerow(summary)
                self.logger.record(
                    f"curriculum/{self.stage_name}/success_rate",
                    summary["success_rate"],
                )
                self.logger.record(
                    f"curriculum/{self.stage_name}/collision_rate",
                    summary["collision_rate"],
                )
                self.logger.record(
                    f"curriculum/{self.stage_name}/stuck_rate",
                    summary["stuck_rate"],
                )
                self.logger.record(
                    f"curriculum/{self.stage_name}/episode_length",
                    summary["episode_length"],
                )
            return True

        def _summary(self) -> Dict:
            rows = self.rows[-self.window :]
            denom = max(1, len(rows))
            return {
                "stage": self.stage_name,
                "timesteps": self.num_timesteps,
                "episodes": self.episodes,
                "success_rate": round(sum(r["success"] for r in rows) / denom, 4),
                "collision_rate": round(sum(r["collision"] for r in rows) / denom, 4),
                "stuck_rate": round(sum(r["stuck"] for r in rows) / denom, 4),
                "episode_length": round(
                    sum(r["episode_length"] for r in rows) / denom, 2
                ),
                "window": len(rows),
            }

    def make_env(rank, env_config=None, use_v2=False):
        if use_v2:
            cfg = dict(env_config or {})
            cfg["seed"] = int(cfg.get("seed", args.seed)) + rank
            env = FairNarrowPassageRewardWrapper(HarderNarrowPassageEnv(cfg))
        else:
            cfg = dict(env_config or DIFFICULTY_CONFIGS[args.difficulty])
            cfg["seed"] = args.seed + rank
            env = ProceduralNarrowPassageEnv(cfg)
        info_keywords = ("success", "collision", "stuck")
        return Monitor(
            apply_ablation(env, args.ablation), info_keywords=info_keywords
        )

    def make_vec_env(env_config=None, use_v2=False):
        return DummyVecEnv(
            [
                lambda rank=i: make_env(rank, env_config=env_config, use_v2=use_v2)
                for i in range(args.num_envs)
            ]
        )

    def make_model(env):
        if args.load_model is not None:
            return PPO.load(args.load_model, env=env, device="cpu")
        return PPO(
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

    def evaluate_stage(model, stage_key, env_config):
        rows = []
        env = make_env(0, env_config=env_config, use_v2=True)
        for ep in range(args.eval_episodes):
            obs, _ = env.reset(seed=args.seed + 10_000 + ep)
            done = False
            info = {}
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            rows.append(
                {
                    "success": float(info.get("success", 0.0)),
                    "collision": float(info.get("collision", 0.0)),
                    "stuck": float(info.get("stuck", 0.0)),
                    "episode_length": steps,
                }
            )
        env.close()
        denom = max(1, len(rows))
        summary = {
            "stage": CURRICULUM_STAGE_CONFIGS[stage_key]["name"],
            "eval_episodes": len(rows),
            "success_rate": round(sum(r["success"] for r in rows) / denom, 4),
            "collision_rate": round(sum(r["collision"] for r in rows) / denom, 4),
            "stuck_rate": round(sum(r["stuck"] for r in rows) / denom, 4),
            "episode_length": round(
                sum(r["episode_length"] for r in rows) / denom, 2
            ),
        }
        write_header = not eval_log_path.exists()
        with eval_log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)
        return summary

    if args.curriculum:
        stage_keys = _resolve_curriculum_stages(args.curriculum_stages)
        curriculum_config = []
        model = None
        env = None
        for stage_idx, stage_key in enumerate(stage_keys, start=1):
            env_config = _stage_to_env_config(stage_key, args.seed + stage_idx * 1000)
            stage_name = CURRICULUM_STAGE_CONFIGS[stage_key]["name"]
            curriculum_config.append({"stage_key": stage_key, **env_config})
            print(
                f"[curriculum] {stage_idx}/{len(stage_keys)} {stage_name} "
                f"steps={args.stage_steps} config={env_config}"
            )
            new_env = make_vec_env(env_config=env_config, use_v2=True)
            if model is None:
                model = make_model(new_env)
            else:
                model.set_env(new_env)
            if env is not None:
                env.close()
            env = new_env
            callback = TrainingStatsCallback(stage_name, train_log_path)
            model.learn(
                total_timesteps=args.stage_steps,
                reset_num_timesteps=(stage_idx == 1 and args.load_model is None),
                callback=callback,
            )
            ckpt_path = (
                args.save_dir / "checkpoints" / f"{stage_idx:02d}_{stage_key}.zip"
            )
            model.save(ckpt_path)
            summary = evaluate_stage(model, stage_key, env_config)
            print(f"[stage_eval] {summary}")
            print(f"[saved] {ckpt_path}")
        _write_json(
            args.save_dir / "curriculum_config.json",
            {
                "seed": args.seed,
                "stage_steps": args.stage_steps,
                "num_envs": args.num_envs,
                "ablation": args.ablation,
                "stages": curriculum_config,
            },
        )
        model.save(args.save_dir / "ppo_narrow_passage")
        if env is not None:
            env.close()
        print(f"[write] {train_log_path}")
        print(f"[write] {eval_log_path}")
        print(f"[saved] {args.save_dir / 'ppo_narrow_passage.zip'}")
        return

    env = make_vec_env(env_config=DIFFICULTY_CONFIGS[args.difficulty], use_v2=False)
    model = make_model(env)
    model.learn(total_timesteps=args.total_steps)
    model.save(args.save_dir / "ppo_narrow_passage")
    env.close()


if __name__ == "__main__":
    main()
