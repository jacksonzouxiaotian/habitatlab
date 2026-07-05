#!/usr/bin/env python3
"""Train SB3 PPO/SAC/TD3 directly in Habitat NarrowPassageNav-v0.

This is a minimal Habitat-native learning baseline.  It deliberately stays out
of Habitat-Baselines internals: official Habitat PPO/DD-PPO can continue to use
habitat_baselines, while SB3 algorithms are connected through a small Gymnasium
wrapper around NarrowPassageNav-v0.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import habitat

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - older env fallback
    import gym
    from gym import spaces


FEATURE_DIM = 19
RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"


def make_habitat_config(data_path: str, split: str, allow_sliding: bool = False):
    return habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            f"habitat.dataset.data_path={data_path}",
            f"habitat.dataset.split={split}",
            "habitat.dataset.type=PointNav-v1",
            f"habitat.simulator.habitat_sim_v0.allow_sliding={str(allow_sliding)}",
        ],
    )


class HabitatNarrowPassageSB3Env(gym.Env):
    """Flat 19-D SB3 wrapper for Habitat NarrowPassageNav-v0."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_steps: int = 500,
        action_space_mode: str = "normalized",
        allow_sliding: bool = False,
    ) -> None:
        super().__init__()
        self._env = habitat.Env(
            config=make_habitat_config(data_path, split, allow_sliding)
        )
        self.max_steps = int(max_steps)
        self.action_space_mode = action_space_mode
        self._steps = 0
        self._episode = None
        self._last_min_clearance = float("inf")
        self._near_collision = False
        self._any_collision = False

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )
        if action_space_mode == "synthetic":
            # Matches HarderNarrowPassageEnv checkpoints so SB3 can load them.
            self.action_space = spaces.Box(
                low=np.array([-0.15, -0.8], dtype=np.float32),
                high=np.array([0.35, 0.8], dtype=np.float32),
                dtype=np.float32,
            )
        elif action_space_mode == "normalized":
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            raise ValueError(
                "--action-space must be 'normalized' or 'synthetic', "
                f"got {action_space_mode!r}"
            )

    @property
    def number_of_episodes(self) -> int:
        return int(self._env.number_of_episodes)

    def close(self) -> None:
        self._env.close()

    def reset(self, *, seed=None, options=None):
        # Habitat Env does not expose per-reset seeding here; SB3 still passes
        # seeds for API compatibility, and dataset episode order remains fixed.
        del seed, options
        obs = self._env.reset()
        self._steps = 0
        self._episode = self._env.current_episode
        self._last_min_clearance = float("inf")
        self._near_collision = False
        self._any_collision = False
        return self._features(obs), self._episode_info()

    def step(self, action):
        lin_norm, ang_norm = self._to_habitat_action(action)
        obs = self._env.step(
            {
                "action": "velocity_control",
                "action_args": {
                    "linear_velocity": lin_norm,
                    "angular_velocity": ang_norm,
                },
            }
        )
        self._steps += 1
        feats = self._features(obs)
        body_margin = float(feats[9])
        self._last_min_clearance = min(self._last_min_clearance, body_margin)
        self._near_collision = self._near_collision or body_margin < 0.05
        self._any_collision = self._any_collision or float(feats[16]) > 0.5

        metrics = self._env.get_metrics()
        reward = float(metrics.get("narrow_passage_reward", 0.0))
        success = float(metrics.get("narrow_passage_success", 0.0)) > 0.5
        collision = float(metrics.get("narrow_passage_collision", 0.0)) > 0.5
        stuck = float(metrics.get("narrow_passage_stuck", 0.0)) > 0.5
        terminated = bool(self._env.episode_over or success or collision or stuck)
        truncated = bool(self._steps >= self.max_steps and not terminated)
        info = self._episode_info(metrics)
        return feats, reward, terminated, truncated, info

    def _features(self, obs: Dict) -> np.ndarray:
        feats = obs.get("narrow_passage_features")
        if feats is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        return np.asarray(feats, dtype=np.float32).reshape(FEATURE_DIM)

    def _to_habitat_action(self, action) -> Tuple[float, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        lin = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        ang = float(np.clip(action[1], self.action_space.low[1], self.action_space.high[1]))
        if self.action_space_mode == "synthetic":
            # Convert synthetic physical action ranges to Habitat's normalized
            # VelocityAction inputs.  Angular synthetic max_wz ~= 45 deg/s.
            lin = 2.0 * ((lin - (-0.15)) / (0.35 - (-0.15))) - 1.0
            ang = float(np.clip(ang / 0.8, -1.0, 1.0))
        return float(np.clip(lin, -1.0, 1.0)), float(np.clip(ang, -1.0, 1.0))

    def _episode_info(self, metrics: Dict = None) -> Dict:
        metrics = metrics or {}
        ep = self._episode or getattr(self._env, "current_episode", None)
        ep_info = getattr(ep, "info", {}) or {}
        scene_id = str(getattr(ep, "scene_id", ""))
        reward_terms = getattr(getattr(self._env, "_task", None), "narrow_passage_reward_terms", {})
        return {
            "episode_id": str(getattr(ep, "episode_id", "")),
            "scene_id": scene_id.split("/")[-2] if "/" in scene_id else scene_id,
            "difficulty": ep_info.get("difficulty", "?"),
            "body_margin": float(ep_info.get("body_margin", np.nan)),
            "steps": self._steps,
            "success": float(metrics.get("narrow_passage_success", 0.0)),
            "collision": float(self._any_collision),
            "stuck": float(metrics.get("narrow_passage_stuck", 0.0)),
            "near_collision": float(self._near_collision),
            "min_clearance": (
                float(self._last_min_clearance)
                if np.isfinite(self._last_min_clearance)
                else 0.0
            ),
            **{str(k): float(v) for k, v in dict(reward_terms).items()},
        }


def build_model(args, env):
    if args.algo == "ppo":
        from stable_baselines3 import PPO

        cls = PPO
        kwargs = dict(
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            verbose=1,
            device=args.device,
            tensorboard_log=str(args.save_dir / "tb"),
        )
    elif args.algo == "sac":
        from stable_baselines3 import SAC

        cls = SAC
        kwargs = dict(
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=(1, "step"),
            gradient_steps=1,
            verbose=1,
            device=args.device,
            tensorboard_log=str(args.save_dir / "tb"),
        )
    else:
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise

        cls = TD3
        noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[-1]),
            sigma=args.action_noise * np.ones(env.action_space.shape[-1]),
        )
        kwargs = dict(
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=noise,
            policy_delay=2,
            verbose=1,
            device=args.device,
            tensorboard_log=str(args.save_dir / "tb"),
        )

    if args.load_model:
        return cls.load(str(args.load_model), env=env, device=args.device, verbose=1)
    return cls("MlpPolicy", env, **kwargs)


def evaluate(model, env, episodes: int, max_steps: int, output_csv: Path):
    rows = []
    total = env.number_of_episodes
    n = total if episodes <= 0 else min(episodes, total)
    for ep_idx in range(n):
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        rows.append(dict(info, eval_index=ep_idx))
        print(
            f"  ep {ep_idx:3d} {info.get('difficulty', '?'):6s} "
            f"success={info.get('success', 0):.0f} steps={info.get('steps', 0):3d}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sr = float(np.mean([r["success"] for r in rows])) if rows else 0.0
    cr = float(np.mean([r["collision"] for r in rows])) if rows else 0.0
    ncr = float(np.mean([r["near_collision"] for r in rows])) if rows else 0.0
    print(f"[eval] episodes={len(rows)} SR={sr:.3f} collision={cr:.3f} near_collision={ncr:.3f}")
    print(f"[write] {output_csv}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac", "td3"], default="td3")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--data-path",
        default="data/datasets/narrow_passage/{split}/{split}.json.gz",
    )
    ap.add_argument("--total-steps", type=int, default=500_000)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=151)
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--load-model", type=Path, default=None)
    ap.add_argument(
        "--action-space",
        choices=["normalized", "synthetic"],
        default="normalized",
        help="Use synthetic when finetuning a model trained in HarderNarrowPassageEnv.",
    )
    ap.add_argument("--allow-sliding", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer-size", type=int, default=200_000)
    ap.add_argument("--action-noise", type=float, default=0.10)
    ap.add_argument("--output-csv", type=Path, default=None)
    args = ap.parse_args()

    if args.save_dir is None:
        args.save_dir = Path(f"data/narrow_passage_habitat_{args.algo}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(args.seed)
    except Exception:
        pass

    train_path = args.data_path.format(split=args.split)
    eval_path = args.data_path.format(split=args.eval_split)
    print(
        f"[train_habitat_sb3] algo={args.algo} split={args.split} "
        f"steps={args.total_steps} action_space={args.action_space}"
    )

    env = HabitatNarrowPassageSB3Env(
        train_path,
        split=args.split,
        max_steps=args.max_steps,
        action_space_mode=args.action_space,
        allow_sliding=args.allow_sliding,
    )
    model = build_model(args, env)
    model.learn(total_timesteps=args.total_steps, progress_bar=False)

    ckpt = args.save_dir / f"{args.algo}_habitat_narrow_passage"
    model.save(str(ckpt))
    print(f"[saved] {ckpt}.zip")
    env.close()

    if args.eval_episodes != 0:
        eval_env = HabitatNarrowPassageSB3Env(
            eval_path,
            split=args.eval_split,
            max_steps=args.max_steps,
            action_space_mode=args.action_space,
            allow_sliding=args.allow_sliding,
        )
        out_csv = args.output_csv or (
            RESULTS / f"habitat_{args.algo}_{args.save_dir.name}_finetune_eval.csv"
        )
        evaluate(model, eval_env, args.eval_episodes, args.max_steps, out_csv)
        eval_env.close()


if __name__ == "__main__":
    main()
