#!/usr/bin/env python3
"""Evaluate a stable-baselines3 policy on the Habitat narrow-passage val set.

The SB3 policy was trained on a 19-dim synthetic narrow-passage env.
Habitat's NarrowPassageGeometrySensor outputs the same 19-dim feature vector,
so the policy can be evaluated directly on real HM3D scenes.

Key fix: NarrowPassageGeometrySensor now uses the correct heading_error formula
  atan2(-delta_x, -delta_z) instead of atan2(delta_x, -delta_z)
Making this evaluation directly comparable to the FSM / APF results.

Run:
    conda run -n habitat python3 examples/narrow_passage_rl/eval_habitat_sb3.py
    conda run -n habitat python3 examples/narrow_passage_rl/eval_habitat_sb3.py \
        --algo ppo --model data/narrow_passage_sb3_hard/ppo_narrow_passage.zip \
        --output-csv results/narrow_passage_rl/habitat_sb3_hard_episodes.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import habitat

sys.path.insert(0, str(Path(__file__).parent))

from narrow_passage.metrics.strict_metrics import (
    StrictMetricConfig,
    compute_strict_metrics,
)
from narrow_passage.models.belief_state import BeliefState

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"
FEATURE_DIM = 19


def _to_habitat_action(action, action_space: str):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    lin = float(action[0])
    ang = float(action[1])
    if action_space == "synthetic":
        # HarderNarrowPassageEnv checkpoints output physical-like synthetic
        # ranges.  Habitat VelocityAction expects normalized [-1, 1].
        lin = 2.0 * ((np.clip(lin, -0.15, 0.35) - (-0.15)) / (0.35 - (-0.15))) - 1.0
        ang = float(np.clip(ang, -0.8, 0.8) / 0.8)
    elif action_space != "normalized":
        raise ValueError(f"Unknown action space: {action_space}")
    return float(np.clip(lin, -1.0, 1.0)), float(np.clip(ang, -1.0, 1.0))


def _load_model(algo: str, model_path: Path):
    """Load PPO/SAC/TD3 from stable-baselines3.

    algo="auto" infers from the model path, which keeps old command lines short.
    """
    try:
        from stable_baselines3 import PPO, SAC, TD3
    except ImportError:
        raise ImportError("Install stable-baselines3: pip install stable-baselines3")

    inferred = algo.lower()
    if inferred == "auto":
        name = str(model_path).lower()
        if "sac" in name:
            inferred = "sac"
        elif "td3" in name:
            inferred = "td3"
        else:
            inferred = "ppo"

    cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[inferred]
    return inferred, cls.load(str(model_path), device="cpu")


def make_env(data_path: str, split: str):
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            f"habitat.dataset.data_path={data_path}",
            f"habitat.dataset.split={split}",
            "habitat.dataset.type=PointNav-v1",
            "habitat.simulator.habitat_sim_v0.allow_sliding=False",
        ],
    )
    return habitat.Env(config=config)


def _belief_metrics(features: np.ndarray, memory_risk: float = 0.0) -> dict:
    belief = BeliefState.from_obs(features, memory_risk=memory_risk)
    risk = float(np.clip(1.0 - belief.p_feas + belief.memory_risk, 0.0, 1.0))
    return {
        "d_hat": belief.d_hat,
        "w_req_cons": belief.w_req_cons,
        "delta_mean": belief.delta_mean,
        "delta_var": belief.delta_var,
        "p_feas": belief.p_feas,
        "risk": risk,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["auto", "ppo", "sac", "td3"], default="auto")
    ap.add_argument("--model", type=Path,
                    default=Path("data/narrow_passage_sb3_hard/ppo_narrow_passage.zip"))
    ap.add_argument("--data-path",
                    default="data/datasets/narrow_passage/{split}/{split}.json.gz")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-episodes", type=int, default=-1, help="-1 = all")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument(
        "--strict-clearance-threshold",
        type=float,
        default=-0.02,
        help=(
            "Minimum body-margin threshold for strict_success.  HM3D depth "
            "features can produce small negative clearance artifacts near "
            "valid navmesh positions, so the default is mildly tolerant."
        ),
    )
    ap.add_argument(
        "--action-space",
        choices=["normalized", "synthetic"],
        default="normalized",
        help=(
            "Use synthetic for checkpoints trained in HarderNarrowPassageEnv; "
            "use normalized for Habitat-native SB3 checkpoints."
        ),
    )
    ap.add_argument("--output-csv", type=Path, default=None)
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    algo, model = _load_model(args.algo, args.model)
    print(f"Algorithm: {algo}")

    all_stats = []

    with make_env(args.data_path.format(split=args.split), args.split) as env:
        total = env.number_of_episodes
        num_ep = total if args.num_episodes <= 0 else min(args.num_episodes, total)
        print(f"[habitat_sb3] evaluating {num_ep}/{total} episodes (split={args.split})")

        for ep_idx in range(num_ep):
            obs_dict = env.reset()
            episode = env.current_episode
            info = episode.info if hasattr(episode, "info") and episode.info else {}
            body_margin = float(info.get("body_margin", float("nan")))
            difficulty = info.get("difficulty", "?")

            steps = 0
            any_collision = False
            min_bm = float("inf")
            done = False
            last_feats = np.array(
                obs_dict.get(
                    "narrow_passage_features",
                    np.zeros(FEATURE_DIM, dtype=np.float32),
                ),
                dtype=np.float32,
            )

            while not done and steps < args.max_steps:
                feats = np.array(
                    obs_dict.get("narrow_passage_features", last_feats),
                    dtype=np.float32,
                )
                last_feats = feats
                bm = float(feats[9])
                if bm < min_bm:
                    min_bm = bm
                if float(feats[16]) > 0.5:
                    any_collision = True

                # SB3 model.predict expects flat numpy array
                action, _ = model.predict(feats, deterministic=True)
                lin_norm, ang_norm = _to_habitat_action(
                    action, args.action_space
                )

                obs_dict = env.step({
                    "action": "velocity_control",
                    "action_args": {
                        "linear_velocity": lin_norm,
                        "angular_velocity": ang_norm,
                    },
                })
                steps += 1
                metrics = env.get_metrics()
                success_now = float(metrics.get("narrow_passage_success", 0.0))
                stuck_now = float(metrics.get("narrow_passage_stuck", 0.0))
                collision_now = float(metrics.get("narrow_passage_collision", 0.0))
                if (
                    env.episode_over
                    or success_now > 0.5
                    or stuck_now > 0.5
                    or collision_now > 0.5
                ):
                    done = True

            metrics = env.get_metrics()
            final_feats = np.array(
                obs_dict.get("narrow_passage_features", last_feats),
                dtype=np.float32,
            )
            success = float(metrics.get("narrow_passage_success", 0.0))
            final_collision = max(
                float(any_collision),
                float(metrics.get("narrow_passage_collision", 0.0)),
            )
            final_stuck = float(metrics.get("narrow_passage_stuck", 0.0))
            min_clearance = float(min_bm) if np.isfinite(min_bm) else 0.0
            strict_metrics = compute_strict_metrics(
                success=success,
                collision=final_collision,
                stuck=final_stuck,
                min_clearance=min_clearance,
                cfg=StrictMetricConfig(
                    strict_clearance_threshold=args.strict_clearance_threshold
                ),
            )

            stat = {
                "episode_id": str(episode.episode_id),
                "scene_id": str(episode.scene_id).split("/")[-2],
                "difficulty": difficulty,
                "body_margin": round(body_margin, 4),
                "steps": steps,
                "success": success,
                "strict_success": strict_metrics["strict_success"],
                "clearance_safe": strict_metrics["clearance_safe"],
                "success_but_unsafe": strict_metrics["success_but_unsafe"],
                "collision": final_collision,
                "stuck": final_stuck,
                "near_collision": strict_metrics["near_collision"],
                "min_clearance": min_clearance,
                **_belief_metrics(final_feats),
                "mode": "direct_velocity",
            }
            all_stats.append(stat)
            print(
                f"  ep {ep_idx:3d}  {difficulty:6s}  bm={body_margin:+.3f}"
                f"  success={success:.0f}  strict={strict_metrics['strict_success']:.0f}"
                f"  steps={steps:3d}"
            )

    n = len(all_stats)
    sr = np.mean([s["success"] for s in all_stats])
    strict_sr = np.mean([s["strict_success"] for s in all_stats])
    unsafe_sr = np.mean([s["success_but_unsafe"] for s in all_stats])
    safe_rate = np.mean([s["clearance_safe"] for s in all_stats])
    cr = np.mean([s["collision"] for s in all_stats])
    ncr = np.mean([s["near_collision"] for s in all_stats])
    avg_bm = np.mean([s["min_clearance"] for s in all_stats])

    print(f"\n=== habitat_sb3_{algo}  |  {n} episodes  |  split={args.split} ===")
    print(f"  model:               {args.model}")
    print(f"  algo:                {algo}")
    print(f"  success_rate:        {sr:.3f}  ({sr*100:.1f}%)")
    print(f"  strict_success_rate: {strict_sr:.3f}  ({strict_sr*100:.1f}%)")
    print(f"  clearance_safe_rate: {safe_rate:.3f}")
    print(f"  success_but_unsafe:  {unsafe_sr:.3f}")
    print(f"  collision_rate:      {cr:.3f}")
    print(f"  near_collision_rate: {ncr:.3f}")
    print(f"  avg_min_clearance:   {avg_bm:.3f}")
    for diff in ["narrow", "normal", "wide"]:
        rows = [s for s in all_stats if s["difficulty"] == diff]
        if rows:
            d_sr = np.mean([r["success"] for r in rows])
            d_strict = np.mean([r["strict_success"] for r in rows])
            print(
                f"  {diff:6s}: SR={d_sr:.3f} "
                f"strict={d_strict:.3f} "
                f"({sum(r['success']>0.5 for r in rows)}/{len(rows)})"
            )

    out_path = args.output_csv or (
        RESULTS / f"habitat_{algo}_{args.model.parent.name}_episodes.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_stats[0].keys()))
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
