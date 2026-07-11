#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import numpy as np
from obs_wrappers import ABLATION_MASKS, apply_ablation
from procedural_env import ProceduralNarrowPassageEnv
from procedural_env_v2 import HarderNarrowPassageEnv
from rl_eval_utils import (
    finalize_episode_info,
    make_episode_trace,
    print_summary,
    summarize_episode_stats,
    update_episode_trace,
    write_summary_csv,
)
from train_sb3 import DIFFICULTY_CONFIGS


FAILURE_TYPES = (
    "collision",
    "stuck",
    "timeout",
    "oscillation",
    "false_feasible_wrong_entry",
    "turn_failure",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _scene_geometry(env, env_version):
    if env_version == "v2":
        params = env._params
        return {
            "env_version": "v2",
            "scene_type": params.ctype.value,
            "path": _jsonable(params.path),
            "width_profile": _jsonable(params.width_profile),
            "blocker": _jsonable(params.blocker),
            "protrusion": _jsonable(params.protrusion),
            "start_world": _jsonable(env._start_world),
            "goal_world": _jsonable(env._goal_world),
            "robot_radius": float(env.robot_radius),
        }

    p = env.params
    return {
        "env_version": "v1",
        "scene_type": "false_feasible" if p.false_feasible else "straight",
        "width": float(p.width),
        "length": float(p.length),
        "obstacle": {
            "x": float(p.obstacle_x),
            "y": float(p.obstacle_y),
            "radius": float(p.obstacle_radius),
        },
        "start": [float(p.start_x), float(p.start_y), float(p.start_yaw)],
        "goal": [0.0, float(p.length + 0.5)],
        "robot_radius": float(env.robot_radius),
    }


def _record_step(env, obs, action, reward, info):
    reward_terms = {
        key: float(val) for key, val in info.items() if key.startswith("reward/")
    }
    pose = env.pose if hasattr(env, "pose") else np.zeros(3, dtype=np.float32)
    return {
        "step": int(getattr(env, "step_count", 0)),
        "pose": _jsonable(pose),
        "obs": _jsonable(obs),
        "action": _jsonable(action),
        "reward": float(reward),
        "reward_breakdown": reward_terms,
        "success": float(info.get("success", 0.0)),
        "collision": float(info.get("collision", 0.0)),
        "stuck": float(info.get("stuck", 0.0)),
        "mode": "PPO",
    }


def _oscillation_count(actions):
    if len(actions) < 3:
        return 0
    turns = [float(a[1]) for a in actions if abs(float(a[1])) > 0.05]
    return sum(
        1 for prev, cur in zip(turns[:-1], turns[1:]) if math.copysign(1.0, prev) != math.copysign(1.0, cur)
    )


def _classify_failure(case, info, done_by_timeout, oscillation_threshold):
    scene_type = case["scene_type"]
    actions = case["actions"]
    if float(info.get("collision", 0.0)) > 0.5:
        return "collision"
    if float(info.get("stuck", 0.0)) > 0.5:
        return "stuck"
    if scene_type == "false_feasible" and len(actions) > 5:
        return "false_feasible_wrong_entry"
    if scene_type in {"l_shaped", "s_shaped"} and not case["success"]:
        return "turn_failure"
    if _oscillation_count(actions) >= oscillation_threshold:
        return "oscillation"
    if done_by_timeout:
        return "timeout"
    return None


def _write_failure_case(case, failure_dir, max_per_type, counts):
    failure_type = case.get("failure_type")
    if failure_type is None:
        return
    if counts.get(failure_type, 0) >= max_per_type:
        return
    counts[failure_type] = counts.get(failure_type, 0) + 1
    type_dir = failure_dir / failure_type
    type_dir.mkdir(parents=True, exist_ok=True)
    path = type_dir / f"{case['episode_id']:06d}_{case['scene_type']}.json"
    with path.open("w") as f:
        json.dump(_jsonable(case), f, indent=2)
    print(f"[failure] {failure_type}: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("data/narrow_passage_sb3/ppo_narrow_passage.zip"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--difficulty", choices=DIFFICULTY_CONFIGS, default="hard")
    parser.add_argument("--ablation", choices=ABLATION_MASKS, default="full")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--env-version", choices=["v1", "v2"], default="v1")
    parser.add_argument(
        "--corridor-types",
        type=str,
        default="straight,l_shaped,s_shaped,false_feasible,asymmetric",
        help="Comma-separated v2 corridor types.",
    )
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument(
        "--failure-dir",
        type=Path,
        default=Path("results/narrow_passage_rl/failure_cases"),
    )
    parser.add_argument("--max-failure-cases", type=int, default=3)
    parser.add_argument("--oscillation-threshold", type=int, default=8)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    if args.env_version == "v2":
        corridor_types = [name.strip() for name in args.corridor_types.split(",") if name.strip()]
        env = HarderNarrowPassageEnv({"corridor_types": corridor_types})
    else:
        env = ProceduralNarrowPassageEnv(DIFFICULTY_CONFIGS[args.difficulty])
    env = apply_ablation(env, args.ablation)
    model = PPO.load(args.model, device="cpu")
    stats = []
    failure_counts = {}
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
        initial_pose = raw_env.pose.copy()
        geometry = _scene_geometry(raw_env, args.env_version)
        trace = make_episode_trace()
        update_episode_trace(trace, obs)
        done = False
        info = {}
        trajectory = []
        actions = []
        rollout_steps = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(np.asarray(action, dtype=np.float32).copy())
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(raw_env.pose.copy())
            rollout_steps.append(_record_step(raw_env, obs, action, reward, info))
            update_episode_trace(trace, obs)
            done = terminated or truncated
        stat = finalize_episode_info(info, trace)
        stats.append(stat)

        scene_type = geometry["scene_type"]
        success = float(info.get("success", 0.0)) > 0.5
        done_by_timeout = (
            len(trajectory) >= int(getattr(raw_env, "max_steps", len(trajectory)))
            and not success
            and float(info.get("collision", 0.0)) <= 0.5
        )
        case = {
            "episode_id": ep,
            "scene_type": scene_type,
            "seed": ep,
            "initial_pose": _jsonable(initial_pose),
            "goal_pose": geometry.get("goal_world", geometry.get("goal")),
            "passage_width": float(info.get("passage_width", stat.get("passage_width", 0.0))),
            "body_margin": float(info.get("body_margin", stat.get("min_body_margin", 0.0))),
            "min_clearance": float(stat.get("min_clearance", 0.0)),
            "final_status": "success" if success else "failure",
            "success": float(success),
            "trajectory": _jsonable(trajectory),
            "actions": _jsonable(actions),
            "reward_breakdown": [step["reward_breakdown"] for step in rollout_steps],
            "steps": rollout_steps,
            "geometry": geometry,
        }
        case["failure_type"] = _classify_failure(
            case, info, done_by_timeout, args.oscillation_threshold
        )
        if case["failure_type"] is not None:
            case["final_status"] = case["failure_type"]
        if args.save_failures and not success and case["failure_type"] is not None:
            _write_failure_case(
                case, args.failure_dir, args.max_failure_cases, failure_counts
            )

    summary = summarize_episode_stats(stats)
    summary.update(
        {
            "policy": "passage_ppo",
            "difficulty": args.difficulty,
            "ablation": args.ablation,
            "model": str(args.model),
        }
    )
    print_summary(summary)
    if args.csv is not None:
        write_summary_csv(args.csv, [summary])


if __name__ == "__main__":
    main()
