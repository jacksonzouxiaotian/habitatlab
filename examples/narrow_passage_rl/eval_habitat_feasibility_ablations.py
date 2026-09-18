#!/usr/bin/env python3
"""Paired RQ1 feasibility ablations on the 151-episode Habitat val split.

The historical ``fsm_full`` stress controller logged a legacy belief but did
not route that belief into its mode decision.  This runner is intentionally a
new, timestamped evaluation: DynamicFeasibilityEstimator and the strict
selector directly choose EXPLORE/COMMIT/RECOVER/REJECT before the shared
velocity controller acts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import habitat

from eval_habitat_stress_validation import (
    FEATURE_DIM,
    _act,
    _dist_to_goal,
    _heading_error,
    _lateral_offset,
    _stop_act,
)
from narrow_passage.models.dynamic_feasibility import DynamicFeasibilityEstimator
from narrow_passage.models.feasibility_ablation import (
    AblationMode,
    SharedDecisionState,
    capability_record,
    select_ablation_mode,
    uses_fixed_uncertainty,
    uses_yaw_prior,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "narrow_passage_rl"
DEFAULT_DATASET = Path(
    "/media/xiaotian/ACD525D1B7A093D9/habitat_data/"
    "datasets/narrow_passage/{split}/{split}.json.gz"
)
DEFAULT_SCENES = Path(
    "/media/xiaotian/ACD525D1B7A093D9/habitat_data/versioned_data/hm3d-0.2"
)
DEFAULT_SCENE_DATASET = Path(
    "/media/xiaotian/ACD525D1B7A093D9/habitat_data/versioned_data/hm3d-0.2/"
    "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
)

METHODS = (
    "full_dynamic_uncertainty",
    "point_estimate",
    "no_uncertainty",
    "no_yaw_prior",
)


def observation_hash(features: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(features, dtype="<f4").tobytes()).hexdigest()


def random_hash(seed: int, episode_id: str) -> str:
    # Nominal protocol has no stochastic observation perturbation.  This token
    # makes the paired seed contract explicit without pretending random draws
    # occurred.
    return hashlib.sha256(f"nominal::{seed}::{episode_id}".encode()).hexdigest()


def make_env(args):
    data_path = str(args.data_path).format(split=args.split)
    overrides = [
        "habitat/task=narrow_passage",
        f"habitat.dataset.data_path={data_path}",
        f"habitat.dataset.split={args.split}",
        "habitat.dataset.type=PointNav-v1",
        f"habitat.dataset.scenes_dir={args.scenes_dir}",
        "habitat.simulator.habitat_sim_v0.allow_sliding=False",
    ]
    if not args.skip_scene_dataset_config:
        overrides.append(
            f"habitat.simulator.scene_dataset={args.scene_dataset_config}"
        )
    if args.agent_radius is not None:
        overrides.append(
            f"habitat.simulator.agents.main_agent.radius={args.agent_radius}"
        )
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=overrides,
    )
    return habitat.Env(config=config), config


def refreshed_features(env, obs: dict) -> np.ndarray:
    features = np.asarray(
        obs.get("narrow_passage_features", np.zeros(FEATURE_DIM, dtype=np.float32)),
        dtype=np.float32,
    ).copy()
    features[10] = _heading_error(env)
    features[11] = _lateral_offset(env)
    features[12] = _dist_to_goal(env)
    return features


def controller_action(env, features: np.ndarray, mode: AblationMode) -> dict:
    heading = _heading_error(env)
    lateral = float(features[11])
    if mode is AblationMode.RECOVER:
        return _act(-0.12, float(np.clip(0.4 * heading, -0.8, 0.8)))
    if mode is AblationMode.EXPLORE:
        # Active sensing/readiness: first align in place, then move slowly while
        # maintaining centering.  EXPLORE never calls the COMMIT controller.
        angular = float(np.clip(1.6 * (0.9 * heading - 1.2 * lateral), -1.2, 1.2))
        linear = 0.0 if abs(heading) > 0.20 else 0.08
        return _act(linear, angular)
    if mode is AblationMode.COMMIT:
        angular = float(np.clip(0.9 * heading - 1.2 * lateral, -1.2, 1.2))
        return _act(0.20, angular)
    return _stop_act()


def run_episode(env, obs: dict, method: str, args) -> tuple[dict, list[dict]]:
    estimator = DynamicFeasibilityEstimator()
    features = refreshed_features(env, obs)
    first_obs_hash = observation_hash(features)
    episode_id = str(env.current_episode.episode_id)
    draw_hash = random_hash(args.seed, episode_id)
    mode_counts: Counter[str] = Counter()
    audit_steps: list[dict] = []
    any_collision = False
    min_margin = float("inf")
    steps = 0
    prior_mode: AblationMode | None = None
    rejected = False

    while not env.episode_over and steps < args.max_steps:
        current_collision = float(features[16]) > 0.5
        current_stuck = float(features[15])
        any_collision = any_collision or current_collision
        min_margin = min(min_margin, float(features[9]))
        metrics = estimator.update(
            features,
            use_yaw_prior=uses_yaw_prior(method),
            fixed_uncertainty=uses_fixed_uncertainty(method),
            memory_risk=0.0,
        )
        selection = select_ablation_mode(
            method,
            metrics,
            SharedDecisionState(
                collision=current_collision,
                stuck_score=current_stuck,
                memory_risk=0.0,
                prior_commitment_failed=(
                    prior_mode is AblationMode.COMMIT
                    and (current_collision or current_stuck > 0.80)
                ),
                explicit_blocker=False,
            ),
        )
        mode_counts[selection.mode.value] += 1
        action = controller_action(env, features, selection.mode)
        action_args = action["action_args"]
        audit_steps.append(
            {
                "episode_id": episode_id,
                "method": method,
                "step": steps,
                "observation_sha256": first_obs_hash,
                "random_draw_sha256": draw_hash,
                "mu_delta_struct": metrics["mu_delta_struct"],
                "sigma_delta_struct": metrics["sigma_delta_struct"],
                "p_feas_struct": metrics["p_feas_struct"],
                "mu_delta_pose": metrics["mu_delta_pose"],
                "sigma_delta_pose": metrics["sigma_delta_pose"],
                "p_feas_pose": metrics["p_feas_pose"],
                "W_required_with_yaw": metrics["W_required_with_yaw"],
                "W_required_without_yaw": metrics["W_required_without_yaw"],
                "yaw_error": metrics["yaw_error"],
                "selected_mode": selection.mode.value,
                "decision_rule": selection.decision_rule,
                "uncertainty_triggered": float(selection.uncertainty_triggered),
                "linear_action_normalized": float(action_args["linear_velocity"]),
                "angular_action_normalized": float(action_args["angular_velocity"]),
            }
        )
        prior_mode = selection.mode
        if selection.mode is AblationMode.REJECT:
            rejected = True
            break
        obs = env.step(action)
        steps += 1
        features = refreshed_features(env, obs)
        env_metrics = env.get_metrics()
        if (
            float(env_metrics.get("narrow_passage_success", 0.0)) > 0.5
            or float(env_metrics.get("narrow_passage_collision", 0.0)) > 0.5
            or float(env_metrics.get("narrow_passage_stuck", 0.0)) > 0.5
        ):
            break

    env_metrics = env.get_metrics()
    if not math.isfinite(min_margin):
        min_margin = float(features[9])
    success = float(env_metrics.get("narrow_passage_success", 0.0))
    collision = max(
        float(any_collision),
        float(env_metrics.get("narrow_passage_collision", 0.0)),
    )
    stuck = float(env_metrics.get("narrow_passage_stuck", 0.0))
    timeout = float(
        steps >= args.max_steps
        and success < 0.5
        and collision < 0.5
        and stuck < 0.5
        and not rejected
    )
    behavior_payload = "\n".join(
        f"{row['selected_mode']}:{row['linear_action_normalized']:.8f}:{row['angular_action_normalized']:.8f}"
        for row in audit_steps
    )
    return (
        {
            "episode_id": episode_id,
            "method": method,
            "seed": args.seed,
            "observation_sha256": first_obs_hash,
            "random_draw_sha256": draw_hash,
            "behavior_sha256": hashlib.sha256(behavior_payload.encode()).hexdigest(),
            "success": success,
            "strict_success": float(
                success > 0.5
                and collision < 0.5
                and stuck < 0.5
                and min_margin >= args.strict_clearance_threshold
            ),
            "collision": collision,
            "stuck": stuck,
            "timeout": timeout,
            "false_reject": float(rejected),
            "rejected": float(rejected),
            "steps": steps,
            "min_clearance": min_margin,
            "commit_steps": mode_counts["COMMIT"],
            "explore_steps": mode_counts["EXPLORE"],
            "recover_steps": mode_counts["RECOVER"],
            "reject_steps": mode_counts["REJECT"],
            "final_mode": prior_mode.value if prior_mode else "unavailable",
        },
        audit_steps,
    )


def evaluate_method(method: str, args) -> tuple[list[dict], list[dict], dict]:
    episodes: list[dict] = []
    steps: list[dict] = []
    env, config = make_env(args)
    try:
        total = env.number_of_episodes
        target = total if args.num_episodes <= 0 else min(total, args.num_episodes)
        for index in range(target):
            obs = env.reset()
            episode = env.current_episode
            info = getattr(episode, "info", {}) or {}
            row, audit = run_episode(env, obs, method, args)
            row.update(
                {
                    "scene_id": str(episode.scene_id),
                    "difficulty": str(info.get("difficulty", "?")),
                    "body_margin_label": float(info.get("body_margin", "nan")),
                    "false_feasible": float(
                        bool(info.get("false_feasible", False))
                    ),
                    "correct_reject": float(
                        row["rejected"] > 0.5
                        and bool(info.get("false_feasible", False))
                    ),
                    "false_reject": float(
                        row["rejected"] > 0.5
                        and not bool(info.get("false_feasible", False))
                    ),
                }
            )
            episodes.append(row)
            steps.extend(audit)
            if args.verbose and (index < 5 or (index + 1) % 25 == 0):
                print(
                    f"[{method}] {index + 1}/{target} success={row['success']:.0f} "
                    f"reject={row['rejected']:.0f} steps={row['steps']}",
                    flush=True,
                )
    finally:
        env.close()
    config_record = {
        "max_episode_steps": int(config.habitat.environment.max_episode_steps),
        "sim_radius_m": float(config.habitat.simulator.agents.main_agent.radius),
        "depth_normalized": bool(
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth
        ),
        "depth_min_m": float(
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth
        ),
        "depth_max_m": float(
            config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth
        ),
        "success_distance_m": float(
            config.habitat.task.measurements.narrow_passage_success.success_distance
        ),
        "success_heading_threshold_rad": float(
            config.habitat.task.measurements.narrow_passage_success.heading_threshold
        ),
        "success_lateral_threshold_m": float(
            config.habitat.task.measurements.narrow_passage_success.lateral_threshold
        ),
    }
    return episodes, steps, config_record


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows: list[dict]) -> list[dict]:
    output = []
    for method in METHODS:
        values = [row for row in rows if row["method"] == method]
        if not values:
            continue
        n = len(values)
        output.append(
            {
                "method": method,
                "episodes": n,
                "success_rate": sum(row["success"] for row in values) / n,
                "collision_rate": sum(row["collision"] for row in values) / n,
                "false_reject_rate": sum(row["false_reject"] for row in values) / n,
                "timeout_rate": sum(row["timeout"] for row in values) / n,
                "avg_steps": sum(row["steps"] for row in values) / n,
                "commit_step_rate": sum(row["commit_steps"] for row in values)
                / max(1, sum(row["steps"] + row["reject_steps"] for row in values)),
                "explore_step_rate": sum(row["explore_steps"] for row in values)
                / max(1, sum(row["steps"] + row["reject_steps"] for row in values)),
            }
        )
    return output


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_table(path: Path, summary: list[dict], alias_equal: bool) -> None:
    lines = [
        "# Compact Habitat RQ1 Feasibility Ablation (new run)",
        "",
        "All rows use the same paired validation episodes, step budget, and Habitat success measure. This is a new belief-gated run and does not overwrite or silently merge with historical rows.",
        "",
        "| Method | N | Success | Collision | False Reject | Timeout | Avg steps | Commit steps | Explore steps |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {method} | {episodes} | {success} | {collision} | {reject} | {timeout} | {steps:.1f} | {commit} | {explore} |".format(
                method=row["method"],
                episodes=row["episodes"],
                success=pct(row["success_rate"]),
                collision=pct(row["collision_rate"]),
                reject=pct(row["false_reject_rate"]),
                timeout=pct(row["timeout_rate"]),
                steps=row["avg_steps"],
                commit=pct(row["commit_step_rate"]),
                explore=pct(row["explore_step_rate"]),
            )
        )
    lines.extend(
        [
            "",
            f"Compatibility check: point-estimate and no-uncertainty episode behavior hashes are {'identical' if alias_equal else 'NOT identical'}. They are behaviorally equivalent aliases and must not be counted as independent paper methods.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--scene-dataset-config", type=Path, default=DEFAULT_SCENE_DATASET)
    parser.add_argument(
        "--skip-scene-dataset-config",
        action="store_true",
        help="Load episode scene .glb paths directly (for example MP3D).",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-episodes", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--strict-clearance-threshold", type=float, default=-0.02)
    parser.add_argument(
        "--agent-radius",
        type=float,
        default=None,
        help="Optional simulator body radius in metres.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    data_path = Path(str(args.data_path).format(split=args.split))
    required_paths = [data_path, args.scenes_dir]
    if not args.skip_scene_dataset_config:
        required_paths.append(args.scene_dataset_config)
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or RESULTS / "audits" / f"habitat_rq1_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    all_episodes: list[dict] = []
    all_steps: list[dict] = []
    config_records: dict[str, dict] = {}
    for method in args.methods:
        episodes, steps, config_record = evaluate_method(method, args)
        all_episodes.extend(episodes)
        all_steps.extend(steps)
        config_records[method] = config_record

    paired_fields = ("observation_sha256", "random_draw_sha256")
    by_method = {
        method: {row["episode_id"]: row for row in all_episodes if row["method"] == method}
        for method in args.methods
    }
    reference = by_method[args.methods[0]]
    paired_equal = all(
        set(rows) == set(reference)
        and all(rows[eid][field] == reference[eid][field] for eid in reference for field in paired_fields)
        for rows in by_method.values()
    )
    alias_equal = True
    if {"point_estimate", "no_uncertainty"}.issubset(by_method):
        point = by_method["point_estimate"]
        no_unc = by_method["no_uncertainty"]
        alias_equal = set(point) == set(no_unc) and all(
            point[eid]["behavior_sha256"] == no_unc[eid]["behavior_sha256"]
            and point[eid]["success"] == no_unc[eid]["success"]
            for eid in point
        )
        if not alias_equal:
            raise RuntimeError("point_estimate/no_uncertainty compatibility aliases diverged")
    if not paired_equal:
        raise RuntimeError("paired initial-observation/random hashes differ across methods")

    summary = summary_rows(all_episodes)
    write_csv(out_dir / "episodes.csv", all_episodes)
    write_csv(out_dir / "steps.csv", all_steps)
    write_csv(out_dir / "summary.csv", summary)
    write_table(out_dir / "compact_habitat_rq1.md", summary, alias_equal)
    manifest = {
        "methods": args.methods,
        "capabilities": {method: capability_record(method) for method in args.methods},
        "dataset": str(data_path),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "scenes_dir": str(args.scenes_dir),
        "scene_dataset_config": str(args.scene_dataset_config),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "agent_radius_m": args.agent_radius,
        "config_records": config_records,
        "paired_initial_observation_and_random_hashes": paired_equal,
        "point_estimate_no_uncertainty_behaviorally_equivalent": alias_equal,
        "historical_results_overwritten": False,
        "scope_note": "new belief-gated nominal Habitat run; not merged with historical stress CSV",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out_dir)


if __name__ == "__main__":
    main()
