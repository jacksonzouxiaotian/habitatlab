#!/usr/bin/env python3
"""Paired 45-episode mechanism validation for strict feasibility ablations.

This is deliberately a decision-mechanism stress test, not an outcome-quality
benchmark.  Every variant receives the exact same synthetic observation and
random corruption at each paired step.  The complete 225-cell Cartesian grid is
written to disk; a deterministic, coverage-checked 45-cell subset is executed
before any full navigation evaluation is allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_harder_benchmark import TurnCommitFSM
from narrow_passage.models.dynamic_feasibility import (
    DynamicFeasibilityConfig,
    required_width_without_yaw,
)
from narrow_passage.models.feasibility_ablation import (
    MAIN_ABLATIONS,
    SelectorThresholds,
)


MARGINS = (-0.10, -0.05, 0.0, 0.05, 0.10)
YAWS_DEG = (0, 15, 30, 45, 60)
NOISE_LEVELS = ("clean", "mild", "severe")
LATERAL_OFFSETS = (0.0, 0.10, 0.20)
STEPS_PER_EPISODE = 4
DEFAULT_PARENT = SCRIPT_DIR / "results" / "ablation_feasibility"


def _default_output_dir() -> Path:
    return DEFAULT_PARENT / datetime.now().strftime("mechanism_%Y%m%d_%H%M%S")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _scenario_key(case: tuple[float, int, str, float]) -> str:
    margin, yaw, noise, lateral = case
    return f"m{margin:+.2f}_y{yaw:02d}_{noise}_l{lateral:.2f}"


def full_grid() -> list[tuple[float, int, str, float]]:
    return list(itertools.product(MARGINS, YAWS_DEG, NOISE_LEVELS, LATERAL_OFFSETS))


def mechanism_subset(size: int = 45) -> list[tuple[float, int, str, float]]:
    if not 20 <= size <= 50:
        raise ValueError("mechanism validation must contain 20-50 episodes")
    if size < len(MARGINS) * len(YAWS_DEG):
        raise ValueError("size must be at least 25 to cover every margin-yaw pair")

    selected: list[tuple[float, int, str, float]] = []
    for mi, margin in enumerate(MARGINS):
        for yi, yaw in enumerate(YAWS_DEG):
            selected.append((
                margin,
                yaw,
                NOISE_LEVELS[(mi + yi) % len(NOISE_LEVELS)],
                LATERAL_OFFSETS[(2 * mi + yi) % len(LATERAL_OFFSETS)],
            ))

    required = [
        (0.05, 0, "severe", 0.0),
        (0.10, 0, "clean", 0.20),
        (0.05, 30, "severe", 0.10),
        (0.10, 30, "mild", 0.0),
        (0.0, 15, "severe", 0.20),
    ]
    for case in required:
        if case not in selected:
            selected.append(case)

    remaining = [case for case in full_grid() if case not in selected]
    remaining.sort(
        key=lambda case: hashlib.sha256(_scenario_key(case).encode()).hexdigest()
    )
    selected.extend(remaining[: size - len(selected)])
    selected = selected[:size]

    for axis_name, expected, index in (
        ("margin", set(MARGINS), 0),
        ("yaw", set(YAWS_DEG), 1),
        ("noise", set(NOISE_LEVELS), 2),
        ("lateral", set(LATERAL_OFFSETS), 3),
    ):
        got = {case[index] for case in selected}
        if got != expected:
            raise RuntimeError(f"stress subset does not cover {axis_name}: {got}")
    return selected


def _noise_draws(case: tuple[float, int, str, float], step: int) -> tuple[np.ndarray, float, list[int]]:
    payload = f"{_scenario_key(case)}:step{step}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    level = case[2]
    ray_sigma = {"clean": 0.0, "mild": 0.035, "severe": 0.10}[level]
    width_sigma = {"clean": 0.0, "mild": 0.018, "severe": 0.060}[level]
    dropout_count = {"clean": 0, "mild": 1, "severe": 3}[level]
    ray_noise = rng.normal(0.0, ray_sigma, size=6)
    width_noise = float(rng.normal(0.0, width_sigma))
    dropout = sorted(
        rng.choice(6, size=dropout_count, replace=False).astype(int).tolist()
    )
    return ray_noise, width_noise, dropout


def make_observation(
    case: tuple[float, int, str, float],
    step: int,
    cfg: DynamicFeasibilityConfig,
) -> tuple[np.ndarray, str]:
    margin, yaw_deg, noise_level, lateral = case
    available_width = required_width_without_yaw(cfg) + margin
    ray_noise, width_noise, dropout = _noise_draws(case, step)
    base_rays = np.asarray([0.72, 0.78, 0.74, 0.96, 1.02, 0.98], dtype=float)
    yaw_scale = 0.12 * abs(math.sin(math.radians(yaw_deg)))
    lateral_skew = np.asarray([-1, 0, 1, -1, 0, 1], dtype=float) * lateral
    rays = base_rays + yaw_scale + lateral_skew + ray_noise
    obs = np.zeros(19, dtype=np.float32)
    obs[:6] = np.clip(rays, 0.02, cfg.max_depth)
    if dropout:
        # Zero is an invalid sensor return.  Max range is reserved for a valid
        # no-return/free-space ray in the repaired estimator.
        obs[dropout] = 0.0
    body_free = (available_width - cfg.body_width) / 2.0
    obs[6] = body_free - lateral
    obs[7] = body_free + lateral
    obs[8] = max(0.05, available_width + width_noise)
    obs[9] = min(float(obs[6]), float(obs[7]))
    obs[10] = math.radians(yaw_deg)
    obs[11] = lateral
    obs[12] = 2.0

    # One explicit paired sequence validates Recover after a prior Commit.  The
    # injected stuck signal is identical for every method and is logged.
    if case == (0.10, 0, "clean", 0.20) and step == STEPS_PER_EPISODE - 1:
        obs[15] = 0.60

    corruption = {
        "ray_noise": [float(value) for value in ray_noise],
        "width_noise": width_noise,
        "dropout_indices": dropout,
        "noise_level": noise_level,
    }
    signature = hashlib.sha256(
        json.dumps(corruption, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return obs, signature


def _observation_hash(obs: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()


def validate_pairing(step_rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in step_rows:
        grouped.setdefault((str(row["episode_id"]), int(row["step"])), []).append(row)
    expected = set(MAIN_ABLATIONS)
    invariant_keys = (
        "observation_sha256",
        "random_draw_sha256",
        "controller_config_sha256",
        "available_width",
        "stress_margin",
        "stress_yaw_deg",
        "noise_level",
        "lateral_offset",
    )
    estimator_invariants = (
        "mu_D",
        "dynamic_sigma_delta",
        "ray_dispersion",
        "valid_depth_ratio",
        "dropout_ratio",
        "boundary_fitting_residual",
        "temporal_width_variation",
        "W_required_with_yaw",
        "W_required_without_yaw",
        "posterior_mu_delta_struct",
        "posterior_var_delta_struct",
        "posterior_sigma_delta_struct",
        "posterior_concentration_struct",
    )
    for key, rows in grouped.items():
        if {str(row["method"]) for row in rows} != expected:
            raise RuntimeError(f"paired methods differ at {key}")
        for field in invariant_keys:
            if len({row[field] for row in rows}) != 1:
                raise RuntimeError(f"paired input {field} differs at {key}")
        for field in estimator_invariants:
            values = np.asarray([float(row[field]) for row in rows], dtype=float)
            if float(np.ptp(values)) > 1e-9:
                raise RuntimeError(f"non-ablated estimator field {field} differs at {key}")


def _paired_disagreement(
    rows: list[dict[str, Any]], left: str, right: str, field: str
) -> tuple[int, int]:
    indexed = {
        (str(row["episode_id"]), int(row["step"]), str(row["method"])): row
        for row in rows
    }
    pairs = sorted({(key[0], key[1]) for key in indexed if key[2] == left})
    count = sum(
        indexed[(episode, step, left)][field]
        != indexed[(episode, step, right)][field]
        for episode, step in pairs
    )
    return int(count), len(pairs)


def run(
    output_dir: Path,
    episode_count: int,
    *,
    enable_belief_propagation: bool = True,
    enable_belief_fusion: bool = True,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output dir {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    feasibility_cfg = DynamicFeasibilityConfig(
        enable_belief_propagation=enable_belief_propagation,
        enable_belief_fusion=enable_belief_fusion,
    )
    selector_cfg = SelectorThresholds()
    controller_config_sha256 = hashlib.sha256(
        json.dumps(
            {
                "controller": "TurnCommitFSM/fsm_action",
                "selector_thresholds": selector_cfg.__dict__,
                "feasibility_estimator": asdict(feasibility_cfg),
                "action_limits": {"linear_commit": 0.20, "linear_explore": 0.08,
                                  "angular_abs_max": 0.80},
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    grid = full_grid()
    subset = mechanism_subset(episode_count)
    _write_csv([
        {
            "episode_id": _scenario_key(case),
            "stress_margin": case[0],
            "stress_yaw_deg": case[1],
            "noise_level": case[2],
            "lateral_offset": case[3],
        }
        for case in grid
    ], output_dir / "stress_grid_225.csv")
    _write_csv([
        {
            "episode_id": _scenario_key(case),
            "stress_margin": case[0],
            "stress_yaw_deg": case[1],
            "noise_level": case[2],
            "lateral_offset": case[3],
        }
        for case in subset
    ], output_dir / "mechanism_scenarios.csv")

    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for case in subset:
        episode_id = _scenario_key(case)
        available_width = required_width_without_yaw(feasibility_cfg) + case[0]
        observations = [
            make_observation(case, step, feasibility_cfg)
            for step in range(STEPS_PER_EPISODE)
        ]
        for method in MAIN_ABLATIONS:
            agent = TurnCommitFSM(
                variant="full",
                strict_ablation=method,
                selector_cfg=selector_cfg,
                feasibility_cfg=feasibility_cfg,
                corridor_type="straight_stress",
            )
            agent.reset()
            counts: Counter[str] = Counter()
            for step, (obs, random_signature) in enumerate(observations):
                action, selected_mode = agent.step(obs.copy())
                audited_mode = str(agent.last_audit["selected_mode"])
                counts[audited_mode] += 1
                step_rows.append({
                    "episode_id": episode_id,
                    "scenario_id": episode_id,
                    "method": method,
                    "step": step,
                    "stress_margin": case[0],
                    "stress_yaw_deg": case[1],
                    "noise_level": case[2],
                    "lateral_offset": case[3],
                    "available_width": available_width,
                    "observation_sha256": _observation_hash(obs),
                    "random_draw_sha256": random_signature,
                    "controller_config_sha256": controller_config_sha256,
                    **agent.last_audit,
                    "controller_mode": selected_mode,
                    "linear_action": float(action[0]),
                    "angular_action": float(action[1]),
                })
            episode_rows.append({
                "episode_id": episode_id,
                "method": method,
                "stress_margin": case[0],
                "stress_yaw_deg": case[1],
                "noise_level": case[2],
                "lateral_offset": case[3],
                "available_width": available_width,
                "steps": STEPS_PER_EPISODE,
                "commit_steps": counts["COMMIT"],
                "explore_steps": counts["EXPLORE"],
                "recover_steps": counts["RECOVER"],
                "reject_steps": counts["REJECT"],
            })

    validate_pairing(step_rows)
    full_rows = [
        row for row in step_rows
        if row["method"] == "full_dynamic_uncertainty"
    ]
    unique_sigma = len({round(float(row["sigma_delta"]), 9) for row in full_rows})
    uncertainty_count = sum(bool(row["uncertainty_triggered"]) for row in full_rows)
    disagreement_count, paired_steps = _paired_disagreement(
        step_rows, "full_dynamic_uncertainty", "mean_only", "selected_mode"
    )
    feasibility_disagreement, _ = _paired_disagreement(
        step_rows, "full_dynamic_uncertainty", "mean_only", "feasibility_mode"
    )
    yaw_changed_count = sum(bool(row["yaw_prior_changed_decision"]) for row in full_rows)

    gates = {
        "sigma_delta_nonconstant": unique_sigma > 1,
        "uncertainty_branch_trigger_rate_gt_zero": uncertainty_count > 0,
        "full_vs_mean_final_mode_disagreement_gt_zero": disagreement_count > 0,
        "full_vs_mean_feasibility_disagreement_gt_zero": feasibility_disagreement > 0,
        "yaw_prior_changed_decision_gt_zero": yaw_changed_count > 0,
        "paired_inputs_and_randomness_identical": True,
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"mechanism validation failed: {failed}")

    method_summary = []
    for method in MAIN_ABLATIONS:
        rows = [row for row in step_rows if row["method"] == method]
        mode_counts = Counter(str(row["selected_mode"]) for row in rows)
        method_summary.append({
            "method": method,
            "episodes": episode_count,
            "steps": len(rows),
            "sigma_min": min(float(row["sigma_delta"]) for row in rows),
            "sigma_max": max(float(row["sigma_delta"]) for row in rows),
            "uncertainty_trigger_rate": (
                sum(bool(row["uncertainty_triggered"]) for row in rows) / len(rows)
            ),
            "yaw_prior_changed_rate": (
                sum(bool(row["yaw_prior_changed_decision"]) for row in rows) / len(rows)
            ),
            "commit_steps": mode_counts["COMMIT"],
            "explore_steps": mode_counts["EXPLORE"],
            "recover_steps": mode_counts["RECOVER"],
            "reject_steps": mode_counts["REJECT"],
        })

    report = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "mechanism_validation_not_navigation_outcome_evaluation",
        "complete_grid_cells": len(grid),
        "executed_paired_episodes": episode_count,
        "methods": list(MAIN_ABLATIONS),
        "steps_per_episode": STEPS_PER_EPISODE,
        "full_sigma_unique_values": unique_sigma,
        "full_uncertainty_trigger_count": uncertainty_count,
        "full_uncertainty_trigger_rate": uncertainty_count / len(full_rows),
        "full_vs_mean_final_mode_disagreement_count": disagreement_count,
        "full_vs_mean_final_mode_disagreement_rate": disagreement_count / paired_steps,
        "full_vs_mean_feasibility_disagreement_count": feasibility_disagreement,
        "yaw_prior_changed_decision_count": yaw_changed_count,
        "yaw_prior_changed_decision_rate": yaw_changed_count / len(full_rows),
        "gates": gates,
        "full_evaluation_permitted": True,
        "full_evaluation_launched_by_this_script": False,
        "selector_config": selector_cfg.__dict__,
        "feasibility_config": asdict(feasibility_cfg),
        "method_summary": method_summary,
    }
    _write_csv(step_rows, output_dir / "steps.csv")
    _write_csv(episode_rows, output_dir / "episodes.csv")
    _write_csv(method_summary, output_dir / "mechanism_summary.csv")
    (output_dir / "mechanism_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=45)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--disable-belief-propagation", action="store_true")
    parser.add_argument("--disable-belief-fusion", action="store_true")
    args = parser.parse_args()
    if not 20 <= args.episodes <= 50:
        parser.error("--episodes must be in [20, 50]")
    return args


def main() -> None:
    args = parse_args()
    run(
        args.output_dir or _default_output_dir(),
        args.episodes,
        enable_belief_propagation=not args.disable_belief_propagation,
        enable_belief_fusion=not args.disable_belief_fusion,
    )


if __name__ == "__main__":
    main()
