#!/usr/bin/env python3
"""Balanced OBB structural-feasibility benchmark with gated full execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cross_episode_memory import CrossEpisodeMemory, MemoryConfig
from eval_harder_benchmark import TurnCommitFSM, run_episode
from procedural_env_v2 import HarderNarrowPassageEnv
from narrow_passage.models.dynamic_feasibility import DynamicFeasibilityConfig
from narrow_passage.models.feasibility_ablation import (
    MAIN_ABLATIONS,
    SelectorThresholds,
    uses_memory,
)
from narrow_passage.models.robot_morphology import RobotMorphology


MARGIN_BINS = (
    (-0.15, -0.10), (-0.10, -0.05), (-0.05, 0.0),
    (0.0, 0.05), (0.05, 0.10), (0.10, 0.15),
)
SCENES = (
    "straight", "l_shaped", "s_shaped", "narrow_entry",
    "narrow_exit", "asymmetric", "false_feasible",
)
YAWS = (0, 15, 30, 45, 60)
NOISE = ("clean", "mild", "severe")
LATERAL = (0.0, 0.10, 0.20)
DEFAULT_PARENT = SCRIPT_DIR / "results" / "ablation_feasibility"
STRICT_GATE_SCHEMA = "strict_structural_repair_v1"
REACTIVE_BASELINE = "reactive_rule_baseline"
EVALUATION_METHODS = (*MAIN_ABLATIONS, REACTIVE_BASELINE)


def _stamp(phase: str) -> Path:
    return DEFAULT_PARENT / f"structural_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class _StreamingCSV:
    def __init__(self, path: Path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self._fields: list[str] = []
        self.count = 0

    def append(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._fields = list(row)
            self._writer = csv.DictWriter(
                self._handle, fieldnames=self._fields, lineterminator="\n"
            )
            self._writer.writeheader()
        extras = set(row) - set(self._fields)
        if extras:
            raise RuntimeError(f"Step schema changed: {sorted(extras)}")
        self._writer.writerow(row)
        self.count += 1
        if self.count % 1000 == 0:
            self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()


def _hash_payload(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _scenario_seed(seed: int, scenario_id: str) -> int:
    return int(_hash_payload([seed, scenario_id])[:8], 16)


def _scenario_rows(
    phase: str, seeds: list[int], factor_design: str = "balanced"
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    axis_grid = list(itertools.product(YAWS, NOISE, LATERAL))
    for seed in seeds:
        for bin_index, (lower, upper) in enumerate(MARGIN_BINS):
            margin = 0.5 * (lower + upper)
            for scene_index, scene in enumerate(SCENES):
                if phase == "full" and factor_design == "cartesian":
                    factors = axis_grid
                elif phase == "full" and factor_design == "publication":
                    # Fifteen paired conditions per bin×scene×seed cell.  The
                    # yaw×noise grid is complete and lateral offset is assigned
                    # by a Latin rule, giving exact marginal balance while
                    # keeping the run one third the full Cartesian cost.
                    factors = [
                        (
                            yaw,
                            noise,
                            LATERAL[
                                (
                                    yaw_index
                                    + noise_index
                                    + scene_index
                                    + seed
                                    + bin_index
                                ) % len(LATERAL)
                            ],
                        )
                        for yaw_index, yaw in enumerate(YAWS)
                        for noise_index, noise in enumerate(NOISE)
                    ]
                elif phase == "full":
                    # Five-run balanced design: every bin×scene×seed cell sees
                    # every yaw. Across the 7 scenes × 3 seeds, noise and
                    # lateral levels are exactly balanced for each yaw/bin.
                    factors = [
                        (
                            yaw,
                            NOISE[(yaw_index + scene_index + seed + bin_index) % 3],
                            LATERAL[(2 * yaw_index + scene_index + seed + bin_index) % 3],
                        )
                        for yaw_index, yaw in enumerate(YAWS)
                    ]
                else:
                    factors = [
                        axis_grid[
                            (bin_index * len(SCENES) + scene_index + seed)
                            % len(axis_grid)
                        ]
                    ]
                for yaw, noise, lateral in factors:
                    scenario_id = (
                        f"seed{seed}_{scene}_bin{bin_index}_m{margin:+.3f}_"
                        f"y{yaw}_{noise}_l{lateral:.2f}"
                    )
                    rows.append({
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "corridor_type": scene,
                        "margin_bin": f"[{lower:.2f},{upper:.2f}{']' if bin_index == 5 else ')'}",
                        "structural_margin_target": margin,
                        "yaw_deg": yaw,
                        "noise_level": noise,
                        "lateral_offset": lateral,
                    })
    return rows


def _ordered_scenarios(
    rows: list[dict[str, Any]], order_mode: str
) -> list[dict[str, Any]]:
    """Return one deterministic per-seed order shared by every method."""

    if order_mode == "legacy_bin_order":
        ordered = list(rows)
    else:
        ordered = []
        seeds = list(dict.fromkeys(int(row["seed"]) for row in rows))
        for seed in seeds:
            seed_rows = [dict(row) for row in rows if int(row["seed"]) == seed]
            rng = np.random.default_rng(seed)
            if order_mode == "seeded_shuffle":
                indices = np.arange(len(seed_rows))
                rng.shuffle(indices)
                ordered.extend(seed_rows[int(index)] for index in indices)
            elif order_mode == "interleaved_margin_sign":
                negative = [row for row in seed_rows if row["structural_margin_target"] < 0.0]
                positive = [row for row in seed_rows if row["structural_margin_target"] >= 0.0]
                rng.shuffle(negative)
                rng.shuffle(positive)
                for index in range(max(len(negative), len(positive))):
                    if index < len(negative):
                        ordered.append(negative[index])
                    if index < len(positive):
                        ordered.append(positive[index])
            else:
                raise ValueError(f"Unknown episode order mode: {order_mode}")
    for order_index, row in enumerate(ordered):
        row["episode_order_index"] = order_index
    return ordered


def _noise_schedule(
    scenario_id: str, level: str, max_depth: float
) -> tuple[Callable[[np.ndarray, int], np.ndarray], str]:
    schedule_descriptor = {
        "scenario_id": scenario_id,
        "level": level,
        "ray_sigma": {"clean": 0.0, "mild": 0.035, "severe": 0.10}[level],
        "width_sigma": {"clean": 0.0, "mild": 0.018, "severe": 0.060}[level],
        "dropout_count": {"clean": 0, "mild": 1, "severe": 3}[level],
        "dropout_encoding": 0.0,
    }
    signature = _hash_payload(schedule_descriptor)

    def transform(obs: np.ndarray, step: int) -> np.ndarray:
        payload = f"{scenario_id}:step{step}".encode("utf-8")
        rng_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)
        rng = np.random.default_rng(rng_seed)
        out = np.asarray(obs, dtype=np.float32).copy()
        ray_sigma = float(schedule_descriptor["ray_sigma"])
        width_sigma = float(schedule_descriptor["width_sigma"])
        dropout_count = int(schedule_descriptor["dropout_count"])
        out[:6] = np.clip(out[:6] + rng.normal(0.0, ray_sigma, 6), 0.0, max_depth)
        if dropout_count:
            indices = rng.choice(6, dropout_count, replace=False)
            # Zero is the declared invalid-depth encoding.  Max range remains a
            # valid no-return/free-space observation.
            out[indices] = float(schedule_descriptor["dropout_encoding"])
        out[8] = max(0.01, float(out[8]) + float(rng.normal(0.0, width_sigma)))
        return out

    return transform, signature


def _initial_observation_hash(
    morphology: RobotMorphology,
    spec: dict[str, Any],
    episode_seed: int,
    width: float,
    transform: Callable[[np.ndarray, int], np.ndarray],
) -> tuple[str, bool, float]:
    env = HarderNarrowPassageEnv({
        "corridor_types": [spec["corridor_type"]],
        "morphology": morphology,
        "max_steps": 200,
    })
    obs, _ = env.reset(seed=episode_seed, options={
        "corridor_type": spec["corridor_type"],
        "passage_width": width,
        "yaw_deg": spec["yaw_deg"],
        "lateral_offset": spec["lateral_offset"],
    })
    observed = transform(obs, 0)
    return (
        hashlib.sha256(observed.astype(np.float32).tobytes()).hexdigest(),
        bool(env.is_passable),
        float(env._episode_W - morphology.structural_required_width),
    )


def _summary(
    rows: list[dict[str, Any]], methods: tuple[str, ...] = MAIN_ABLATIONS
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for subset_name, passable, metrics in (
            ("feasible", True, ("success", "collision", "false_reject", "timeout", "alignment_time", "oscillation_count")),
            ("infeasible", False, ("correct_reject", "collision", "wasted_commitment", "time_to_reject", "recovery_attempts")),
        ):
            selected = [row for row in method_rows if bool(row["passable"]) is passable]
            result: dict[str, Any] = {
                "method": method, "subset": subset_name, "episodes": len(selected)
            }
            for metric in metrics:
                values = np.asarray([float(row.get(metric, math.nan)) for row in selected])
                values = values[np.isfinite(values)]
                result[metric] = float(np.mean(values)) if len(values) else math.nan
            out.append(result)
    return out


def _gate_report(
    rows: list[dict[str, Any]],
    methods: tuple[str, ...] = MAIN_ABLATIONS,
    step_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate the smoke-to-full fail-fast contract.

    The statistical checks prevent an unbalanced or unpaired smoke run from
    being promoted.  The engineering checks encode the acceptance targets in
    the benchmark specification; they are deliberately *not* relaxed when a
    controller underperforms.  Geometry unit tests remain a separate required
    command because their result cannot be inferred from an episode CSV.
    """
    full = [row for row in rows if row["method"] == "full_dynamic_uncertainty"]
    paired_hashes = defaultdict(lambda: {"obs": set(), "random": set()})
    for row in rows:
        paired_hashes[row["scenario_id"]]["obs"].add(row["observation_sha256"])
        paired_hashes[row["scenario_id"]]["random"].add(row["random_draw_sha256"])
    hash_ok = all(
        len(item["obs"]) == 1 and len(item["random"]) == 1
        for item in paired_hashes.values()
    )
    per_method_success = {
        method: float(np.mean([float(row["success"]) for row in rows if row["method"] == method]))
        for method in methods
    }
    all_degenerate = all(value <= 0.02 for value in per_method_success.values()) or all(
        value >= 0.98 for value in per_method_success.values()
    )
    bin_counts = defaultdict(lambda: {"all": 0, "feasible": 0, "infeasible": 0})
    for row in full:
        item = bin_counts[row["margin_bin"]]
        item["all"] += 1
        item["feasible" if row["passable"] else "infeasible"] += 1
    sampling_ok = len(bin_counts) == len(MARGIN_BINS) and all(
        item["all"] >= len(SCENES) for item in bin_counts.values()
    )

    feasible = [row for row in full if bool(row["passable"])]
    infeasible = [row for row in full if not bool(row["passable"])]

    def _rate(selected: list[dict[str, Any]], field: str) -> float:
        if not selected:
            return math.nan
        return float(np.mean([float(row.get(field, 0.0)) for row in selected]))

    feasible_metrics = {
        "episodes": len(feasible),
        "success": _rate(feasible, "success"),
        "false_reject": _rate(feasible, "false_reject"),
        "collision": _rate(feasible, "collision"),
        "timeout": _rate(feasible, "timeout"),
    }
    infeasible_metrics = {
        "episodes": len(infeasible),
        "correct_reject": _rate(infeasible, "correct_reject"),
    }
    scene_success: dict[str, float | None] = {}
    for scene in SCENES:
        value = _rate(
            [row for row in feasible if row["corridor_type"] == scene],
            "success",
        )
        scene_success[scene] = value if math.isfinite(value) else None

    full_steps = [
        row for row in (step_rows or [])
        if row.get("method") == "full_dynamic_uncertainty"
    ]
    large_yaw_ids = {
        row["episode_id"] for row in feasible if float(row["yaw_deg"]) >= 45.0
    }
    align_then_commit: dict[str, bool] = {}
    for episode_id in large_yaw_ids:
        trace = sorted(
            [row for row in full_steps if row.get("episode_id") == episode_id],
            key=lambda row: int(float(row.get("step", 0))),
        )
        phases = [str(row.get("passage_control_state", "")) for row in trace]
        modes = [str(row.get("selected_mode", "")) for row in trace]
        first_nonempty_phase = next((phase for phase in phases if phase), "")
        align_then_commit[episode_id] = (
            first_nonempty_phase == "ALIGN" and "COMMIT" in modes
        )
    mechanism_metrics = {
        "large_yaw_feasible_episodes": len(large_yaw_ids),
        "large_yaw_align_then_commit_rate": (
            float(np.mean(list(align_then_commit.values())))
            if align_then_commit else math.nan
        ),
        "truly_narrow_correct_reject_count": int(sum(
            float(row.get("correct_reject", 0.0)) > 0.5 for row in infeasible
        )),
        "uncertainty_trigger_rate": (
            _rate(full_steps, "uncertainty_triggered") if full_steps else math.nan
        ),
    }

    protocol_gates = {
        "success_not_all_near_zero_or_one": not all_degenerate,
        "margin_bins_have_minimum_samples": sampling_ok,
        "feasible_and_infeasible_present_overall": (
            bool(feasible) and bool(infeasible)
        ),
        "paired_observation_and_random_hashes_identical": hash_ok,
    }
    engineering_gates = {
        "feasible_success_at_least_60_percent": (
            feasible_metrics["success"] >= 0.60
        ),
        "feasible_false_reject_below_5_percent": (
            feasible_metrics["false_reject"] < 0.05
        ),
        "feasible_collision_below_10_percent": (
            feasible_metrics["collision"] < 0.10
        ),
        "feasible_timeout_below_20_percent": (
            feasible_metrics["timeout"] < 0.20
        ),
        "l_shaped_has_nonzero_feasible_success": (
            scene_success["l_shaped"] is not None
            and scene_success["l_shaped"] > 0.0
        ),
        "s_shaped_has_nonzero_feasible_success": (
            scene_success["s_shaped"] is not None
            and scene_success["s_shaped"] > 0.0
        ),
        "large_yaw_feasible_aligns_then_commits": (
            bool(align_then_commit) and all(align_then_commit.values())
        ),
        "truly_narrow_passage_can_reject": (
            mechanism_metrics["truly_narrow_correct_reject_count"] > 0
        ),
        "uncertainty_branch_triggers": (
            mechanism_metrics["uncertainty_trigger_rate"] > 0.0
        ),
    }
    gates = {**protocol_gates, **engineering_gates}
    return {
        "gate_schema": STRICT_GATE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "protocol_gates": protocol_gates,
        "engineering_gates": engineering_gates,
        "per_method_success": per_method_success,
        "feasible_metrics": feasible_metrics,
        "infeasible_metrics": infeasible_metrics,
        "scene_wise_feasible_success": scene_success,
        "mechanism_metrics": mechanism_metrics,
        "large_yaw_align_then_commit": align_then_commit,
        "margin_bin_counts": dict(bin_counts),
        "full_evaluation_permitted": all(gates.values()),
        "note": (
            "Negative structural-margin bins are necessarily OBB-infeasible; "
            "requiring passable samples inside those bins would contradict the label definition. "
            "Passing the geometry/label unit-test command is an additional prerequisite and is "
            "reported separately from this runtime-derived gate."
        ),
    }


def run(args: argparse.Namespace) -> Path:
    started_at_utc = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    output_dir = args.output_dir or _stamp(args.phase)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty directory {output_dir}")
    if args.phase == "full":
        if args.gate_report is None:
            raise RuntimeError("--phase full requires --gate-report from a passed smoke run")
        gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
        if gate.get("gate_schema") != STRICT_GATE_SCHEMA:
            raise RuntimeError(
                "Full evaluation blocked: gate report predates the strict structural-repair schema"
            )
        if not gate.get("full_evaluation_permitted", False):
            raise RuntimeError("Full evaluation blocked by failed smoke gates")

    methods = tuple(args.methods)
    morphology = RobotMorphology(
        width=args.body_width,
        length=args.body_length,
        safety_margin=args.safety_margin,
    )
    selector_cfg = SelectorThresholds(
        kappa=args.kappa,
        tau_commit=args.tau_commit,
        tau_reject=args.tau_reject,
    )
    specs = _ordered_scenarios(
        _scenario_rows(args.phase, args.seeds, args.factor_design),
        args.episode_order,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_cfg = MemoryConfig(frozen=args.memory_policy == "frozen_memory")
    memories = {
        (seed, method): CrossEpisodeMemory(memory_cfg)
        for seed in args.seeds for method in methods
    }
    for memory in memories.values():
        memory.reset()
    order_rows = [{
        "episode_order_index": int(index),
        "seed_order_index": int(sum(
            1 for prior in specs[:index]
            if int(prior["seed"]) == int(spec["seed"])
        )),
        **spec,
    } for index, spec in enumerate(specs)]
    _write_csv(order_rows, output_dir / "episode_order.csv")
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] | _StreamingCSV = (
        _StreamingCSV(output_dir / "steps.csv") if args.phase == "full" else []
    )
    try:
      for index, spec in enumerate(specs):
        width = morphology.structural_required_width + float(spec["structural_margin_target"])
        episode_seed = _scenario_seed(int(spec["seed"]), str(spec["scenario_id"]))
        transform, random_hash = _noise_schedule(
            str(spec["scenario_id"]), str(spec["noise_level"]), args.max_depth
        )
        observation_hash, passable, actual_margin = _initial_observation_hash(
            morphology, spec, episode_seed, width, transform
        )
        for method in methods:
            env = HarderNarrowPassageEnv({
                "corridor_types": [spec["corridor_type"]],
                "morphology": morphology,
                "max_steps": args.max_steps,
                "max_depth": args.max_depth,
            })
            is_reactive = method == REACTIVE_BASELINE
            memory = None if is_reactive else (
                memories[(int(spec["seed"]), method)]
                if uses_memory(method) and args.memory_policy != "no_memory"
                else None
            )
            agent = None if is_reactive else TurnCommitFSM(
                    variant="full",
                    cross_mem=memory,
                    strict_ablation=method,
                    selector_cfg=selector_cfg,
                    feasibility_cfg=DynamicFeasibilityConfig(
                        morphology=env.morphology,
                        enable_belief_propagation=not args.disable_belief_propagation,
                        enable_belief_fusion=not args.disable_belief_fusion,
                    ),
                    corridor_type=str(spec["corridor_type"]),
                    env=env,
                )
            trace_context = {
                **spec,
                "episode_id": spec["scenario_id"],
                "observation_sha256": observation_hash,
                "random_draw_sha256": random_hash,
            }
            entry_obs_ref: list[np.ndarray] = []
            stat = run_episode(
                env,
                "rule_baseline" if is_reactive else "fsm_cross_memory",
                None,
                memory,
                max_steps=args.max_steps,
                entry_obs_ref=entry_obs_ref,
                variant="full",
                agent=agent,
                episode_seed=episode_seed,
                reset_options={
                    "corridor_type": spec["corridor_type"],
                    "passage_width": width,
                    "yaw_deg": spec["yaw_deg"],
                    "lateral_offset": spec["lateral_offset"],
                },
                observation_transform=transform,
                step_trace_out=step_rows,
                step_trace_context=trace_context,
            )
            if memory is not None and entry_obs_ref:
                memory_audit = memory.record_episode(
                    entry_obs_ref[0],
                    bool(float(stat["success"]) > 0.5),
                    float(stat["passage_width"]),
                    steps=int(stat["steps"]),
                    corridor_type=str(stat["corridor_type"]),
                    outcome=str(stat["episode_outcome"]),
                    attempted=bool(stat["attempted"]),
                    env_step_count=int(stat["env_step_count"]),
                    geometry_caused=bool(stat["geometry_caused"]),
                    structural_margin=float(stat["structural_margin_gt"]),
                    yaw_error=float(stat["entry_heading_error"]),
                    morphology=morphology,
                )
                stat.update(memory_audit)
            elif memory is None:
                stat.update({
                    "geometry_memory_write": 0.0,
                    "memory_write_type": "not_applicable" if is_reactive else "disabled",
                    "memory_write_reason": (
                        "reactive baseline has no memory"
                        if is_reactive
                        else "no_memory ablation or run policy"
                    ),
                })
            episode_rows.append({
                **spec,
                "episode_id": spec["scenario_id"],
                "method": method,
                "episode_seed": episode_seed,
                "observation_sha256": observation_hash,
                "random_draw_sha256": random_hash,
                "passable": passable,
                "passable_label": float(passable),
                "structural_margin_gt": actual_margin,
                **{key: value for key, value in stat.items() if not key.startswith("_")},
                **dict(stat.get("_belief_row", {})),
            })
        if index < 3 or (index + 1) % 25 == 0:
            print(f"[paired] {index + 1}/{len(specs)} {spec['scenario_id']}", flush=True)
    finally:
        if isinstance(step_rows, _StreamingCSV):
            step_rows.close()

    _write_csv(episode_rows, output_dir / "episodes.csv")
    if isinstance(step_rows, list):
        _write_csv(step_rows, output_dir / "steps.csv")
    _write_csv(_summary(episode_rows, methods), output_dir / "subset_metrics.csv")
    gate = _gate_report(
        episode_rows,
        methods,
        step_rows if isinstance(step_rows, list) else None,
    )
    (output_dir / "gate_report.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    finished_at_utc = datetime.now(timezone.utc)
    metadata = {
        "created_at_utc": finished_at_utc.isoformat(),
        "started_at_utc": started_at_utc.isoformat(),
        "phase": args.phase,
        "morphology": morphology.record(),
        "formulas": {
            "delta_struct": "passage_width - (body_width + 2*safety_margin)",
            "delta_pose": (
                "passage_width - (body_width*abs(cos(yaw)) + "
                "body_length*abs(sin(yaw)) + 2*safety_margin)"
            ),
        },
        "margin_bins": MARGIN_BINS,
        "yaw_degrees": YAWS,
        "noise": NOISE,
        "lateral_offsets_m": LATERAL,
        "factor_design": args.factor_design,
        "episode_order": args.episode_order,
        "episode_order_sha256": _hash_payload([
            row["scenario_id"] for row in specs
        ]),
        "memory_policy": args.memory_policy,
        "memory_config": asdict(memory_cfg),
        "methods": methods,
        "selector_thresholds": asdict(selector_cfg),
        "scenario_count": len(specs),
        "episode_rows": len(episode_rows),
        "step_rows": (
            step_rows.count if isinstance(step_rows, _StreamingCSV) else len(step_rows)
        ),
        "dynamic_feasibility": asdict(DynamicFeasibilityConfig(
            morphology=morphology,
            enable_belief_propagation=not args.disable_belief_propagation,
            enable_belief_fusion=not args.disable_belief_fusion,
        )),
        "protected_legacy_result": "paper_dynamic_20260818_170356",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "run_command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_diff = subprocess.run(
            ["git", "diff", "--stat"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_diff = "unavailable", "unavailable\n"
    (output_dir / "git_commit.txt").write_text(
        git_commit + "\n", encoding="utf-8"
    )
    (output_dir / "git_diff_summary.txt").write_text(
        git_diff, encoding="utf-8"
    )
    runtime = {
        "started_at_utc": started_at_utc.isoformat(),
        "finished_at_utc": finished_at_utc.isoformat(),
        "wall_time_seconds": time.perf_counter() - started_perf,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--body-width", type=float, default=0.36)
    parser.add_argument("--body-length", type=float, default=0.60)
    parser.add_argument("--safety-margin", type=float, default=0.03)
    parser.add_argument("--kappa", type=float, default=1.645)
    parser.add_argument("--tau-commit", type=float, default=0.020)
    parser.add_argument("--tau-reject", type=float, default=0.020)
    parser.add_argument("--disable-belief-propagation", action="store_true")
    parser.add_argument("--disable-belief-fusion", action="store_true")
    parser.add_argument(
        "--methods", nargs="+", choices=EVALUATION_METHODS,
        default=list(MAIN_ABLATIONS),
    )
    parser.add_argument(
        "--factor-design",
        choices=("balanced", "publication", "cartesian"),
        default="balanced",
    )
    parser.add_argument(
        "--episode-order",
        choices=("seeded_shuffle", "interleaved_margin_sign", "legacy_bin_order"),
        default="seeded_shuffle",
    )
    parser.add_argument(
        "--memory-policy",
        choices=("empty_online_memory", "frozen_memory", "no_memory"),
        default="empty_online_memory",
    )
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--seed-shard",
        action="store_true",
        help="Allow one full-evaluation seed for parallel sharding; merge all three before analysis.",
    )
    args = parser.parse_args()
    if args.body_width <= 0.0 or args.body_length <= 0.0:
        parser.error("body dimensions must be positive")
    if args.safety_margin < 0.0:
        parser.error("--safety-margin must be non-negative")
    if args.phase == "full" and len(args.seeds) != 3:
        if not (args.seed_shard and len(args.seeds) == 1):
            parser.error("full evaluation requires three seeds, or one with --seed-shard")
    return args


if __name__ == "__main__":
    run(parse_args())
