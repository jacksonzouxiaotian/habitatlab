#!/usr/bin/env python3
"""Paired evaluation for the five strict DEGNav feasibility ablations.

This runner intentionally lives beside, rather than replacing, the legacy
``eval_harder_benchmark.py`` entry point.  For each generated scenario it runs
the selected ablations consecutively with the same explicit environment seed,
initial yaw distribution, lateral perturbation, action budget, and controller.
Outputs are refused when a target file already exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cross_episode_memory import CrossEpisodeMemory
from eval_harder_benchmark import eval_method
from evaluation.logging_schema import fieldnames_for_rows
from narrow_passage.models.dynamic_feasibility import DynamicFeasibilityConfig
from narrow_passage.models.feasibility_ablation import (
    ABLATIONS,
    MAIN_ABLATIONS,
    PAPER_METHOD_NAMES,
    SelectorThresholds,
    capability_record,
    canonical_ablation,
)


DEFAULT_CORRIDORS = (
    "straight",
    "l_shaped",
    "s_shaped",
    "narrow_entry",
    "narrow_exit",
    "asymmetric",
    "false_feasible",
)
DEFAULT_RESULTS_PARENT = SCRIPT_DIR / "results" / "ablation_feasibility"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return DEFAULT_RESULTS_PARENT / stamp


def _assert_new_path(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing ablation output: {path}. "
            "Choose a new --output-dir."
        )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    _assert_new_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames_for_rows(rows),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


class _StreamingCSV:
    """Append-only CSV sink used to keep multi-million-step logs bounded."""

    def __init__(self, path: Path):
        _assert_new_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer: csv.DictWriter | None = None
        self.fields: list[str] = []
        self.count = 0

    def append(self, row: dict[str, Any]) -> None:
        if self.writer is None:
            self.fields = list(row.keys())
            self.writer = csv.DictWriter(
                self.handle,
                fieldnames=self.fields,
                lineterminator="\n",
            )
            self.writer.writeheader()
        extras = set(row) - set(self.fields)
        if extras:
            raise RuntimeError(f"Step-log schema changed during run: {sorted(extras)}")
        self.writer.writerow(row)
        self.count += 1
        if self.count % 1000 == 0:
            self.handle.flush()

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


def _episode_seed(run_seed: int, scene_index: int, episode_index: int) -> int:
    """Stable, method-independent seed for one paired procedural scenario."""

    if episode_index >= 1_000_000:
        raise ValueError("episode_index must be below 1,000,000")
    payload = f"{run_seed}:{scene_index}:{episode_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _episode_id(run_seed: int, corridor: str, episode_index: int) -> str:
    return f"procedural_v2_seed{run_seed}_{corridor}_ep{episode_index:06d}"


def _coerce_ablation_rows(
    rows: Iterable[dict[str, Any]],
    ablation: str,
) -> list[dict[str, Any]]:
    out = []
    for original in rows:
        row = dict(original)
        row["method"] = ablation
        row["ablation"] = ablation
        row["canonical_method"] = canonical_ablation(ablation)
        row["paper_method"] = PAPER_METHOD_NAMES[canonical_ablation(ablation)]
        row["behaviorally_equivalent_alias"] = (
            "mean_only" if ablation in {"point_estimate", "no_uncertainty"} else ""
        )
        row["scene_type"] = row.get("corridor_type", "unknown")
        row["available_width"] = row.get(
            "available_width", row.get("passage_width", float("nan"))
        )
        row["mu_D"] = row.get("mu_D", row.get("d_hat", float("nan")))
        row["var_D"] = row.get("var_D", float("nan"))
        row["mu_W"] = row.get("mu_W", row.get("w_req_cons", float("nan")))
        row["var_W"] = row.get("var_W", float("nan"))
        row["mu_delta"] = row.get(
            "mu_delta", row.get("delta_mean", float("nan"))
        )
        row["var_delta"] = row.get(
            "var_delta", row.get("delta_var", float("nan"))
        )
        row["initial_yaw"] = row.get(
            "initial_yaw", row.get("entry_heading_error", float("nan"))
        )
        row["initial_lateral_offset"] = row.get(
            "initial_lateral_offset", row.get("lateral_offset", float("nan"))
        )
        row["selected_mode"] = row.get(
            "selected_mode", row.get("final_mode", "unavailable")
        )
        out.append(row)
    return out


def validate_pairing(rows: list[dict[str, Any]], methods: list[str]) -> None:
    """Fail fast unless every episode is a genuine controlled pair."""

    expected = set(methods)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["episode_id"]), []).append(row)

    for episode_id, episode_rows in grouped.items():
        got = {str(row["method"]) for row in episode_rows}
        if got != expected:
            raise RuntimeError(
                f"Pairing failure for {episode_id}: expected={expected}, got={got}"
            )
        if len(episode_rows) != len(methods):
            raise RuntimeError(
                f"Duplicate paired row for {episode_id}: n={len(episode_rows)}"
            )
        for key in (
            "seed",
            "episode_seed",
            "corridor_type",
            "available_width",
            "initial_yaw",
            "initial_lateral_offset",
        ):
            values = [row.get(key) for row in episode_rows]
            if key in {"available_width", "initial_yaw", "initial_lateral_offset"}:
                numeric = np.asarray([float(value) for value in values], dtype=float)
                if not np.all(np.isfinite(numeric)) or float(np.ptp(numeric)) > 1e-7:
                    raise RuntimeError(
                        f"Pairing failure for {episode_id}: {key}={values}"
                    )
            elif len(set(values)) != 1:
                raise RuntimeError(
                    f"Pairing failure for {episode_id}: {key}={values}"
                )


def _write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    methods: list[str],
    rows: list[dict[str, Any]],
    step_row_count: int,
) -> None:
    _assert_new_path(path)
    selector = SelectorThresholds()
    belief = DynamicFeasibilityConfig()
    episode_ids = sorted({str(row["episode_id"]) for row in rows})
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "paired_evaluation": True,
        "pairing_validation": "passed",
        "episode_id_sha256": hashlib.sha256(
            "\n".join(episode_ids).encode("utf-8")
        ).hexdigest(),
        "episodes_per_scene_per_seed_per_method": args.episodes_per_scene,
        "paired_episode_count": len(episode_ids),
        "row_count": len(rows),
        "step_row_count": int(step_row_count),
        "step_log_write_mode": "incremental_atomic_partial_then_rename",
        "seeds": args.seeds,
        "methods": methods,
        "corridor_types": args.corridor_types,
        "shared_environment": {
            "max_steps": args.max_steps,
            "width_range_m": args.width_range,
            "yaw_noise_deg": args.yaw_noise_deg,
            "entry_jitter_sigma_m": args.entry_jitter,
            "collision_geometry": "oriented_rectangle",
            "morphology": belief.morphology.record(),
            "dt_s": 0.25,
        },
        "shared_selector_thresholds": {
            "kappa": selector.kappa,
            "tau_commit_m": selector.tau_commit,
            "tau_reject_m": selector.tau_reject,
            "commit_rule": (
                "LCB_struct > tau_commit and LCB_pose > tau_commit"
            ),
            "reject_rule": "UCB_struct < -tau_reject",
            "explore_rule": (
                "structural interval overlap or pose-conditioned not ready"
            ),
        },
        "belief_model": {
            "mu_D": "EWMA(narrow_passage_features[8])",
            "uncertainty_terms": [
                "depth sector/ray dispersion",
                "valid depth and dropout ratio",
                "boundary fitting residual",
                "temporal width variation",
                "yaw and lateral pose uncertainty",
            ],
            "variance_is_constant": False,
            "fixed_uncertainty_sigma_m": belief.fixed_sigma_delta,
            "yaw_prior_formula_in_code": (
                "A*abs(cos(yaw)) + B*abs(sin(yaw)) + 2*safety_margin"
            ),
            "body_width_m": belief.body_width,
            "body_length_m": belief.body_length,
            "safety_margin_each_side_m": belief.safety_margin,
        },
        "capabilities": {
            method: capability_record(method) for method in methods
        },
        "controlled_components": [
            "scenario and episode IDs",
            "environment seed and initial pose",
            "yaw and lateral perturbation",
            "action budget and outcome criteria",
            "geometry-indexed CrossEpisodeMemory implementation",
            "alignment, recovery, turn commitment, and low-level controller",
        ],
        "audit_notes": [
            (
                "Collision, clearance, feasibility label, and strict controller "
                "share one immutable 0.36 x 0.60 m OBB morphology object."
            ),
            (
                "CrossEpisodeMemory owns a separate D_min calibrator, but its "
                "posterior is not wired into BeliefState p_t(W_req). No online "
                "required-width posterior is claimed by these runs."
            ),
            (
                "No explicit generic mode hysteresis implementation exists in "
                "the current controller; all five variants share the same turn-"
                "commitment state machine only."
            ),
        ],
        "legacy_result_directories_unchanged": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[write] {path}")


def run(args: argparse.Namespace, step_trace_out: Any = None) -> list[dict[str, Any]]:
    methods = list(MAIN_ABLATIONS) if args.all_ablations else [args.ablation]
    memories = {
        (seed, method): CrossEpisodeMemory()
        for seed in args.seeds
        for method in methods
    }
    rows: list[dict[str, Any]] = []
    total_pairs = len(args.seeds) * len(args.corridor_types) * args.episodes_per_scene
    completed_pairs = 0

    for run_seed in args.seeds:
        for scene_index, corridor in enumerate(args.corridor_types):
            for episode_index in range(args.episodes_per_scene):
                ep_seed = _episode_seed(run_seed, scene_index, episode_index)
                ep_id = _episode_id(run_seed, corridor, episode_index)
                # Paired ordering: one scenario, then every requested method.
                for method in methods:
                    method_rows = eval_method(
                        method,
                        [corridor],
                        1,
                        seed=run_seed,
                        entry_jitter_sigma=args.entry_jitter,
                        log_belief_diagnostics=True,
                        width_range=tuple(args.width_range),
                        max_steps=args.max_steps,
                        strict_ablation=method,
                        cross_mem_override=memories[(run_seed, method)],
                        episode_seeds=[ep_seed],
                        episode_ids=[ep_id],
                        progress=False,
                        yaw_noise=math.radians(args.yaw_noise_deg),
                        step_trace_out=step_trace_out,
                    )
                    rows.extend(_coerce_ablation_rows(method_rows, method))
                completed_pairs += 1
                if completed_pairs <= 3 or completed_pairs % 100 == 0:
                    print(
                        f"[paired] {completed_pairs}/{total_pairs} "
                        f"seed={run_seed} scene={corridor} episode={episode_index}",
                        flush=True,
                    )

    validate_pairing(rows, methods)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict paired DEGNav feasibility ablations."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ablation", choices=ABLATIONS, default="full_dynamic_uncertainty"
    )
    group.add_argument(
        "--all-ablations",
        action="store_true",
        help="Run all five methods consecutively for every episode.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes-per-scene", type=int, default=100)
    parser.add_argument(
        "--corridor-types",
        nargs="+",
        choices=DEFAULT_CORRIDORS,
        default=list(DEFAULT_CORRIDORS),
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--width-range", nargs=2, type=float, default=[0.30, 0.70])
    parser.add_argument("--yaw-noise-deg", type=float, default=90.0)
    parser.add_argument("--entry-jitter", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes_per_scene <= 0:
        parser.error("--episodes-per-scene must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.width_range[0] <= 0 or args.width_range[0] >= args.width_range[1]:
        parser.error("--width-range must satisfy 0 < MIN < MAX")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or _default_output_dir()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse non-empty output directory: {output_dir}"
        )

    partial_steps = output_dir / "steps.partial.csv"
    final_steps = output_dir / "steps.csv"
    _assert_new_path(final_steps)
    step_sink = _StreamingCSV(partial_steps)
    completed = False
    try:
        rows = run(args, step_trace_out=step_sink)
        completed = True
    finally:
        step_sink.close()
    if not completed:
        raise RuntimeError(f"Run incomplete; partial step log retained at {partial_steps}")
    partial_steps.rename(final_steps)
    print(f"[write] {final_steps} ({step_sink.count} rows)")
    methods = list(MAIN_ABLATIONS) if args.all_ablations else [args.ablation]
    _write_csv(rows, output_dir / "episodes.csv")
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        _write_csv(method_rows, output_dir / method / "episodes.csv")
    _write_metadata(
        output_dir / "run_metadata.json",
        args=args,
        methods=methods,
        rows=rows,
        step_row_count=step_sink.count,
    )
    print(
        f"[done] paired episodes={len({row['episode_id'] for row in rows})} "
        f"rows={len(rows)} methods={methods}"
    )


if __name__ == "__main__":
    main()
