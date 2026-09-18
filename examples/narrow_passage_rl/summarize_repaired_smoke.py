#!/usr/bin/env python3
"""Build an auditable report for a repaired strict-feasibility smoke run.

The script is intentionally descriptive: it never changes controller thresholds,
episode budgets, labels, or historical results.  Every output directory must be
new and empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from classify_feasibility_failures import classify


METHOD = "full_dynamic_uncertainty"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _number(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _pct(rows: list[dict[str, str]], key: str) -> float:
    return (
        100.0 * float(np.mean([_number(row, key, 0.0) for row in rows]))
        if rows
        else math.nan
    )


def _bool_pct(values: Iterable[bool]) -> float:
    values = list(values)
    return 100.0 * float(np.mean(values)) if values else math.nan


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _outcome_row(name: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "subset": name,
        "episodes": len(rows),
        "success_pct": _pct(rows, "success"),
        "collision_pct": _pct(rows, "collision"),
        "false_reject_pct": _pct(rows, "false_reject"),
        "correct_reject_pct": _pct(rows, "correct_reject"),
        "timeout_pct": _pct(rows, "timeout"),
        "mean_steps": float(np.mean([_number(row, "steps") for row in rows])),
        "mean_path_length_m": float(
            np.mean([_number(row, "path_length") for row in rows])
        ),
        "mean_min_body_margin_m": float(
            np.mean([_number(row, "min_body_margin") for row in rows])
        ),
    }


def _group_metrics(
    rows: list[dict[str, str]], factor: str, subset: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for value in sorted({row.get(factor, "") for row in rows}):
        selected = [row for row in rows if row.get(factor, "") == value]
        output.append({
            "subset": subset,
            "factor": factor,
            "value": value,
            "episodes": len(selected),
            "success_pct": _pct(selected, "success"),
            "collision_pct": _pct(selected, "collision"),
            "false_reject_pct": _pct(selected, "false_reject"),
            "timeout_pct": _pct(selected, "timeout"),
            "mean_steps": float(np.mean([_number(row, "steps") for row in selected])),
            "mean_path_length_m": float(
                np.mean([_number(row, "path_length") for row in selected])
            ),
        })
    return output


def _final_goal_distance(rows: list[dict[str, str]]) -> float:
    if not rows:
        return math.nan
    try:
        return float(json.loads(rows[-1]["observation"])[12])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return math.nan


def _failure_rows(
    episodes: list[dict[str, str]], steps: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in steps:
        grouped[row["episode_id"]].append(row)
    detailed: list[dict[str, Any]] = []
    for episode in episodes:
        if _number(episode, "success", 0.0) > 0.5:
            continue
        episode_steps = grouped.get(episode["episode_id"], [])
        category, reason = classify(episode, episode_steps)
        detailed.append({
            "episode_id": episode["episode_id"],
            "corridor_type": episode.get("corridor_type", ""),
            "margin_bin": episode.get("margin_bin", ""),
            "yaw_deg": episode.get("yaw_deg", ""),
            "noise_level": episode.get("noise_level", ""),
            "lateral_offset": episode.get("lateral_offset", ""),
            "passable": _truth(episode.get("passable")),
            "final_outcome": episode.get("final_outcome", ""),
            "failure_category": category,
            "classification_reason": reason,
            "steps": int(_number(episode, "steps", 0.0)),
            "final_goal_distance_m": _final_goal_distance(episode_steps),
            "min_body_margin_m": _number(episode, "min_body_margin"),
            "corner_turn_locked_steps": sum(
                _truth(row.get("corner_turn_locked")) for row in episode_steps
            ),
            "safety_stop_steps": sum(
                row.get("safe_action_projection_reason")
                in {"stop", "no_safe_candidate"}
                for row in episode_steps
            ),
            "recover_steps": sum(
                row.get("controller_mode") == "RECOVER" for row in episode_steps
            ),
            "uncertainty_triggered_steps": sum(
                _truth(row.get("uncertainty_triggered")) for row in episode_steps
            ),
        })
    counts = Counter(row["failure_category"] for row in detailed)
    summary = [{
        "failure_category": category,
        "episodes": count,
        "share_of_all_non_success_pct": 100.0 * count / max(len(detailed), 1),
        "share_of_feasible_failures_pct": 100.0 * count / max(
            sum(row["passable"] for row in detailed), 1
        ) if any(
            row["failure_category"] == category and row["passable"]
            for row in detailed
        ) else 0.0,
    } for category, count in counts.most_common()]
    return detailed, summary


def _historical_row(label: str, path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = [row for row in _read(path) if row.get("method") == METHOD]
    feasible = [row for row in rows if _truth(row.get("passable"))]
    if not feasible:
        return None
    return {
        "stage": label,
        "source": str(path.parent),
        "feasible_episodes": len(feasible),
        "feasible_success_pct": _pct(feasible, "success"),
        "feasible_false_reject_pct": _pct(feasible, "false_reject"),
        "feasible_collision_pct": _pct(feasible, "collision"),
        "feasible_timeout_pct": _pct(feasible, "timeout"),
    }


def run(result_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = result_dir / "episodes.csv"
    steps_path = result_dir / "steps.csv"
    episodes = [row for row in _read(episodes_path) if row.get("method") == METHOD]
    steps = [row for row in _read(steps_path) if row.get("method") == METHOD]
    gate = json.loads((result_dir / "gate_report.json").read_text(encoding="utf-8"))
    run_metadata = json.loads(
        (result_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    feasible = [row for row in episodes if _truth(row.get("passable"))]
    infeasible = [row for row in episodes if not _truth(row.get("passable"))]
    failures = [row for row in feasible if _number(row, "success", 0.0) < 0.5]
    success_ids = {
        row["episode_id"] for row in feasible if _number(row, "success", 0.0) > 0.5
    }
    failure_ids = {row["episode_id"] for row in failures}
    feasible_steps = [row for row in steps if row["episode_id"] in success_ids | failure_ids]
    failed_steps = [row for row in steps if row["episode_id"] in failure_ids]

    summary_rows = [
        _outcome_row("overall", episodes),
        _outcome_row("feasible", feasible),
        _outcome_row("infeasible", infeasible),
    ]
    _write_csv(summary_rows, output_dir / "summary.csv")
    grouped_rows = _group_metrics(feasible, "corridor_type", "feasible")
    for factor in ("noise_level", "yaw_deg", "margin_bin", "lateral_offset"):
        grouped_rows.extend(_group_metrics(feasible, factor, "feasible"))
    _write_csv(grouped_rows, output_dir / "grouped_metrics.csv")

    mode_counts = Counter(row.get("controller_mode", "") for row in feasible_steps)
    explore_steps = [
        row for row in feasible_steps if row.get("selected_mode") == "EXPLORE"
    ]
    mechanism_rows = [{
        "subset": "feasible_steps",
        "steps": len(feasible_steps),
        "uncertainty_triggered_pct": _bool_pct(
            _truth(row.get("uncertainty_triggered")) for row in feasible_steps
        ),
        "width_observable_pct": _bool_pct(
            _truth(row.get("width_observable")) for row in feasible_steps
        ),
        "invalid_depth_mean_pct": 100.0 * float(np.mean([
            _number(row, "invalid_ratio", 0.0) for row in feasible_steps
        ])),
        "safety_projection_pct": _bool_pct(
            _truth(row.get("swept_obb_safety_projected")) for row in feasible_steps
        ),
        "safety_stop_pct": _bool_pct(
            row.get("safe_action_projection_reason") in {"stop", "no_safe_candidate"}
            for row in feasible_steps
        ),
        "recover_pct": 100.0 * mode_counts["RECOVER"] / max(len(feasible_steps), 1),
        "corner_turn_locked_pct": _bool_pct(
            _truth(row.get("corner_turn_locked")) for row in feasible_steps
        ),
        "zero_linear_action_pct": _bool_pct(
            abs(_number(row, "linear_action", 0.0)) < 1e-9 for row in feasible_steps
        ),
        "explore_zero_linear_action_pct": _bool_pct(
            abs(_number(row, "linear_action", 0.0)) < 1e-9
            for row in explore_steps
        ),
    }, {
        "subset": "feasible_failure_steps",
        "steps": len(failed_steps),
        "uncertainty_triggered_pct": _bool_pct(
            _truth(row.get("uncertainty_triggered")) for row in failed_steps
        ),
        "width_observable_pct": _bool_pct(
            _truth(row.get("width_observable")) for row in failed_steps
        ),
        "invalid_depth_mean_pct": 100.0 * float(np.mean([
            _number(row, "invalid_ratio", 0.0) for row in failed_steps
        ])),
        "safety_projection_pct": _bool_pct(
            _truth(row.get("swept_obb_safety_projected")) for row in failed_steps
        ),
        "safety_stop_pct": _bool_pct(
            row.get("safe_action_projection_reason") in {"stop", "no_safe_candidate"}
            for row in failed_steps
        ),
        "recover_pct": _bool_pct(
            row.get("controller_mode") == "RECOVER" for row in failed_steps
        ),
        "corner_turn_locked_pct": _bool_pct(
            _truth(row.get("corner_turn_locked")) for row in failed_steps
        ),
        "zero_linear_action_pct": _bool_pct(
            abs(_number(row, "linear_action", 0.0)) < 1e-9 for row in failed_steps
        ),
        "explore_zero_linear_action_pct": _bool_pct(
            abs(_number(row, "linear_action", 0.0)) < 1e-9
            for row in failed_steps if row.get("selected_mode") == "EXPLORE"
        ),
    }]
    _write_csv(mechanism_rows, output_dir / "mechanism_metrics.csv")

    detailed_failures, failure_summary = _failure_rows(episodes, steps)
    _write_csv(detailed_failures, output_dir / "failure_modes.csv")
    _write_csv(failure_summary, output_dir / "failure_mode_summary.csv")

    root = result_dir.parent
    stage_rows = [row for row in (
        _historical_row(
            "pre-repair formal",
            root / "structural_full_20260818_194833_final" / "episodes.csv",
        ),
        _historical_row(
            "stage-1 memory repair",
            root / "repaired_stage1_20260820_153313" / "episodes.csv",
        ),
        _historical_row("current stage-3 smoke", episodes_path),
    ) if row is not None]
    _write_csv(stage_rows, output_dir / "stage_comparison.csv")

    scene = {row["value"]: row for row in grouped_rows if row["factor"] == "corridor_type"}
    failure_counts = Counter(row["failure_category"] for row in detailed_failures)
    feasible_failure_counts = Counter(
        row["failure_category"] for row in detailed_failures if row["passable"]
    )
    cause_rows = [
        {
            "id": "R1", "status": "confirmed protocol denominator",
            "cause": "Mixed feasible/infeasible denominator",
            "evidence": f"Only {len(feasible)}/{len(episodes)} episodes are OBB-labelled passable; Overall Success ceiling is {100.0*len(feasible)/len(episodes):.2f}%.",
            "effect": "Overall Success is not comparable to feasible-subset Success.",
        },
        {
            "id": "R2", "status": "fixed in current smoke",
            "cause": "Cross-episode memory false reject",
            "evidence": f"Current feasible False Reject is {_pct(feasible, 'false_reject'):.2f}% and forbidden censored/control geometry writes are {sum(_truth(row.get('geometry_memory_write')) and row.get('episode_outcome') not in {'success','geometric_infeasible','collision_geometry'} for row in episodes)}.",
            "effect": "No longer the current Success bottleneck, but remains the dominant explanation of the archived formal run.",
        },
        {
            "id": "R3", "status": "confirmed current dominant controller cause",
            "cause": "L/S corner traversal remains unsolved",
            "evidence": f"Feasible l_shaped Success={float(scene['l_shaped']['success_pct']):.2f}% and s_shaped Success={float(scene['s_shaped']['success_pct']):.2f}%; {feasible_failure_counts['corner traversal / OBB projection stall']} feasible failures have a corner-lock/timeout signature.",
            "effect": "The local turn lock makes progress but cannot complete one/two 90-degree turns within the unchanged 200-step budget.",
        },
        {
            "id": "R4", "status": "confirmed current action-path cause",
            "cause": "Swept-OBB projection saturation",
            "evidence": f"{mechanism_rows[1]['safety_projection_pct']:.2f}% of feasible-failure steps are projected and {mechanism_rows[1]['safety_stop_pct']:.2f}% become STOP.",
            "effect": "Safe but inconsistent advance/turn candidates consume the budget; collision stays 0% while timeout remains high.",
        },
        {
            "id": "R5", "status": "confirmed current scene cause",
            "cause": "Asymmetric protrusion avoidance is incomplete",
            "evidence": f"Feasible asymmetric Success={float(scene['asymmetric']['success_pct']):.2f}%; {feasible_failure_counts['asymmetric clearance-control stall']} feasible failures are labelled clearance-control stalls.",
            "effect": "Swept OBB prevents the old 100% collision outcome but the controller often stops instead of finding a max-clearance lateral pose.",
        },
        {
            "id": "R6", "status": "confirmed current entry cause",
            "cause": "Some high-yaw/lateral starts spend too much budget before entry",
            "evidence": f"{feasible_failure_counts['pre-entry readiness / OBB projection stall']} feasible failures never reach ENTER/TRAVERSE.",
            "effect": "The entry waypoint repair helps narrow_entry, but sparse/noisy aperture rays can still leave a safety-stop tail.",
        },
        {
            "id": "R7", "status": "supported sensor/control cause",
            "cause": "Six-ray local geometry is sparse and far-ray origins can cross a wall",
            "evidence": f"Mean invalid depth is {mechanism_rows[1]['invalid_depth_mean_pct']:.2f}% on failed feasible steps; local steering still depends on six sector values and a synthetic far origin one metre ahead.",
            "effect": "Dropout and discontinuous corner observations produce direction jitter and large requested angular actions.",
        },
        {
            "id": "R8", "status": "confirmed implementation limitation",
            "cause": "Junction collision/feasibility is not a full polygon-union OBB proof",
            "evidence": "_collision_at_pose projects the centre to one nearest segment and checks two local parallel walls; ground_truth_passable uses minimum cross-section/blocker gaps and does not solve an OBB turning path.",
            "effect": "Straight-wall yaw tests pass, but L/S passable labels and corner collision checks are not yet a complete configuration-space reachability certificate.",
        },
        {
            "id": "R9", "status": "secondary, not dominant",
            "cause": "Dynamic uncertainty and injected depth noise",
            "evidence": f"Uncertainty is triggered on {mechanism_rows[1]['uncertainty_triggered_pct']:.2f}% of failed feasible steps, but current False Reject is 0% and clean curved scenes also time out.",
            "effect": "Noise increases control jitter/indecision; it cannot alone explain the scene-structured failures.",
        },
        {
            "id": "R10", "status": "confirmed exposure, not a permitted shortcut",
            "cause": "Unchanged 200-step/goal-pose termination exposes slow control",
            "evidence": f"{len(failures)}/{len(feasible)} feasible episodes terminate by timeout; mean failed path length is {float(np.mean([_number(row,'path_length') for row in failures])):.2f} m.",
            "effect": "Extending max_steps would hide slow/oscillatory control and was deliberately not used as a repair.",
        },
        {
            "id": "R11", "status": "evaluation-validity caveat",
            "cause": "Path-frame heading/lateral observations are derived from procedural centreline",
            "evidence": "HarderNarrowPassageEnv._obs calls _project on the ground-truth polyline for heading_error and lateral_offset.",
            "effect": "This makes the current controller an optimistic procedural diagnostic, not yet a sensor-only Habitat claim; it does not explain low Success but limits paper interpretation.",
        },
        {
            "id": "R12", "status": "confirmed current stage-2 gate failure",
            "cause": "Active Explore is frequently projected back to zero translation",
            "evidence": f"Among {len(explore_steps)} feasible selected_mode=EXPLORE steps, {mechanism_rows[0]['explore_zero_linear_action_pct']:.2f}% have final linear_action=0 (target <10%).",
            "effect": "The selector asks for information, but the safety/controller composition often collects it without forward progress and consumes the fixed budget.",
        },
        {
            "id": "R13", "status": "supported observability-gate weakness",
            "cause": "Passage width observability is almost saturated",
            "evidence": f"width_observable is true on {mechanism_rows[0]['width_observable_pct']:.2f}% of feasible steps despite open-space and corner discontinuities.",
            "effect": "The gate rarely suppresses structural CI decisions when six-ray geometry is weak, so downstream control receives overconfident/discontinuous aperture state.",
        },
        {
            "id": "R14", "status": "measurement limitation",
            "cause": "The 42-episode smoke is a gate, not a stable effect estimate",
            "evidence": f"Only {len(feasible)} feasible episodes (three per represented feasible scene) and one morphology/seed are present; noise, yaw, lateral offset, scene, and margin are jointly assigned.",
            "effect": "Noise/yaw subgroup percentages are descriptive and confounded; they cannot support paper-level causal comparisons or morphology generalization.",
        },
    ]
    _write_csv(cause_rows, output_dir / "cause_registry.csv")

    def pct(value: Any) -> str:
        return f"{float(value):.2f}%"

    lines = [
        "# DEGNav strict feasibility repaired smoke：结果与低成功率诊断",
        "",
        f"Source: `{result_dir}`. Method: `{METHOD}`. This report is descriptive; no threshold, OBB, label, collision rule, or episode budget was changed while producing it.",
        "",
        "## Outcome",
        "",
        "| Subset | N | Success | Collision | False Reject | Correct Reject | Timeout |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['subset']} | {row['episodes']} | {pct(row['success_pct'])} | "
            f"{pct(row['collision_pct'])} | {pct(row['false_reject_pct'])} | "
            f"{pct(row['correct_reject_pct'])} | {pct(row['timeout_pct'])} |"
        )
    lines.extend([
        "",
        f"The current feasible Success is {int(sum(_number(row,'success',0)>0.5 for row in feasible))}/{len(feasible)} ({_pct(feasible,'success'):.2f}%). Memory False Reject and collision are both 0%, but {int(sum(_number(row,'timeout',0)>0.5 for row in feasible))}/{len(feasible)} feasible episodes time out. The run therefore improves the failure semantics without meeting the >=60% engineering target.",
        "",
        "## Feasible scene breakdown",
        "",
        "| Scene | N | Success | Collision | Timeout | Mean steps | Path length |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in sorted(scene):
        row = scene[name]
        lines.append(
            f"| {name} | {row['episodes']} | {pct(row['success_pct'])} | "
            f"{pct(row['collision_pct'])} | {pct(row['timeout_pct'])} | "
            f"{float(row['mean_steps']):.1f} | {float(row['mean_path_length_m']):.2f} m |"
        )
    noise_rows = [
        row for row in grouped_rows if row["factor"] == "noise_level"
    ]
    morphology = run_metadata.get("morphology", {})
    lines.extend([
        "",
        "## Feasible noise breakdown (descriptive only)",
        "",
        "| Noise | N | Success | Collision | Timeout |",
        "|:---|---:|---:|---:|---:|",
    ])
    for row in noise_rows:
        lines.append(
            f"| {row['value']} | {row['episodes']} | {pct(row['success_pct'])} | "
            f"{pct(row['collision_pct'])} | {pct(row['timeout_pct'])} |"
        )
    lines.extend([
        "",
        "This one-seed balanced smoke jointly varies scene, margin, yaw, noise, and lateral offset. The noise rows are not controlled causal estimates.",
        "",
        "## Morphology scope",
        "",
        f"Only one morphology was run: width={float(morphology.get('width', math.nan)):.2f} m, length={float(morphology.get('length', math.nan)):.2f} m, safety_margin={float(morphology.get('safety_margin', math.nan)):.2f} m. Cross-morphology results are unavailable because the strict gate failed before phase 4.",
        "",
        "## Mutually exclusive failure modes",
        "",
        "| Category | Episodes | Share of all non-success |",
        "|:---|---:|---:|",
    ])
    for row in failure_summary:
        lines.append(
            f"| {row['failure_category']} | {row['episodes']} | "
            f"{float(row['share_of_all_non_success_pct']):.2f}% |"
        )
    lines.extend([
        "",
        "## All current causes and caveats",
        "",
        "| ID | Status | Cause | Evidence and effect |",
        "|:---|:---|:---|:---|",
    ])
    for row in cause_rows:
        lines.append(
            f"| {row['id']} | {row['status']} | {row['cause']} | "
            f"{row['evidence']} {row['effect']} |"
        )
    lines.extend([
        "",
        "## Gate decision",
        "",
        "- Passed: structured memory semantics, zero forbidden negative writes, seeded pairing, OBB straight-wall tests, False Reject=0%, Collision=0%, large-yaw Align→Commit, truly narrow Reject, uncertainty trigger >0, narrow-entry 3/3, narrow-exit 3/3.",
        f"- Failed: feasible Timeout is {_pct(feasible, 'timeout'):.2f}% (target <20%, priority <30%); feasible Success is {_pct(feasible, 'success'):.2f}% (target >=60%); L/S are both 0/3; Explore zero-linear ratio is {mechanism_rows[0]['explore_zero_linear_action_pct']:.2f}% (stage-2 target <10%).",
        f"- The versioned runtime gate records `full_evaluation_permitted={str(bool(gate.get('full_evaluation_permitted'))).lower()}` and actively blocks phase 4/full. Consequently uncertainty calibration, cross-morphology evaluation, and the three-seed formal evaluation were **not run**.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xiaotian/miniconda3/envs/navila/bin/python -m pytest -q \\",
        "  examples/narrow_passage_rl/tests/test_repaired_feasibility.py \\",
        "  examples/narrow_passage_rl/tests/test_feasibility_ablation.py \\",
        "  examples/narrow_passage_rl/tests/test_strict_obb_geometry.py \\",
        "  examples/narrow_passage_rl/tests/test_structural_evaluation_protocol.py",
        "",
        "python examples/narrow_passage_rl/eval_structural_feasibility.py \\",
        "  --phase smoke --seeds 0 --max-steps 200 \\",
        "  --methods full_dynamic_uncertainty \\",
        "  --output-dir examples/narrow_passage_rl/results/ablation_feasibility/repaired_stage3_smoke_NEW_TIMESTAMP",
        "```",
        "",
    ])
    report = output_dir / "repaired_success_diagnosis.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_result_dir": str(result_dir),
        "method": METHOD,
        "source_episodes_sha256_not_recomputed": True,
        "phase4_run": False,
        "three_seed_run": False,
        "gate_reason": "feasible Success/Timeout and L/S course targets failed",
    }
    (output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = (
        "python examples/narrow_passage_rl/summarize_repaired_smoke.py "
        f"--result-dir {result_dir} --output-dir {output_dir}\n"
    )
    (output_dir / "run_command.txt").write_text(command, encoding="utf-8")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--stat"], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        commit, diff = "unavailable", "unavailable\n"
    (output_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output_dir / "git_diff_summary.txt").write_text(diff, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.result_dir.parent / datetime.now().strftime(
        "repaired_diagnosis_%Y%m%d_%H%M%S"
    )
    report = run(args.result_dir, output_dir)
    print(f"[write] {report}")


if __name__ == "__main__":
    main()
