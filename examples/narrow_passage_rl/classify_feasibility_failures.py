#!/usr/bin/env python3
"""Assign one auditable, mutually exclusive label to each non-success episode."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


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


def _high_depth_corruption(row: dict[str, str]) -> bool:
    return (
        row.get("noise_level") == "severe"
        and (
            _number(row, "dropout_ratio", 0.0) >= 0.50
            or _number(row, "boundary_fitting_residual", 0.0) >= 0.05
            or _number(row, "ray_dispersion", 0.0) >= 0.25
        )
    )


def classify(
    episode: dict[str, str], steps: list[dict[str, str]]
) -> tuple[str, str]:
    """Return one label and its directly auditable reason, in priority order."""

    reject_steps = [row for row in steps if row.get("controller_mode") == "REJECT"]
    if _truth(episode.get("correct_reject")):
        return (
            "correct infeasible reject (policy-correct non-success)",
            "ground-truth infeasible and controller rejected",
        )
    if _truth(episode.get("collision")):
        return "genuine OBB collision", "episode collision flag is true"
    if (
        _number(episode, "min_body_margin", 0.0) < 0.0
        and not _truth(episode.get("collision"))
    ):
        return (
            "body-margin conversion offset",
            "negative clearance diagnostic without an OBB collision",
        )
    if _truth(episode.get("false_reject")):
        if any(
            row.get("controller_mode") == "REJECT"
            and row.get("feasibility_mode") != "REJECT"
            for row in reject_steps
        ):
            return (
                "cross-episode memory-driven false reject",
                "controller rejected while the feasibility selector did not",
            )
        if any(_truth(row.get("explicit_blocker")) for row in reject_steps):
            return (
                "explicit-blocker false positive",
                "explicit blocker override on a passable episode",
            )
        if any(_high_depth_corruption(row) for row in reject_steps):
            return (
                "depth-noise-driven boundary misjudgment",
                "severe corruption coincided with structural reject evidence",
            )
        return (
            "conservative-threshold false reject",
            "selector structural reject evidence on a passable episode",
        )
    if _truth(episode.get("timeout")):
        tail = steps[max(0, len(steps) * 3 // 4):]
        passable = _number(episode, "passable_label", 0.0) > 0.5
        ever_entered = any(
            row.get("passage_control_state") in {"ENTER", "TRAVERSE", "EXIT"}
            or _number(row, "body_margin", 5.0) <= 0.30
            for row in steps
        )
        corner_locked = any(_truth(row.get("corner_turn_locked")) for row in steps)
        tail_stop_fraction = sum(
            row.get("safe_action_projection_reason") in {"stop", "no_safe_candidate"}
            for row in tail
        ) / max(len(tail), 1)
        tail_recover_fraction = sum(
            row.get("controller_mode") == "RECOVER" for row in tail
        ) / max(len(tail), 1)
        try:
            final_goal_distance = float(
                json.loads(steps[-1].get("observation", "[]"))[12]
            )
        except (IndexError, TypeError, ValueError, json.JSONDecodeError):
            final_goal_distance = math.nan
        scene = episode.get("corridor_type", "")
        if not passable:
            return (
                "missed infeasible reject / interval indecision",
                "ground-truth infeasible episode exhausted the budget without Reject",
            )
        if not ever_entered:
            return (
                "pre-entry readiness / OBB projection stall",
                "passable episode never reached ENTER/TRAVERSE; entry waypoint or safety projection stalled",
            )
        if scene in {"l_shaped", "s_shaped"} and corner_locked:
            return (
                "corner traversal / OBB projection stall",
                f"local corner lock fired but traversal timed out; tail safety-stop fraction={tail_stop_fraction:.2f}",
            )
        if scene == "asymmetric":
            return (
                "asymmetric clearance-control stall",
                f"passable protrusion scene timed out; tail safety-stop fraction={tail_stop_fraction:.2f}",
            )
        if tail_stop_fraction >= 0.50:
            return (
                "swept-OBB projection saturation",
                f"{tail_stop_fraction:.0%} of final-quarter actions were projected to STOP",
            )
        if math.isfinite(final_goal_distance) and final_goal_distance <= 0.50:
            return (
                "terminal goal-pose tolerance miss",
                f"budget expired near goal at {final_goal_distance:.3f} m without satisfying distance/yaw success",
            )
        if any(_high_depth_corruption(row) for row in tail):
            return (
                "depth-noise-driven boundary misjudgment",
                "severe corrupted belief persisted into timeout tail",
            )
        alignment_fraction = sum(
            row.get("controller_mode") == "ALIGN" for row in tail
        ) / max(len(tail), 1)
        max_abs_yaw = max(
            (abs(_number(row, "yaw_error", 0.0)) for row in tail), default=0.0
        )
        if alignment_fraction >= 0.25 or max_abs_yaw >= math.radians(45.0):
            return (
                "yaw-readiness/alignment stall",
                "large yaw or repeated Align persisted until timeout",
            )
        if tail_recover_fraction >= 0.20:
            return (
                "bounded-Recover budget consumption",
                f"Recover occupied {tail_recover_fraction:.0%} of the timeout tail",
            )
        return "controller timeout (other)", "timeout without another audited signature"
    return "controller failure (other)", str(episode.get("final_outcome", "unknown"))


def run(episodes_csv: Path, steps_csv: Path, output_dir: Path) -> list[dict[str, Any]]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = [
        row for row in _read(episodes_csv)
        if row.get("method") == METHOD and _number(row, "success", 0.0) < 0.5
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read(steps_csv):
        if row.get("method") == METHOD:
            grouped[str(row["episode_id"])].append(row)

    classified: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        episode_steps = grouped.get(episode_id, [])
        label, reason = classify(episode, episode_steps)
        try:
            final_goal_distance = float(
                json.loads(episode_steps[-1].get("observation", "[]"))[12]
            )
        except (IndexError, TypeError, ValueError, json.JSONDecodeError):
            final_goal_distance = math.nan
        classified.append({
            "episode_id": episode_id,
            "method": METHOD,
            "failure_category": label,
            "classification_reason": reason,
            "success": _number(episode, "success", 0.0),
            "final_outcome": episode.get("final_outcome", ""),
            "corridor_type": episode.get("corridor_type", ""),
            "passable_label": _number(episode, "passable_label", 0.0),
            "structural_margin_gt": _number(episode, "structural_margin_gt"),
            "yaw_deg": _number(episode, "yaw_deg"),
            "noise_level": episode.get("noise_level", ""),
            "min_body_margin": _number(episode, "min_body_margin"),
            "final_goal_distance": final_goal_distance,
            "corner_turn_steps": sum(
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

    counts = Counter(str(row["failure_category"]) for row in classified)
    summary: list[dict[str, Any]] = []
    for category, count in counts.most_common():
        rows = [row for row in classified if row["failure_category"] == category]
        summary.append({
            "failure_category": category,
            "episodes": count,
            "share_of_non_success_pct": 100.0 * count / max(len(classified), 1),
            "success_rate_pct": 100.0 * sum(float(row["success"]) for row in rows) / count,
        })
    # Keep requested diagnostic hypotheses visible even when the audit finds no
    # matching episode; zero is an important negative result, not a missing row.
    for category in (
        "body-margin conversion offset",
        "conservative-threshold false reject",
        "explicit-blocker false positive",
    ):
        if category not in counts:
            summary.append({
                "failure_category": category,
                "episodes": 0,
                "share_of_non_success_pct": 0.0,
                "success_rate_pct": math.nan,
            })

    def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(classified, output_dir / "classified_failures.csv")
    write_csv(summary, output_dir / "failure_mode_table.csv")
    dominant_all = summary[0]
    error_rows = [
        row for row in summary
        if not str(row["failure_category"]).startswith("correct infeasible reject")
    ]
    dominant_error = max(error_rows, key=lambda row: int(row["episodes"]))
    lines = [
        "# Strict feasibility failure-mode analysis",
        "",
        f"Input method: `{METHOD}`. The table conditions on all {len(classified)} episodes with "
        "`success=False`; therefore the within-category Success column is 0% by construction.",
        "A correct rejection is retained because the requested conditioning uses navigation",
        "Success, but it is marked policy-correct rather than treated as an error.",
        "",
        "| Mutually exclusive category | Episodes | Share of non-success | Success rate |",
        "|:---|---:|---:|---:|",
    ]
    for row in summary:
        success_text = (
            "n/a"
            if not math.isfinite(float(row["success_rate_pct"]))
            else f"{float(row['success_rate_pct']):.2f}%"
        )
        lines.append(
            f"| {row['failure_category']} | {row['episodes']} | "
            f"{float(row['share_of_non_success_pct']):.2f}% | "
            f"{success_text} |"
        )
    lines.extend([
        "",
        "Classification priority is: correct infeasible reject → genuine OBB collision →",
        "clearance/collision mismatch → false-reject subtypes → timeout subtypes → other.",
        "A memory false reject is identified directly by `controller_mode=REJECT` while",
        "`feasibility_mode!=REJECT`. A yaw *estimation error* cannot be measured because the",
        "log has no separate ground-truth yaw; the table uses the narrower, auditable",
        "yaw-readiness/alignment-stall label instead.",
        "",
        f"总体最多的是 {dominant_all['failure_category']}（"
        f"{float(dominant_all['share_of_non_success_pct']):.2f}%）；排除策略正确的不可行通道拒绝后，"
        f"主失效是 {dominant_error['failure_category']}（"
        f"{float(dominant_error['share_of_non_success_pct']):.2f}% of all non-success）。",
        "",
    ])
    (output_dir / "failure_mode_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-csv", type=Path, required=True)
    parser.add_argument("--steps-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        args.episodes_csv.parent
        / datetime.now().strftime("failure_modes_%Y%m%d_%H%M%S")
    )
    for row in run(args.episodes_csv, args.steps_csv, output_dir):
        print(row)
    print(f"[write] {output_dir / 'failure_mode_analysis.md'}")


if __name__ == "__main__":
    main()
