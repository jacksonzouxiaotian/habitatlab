#!/usr/bin/env python3
"""Evaluate the system-level VLN safety adapter on saved NaVILA outputs.

The comparison is paired: baseline and adapted rows use the exact same model
response.  Improvements therefore measure deterministic interface behavior,
not improved NaVILA weights or formal navigation performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Optional

from narrow_passage.models.policy import DecisionMode
from narrow_passage.models.vln_safety_adapter import (
    VLNAdapterObservation,
    VLNSafetyAdapter,
    VLNSafetyAdapterConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "examples/narrow_passage_rl/results/vln_batch_diagnostic"
            "/predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "examples/narrow_passage_rl/results/vln_safety_adapter_smoke"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if row.get(key) is None
                        or (
                            isinstance(row.get(key), float)
                            and not math.isfinite(row[key])
                        )
                        else row.get(key)
                    )
                    for key in fields
                }
            )


def synthetic_geometry(row: dict[str, str]) -> dict[str, Optional[float]]:
    """Provide measured-fixture equivalents, not hidden outcome labels."""

    if row["source"] != "controlled_synthetic":
        return {
            "p_feas": None,
            "risk": None,
            "min_clearance_m": None,
            "body_margin_m": None,
        }
    if row["subgroup"] == "blocked":
        return {
            "p_feas": 0.05,
            "risk": 0.95,
            "min_clearance_m": 0.02,
            "body_margin_m": -0.02,
        }
    return {
        "p_feas": 0.95,
        "risk": 0.05,
        "min_clearance_m": 0.35,
        "body_margin_m": 0.20,
    }


def adapt_rows(
    raw_rows: list[dict[str, str]],
    adapter: VLNSafetyAdapter,
) -> list[dict[str, Any]]:
    adapted: list[dict[str, Any]] = []
    for row in raw_rows:
        geometry = synthetic_geometry(row)
        trusted = row["label_scope"] in {
            "explicit_local_command",
            "explicit_stop_override_only",
        }
        model_value = optional_float(row["action_value"])
        decision = adapter.adapt(
            row["parsed_action"],
            model_value,
            VLNAdapterObservation(
                instruction=row["instruction"],
                trusted_control_directive=trusted,
                **geometry,
            ),
        )
        expected = row["expected_action"]
        adapted.append(
            {
                "case_id": row["case_id"],
                "suite": row["suite"],
                "subgroup": row["subgroup"],
                "source": row["source"],
                "dataset_episode_id": row["dataset_episode_id"],
                "scene_id": row["scene_id"],
                "pair_id": row["pair_id"],
                "visual_id": row["visual_id"],
                "perturbation": row["perturbation"],
                "label_scope": row["label_scope"],
                "instruction": row["instruction"],
                "expected_action": expected,
                "baseline_action": row["parsed_action"],
                "baseline_value": model_value,
                "adapted_action": decision.adapted_action,
                "adapted_value": decision.adapted_value,
                "baseline_exact_match": (
                    row["parsed_action"] == expected if expected else None
                ),
                "adapted_exact_match": (
                    decision.adapted_action == expected if expected else None
                ),
                "mode": decision.mode.value,
                "intervention": decision.intervention,
                "reason": decision.reason,
                "overridden": decision.overridden,
                "terminal_stop": decision.terminal_stop,
                "effective_risk": decision.effective_risk,
                "p_feas": geometry["p_feas"],
                "risk": geometry["risk"],
                "min_clearance_m": geometry["min_clearance_m"],
                "body_margin_m": geometry["body_margin_m"],
            }
        )
    return adapted


def rate(rows: Iterable[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> tuple[int, float]:
    items = list(rows)
    return len(items), mean(predicate(row) for row in items) if items else math.nan


def metric_row(
    metric: str,
    direction: str,
    rows: list[dict[str, Any]],
    baseline_predicate: Callable[[dict[str, Any]], bool],
    adapted_predicate: Callable[[dict[str, Any]], bool],
    interpretation: str,
) -> dict[str, Any]:
    count, baseline = rate(rows, baseline_predicate)
    _, adapted = rate(rows, adapted_predicate)
    return {
        "metric": metric,
        "direction": direction,
        "cases": count,
        "baseline": baseline,
        "adapted": adapted,
        "delta": adapted - baseline,
        "interpretation": interpretation,
    }


def visual_consistency(
    rows: list[dict[str, Any]],
    action_key: str,
) -> tuple[int, float]:
    visual = [row for row in rows if row["suite"] == "visual_robustness"]
    clean = {
        row["pair_id"]: row
        for row in visual
        if row["perturbation"] == "clean"
    }
    perturbed = [row for row in visual if row["perturbation"] != "clean"]
    return rate(
        perturbed,
        lambda row: row[action_key] == clean[row["pair_id"]][action_key],
    )


def build_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    command = [row for row in rows if row["suite"] == "command_grounding"]
    command_nonblocked = [row for row in command if row["subgroup"] != "blocked"]
    local_stops = [
        row for row in command if row["expected_action"] == "stop"
    ]
    r2r_overrides = [
        row
        for row in rows
        if row["suite"] == "r2r_language_probe"
        and row["subgroup"] == "stop_override"
    ]
    all_stops = [*local_stops, *r2r_overrides]
    r2r_original = [
        row
        for row in rows
        if row["suite"] == "r2r_language_probe"
        and row["subgroup"] == "original_instruction"
    ]
    blocked_forward = [
        row
        for row in command
        if row["subgroup"] == "blocked"
        and row["expected_action"] == "move_forward"
    ]
    comparison = [
        metric_row(
            "Controlled command exact match",
            "higher",
            command,
            lambda row: bool(row["baseline_exact_match"]),
            lambda row: bool(row["adapted_exact_match"]),
            "Includes safety-command conflicts in the blocked fixture.",
        ),
        metric_row(
            "Non-blocked command exact match",
            "higher",
            command_nonblocked,
            lambda row: bool(row["baseline_exact_match"]),
            lambda row: bool(row["adapted_exact_match"]),
            "Trusted immediate commands with no measured geometry conflict.",
        ),
        metric_row(
            "Explicit stop compliance",
            "higher",
            all_stops,
            lambda row: row["baseline_action"] == "stop",
            lambda row: row["adapted_action"] == "stop",
            "Local stop commands plus paired R2R stop overrides.",
        ),
        metric_row(
            "R2R stop-override compliance",
            "higher",
            r2r_overrides,
            lambda row: row["baseline_action"] == "stop",
            lambda row: row["adapted_action"] == "stop",
            "A trusted runtime override, not route-instruction accuracy.",
        ),
        metric_row(
            "Blocked-fixture forward output",
            "lower",
            blocked_forward,
            lambda row: row["baseline_action"] == "move_forward",
            lambda row: row["adapted_action"] == "move_forward",
            "Uses measured synthetic clearance inputs; only four controlled cases.",
        ),
        metric_row(
            "Original R2R proposal preservation",
            "higher",
            r2r_original,
            lambda row: True,
            lambda row: row["baseline_action"] == row["adapted_action"],
            "Untrusted route text is not parsed as an immediate command.",
        ),
    ]
    visual_count, baseline_visual = visual_consistency(rows, "baseline_action")
    _, adapted_visual = visual_consistency(rows, "adapted_action")
    comparison.append(
        {
            "metric": "Visual perturbation action consistency",
            "direction": "higher",
            "cases": visual_count,
            "baseline": baseline_visual,
            "adapted": adapted_visual,
            "delta": adapted_visual - baseline_visual,
            "interpretation": (
                "The adapter does not claim to repair visual representation "
                "robustness."
            ),
        }
    )
    return comparison


def per_action_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    command = [row for row in rows if row["suite"] == "command_grounding"]
    result = []
    for expected in ("move_forward", "turn_left", "turn_right", "stop"):
        subset = [row for row in command if row["expected_action"] == expected]
        count, baseline = rate(
            subset, lambda row: bool(row["baseline_exact_match"])
        )
        _, adapted = rate(subset, lambda row: bool(row["adapted_exact_match"]))
        result.append(
            {
                "expected_action": expected,
                "cases": count,
                "baseline_exact_match": baseline,
                "adapted_exact_match": adapted,
                "delta": adapted - baseline,
            }
        )
    return result


def mode_path_probes(adapter: VLNSafetyAdapter) -> list[dict[str, Any]]:
    probes = [
        (
            "safe_commit",
            VLNAdapterObservation(
                "Continue.", p_feas=0.95, risk=0.05, min_clearance_m=0.30
            ),
            DecisionMode.COMMIT,
        ),
        (
            "uncertain_explore",
            VLNAdapterObservation("Continue.", p_feas=0.55, risk=0.45),
            DecisionMode.EXPLORE,
        ),
        (
            "stuck_recover",
            VLNAdapterObservation("Continue.", stuck_score=0.90),
            DecisionMode.RECOVER,
        ),
        (
            "collision_recover",
            VLNAdapterObservation("Continue.", collision_flag=True),
            DecisionMode.RECOVER,
        ),
        (
            "memory_reject",
            VLNAdapterObservation("Continue.", memory_risk=0.90),
            DecisionMode.REJECT,
        ),
        (
            "clearance_reject",
            VLNAdapterObservation(
                "Continue.",
                p_feas=0.05,
                risk=0.95,
                min_clearance_m=0.02,
                body_margin_m=-0.01,
            ),
            DecisionMode.REJECT,
        ),
    ]
    rows = []
    for probe_id, observation, expected_mode in probes:
        decision = adapter.adapt("move_forward", 75.0, observation)
        rows.append(
            {
                "probe_id": probe_id,
                "expected_mode": expected_mode.value,
                "actual_mode": decision.mode.value,
                "adapted_action": decision.adapted_action,
                "intervention": decision.intervention,
                "reason": decision.reason,
                "passed": decision.mode == expected_mode,
            }
        )
    return rows


def percent(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def markdown_report(
    metadata: dict[str, Any],
    comparison: list[dict[str, Any]],
    per_action: list[dict[str, Any]],
    mode_probes: list[dict[str, Any]],
) -> str:
    lines = [
        "# NaVILA + DEGNAV Safety Adapter Smoke Test",
        "",
        "## Scope",
        "",
        (
            "This is a paired system-interface smoke test over saved NaVILA "
            "outputs. The adapter does not change model weights. Reported gains "
            "come from trusted runtime directives and measured geometry gates, "
            "not learned VLN improvement or simulator navigation."
        ),
        "",
        "## Paired Comparison",
        "",
        "| Metric | Direction | Cases | NaVILA direct | + safety adapter | Delta |",
        "| :--- | :---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(
            f"| {row['metric']} | {row['direction']} | {row['cases']} | "
            f"{percent(row['baseline'])} | {percent(row['adapted'])} | "
            f"{100.0 * row['delta']:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## Controlled Command Breakdown",
            "",
            "| Expected action | Cases | NaVILA direct | + safety adapter |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for row in per_action:
        lines.append(
            f"| {row['expected_action'].replace('_', ' ')} | {row['cases']} | "
            f"{percent(row['baseline_exact_match'])} | "
            f"{percent(row['adapted_exact_match'])} |"
        )
    passed = sum(bool(row["passed"]) for row in mode_probes)
    lines.extend(
        [
            "",
            "## Mode-Path Smoke Test",
            "",
            f"Deterministic mode-path checks: **{passed}/{len(mode_probes)} passed**.",
            "",
            "| Probe | Expected | Actual | Realized action | Pass |",
            "| :--- | :--- | :--- | :--- | :---: |",
        ]
    )
    for row in mode_probes:
        lines.append(
            f"| {row['probe_id']} | {row['expected_mode']} | "
            f"{row['actual_mode']} | {row['adapted_action']} | "
            f"{'yes' if row['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- The adapter closes explicit control-channel failures and "
                "blocks forward motion under the controlled low-clearance input."
            ),
            (
                "- Original R2R route proposals remain unchanged when no trusted "
                "directive or measured geometry evidence is available."
            ),
            (
                "- Visual perturbation robustness is unchanged; improving the "
                "visual representation still requires training and paired data."
            ),
            (
                "- Recover and Reject are deterministic interface paths here, not "
                "evidence that NaVILA learned failure recovery or rejection."
            ),
            "",
            "## Provenance",
            "",
            f"- Input predictions: `{metadata['input_predictions']}`",
            f"- Input SHA-256: `{metadata['input_sha256']}`",
            f"- Paired rows: {metadata['input_rows']}",
            f"- Generated: `{metadata['generated_at']}`",
            "",
        ]
    )
    return "\n".join(lines)


def plot_comparison(
    comparison: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    selected_names = {
        "Controlled command exact match",
        "Explicit stop compliance",
        "R2R stop-override compliance",
        "Blocked-fixture forward output",
        "Original R2R proposal preservation",
    }
    selected = [row for row in comparison if row["metric"] in selected_names]
    labels = [
        row["metric"]
        .replace("Controlled command ", "Command ")
        .replace("R2R ", "")
        .replace("Original ", "")
        for row in selected
    ]
    x = np.arange(len(selected))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.4, 4.0))
    axis.bar(
        x - width / 2,
        [row["baseline"] for row in selected],
        width,
        label="NaVILA direct",
    )
    axis.bar(
        x + width / 2,
        [row["adapted"] for row in selected],
        width,
        label="+ safety adapter",
    )
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Rate")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "vln_safety_adapter_comparison.png", dpi=240)
    fig.savefig(figure_dir / "vln_safety_adapter_comparison.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"No predictions found in {args.predictions}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = VLNSafetyAdapter()
    rows = adapt_rows(raw_rows, adapter)
    comparison = build_comparison(rows)
    per_action = per_action_comparison(rows)
    probes = mode_path_probes(adapter)
    if not all(bool(row["passed"]) for row in probes):
        raise RuntimeError("One or more deterministic mode-path probes failed.")

    write_csv(rows, args.output_dir / "adapted_predictions.csv")
    write_csv(comparison, args.output_dir / "comparison_summary.csv")
    write_csv(per_action, args.output_dir / "command_breakdown.csv")
    write_csv(probes, args.output_dir / "mode_path_smoke.csv")

    mode_counts = Counter(row["mode"] for row in rows)
    intervention_counts = Counter(row["intervention"] for row in rows)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "scope": "paired deterministic system-interface smoke test",
        "input_predictions": str(args.predictions.resolve()),
        "input_sha256": sha256(args.predictions),
        "input_rows": len(raw_rows),
        "paired_rows": len(rows),
        "adapter_config": asdict(adapter.cfg),
        "mode_distribution": dict(sorted(mode_counts.items())),
        "intervention_distribution": dict(sorted(intervention_counts.items())),
        "overridden_rows": sum(bool(row["overridden"]) for row in rows),
        "comparison": comparison,
        "per_action": per_action,
        "mode_path_smoke": {
            "passed": sum(bool(row["passed"]) for row in probes),
            "total": len(probes),
        },
        "runtime": {
            "python": platform.python_version(),
        },
        "limitations": [
            "The adapter is deterministic and does not improve NaVILA weights.",
            "Synthetic geometry values emulate fixture measurements.",
            "R2R instructions remain paired with unmatched demo frames.",
            "No SR, SPL, NE, nDTW, collision, or trajectory result is inferred.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        markdown_report(metadata, comparison, per_action, probes),
        encoding="utf-8",
    )
    plot_comparison(comparison, args.output_dir)
    print(json.dumps(metadata, indent=2))
    print(f"[write] {args.output_dir / 'adapted_predictions.csv'}")
    print(f"[write] {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
