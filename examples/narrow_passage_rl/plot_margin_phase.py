#!/usr/bin/env python3
"""Plot width-margin phase diagrams from per-episode evaluation CSV files.

The x-axis is the estimated feasibility margin:

    delta = D_hat - W_req_cons

where available either as ``delta_mean``/``final_delta_mean`` or computed from
``d_hat - w_req_cons``.  The script is deliberately dependency-light: it uses
only the Python standard library, NumPy, and Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    Path(__file__).parent / "results" / "narrow_passage_rl"
)

BOOL_KEYS = {
    "success": ("success",),
    "strict_success": ("strict_success",),
    "collision": ("collision",),
    "near_collision": ("near_collision",),
    "reject": ("reject", "rejected"),
    "correct_reject": ("correct_reject",),
    "false_reject": ("false_reject",),
    "timeout": ("timeout",),
    "stuck": ("stuck",),
}

MODE_COUNT_KEYS = {
    "commit": ("mode_commit_count", "commit_count"),
    "explore": ("mode_explore_count", "explore_count"),
    "recover": ("mode_recover_count", "recover_count"),
    "reject": ("mode_reject_count", "reject_count"),
}

DELTA_DIRECT_KEYS = ("delta_mean", "Delta_mean", "delta", "margin")
D_HAT_KEYS = ("d_hat", "D_hat")
W_REQ_KEYS = ("w_req_cons", "W_req_cons")
P_FEAS_KEYS = ("p_feas", "P_feas")

PAPER_METHOD_LABELS = {
    "rule_baseline": "Reactive rule",
    "geometry_fsm": "DEGNAV-Rule",
}


def _split_cli_values(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _get_first_float(row: dict[str, str], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in row:
            value = _finite_float(row.get(key))
            if value is not None:
                return value
    return None


def _metric(row: dict[str, str], metric_name: str) -> float | None:
    return _get_first_float(row, BOOL_KEYS[metric_name])


def _delta(row: dict[str, str]) -> float | None:
    direct = _get_first_float(row, DELTA_DIRECT_KEYS)
    if direct is not None:
        return direct

    d_hat = _get_first_float(row, D_HAT_KEYS)
    w_req = _get_first_float(row, W_REQ_KEYS)
    if d_hat is None or w_req is None:
        return None
    return d_hat - w_req


def _mode_counts(row: dict[str, str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for mode, keys in MODE_COUNT_KEYS.items():
        value = _get_first_float(row, keys)
        if value is not None:
            counts[mode] = max(0.0, value)

    if counts:
        return counts

    # Some diagnostic CSVs only log a final mode string.  Treat that as a single
    # mode observation so the script can still provide a coarse phase diagram.
    raw_mode = str(row.get("mode") or row.get("mode_name") or "").lower()
    if not raw_mode:
        return {}
    if "reject" in raw_mode:
        return {"reject": 1.0}
    if "recover" in raw_mode:
        return {"recover": 1.0}
    if "explore" in raw_mode or "align" in raw_mode:
        return {"explore": 1.0}
    if "commit" in raw_mode or "stop" in raw_mode:
        return {"commit": 1.0}
    return {}


def _validate_required_columns(
    path: Path,
    fieldnames: list[str] | None,
    *,
    require_method: bool,
) -> None:
    fields = set(fieldnames or [])
    missing: list[str] = []
    if require_method and "method" not in fields:
        missing.append("method")
    if not any(key in fields for key in DELTA_DIRECT_KEYS):
        if not any(key in fields for key in D_HAT_KEYS):
            missing.append("delta_mean or d_hat")
        if not any(key in fields for key in W_REQ_KEYS):
            missing.append("delta_mean or w_req_cons")
    for key in ("success", "reject"):
        if key not in fields:
            missing.append(key)
    if "collision" not in fields and "near_collision" not in fields:
        missing.append("collision or near_collision")
    if missing:
        raise RuntimeError(
            f"{path} is missing required margin-phase columns: {', '.join(missing)}"
        )


def load_records(
    inputs: list[Path],
    labels: list[str] | None,
    methods: list[str] | None,
    *,
    require_method: bool = True,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    records: list[dict[str, object]] = []
    stats = {
        "total_rows": 0,
        "filtered_by_method": 0,
        "dropped_missing_margin": 0,
    }
    method_filter = set(methods or [])
    method_labels: dict[str, str] = {}
    file_labels: dict[int, str] = {}

    if labels is not None and methods is not None:
        if len(labels) != len(methods):
            raise ValueError("--labels must match --methods when --methods is provided")
        method_labels = dict(zip(methods, labels))
    elif labels is not None:
        if len(labels) == 1:
            file_labels = {idx: labels[0] for idx, _ in enumerate(inputs)}
        elif len(labels) != len(inputs):
            raise ValueError("--labels must match --inputs when --methods is omitted")
        else:
            file_labels = {idx: label for idx, label in enumerate(labels)}

    for idx, path in enumerate(inputs):
        file_label = file_labels.get(idx)
        with Path(path).open(newline="") as f:
            reader = csv.DictReader(f)
            _validate_required_columns(
                Path(path),
                reader.fieldnames,
                require_method=require_method,
            )
            for row_idx, row in enumerate(reader):
                stats["total_rows"] += 1
                raw_method = str(row.get("method") or row.get("ablation") or Path(path).stem)
                if method_filter and raw_method not in method_filter:
                    stats["filtered_by_method"] += 1
                    continue
                delta = _delta(row)
                if delta is None:
                    stats["dropped_missing_margin"] += 1
                    continue
                method = method_labels.get(
                    raw_method,
                    file_label or PAPER_METHOD_LABELS.get(raw_method, raw_method),
                )
                mode_counts = _mode_counts(row)
                timeout = _metric(row, "timeout")
                stuck = _metric(row, "stuck")
                records.append(
                    {
                        "source": str(path),
                        "row_idx": row_idx,
                        "raw_method": raw_method,
                        "method": str(method),
                        "seed": str(row.get("seed") or ""),
                        "scenario_id": str(
                            row.get("scenario_id")
                            or row.get("episode_id")
                            or row.get("episode")
                            or row_idx
                        ),
                        "delta": float(delta),
                        "delta_logged": _get_first_float(row, DELTA_DIRECT_KEYS),
                        "success": _metric(row, "success"),
                        "strict_success": _metric(row, "strict_success"),
                        "collision": _metric(row, "collision"),
                        "near_collision": _metric(row, "near_collision"),
                        "reject": _metric(row, "reject"),
                        "correct_reject": _metric(row, "correct_reject"),
                        "false_reject": _metric(row, "false_reject"),
                        "timeout": timeout,
                        "stuck": stuck,
                        "timeout_or_stuck": (
                            None
                            if timeout is None and stuck is None
                            else float((timeout or 0.0) > 0.5 or (stuck or 0.0) > 0.5)
                        ),
                        "p_feas": _get_first_float(row, P_FEAS_KEYS),
                        "d_hat": _get_first_float(row, D_HAT_KEYS),
                        "w_req_cons": _get_first_float(row, W_REQ_KEYS),
                        "mode_counts": mode_counts,
                    }
                )
    return records, stats


def _raw_audit(
    path: Path,
    *,
    bin_width: float,
    margin_min: float,
    margin_max: float,
    label_override: str | None = None,
) -> dict[str, object]:
    rows = 0
    finite = 0
    methods: dict[str, int] = defaultdict(int)
    raw_methods: dict[str, int] = defaultdict(int)
    seeds: dict[str, int] = defaultdict(int)
    deltas: list[float] = []
    bin_counts: dict[str, int] = defaultdict(int)
    fields: list[str] = []
    edges = np.arange(margin_min, margin_max + bin_width * 0.5, bin_width)
    if len(edges) < 2 or edges[-1] < margin_max:
        edges = np.append(edges, margin_max)

    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "rows": 0,
            "methods": {},
            "raw_methods": {},
            "rows_per_seed": {},
            "finite_delta_coverage": 0.0,
            "delta_min": None,
            "delta_median": None,
            "delta_max": None,
            "bin_counts": {},
            "has_rule_and_degnav": False,
        }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for idx, row in enumerate(reader):
            rows += 1
            raw_method = str(row.get("method") or row.get("ablation") or path.stem)
            method = label_override or raw_method
            raw_methods[raw_method] += 1
            methods[method] += 1
            seeds[str(row.get("seed") or "")] += 1
            delta = _delta(row)
            if delta is None:
                continue
            finite += 1
            deltas.append(float(delta))
            bin_idx = _bin_index(float(delta), edges)
            if bin_idx is not None:
                key = f"[{edges[bin_idx]:.2f}, {edges[bin_idx + 1]:.2f})"
                if bin_idx == len(edges) - 2:
                    key = f"[{edges[bin_idx]:.2f}, {edges[bin_idx + 1]:.2f}]"
                bin_counts[key] += 1

    method_keys = set(raw_methods) | set(methods)
    return {
        "path": str(path),
        "exists": True,
        "fields": fields,
        "rows": rows,
        "methods": dict(methods),
        "raw_methods": dict(raw_methods),
        "rows_per_seed": dict(seeds),
        "finite_delta_coverage": (finite / rows) if rows else 0.0,
        "finite_delta_count": finite,
        "delta_min": min(deltas) if deltas else None,
        "delta_median": float(np.median(deltas)) if deltas else None,
        "delta_max": max(deltas) if deltas else None,
        "bin_counts": dict(sorted(bin_counts.items())),
        "has_rule_and_degnav": {"rule_baseline", "geometry_fsm"}.issubset(method_keys),
    }


def _fmt_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value_f):
        return "n/a"
    return f"{value_f:.{digits}f}"


def _format_mapping(mapping: dict[str, int], max_items: int = 12) -> str:
    if not mapping:
        return "none"
    items = sorted(mapping.items())
    shown = ", ".join(f"{key}: {value}" for key, value in items[:max_items])
    if len(items) > max_items:
        shown += f", ... ({len(items) - max_items} more)"
    return shown


def write_input_audit(
    *,
    table_dir: Path,
    inputs: list[Path],
    bin_width: float,
    margin_min: float,
    margin_max: float,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    result_root = Path(__file__).parent / "results" / "narrow_passage_rl"
    current_figure = result_root / "figures" / "margin_phase_all_methods.pdf"
    current_summary = result_root / "tables" / "margin_phase_summary.csv"
    inferred_current_input = result_root / "belief_mode_full_seed0_eval.csv"
    audits = [
        (
            "Current DEGNAV-RL-only phase plot source",
            _raw_audit(
                inferred_current_input,
                bin_width=bin_width,
                margin_min=margin_min,
                margin_max=margin_max,
                label_override="DEGNAV-RL",
            ),
        )
    ]
    for path in inputs:
        audits.append(
            (
                f"Requested input: {path}",
                _raw_audit(
                    path,
                    bin_width=bin_width,
                    margin_min=margin_min,
                    margin_max=margin_max,
                ),
            )
        )

    lines = [
        "# Margin-Phase Input Audit",
        "",
        "This audit records which data sources are suitable for the paper-facing "
        "Rule-vs-DEGNAV margin-phase plot.",
        "",
        "## Current Four-Panel Figure",
        "",
        f"- Figure: `{current_figure}`",
        f"- Existing summary: `{current_summary}`",
        f"- Identified raw input: `{inferred_current_input}`",
        "- Provenance note: the figure itself does not store its command.  The "
        "co-located summary table contains only `DEGNAV-RL` rows, and its bin "
        "support matches the identified seed-0 DEGNAV-RL evaluation CSV.",
        "",
    ]
    for title, audit in audits:
        lines.extend(
            [
                f"## {title}",
                "",
                f"- input path: `{audit['path']}`",
                f"- exists: {audit['exists']}",
                f"- methods: {_format_mapping(audit.get('methods', {}))}",
                f"- raw method values: {_format_mapping(audit.get('raw_methods', {}))}",
                f"- rows per seed: {_format_mapping(audit.get('rows_per_seed', {}))}",
                f"- finite delta_mean coverage: {100.0 * float(audit.get('finite_delta_coverage', 0.0)):.1f}% "
                f"({audit.get('finite_delta_count', 0)}/{audit.get('rows', 0)})",
                f"- delta min / median / max: {_fmt_float(audit.get('delta_min'))} / "
                f"{_fmt_float(audit.get('delta_median'))} / {_fmt_float(audit.get('delta_max'))}",
                f"- contains both rule_baseline and geometry_fsm: {audit.get('has_rule_and_degnav')}",
                "",
                "Bin counts:",
                "",
            ]
        )
        if audit.get("bin_counts"):
            lines.append("| Bin | Count |")
            lines.append("| :--- | ---: |")
            for bin_name, count in audit["bin_counts"].items():
                lines.append(f"| `{bin_name}` | {count} |")
        else:
            lines.append("No finite-margin rows inside the requested range.")
        lines.append("")

    path = table_dir / "margin_phase_input_audit.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def write_main_missing_data_report(table_dir: Path, inputs: list[Path]) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Missing Rule-vs-DEGNAV Margin-Phase Data",
        "",
        "The requested main-paper figure was not generated because the input did "
        "not contain paired `rule_baseline` and `geometry_fsm` rows.",
        "",
        "Inputs:",
    ]
    lines.extend(f"- `{path}`" for path in inputs)
    lines.extend(
        [
            "",
            "Regenerate the required raw CSV with:",
            "",
            "```bash",
            "python examples/narrow_passage_rl/eval_harder_benchmark.py \\",
            "  --methods rule_baseline geometry_fsm \\",
            "  --episodes 500 \\",
            "  --seeds 0 1 2 \\",
            "  --log-belief-diagnostics \\",
            "  --log-outcome-decomposition \\",
            "  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv",
            "```",
        ]
    )
    path = table_dir / "margin_phase_main_missing_data.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def validate_main_records(
    records: list[dict[str, object]],
    summary: list[dict[str, object]],
    *,
    methods_filter: list[str] | None,
    min_bin_count: int,
) -> None:
    raw_methods = {str(rec.get("raw_method")) for rec in records}
    if raw_methods == {"DEGNAV-RL"} or raw_methods == {"full"}:
        raise RuntimeError(
            "Main plot input appears to contain only DEGNAV-RL diagnostic data; "
            "use --plot-mode rl-diagnostic instead."
        )
    required = {"rule_baseline", "geometry_fsm"}
    missing = sorted(required - raw_methods)
    if missing:
        raise RuntimeError(f"Main plot requires paired methods {sorted(required)}; missing {missing}")
    if methods_filter and set(methods_filter) != required:
        raise RuntimeError("--plot-mode main must be run with --methods rule_baseline geometry_fsm")

    seen: set[tuple[str, str, str]] = set()
    duplicates: list[tuple[str, str, str]] = []
    scenario_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    consistency_failures = 0
    finite_consistency_checks = 0
    for rec in records:
        raw_method = str(rec.get("raw_method"))
        if raw_method not in required:
            continue
        seed = str(rec.get("seed") or "")
        scenario_id = str(rec.get("scenario_id") or "")
        key = (raw_method, seed, scenario_id)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        scenario_sets[(raw_method, seed)].add(scenario_id)
        delta_logged = rec.get("delta_logged")
        d_hat = rec.get("d_hat")
        w_req = rec.get("w_req_cons")
        if delta_logged is None or d_hat is None or w_req is None:
            continue
        finite_consistency_checks += 1
        if abs(float(delta_logged) - (float(d_hat) - float(w_req))) > 1e-6:
            consistency_failures += 1

    if duplicates:
        raise RuntimeError(f"Duplicate method+seed+scenario_id rows found, first duplicate: {duplicates[0]}")
    if consistency_failures:
        raise RuntimeError(
            f"delta_mean consistency failed for {consistency_failures}/"
            f"{finite_consistency_checks} finite rows"
        )
    seeds = sorted({seed for _, seed in scenario_sets})
    for seed in seeds:
        rule_set = scenario_sets.get(("rule_baseline", seed), set())
        degnav_set = scenario_sets.get(("geometry_fsm", seed), set())
        if rule_set != degnav_set:
            raise RuntimeError(
                f"Unpaired scenario_id sets for seed {seed}: "
                f"rule={len(rule_set)} geometry_fsm={len(degnav_set)} "
                f"intersection={len(rule_set & degnav_set)}"
            )
    total_bin_support: dict[float, int] = defaultdict(int)
    for row in summary:
        total_bin_support[float(row["bin_center"])] += int(row.get("episode_count", 0))
    sufficient_bins = [
        center for center, count in total_bin_support.items()
        if count >= int(min_bin_count)
    ]
    if len(sufficient_bins) < 3:
        raise RuntimeError(
            f"Need at least three bins with >= {min_bin_count} total episodes; "
            f"found {len(sufficient_bins)}"
        )
    print("[validate] main input has paired rule_baseline and geometry_fsm scenarios")
    print(f"[validate] delta consistency checks passed: {finite_consistency_checks}")
    print(f"[validate] bins with >={min_bin_count} episodes: {len(sufficient_bins)}")


def _bin_index(delta: float, edges: np.ndarray) -> int | None:
    if delta < edges[0] or delta > edges[-1]:
        return None
    if delta == edges[-1]:
        return len(edges) - 2
    idx = int(np.searchsorted(edges, delta, side="right") - 1)
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return idx


def _mean_available(records: list[dict[str, object]], key: str) -> float | None:
    values = [
        float(rec[key])
        for rec in records
        if rec.get(key) is not None and math.isfinite(float(rec[key]))
    ]
    if not values:
        return None
    return float(np.mean(values))


def _wilson_interval(successes: float, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """Dependency-free Wilson score interval for a binomial rate."""

    if n <= 0:
        return None, None
    phat = float(successes) / float(n)
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _rate_and_ci(records: list[dict[str, object]], key: str) -> tuple[float | None, float | None, float | None]:
    values = [
        float(rec[key])
        for rec in records
        if rec.get(key) is not None and math.isfinite(float(rec[key]))
    ]
    if not values:
        return None, None, None
    rate = float(np.mean(values))
    lo, hi = _wilson_interval(float(np.sum(values)), len(values))
    return rate, lo, hi


def summarize_records(
    records: list[dict[str, object]],
    bin_width: float,
    margin_min: float,
    margin_max: float,
    min_bin_count: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    edges = np.arange(margin_min, margin_max + bin_width * 0.5, bin_width)
    if len(edges) < 2 or edges[-1] < margin_max:
        edges = np.append(edges, margin_max)

    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        idx = _bin_index(float(rec["delta"]), edges)
        if idx is None:
            continue
        groups[(str(rec["method"]), idx)].append(rec)

    summary: list[dict[str, object]] = []
    for (method, idx), vals in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        mode_totals = {mode: 0.0 for mode in MODE_COUNT_KEYS}
        total_mode_count = 0.0
        for rec in vals:
            counts = rec.get("mode_counts") or {}
            if not isinstance(counts, dict):
                continue
            for mode in mode_totals:
                value = float(counts.get(mode, 0.0))
                mode_totals[mode] += value
                total_mode_count += value

        row: dict[str, object] = {
            "method": method,
            "bin_left": round(float(edges[idx]), 6),
            "bin_right": round(float(edges[idx + 1]), 6),
            "bin_center": round(float((edges[idx] + edges[idx + 1]) / 2.0), 6),
            "episode_count": len(vals),
            "n": len(vals),
            "low_support": len(vals) < min_bin_count,
        }
        for metric_name in BOOL_KEYS:
            row[f"{metric_name}_rate"] = _mean_available(vals, metric_name)
        for metric_name in ("success", "collision", "reject"):
            rate, lo, hi = _rate_and_ci(vals, metric_name)
            row[f"{metric_name}_rate"] = rate
            row[f"{metric_name}_ci_low"] = lo
            row[f"{metric_name}_ci_high"] = hi
        row["mean_p_feas"] = _mean_available(vals, "p_feas")
        row["mean_d_hat"] = _mean_available(vals, "d_hat")
        row["mean_w_req_cons"] = _mean_available(vals, "w_req_cons")
        for mode, total in mode_totals.items():
            row[f"{mode}_freq"] = (
                float(total / total_mode_count) if total_mode_count > 0 else None
            )
        summary.append(row)
    return summary, edges


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "bin_left",
        "bin_right",
        "bin_center",
        "episode_count",
        "low_support",
        "success_rate",
        "success_ci_low",
        "success_ci_high",
        "strict_success_rate",
        "collision_rate",
        "collision_ci_low",
        "collision_ci_high",
        "near_collision_rate",
        "reject_rate",
        "reject_ci_low",
        "reject_ci_high",
        "correct_reject_rate",
        "false_reject_rate",
        "timeout_rate",
        "stuck_rate",
        "mean_p_feas",
        "mean_d_hat",
        "mean_w_req_cons",
        "commit_freq",
        "explore_freq",
        "recover_freq",
        "reject_freq",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key)
                    for key in fieldnames
                }
            )
    print(f"[write] {path}")


def _series(
    rows: list[dict[str, object]],
    method: str,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    vals = [
        (float(row["bin_center"]), row.get(key))
        for row in rows
        if row.get("method") == method and row.get(key) is not None
    ]
    if not vals:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    vals.sort(key=lambda x: x[0])
    x = np.asarray([v[0] for v in vals], dtype=np.float32)
    y = np.asarray([float(v[1]) for v in vals], dtype=np.float32)
    return x, y


def _series_with_support(
    rows: list[dict[str, object]],
    method: str,
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vals = [
        (
            float(row["bin_center"]),
            row.get(key),
            bool(row.get("low_support", False)),
            row.get(key.replace("_rate", "_ci_low")),
            row.get(key.replace("_rate", "_ci_high")),
        )
        for row in rows
        if row.get("method") == method and row.get(key) is not None
    ]
    if not vals:
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=bool),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )
    vals.sort(key=lambda x: x[0])
    x = np.asarray([v[0] for v in vals], dtype=np.float32)
    y = np.asarray([float(v[1]) for v in vals], dtype=np.float32)
    low = np.asarray([bool(v[2]) for v in vals], dtype=bool)
    ci_low = np.asarray([
        float(v[3]) if v[3] is not None else float(v[1])
        for v in vals
    ], dtype=np.float32)
    ci_high = np.asarray([
        float(v[4]) if v[4] is not None else float(v[1])
        for v in vals
    ], dtype=np.float32)
    return x, y, low, ci_low, ci_high


def _plot_metric_lines(
    ax,
    rows: list[dict[str, object]],
    methods: list[str],
    metric_keys: list[tuple[str, str]],
    title: str,
    min_bin_count: int,
    *,
    show_xlabel: bool = True,
    show_legend: bool = True,
) -> None:
    any_line = False
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", ":", "-."]
    for method_idx, method in enumerate(methods):
        for metric_idx, (key, label) in enumerate(metric_keys):
            color_idx = method_idx if len(metric_keys) == 1 else metric_idx
            color = colors[color_idx % len(colors)] if colors else None
            linestyle = linestyles[method_idx % len(linestyles)]
            x, y, low, ci_low, ci_high = _series_with_support(rows, method, key)
            if len(x) == 0:
                continue
            any_line = True
            marker = markers[method_idx % len(markers)]
            high = ~low
            label_text = method if len(metric_keys) == 1 else f"{method}: {label}"
            if np.any(high):
                first_segment = True
                start = None
                for idx, is_high in enumerate(high):
                    if is_high and start is None:
                        start = idx
                    at_end = idx == len(high) - 1
                    if start is not None and ((not is_high) or at_end):
                        end = idx + 1 if is_high and at_end else idx
                        seg = slice(start, end)
                        yerr = np.vstack([
                            np.maximum(0.0, y[seg] - ci_low[seg]),
                            np.maximum(0.0, ci_high[seg] - y[seg]),
                        ])
                        ax.errorbar(
                            x[seg],
                            y[seg],
                            yerr=yerr,
                            linestyle=linestyle,
                            linewidth=1.8,
                            marker=marker,
                            markersize=4.5,
                            capsize=2.5,
                            alpha=0.85,
                            color=color,
                            label=label_text if first_segment else None,
                        )
                        first_segment = False
                        start = None
            if np.any(low):
                yerr_low = np.vstack([
                    np.maximum(0.0, y[low] - ci_low[low]),
                    np.maximum(0.0, ci_high[low] - y[low]),
                ])
                ax.errorbar(
                    x[low],
                    y[low],
                    yerr=yerr_low,
                    linestyle="none",
                    marker=marker,
                    markersize=5.0,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.4,
                    ecolor=color,
                    elinewidth=0.9,
                    capsize=2.0,
                    label=None,
                )
    ax.axvspan(-0.05, 0.05, color="0.8", alpha=0.22, zorder=0)
    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel(r"Estimated margin $\hat{D} - W_{\mathrm{req,cons}}$ (m)")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    if any_line:
        if show_legend:
            ax.legend(fontsize=7, loc="upper left", frameon=False, handlelength=2.2)
    else:
        ax.text(0.5, 0.5, "No available data", ha="center", va="center", transform=ax.transAxes)


def plot_summary(
    rows: list[dict[str, object]],
    figures_dir: Path,
    methods: list[str],
    *,
    min_bin_count: int,
    stem: str,
    title: str,
    xlim: tuple[float, float] | None = None,
    write_png: bool = True,
    write_pdf: bool = True,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.35), sharex=True, constrained_layout=True)

    _plot_metric_lines(
        axes[0],
        rows,
        methods,
        [("success_rate", "success")],
        "A. Success rate",
        min_bin_count,
        show_xlabel=False,
        show_legend=True,
    )
    _plot_metric_lines(
        axes[1],
        rows,
        methods,
        [("collision_rate", "collision")],
        "B. Collision rate",
        min_bin_count,
        show_xlabel=True,
        show_legend=False,
    )
    if xlim is not None:
        for ax in axes:
            ax.set_xlim(*xlim)
            ax.locator_params(axis="x", nbins=5)

    figures_dir.mkdir(parents=True, exist_ok=True)
    if write_png:
        png_path = figures_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=450)
        print(f"[write] {png_path}")
    if write_pdf:
        pdf_path = figures_dir / f"{stem}.pdf"
        fig.savefig(pdf_path)
        print(f"[write] {pdf_path}")
    plt.close(fig)


def _available_metric_keys(
    rows: list[dict[str, object]],
    candidates: list[tuple[str, str]],
    *,
    show_zero_metrics: bool,
) -> list[tuple[str, str]]:
    available: list[tuple[str, str]] = []
    for key, label in candidates:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]
        if not values:
            continue
        if not show_zero_metrics and all(abs(value) < 1e-12 for value in values):
            continue
        available.append((key, label))
    return available


def plot_rl_diagnostic(
    rows: list[dict[str, object]],
    figures_dir: Path,
    methods: list[str],
    *,
    min_bin_count: int,
    xlim: tuple[float, float] | None,
    show_zero_metrics: bool,
) -> None:
    diagnostic_dir = (
        figures_dir.parent / "diagnostic"
        if figures_dir.name == "figures"
        else figures_dir / "diagnostic"
    )
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = _available_metric_keys(
        rows,
        [
            ("success_rate", "success"),
            ("collision_rate", "collision"),
            ("near_collision_rate", "near collision"),
        ],
        show_zero_metrics=show_zero_metrics,
    )
    mode_keys = _available_metric_keys(
        rows,
        [
            ("commit_freq", "Commit"),
            ("explore_freq", "Explore"),
            ("recover_freq", "Recover"),
            ("reject_freq", "Reject"),
        ],
        show_zero_metrics=show_zero_metrics,
    )

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True, constrained_layout=True)
    _plot_metric_lines(
        axes[0],
        rows,
        methods,
        metric_keys,
        "Outcome diagnostics",
        min_bin_count,
    )
    _plot_metric_lines(
        axes[1],
        rows,
        methods,
        mode_keys,
        "Mode usage",
        min_bin_count,
    )
    for ax in axes:
        if xlim is not None:
            ax.set_xlim(*xlim)
    fig.suptitle("Diagnostic Behavior of DEGNAV-RL Across Feasibility Margins", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Appendix diagnostic: current DEGNAV-RL collapses to Commit/Explore and does not "
        "demonstrate meaningful Recover or Reject behavior.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    png_path = diagnostic_dir / "degnav_rl_margin_diagnostic.png"
    pdf_path = diagnostic_dir / "degnav_rl_margin_diagnostic.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[write] {png_path}")
    print(f"[write] {pdf_path}")


def _rate(records: list[dict[str, object]], key: str) -> float | None:
    values = [
        float(rec[key])
        for rec in records
        if rec.get(key) is not None and math.isfinite(float(rec[key]))
    ]
    if not values:
        return None
    return float(np.mean(values))


def _fmt_rate(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _fmt_tex_rate(value: float | None) -> str:
    return _fmt_rate(value).replace("%", r"\%")


def _paper_table_rows(
    records: list[dict[str, object]],
    summary: list[dict[str, object]],
    methods: list[str],
    margin_min: float,
    margin_max: float,
    min_bin_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    regimes = [
        ("Negative margin", lambda d: d < -0.05),
        ("Near-boundary", lambda d: -0.05 <= d <= 0.05),
        ("Positive margin", lambda d: d > 0.05),
    ]
    for method in methods:
        for regime_name, predicate in regimes:
            method_records = [
                rec for rec in records
                if rec.get("method") == method
                and margin_min <= float(rec["delta"]) <= margin_max
                and predicate(float(rec["delta"]))
            ]
            rows.append(
                {
                    "method": method,
                    "regime": regime_name,
                    "episodes": len(method_records),
                    "success": _rate(method_records, "success"),
                    "collision": _rate(method_records, "collision"),
                    "near_collision": _rate(method_records, "near_collision"),
                    "reject": _rate(method_records, "reject"),
                    "correct_reject": _rate(method_records, "correct_reject"),
                    "false_reject": _rate(method_records, "false_reject"),
                    "timeout_stuck": _rate(method_records, "timeout_or_stuck"),
                }
            )
    return rows


def write_paper_tables(
    records: list[dict[str, object]],
    summary: list[dict[str, object]],
    methods: list[str],
    table_dir: Path,
    margin_min: float,
    margin_max: float,
    min_bin_count: int,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = _paper_table_rows(
        records, summary, methods, margin_min, margin_max, min_bin_count
    )
    md_lines = [
        "# Table: Margin-Phase Summary",
        "",
        "Regime summary by `delta_mean = d_hat - w_req_cons`. Episode counts are rows inside "
        f"the plotted margin range [{margin_min:.2f}, {margin_max:.2f}] m.",
        "",
        "| Method | Regime | Episodes | Success | Collision | Near collision | Reject | Correct reject | False reject | Timeout/stuck |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {method} | {regime} | {episodes} | {success} | {collision} | {near_collision} | "
            "{reject} | {correct_reject} | {false_reject} | {timeout_stuck} |".format(
                method=row["method"],
                regime=row["regime"],
                episodes=row["episodes"],
                success=_fmt_rate(row["success"]),
                collision=_fmt_rate(row["collision"]),
                near_collision=_fmt_rate(row["near_collision"]),
                reject=_fmt_rate(row["reject"]),
                correct_reject=_fmt_rate(row["correct_reject"]),
                false_reject=_fmt_rate(row["false_reject"]),
                timeout_stuck=_fmt_rate(row["timeout_stuck"]),
            )
        )
    md_lines.extend([
        "",
        "Notes:",
        "- Margin is computed at the shared pre-divergence decision snapshot.",
        "- Belief diagnostics for the rule baseline are used only for stratification and do not influence its actions.",
        f"- Bins with fewer than {min_bin_count} episodes are marked low support.",
        "- Low-support bins are marked and not interpreted as reliable trends.",
        "- Reject denotes an explicit Reject mode, not generic failure.",
    ])
    md_path = table_dir / "paper_table_margin_phase_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[write] {md_path}")

    tex_lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Method & Regime & Episodes & Success & Collision & Near collision & Reject & Correct reject & False reject & Timeout/stuck \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{str(row['method']).replace('_', r'\\_')} & {row['regime']} & {row['episodes']} "
            f"& {_fmt_tex_rate(row['success'])} "
            f"& {_fmt_tex_rate(row['collision'])} "
            f"& {_fmt_tex_rate(row['near_collision'])} "
            f"& {_fmt_tex_rate(row['reject'])} "
            f"& {_fmt_tex_rate(row['correct_reject'])} "
            f"& {_fmt_tex_rate(row['false_reject'])} "
            f"& {_fmt_tex_rate(row['timeout_stuck'])} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path = table_dir / "paper_table_margin_phase_summary.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"[write] {tex_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate width-margin phase diagram from evaluation CSVs."
    )
    parser.add_argument(
        "--plot-mode",
        choices=("main", "rl-diagnostic"),
        default="main",
        help="main requires paired rule_baseline/geometry_fsm; rl-diagnostic is appendix-only.",
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels. With --methods, provide one per method; comma-separated is allowed.",
    )
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-dir", type=Path, default=None)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--margin-min", type=float, default=-0.10)
    parser.add_argument("--margin-max", type=float, default=0.30)
    parser.add_argument("--min-bin-count", type=int, default=1)
    parser.add_argument(
        "--show-zero-metrics",
        action="store_true",
        help="In rl-diagnostic mode, include all-zero diagnostic curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")
    if args.margin_max <= args.margin_min:
        raise ValueError("--margin-max must be greater than --margin-min")
    if args.min_bin_count < 1:
        raise ValueError("--min-bin-count must be >= 1")

    labels = _split_cli_values(args.labels)
    methods_filter = _split_cli_values(args.methods)

    output_dir = Path(args.output_dir)
    if args.table_dir is None:
        figures_dir = output_dir / "figures"
        table_dir = output_dir / "tables"
        summary_name = "margin_phase_summary.csv"
    else:
        figures_dir = output_dir
        table_dir = Path(args.table_dir)
        summary_name = "margin_phase_rule_vs_degnav_rule.csv"

    write_input_audit(
        table_dir=table_dir,
        inputs=list(args.inputs),
        bin_width=float(args.bin_width),
        margin_min=float(args.margin_min),
        margin_max=float(args.margin_max),
    )

    records, load_stats = load_records(
        args.inputs,
        labels,
        methods_filter,
        require_method=args.plot_mode == "main",
    )
    if not records:
        raise RuntimeError(
            "No rows with usable margin data found. Expected delta_mean, "
            "or d_hat/w_req_cons columns."
        )
    if methods_filter:
        present_raw_methods = {str(rec["raw_method"]) for rec in records}
        missing_methods = [method for method in methods_filter if method not in present_raw_methods]
        if missing_methods:
            raise RuntimeError(f"Requested methods missing from finite-margin data: {missing_methods}")
        for method in methods_filter:
            count = sum(1 for rec in records if str(rec["raw_method"]) == method)
            if count <= 0:
                raise RuntimeError(f"Requested method has no finite-margin episodes: {method}")
    methods = []
    seen = set()
    for rec in records:
        method = str(rec["method"])
        if method not in seen:
            methods.append(method)
            seen.add(method)

    in_range_records = [
        rec for rec in records
        if float(args.margin_min) <= float(rec["delta"]) <= float(args.margin_max)
    ]
    dropped_out_of_range = len(records) - len(in_range_records)
    print(
        "[load] total_rows={total} filtered_by_method={filtered} "
        "dropped_missing_margin={missing} dropped_out_of_range={out_of_range}".format(
            total=load_stats["total_rows"],
            filtered=load_stats["filtered_by_method"],
            missing=load_stats["dropped_missing_margin"],
            out_of_range=dropped_out_of_range,
        )
    )

    summary, _edges = summarize_records(
        records=records,
        bin_width=float(args.bin_width),
        margin_min=float(args.margin_min),
        margin_max=float(args.margin_max),
        min_bin_count=int(args.min_bin_count),
    )
    if not summary:
        raise RuntimeError(
            "No rows fell inside the requested margin range. Adjust "
            "--margin-min/--margin-max."
        )
    total_bin_support: dict[float, int] = defaultdict(int)
    for row in summary:
        total_bin_support[float(row["bin_center"])] += int(row.get("episode_count", 0))
    sufficient_bins = [
        center for center, count in total_bin_support.items()
        if count >= int(args.min_bin_count)
    ]
    if len(sufficient_bins) < 3:
        raise RuntimeError(
            f"Need at least three bins with >= {args.min_bin_count} total episodes; "
            f"found {len(sufficient_bins)}"
        )

    if args.plot_mode == "main":
        try:
            validate_main_records(
                records,
                summary,
                methods_filter=methods_filter,
                min_bin_count=int(args.min_bin_count),
            )
        except RuntimeError:
            write_main_missing_data_report(table_dir, list(args.inputs))
            raw_methods = {str(rec.get("raw_method")) for rec in records}
            if not {"rule_baseline", "geometry_fsm"}.issubset(raw_methods):
                diagnostic_dir = (
                    figures_dir.parent / "diagnostic"
                    if figures_dir.name == "figures"
                    else figures_dir / "diagnostic"
                )
                write_summary_csv(summary, diagnostic_dir / "degnav_rl_margin_summary.csv")
                plot_rl_diagnostic(
                    summary,
                    figures_dir,
                    methods,
                    min_bin_count=int(args.min_bin_count),
                    xlim=(float(args.margin_min), float(args.margin_max)),
                    show_zero_metrics=bool(args.show_zero_metrics),
                )
            raise
    else:
        diagnostic_dir = (
            figures_dir.parent / "diagnostic"
            if figures_dir.name == "figures"
            else figures_dir / "diagnostic"
        )
        write_summary_csv(summary, diagnostic_dir / "degnav_rl_margin_summary.csv")
        plot_rl_diagnostic(
            summary,
            figures_dir,
            methods,
            min_bin_count=int(args.min_bin_count),
            xlim=(float(args.margin_min), float(args.margin_max)),
            show_zero_metrics=bool(args.show_zero_metrics),
        )
        return

    write_summary_csv(summary, table_dir / summary_name)
    write_paper_tables(
        records=records,
        summary=summary,
        methods=methods,
        table_dir=table_dir,
        margin_min=float(args.margin_min),
        margin_max=float(args.margin_max),
        min_bin_count=int(args.min_bin_count),
    )
    plot_summary(
        summary,
        figures_dir,
        methods,
        min_bin_count=int(args.min_bin_count),
        stem="margin_phase_rule_vs_degnav_rule",
        title="",
        xlim=(float(args.margin_min), float(args.margin_max)),
        write_png=True,
        write_pdf=True,
    )

    zoom_rows = [
        row for row in summary
        if -0.10 <= float(row["bin_center"]) <= 0.10
    ]
    zoom_supported_rows = [
        row for row in zoom_rows
        if int(row.get("episode_count", 0)) >= int(args.min_bin_count)
    ]
    if zoom_supported_rows:
        plot_summary(
            zoom_rows,
            figures_dir,
            methods,
            min_bin_count=int(args.min_bin_count),
            stem="margin_phase_near_boundary_zoom",
            title="",
            xlim=(-0.10, 0.10),
            write_png=True,
            write_pdf=True,
        )
    else:
        print("[warn] near-boundary zoom skipped: no bins in [-0.10, 0.10] "
              f"with at least {args.min_bin_count} episodes")


if __name__ == "__main__":
    main()
