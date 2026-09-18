#!/usr/bin/env python3
"""Statistics, paper tables, and plots for strict feasibility ablations."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from narrow_passage.models.feasibility_ablation import (
    ABLATIONS,
    MAIN_ABLATIONS,
    PAPER_METHOD_NAMES,
    SelectorThresholds,
    canonical_ablation,
)


METRICS: tuple[tuple[str, str, bool], ...] = (
    ("success", "Success Rate", True),
    ("collision", "Collision Rate", True),
    ("correct_reject", "Correct Reject Rate", True),
    ("false_reject", "False Reject Rate", True),
    ("timeout_stuck", "Timeout/Stuck Rate", True),
    ("steps", "Average Steps", False),
    ("completion_time", "Average Time", False),
    ("oscillation_count", "Oscillation Count", False),
)

MARGIN_EDGES = (-0.1500001, -0.10, -0.05, 0.0, 0.05, 0.10, 0.1500001)
MARGIN_LABELS = (
    "[-0.15,-0.10)",
    "[-0.10,-0.05)",
    "[-0.05,0)",
    "[0,0.05)",
    "[0.05,0.10)",
    "[0.10,0.15]",
)
YAW_EDGES_DEG = (0.0, 10.0, 30.0, 45.0, 60.0, math.inf)
YAW_LABELS = ("0-10", "10-30", "30-45", "45-60", ">60")


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value


def _metric(row: dict[str, str], key: str) -> float:
    if key == "timeout_stuck":
        return float(_float(row, "timeout", 0.0) > 0.5 or _float(row, "stuck", 0.0) > 0.5)
    return _float(row, key)


def _read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("No episode rows were found")
    return rows


def _canonicalize_alias_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Collapse compatibility aliases so they cannot become separate paper rows."""

    collapsed: dict[tuple[str, str], dict[str, str]] = {}
    audit: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        raw = str(row.get("method", row.get("ablation", "")))
        canonical = canonical_ablation(raw)
        row["source_method"] = raw
        row["method"] = canonical
        row["canonical_method"] = canonical
        alias = raw != canonical
        if alias:
            item = audit.setdefault(
                raw,
                {
                    "alias": raw,
                    "canonical_method": canonical,
                    "behaviorally_equivalent": True,
                    "reported_as_independent_result": False,
                    "rows_seen": 0,
                },
            )
            item["rows_seen"] += 1
        key = (canonical, str(row.get("episode_id", "")))
        if key in collapsed:
            # Alias duplication is allowed only when observable outcomes agree.
            for field in ("success", "collision", "reject", "selected_mode"):
                if str(collapsed[key].get(field, "")) != str(row.get(field, "")):
                    raise RuntimeError(
                        f"Behaviorally-equivalent aliases disagree at {key}: {field}"
                    )
            continue
        collapsed[key] = row
    return list(collapsed.values()), list(audit.values())


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def _write_text(text: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"[write] {path}")


def _t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    return table.get(df, 1.96)


def _seed_summary(rows: list[dict[str, str]], key: str) -> dict[str, float | int]:
    by_seed: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _metric(row, key)
        if np.isfinite(value):
            by_seed[str(row["seed"])].append(value)
    seed_means = [float(np.mean(values)) for values in by_seed.values() if values]
    if not seed_means:
        return {"mean": math.nan, "std": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n_seeds": 0, "episodes": 0}
    mean = float(np.mean(seed_means))
    std = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
    if len(seed_means) > 1:
        half = _t_critical_95(len(seed_means) - 1) * std / math.sqrt(len(seed_means))
        ci_low, ci_high = mean - half, mean + half
    else:
        ci_low = ci_high = math.nan
    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_seeds": len(seed_means),
        "episodes": sum(len(values) for values in by_seed.values()),
    }


def overall_rows(rows: list[dict[str, str]], methods: list[str]) -> list[dict[str, Any]]:
    out = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        result: dict[str, Any] = {
            "method": method,
            "paper_method": PAPER_METHOD_NAMES.get(method, method),
            "episodes": len(subset),
            "seeds": len({row["seed"] for row in subset}),
        }
        for key, _label, _is_rate in METRICS:
            summary = _seed_summary(subset, key)
            for stat in ("mean", "std", "ci_low", "ci_high"):
                result[f"{key}_{stat}"] = summary[stat]
        out.append(result)
    return out


def paired_rows(rows: list[dict[str, str]], methods: list[str]) -> list[dict[str, Any]]:
    reference = "full_dynamic_uncertainty"
    if reference not in methods:
        return []
    index = {(row["method"], row["episode_id"]): row for row in rows}
    out = []
    for ablation in methods:
        if ablation == reference:
            continue
        episode_ids = sorted(
            set(row["episode_id"] for row in rows if row["method"] == reference)
            & set(row["episode_id"] for row in rows if row["method"] == ablation)
        )
        for key, label, is_rate in METRICS:
            by_seed: dict[str, list[float]] = defaultdict(list)
            for episode_id in episode_ids:
                full = index[(reference, episode_id)]
                other = index[(ablation, episode_id)]
                diff = _metric(full, key) - _metric(other, key)
                if np.isfinite(diff):
                    by_seed[str(full["seed"])].append(diff)
            seed_diffs = [float(np.mean(values)) for values in by_seed.values() if values]
            mean = float(np.mean(seed_diffs)) if seed_diffs else math.nan
            std = float(np.std(seed_diffs, ddof=1)) if len(seed_diffs) > 1 else 0.0
            if len(seed_diffs) > 1:
                half = _t_critical_95(len(seed_diffs) - 1) * std / math.sqrt(len(seed_diffs))
                lo, hi = mean - half, mean + half
            else:
                lo = hi = math.nan
            out.append(
                {
                    "comparison": f"{reference} - {ablation}",
                    "ablation": ablation,
                    "metric": key,
                    "metric_label": label,
                    "rate_metric": int(is_rate),
                    "paired_episodes": sum(len(values) for values in by_seed.values()),
                    "n_seeds": len(seed_diffs),
                    "mean_difference": mean,
                    "std_across_seeds": std,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return out


def _format_stat(row: dict[str, Any], key: str, rate: bool) -> str:
    mean = float(row[f"{key}_mean"])
    std = float(row[f"{key}_std"])
    lo = float(row[f"{key}_ci_low"])
    hi = float(row[f"{key}_ci_high"])
    if rate:
        base = f"{100 * mean:.1f} ± {100 * std:.1f}%"
        return base + (f" [{100 * lo:.1f}, {100 * hi:.1f}]" if np.isfinite(lo) else " [CI n/a]")
    base = f"{mean:.2f} ± {std:.2f}"
    return base + (f" [{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else " [CI n/a]")


def overall_markdown(summary: list[dict[str, Any]]) -> str:
    labels = [label for _key, label, _rate in METRICS]
    lines = [
        "# Strict Feasibility Ablation",
        "",
        "Values are mean ± sample standard deviation across seeds, followed by the 95% Student-t confidence interval. `CI n/a` means only one seed was run.",
        "",
        "| Method | Episodes | Seeds | " + " | ".join(labels) + " |",
        "|:---|---:|---:|" + "---:|" * len(labels),
    ]
    for row in summary:
        values = [
            _format_stat(row, key, is_rate)
            for key, _label, is_rate in METRICS
        ]
        lines.append(
            f"| {row['paper_method']} | {row['episodes']} | {row['seeds']} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "Correct Reject Rate uses all episodes as denominator; false-feasible-only outcome rows remain available in the episode CSV. Timeout/stuck is not counted as correct rejection.",
        ]
    )
    return "\n".join(lines)


def overall_latex(summary: list[dict[str, Any]]) -> str:
    def compact(row: dict[str, Any], key: str, rate: bool) -> str:
        mean = float(row[f"{key}_mean"])
        std = float(row[f"{key}_std"])
        scale = 100.0 if rate else 1.0
        suffix = r"\%" if rate else ""
        return f"{scale * mean:.1f} $\\pm$ {scale * std:.1f}{suffix}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Strict paired feasibility ablation. Values are mean $\pm$ standard deviation across seeds; confidence intervals and paired differences are provided in the accompanying CSV.}",
        r"\label{tab:feasibility-ablation}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        "Method & $N$ & SR & Collision & Correct reject & False reject & Timeout/stuck & Steps & Time & Osc. \\\\",
        r"\midrule",
    ]
    for row in summary:
        values = [compact(row, key, rate) for key, _label, rate in METRICS]
        method = str(row["paper_method"]).replace("_", r"\_")
        lines.append(f"{method} & {row['episodes']} & " + " & ".join(values) + " \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def _bin_label(value: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str | None:
    for index, label in enumerate(labels):
        if edges[index] <= value < edges[index + 1]:
            return label
    return None


def grouped_rows(
    rows: list[dict[str, str]],
    methods: list[str],
    *,
    value_fn: Callable[[dict[str, str]], float],
    edges: tuple[float, ...],
    labels: tuple[str, ...],
    metrics: tuple[str, ...],
    group_name: str,
) -> list[dict[str, Any]]:
    out = []
    for method in methods:
        for label in labels:
            subset = [
                row
                for row in rows
                if row["method"] == method
                and _bin_label(value_fn(row), edges, labels) == label
            ]
            result: dict[str, Any] = {
                "method": method,
                group_name: label,
                "episodes": len(subset),
            }
            for metric in metrics:
                values = [_metric(row, metric) for row in subset]
                values = [value for value in values if np.isfinite(value)]
                result[metric] = float(np.mean(values)) if values else math.nan
            out.append(result)
    return out


def calibration_rows(rows: list[dict[str, str]], methods: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = []
    reliability = []
    eps = 1e-9
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        if method == "mean_only":
            continue
        probabilities = np.asarray([_float(row, "p_feas") for row in subset], dtype=float)
        labels = np.asarray([_float(row, "passable_label") for row in subset], dtype=float)
        valid = np.isfinite(probabilities) & np.isfinite(labels)
        probabilities = np.clip(probabilities[valid], eps, 1.0 - eps)
        labels = labels[valid]
        brier = float(np.mean((probabilities - labels) ** 2)) if len(labels) else math.nan
        nll = float(-np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))) if len(labels) else math.nan
        ece = 0.0
        for bin_index in range(10):
            lo, hi = bin_index / 10.0, (bin_index + 1) / 10.0
            mask = (probabilities >= lo) & ((probabilities < hi) if hi < 1.0 else (probabilities <= hi))
            count = int(np.sum(mask))
            if count:
                mean_p = float(np.mean(probabilities[mask]))
                empirical = float(np.mean(labels[mask]))
                ece += count / max(len(labels), 1) * abs(mean_p - empirical)
            else:
                mean_p = empirical = math.nan
            reliability.append(
                {
                    "method": method,
                    "bin_lower": lo,
                    "bin_upper": hi,
                    "count": count,
                    "mean_p_feas": mean_p,
                    "empirical_feasible": empirical,
                }
            )
        metrics.append(
            {"method": method, "episodes": len(labels), "brier": brier, "ece": ece, "nll": nll}
        )

    point = [row for row in rows if row["method"] == "mean_only"]
    if point:
        threshold = SelectorThresholds().tau_commit
        y = np.asarray([_float(row, "passable_label") > 0.5 for row in point], dtype=bool)
        pred = np.asarray(
            [_float(row, "mu_delta_struct") >= threshold for row in point],
            dtype=bool,
        )
        tp = int(np.sum(pred & y))
        fp = int(np.sum(pred & ~y))
        fn = int(np.sum(~pred & y))
        metrics.append(
            {
                "method": "mean_only",
                "episodes": len(point),
                "classification_accuracy": float(np.mean(pred == y)),
                "classification_precision": tp / max(tp + fp, 1),
                "classification_recall": tp / max(tp + fn, 1),
                "classification_threshold_mu_delta": threshold,
                "brier": math.nan,
                "ece": math.nan,
                "nll": math.nan,
            }
        )
    return metrics, reliability


def subset_metric_rows(
    rows: list[dict[str, str]], methods: list[str]
) -> list[dict[str, Any]]:
    """Emit the two label-conditional metric panels required by the protocol."""

    feasible_metrics = (
        "success", "collision", "false_reject", "timeout",
        "alignment_time", "oscillation_count",
    )
    infeasible_metrics = (
        "correct_reject", "collision", "wasted_commitment",
        "time_to_reject", "recovery_attempts",
    )
    out: list[dict[str, Any]] = []
    for method in methods:
        for subset_name, passable, metric_names in (
            ("feasible", True, feasible_metrics),
            ("infeasible", False, infeasible_metrics),
        ):
            subset = [
                row for row in rows
                if row["method"] == method
                and (_float(row, "passable_label") > 0.5) is passable
            ]
            result: dict[str, Any] = {
                "method": method,
                "subset": subset_name,
                "episodes": len(subset),
            }
            for metric in metric_names:
                values = [_metric(row, metric) for row in subset]
                values = [value for value in values if np.isfinite(value)]
                result[metric] = float(np.mean(values)) if values else math.nan
            out.append(result)
    return out


def risk_coverage_rows(
    rows: list[dict[str, str]], methods: list[str]
) -> list[dict[str, Any]]:
    """Selective feasibility-classification risk as confidence coverage changes."""

    out: list[dict[str, Any]] = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        probabilities = np.asarray([_float(row, "p_feas_struct") for row in subset])
        if not np.any(np.isfinite(probabilities)):
            probabilities = np.asarray([_float(row, "p_feas") for row in subset])
        labels = np.asarray([_float(row, "passable_label") > 0.5 for row in subset])
        valid = np.isfinite(probabilities)
        probabilities = np.clip(probabilities[valid], 0.0, 1.0)
        labels = labels[valid]
        confidence = np.maximum(probabilities, 1.0 - probabilities)
        predictions = probabilities >= 0.5
        for threshold in np.linspace(0.50, 0.95, 10):
            covered = confidence >= threshold
            count = int(np.sum(covered))
            out.append({
                "method": method,
                "confidence_threshold": float(threshold),
                "coverage": count / max(len(labels), 1),
                "selective_risk": (
                    float(np.mean(predictions[covered] != labels[covered]))
                    if count else math.nan
                ),
                "covered_episodes": count,
            })
    return out


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix, kwargs in ((".png", {"dpi": 220}), (".pdf", {})):
        path = stem.with_suffix(suffix)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing figure: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"[write] {path}")
    plt.close(fig)


def _plot_grouped(
    grouped: list[dict[str, Any]],
    methods: list[str],
    group_key: str,
    labels: tuple[str, ...],
    metric_specs: tuple[tuple[str, str], ...],
    stem: Path,
) -> None:
    cols = 2
    rows_n = math.ceil(len(metric_specs) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(8.2, 3.0 * rows_n), squeeze=False)
    x = np.arange(len(labels))
    for axis, (metric, title) in zip(axes.flat, metric_specs):
        for method in methods:
            values = []
            for label in labels:
                match = next(
                    item for item in grouped
                    if item["method"] == method and item[group_key] == label
                )
                values.append(float(match[metric]))
            axis.plot(x, values, marker="o", linewidth=1.5, label=PAPER_METHOD_NAMES.get(method, method))
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(alpha=0.3)
        axis.set_ylim(bottom=0.0)
    for axis in axes.flat[len(metric_specs):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    _save_figure(fig, stem)


def _uncertainty_groups(rows: list[dict[str, str]], methods: list[str]) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    compared = {"full_dynamic_uncertainty", "mean_only"}
    selected = [row for row in rows if row["method"] in compared]
    values = sorted({round(math.sqrt(max(0.0, _float(row, "var_delta"))), 9) for row in selected})
    if len(values) <= 1:
        value = values[0] if values else math.nan
        labels = (f"{value:.3f}" if np.isfinite(value) else "unavailable",)
        edges = (-math.inf, math.inf)
    else:
        quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, min(6, len(values) + 1))))
        if len(quantiles) < 2:
            labels = (f"{values[0]:.3f}",)
            edges = (-math.inf, math.inf)
        else:
            inner = tuple(float(value) for value in quantiles[1:-1])
            edges = (-math.inf, *inner, math.inf)
            labels = tuple(f"q{index + 1}" for index in range(len(edges) - 1))
    grouped = grouped_rows(
        selected,
        [method for method in methods if method in compared],
        value_fn=lambda row: math.sqrt(max(0.0, _float(row, "var_delta"))),
        edges=edges,
        labels=labels,
        metrics=("explore_ratio", "collision", "success", "false_reject"),
        group_name="uncertainty_bin",
    )
    return grouped, labels


def _conclusion(summary: list[dict[str, Any]], paired: list[dict[str, Any]], rows: list[dict[str, str]]) -> str:
    seeds = len({row["seed"] for row in rows})
    episodes_per_method = min(
        (sum(row["method"] == method for row in rows) for method in {row["method"] for row in rows}),
        default=0,
    )
    paper_scale = seeds >= 3 and episodes_per_method >= 630
    lines = [
        "# Objective Result Note",
        "",
        f"This is a {'paper-scale' if paper_scale else 'smoke/diagnostic'} run with {seeds} seed(s) and at least {episodes_per_method} episodes per method.",
        "",
    ]
    for row in summary:
        lines.append(
            f"- {row['paper_method']}: success={100 * float(row['success_mean']):.1f}%, collision={100 * float(row['collision_mean']):.1f}%, correct reject={100 * float(row['correct_reject_mean']):.1f}%."
        )
    full_no_unc = [
        row for row in paired
        if row["ablation"] == "mean_only" and row["metric"] == "success"
    ]
    if full_no_unc:
        lines.append("")
    full_point = full_no_unc
    if full_point:
        diff = float(full_point[0]["mean_difference"])
        lo = float(full_point[0]["ci95_low"])
        hi = float(full_point[0]["ci95_high"])
        lines.append(
            f"The observed paired success difference `full_dynamic_uncertainty - mean_only` is {100 * diff:.2f} percentage points (95% CI [{100 * lo:.2f}, {100 * hi:.2f}])."
        )
    full_fixed = [
        row for row in paired
        if row["ablation"] == "fixed_uncertainty" and row["metric"] == "success"
    ]
    if full_fixed:
        diff = float(full_fixed[0]["mean_difference"])
        lo = float(full_fixed[0]["ci95_low"])
        hi = float(full_fixed[0]["ci95_high"])
        lines.append(
            f"The observed paired success difference `full_dynamic_uncertainty - fixed_uncertainty` is {100 * diff:.2f} percentage points (95% CI [{100 * lo:.2f}, {100 * hi:.2f}])."
        )
    full_no_yaw = [
        row for row in paired
        if row["ablation"] == "no_yaw_aware_readiness" and row["metric"] == "success"
    ]
    if full_no_yaw:
        diff = float(full_no_yaw[0]["mean_difference"])
        lo = float(full_no_yaw[0]["ci95_low"])
        hi = float(full_no_yaw[0]["ci95_high"])
        lines.append(
            f"The observed paired success difference `full_dynamic_uncertainty - no_yaw_aware_readiness` is {100 * diff:.2f} percentage points (95% CI [{100 * lo:.2f}, {100 * hi:.2f}])."
        )
    unique_uncertainty = {
        round(math.sqrt(max(0.0, _float(row, "var_delta"))), 9)
        for row in rows
        if row["method"] in {"full_dynamic_uncertainty", "mean_only"}
    }
    if len(unique_uncertainty) <= 1:
        lines.append(
            "The run contains only one uncertainty level. It therefore cannot identify an uncertainty-aware gating effect; equality or difference in other outcomes must not be presented as uncertainty calibration evidence."
        )
        if full_point and float(full_point[0]["mean_difference"]) == 0.0:
            lines.append(
                "With fixed variance and analytically mapped thresholds, the probability gate is a monotone reparameterization of mean margin; the exact full/point-estimate equality is therefore an implementation limitation, not evidence that distributions are generally unnecessary."
            )
    if full_no_yaw:
        lo = float(full_no_yaw[0]["ci95_low"])
        hi = float(full_no_yaw[0]["ci95_high"])
        if lo <= 0.0 <= hi:
            lines.append(
                "The paired success interval for the yaw ablation includes zero, so this run does not support a claim that the projected-rectangle yaw prior improves success."
            )
    if paper_scale:
        lines.append(
            "The requested sample scale was completed. Causal claims must follow the paired confidence intervals; passing mechanism validation establishes that the ablations are distinguishable, not that any variant is superior."
        )
    else:
        lines.append(
            "No causal superiority claim should be copied into the paper unless the paper-scale run is completed and its paired confidence interval supports that claim."
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output directory: {args.output_dir}")
    rows, alias_audit = _canonicalize_alias_rows(_read_rows(args.input_csv))
    methods = [
        method for method in MAIN_ABLATIONS
        if any(row["method"] == method for row in rows)
    ]
    if alias_audit:
        _write_csv(alias_audit, args.output_dir / "compatibility_alias_audit.csv")
    summary = overall_rows(rows, methods)
    paired = paired_rows(rows, methods)
    _write_csv(summary, args.output_dir / "overall_ablation_summary.csv")
    if paired:
        _write_csv(paired, args.output_dir / "paired_differences.csv")
    _write_text(overall_markdown(summary), args.output_dir / "overall_ablation_table.md")
    _write_text(overall_latex(summary), args.output_dir / "overall_ablation_table.tex")
    _write_csv(
        subset_metric_rows(rows, methods),
        args.output_dir / "label_conditional_metrics.csv",
    )

    full_margin_by_episode = {
        row["episode_id"]: _float(row, "structural_margin_gt")
        for row in rows
        if row["method"] == "full_dynamic_uncertainty"
    }
    for row in rows:
        row["_reference_mu_delta"] = str(
            full_margin_by_episode.get(
                row["episode_id"], _float(row, "structural_margin_gt")
            )
        )
    margin = grouped_rows(
        rows,
        methods,
        value_fn=lambda row: _float(row, "_reference_mu_delta"),
        edges=MARGIN_EDGES,
        labels=MARGIN_LABELS,
        metrics=("success", "collision", "correct_reject", "false_reject", "explore_ratio"),
        group_name="margin_bin_m",
    )
    for row in margin:
        row["margin_source"] = "unified_obb_structural_margin_ground_truth"
    _write_csv(margin, args.output_dir / "margin_phase.csv")
    _plot_grouped(
        margin,
        methods,
        "margin_bin_m",
        MARGIN_LABELS,
        (
            ("success", "Success rate"),
            ("collision", "Collision rate"),
            ("correct_reject", "Correct reject rate"),
            ("false_reject", "False reject rate"),
            ("explore_ratio", "Explore rate"),
        ),
        args.output_dir / "margin_phase",
    )

    yaw_methods = list(methods)
    yaw = grouped_rows(
        rows,
        yaw_methods,
        value_fn=lambda row: abs(math.degrees(_float(row, "initial_yaw"))),
        edges=YAW_EDGES_DEG,
        labels=YAW_LABELS,
        metrics=("success", "collision", "timeout_stuck", "steps"),
        group_name="yaw_bin_deg",
    )
    _write_csv(yaw, args.output_dir / "yaw_sensitivity.csv")
    _plot_grouped(
        yaw,
        yaw_methods,
        "yaw_bin_deg",
        YAW_LABELS,
        (("success", "Success rate"), ("collision", "Collision rate"), ("timeout_stuck", "Timeout/stuck rate"), ("steps", "Average steps")),
        args.output_dir / "yaw_sensitivity",
    )

    uncertainty, uncertainty_labels = _uncertainty_groups(rows, methods)
    _write_csv(uncertainty, args.output_dir / "uncertainty_behavior.csv")
    uncertainty_methods = [
        method for method in methods
        if method in {"full_dynamic_uncertainty", "mean_only"}
    ]
    _plot_grouped(
        uncertainty,
        uncertainty_methods,
        "uncertainty_bin",
        uncertainty_labels,
        (("explore_ratio", "Explore rate"), ("collision", "Collision rate"), ("success", "Success rate"), ("false_reject", "False reject rate")),
        args.output_dir / "uncertainty_behavior",
    )

    calibration, reliability = calibration_rows(rows, methods)
    _write_csv(calibration, args.output_dir / "calibration_metrics.csv")
    _write_csv(reliability, args.output_dir / "reliability.csv")
    if reliability:
        fig, axis = plt.subplots(figsize=(4.8, 4.2))
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        for method in methods:
            method_rows = [row for row in reliability if row["method"] == method and row["count"] > 0]
            if method_rows:
                axis.plot(
                    [row["mean_p_feas"] for row in method_rows],
                    [row["empirical_feasible"] for row in method_rows],
                    marker="o",
                    label=PAPER_METHOD_NAMES.get(method, method),
                )
        axis.set(xlabel="Mean predicted p_feas", ylabel="Empirical feasible rate", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
        _save_figure(fig, args.output_dir / "reliability_diagram")

    risk_coverage = risk_coverage_rows(rows, methods)
    _write_csv(risk_coverage, args.output_dir / "risk_coverage.csv")
    if risk_coverage:
        fig, axis = plt.subplots(figsize=(5.0, 4.2))
        for method in methods:
            method_rows = [row for row in risk_coverage if row["method"] == method]
            axis.plot(
                [row["coverage"] for row in method_rows],
                [row["selective_risk"] for row in method_rows],
                marker="o",
                label=PAPER_METHOD_NAMES.get(method, method),
            )
        axis.set(xlabel="Coverage", ylabel="Selective feasibility risk")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
        _save_figure(fig, args.output_dir / "risk_coverage")

    _write_text(_conclusion(summary, paired, rows), args.output_dir / "objective_conclusion.md")


if __name__ == "__main__":
    main()
