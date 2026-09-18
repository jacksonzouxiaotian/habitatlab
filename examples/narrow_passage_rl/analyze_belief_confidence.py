#!/usr/bin/env python3
"""Compare engineering and recursive-posterior confidence audit signals."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


FULL_METHOD = "full_dynamic_uncertainty"
MEASURES = (
    ("engineering feature sigma", "sigma_delta_pose"),
    ("recursive posterior sigma", "posterior_sigma_delta_pose"),
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _correlation_ratio_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Return categorical-vs-continuous effect size eta squared."""

    grand = float(np.mean(values))
    total = float(np.sum((values - grand) ** 2))
    if total <= 0.0:
        return 0.0
    between = 0.0
    for group in np.unique(groups):
        selected = values[groups == group]
        between += len(selected) * (float(np.mean(selected)) - grand) ** 2
    return float(between / total)


def _point_biserial(values: np.ndarray, labels: np.ndarray) -> float:
    """Pearson/point-biserial correlation for a continuous/binary pair."""

    if len(np.unique(labels)) < 2 or float(np.std(values)) <= 0.0:
        return 0.0
    return float(np.corrcoef(values, labels.astype(float))[0, 1])


def analyze(input_csv: Path, output_dir: Path) -> tuple[list[dict[str, Any]], str]:
    rows = [row for row in _read_rows(input_csv) if row.get("method") == FULL_METHOD]
    if not rows:
        raise RuntimeError(f"No {FULL_METHOD!r} rows in {input_csv}")
    required = {"selected_mode", "uncertainty_triggered", *(field for _, field in MEASURES)}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"Missing confidence audit columns: {missing}")

    table: list[dict[str, Any]] = []
    for label, field in MEASURES:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        modes = np.asarray([str(row["selected_mode"]) for row in rows], dtype=object)
        triggered = np.asarray(
            [_as_bool(row["uncertainty_triggered"]) for row in rows], dtype=bool
        )
        valid = np.isfinite(values)
        values, modes, triggered = values[valid], modes[valid], triggered[valid]
        eta2 = _correlation_ratio_squared(values, modes)
        r = _point_biserial(values, triggered)
        table.append({
            "confidence_measure": label,
            "source_field": field,
            "steps": len(values),
            "selected_mode_eta_squared": eta2,
            "uncertainty_triggered_point_biserial_r": r,
            "uncertainty_triggered_r_squared": r**2,
            "mean_explanatory_score": 0.5 * (eta2 + r**2),
        })

    winner = max(table, key=lambda row: float(row["mean_explanatory_score"]))
    loser = min(table, key=lambda row: float(row["mean_explanatory_score"]))
    conclusion = (
        f"在这次 mechanism validation 中，{winner['confidence_measure']} 对模式/"
        f"uncertainty branch 的联合解释力更强（平均解释分数 "
        f"{float(winner['mean_explanatory_score']):.3f} vs "
        f"{float(loser['mean_explanatory_score']):.3f}）；这是相关性诊断，不是因果效应。"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(table, output_dir / "belief_confidence_comparison.csv")
    markdown = [
        "# Belief confidence comparison",
        "",
        "只分析 `full_dynamic_uncertainty` 的逐步记录。`eta²` 衡量连续置信度在不同",
        "`selected_mode` 组之间的方差解释比例；点二列相关衡量它与",
        "`uncertainty_triggered` 的关联。两者都不改变 selector 输入。",
        "",
        "| confidence | steps | selected-mode η² | uncertainty-triggered r | r² | mean score |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        markdown.append(
            f"| {row['confidence_measure']} | {row['steps']} | "
            f"{float(row['selected_mode_eta_squared']):.3f} | "
            f"{float(row['uncertainty_triggered_point_biserial_r']):.3f} | "
            f"{float(row['uncertainty_triggered_r_squared']):.3f} | "
            f"{float(row['mean_explanatory_score']):.3f} |"
        )
    markdown.extend(["", conclusion, ""])
    md_path = output_dir / "belief_confidence_comparison.md"
    if md_path.exists():
        raise FileExistsError(f"Refusing to overwrite {md_path}")
    md_path.write_text("\n".join(markdown), encoding="utf-8")
    return table, conclusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_csv.parent
    table, conclusion = analyze(args.input_csv, output_dir)
    for row in table:
        print(row)
    print(conclusion)


if __name__ == "__main__":
    main()
