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
}

MODE_COUNT_KEYS = {
    "commit": ("mode_commit_count", "commit_count"),
    "explore": ("mode_explore_count", "explore_count"),
    "recover": ("mode_recover_count", "recover_count"),
    "reject": ("mode_reject_count", "reject_count"),
}


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
    direct = _get_first_float(
        row,
        (
            "delta_mean",
            "Delta_mean",
            "final_delta_mean",
            "delta",
            "margin",
        ),
    )
    if direct is not None:
        return direct

    d_hat = _get_first_float(row, ("d_hat", "D_hat", "final_d_hat"))
    w_req = _get_first_float(
        row,
        ("w_req_cons", "W_req_cons", "final_w_req_cons"),
    )
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


def load_records(inputs: list[Path], labels: list[str] | None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if labels is not None and len(labels) != len(inputs):
        raise ValueError("--labels must have the same length as --inputs")

    for idx, path in enumerate(inputs):
        label = labels[idx] if labels is not None else None
        with Path(path).open(newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                delta = _delta(row)
                if delta is None:
                    continue
                method = label or row.get("method") or Path(path).stem
                mode_counts = _mode_counts(row)
                records.append(
                    {
                        "source": str(path),
                        "row_idx": row_idx,
                        "method": str(method),
                        "delta": float(delta),
                        "success": _metric(row, "success"),
                        "strict_success": _metric(row, "strict_success"),
                        "collision": _metric(row, "collision"),
                        "near_collision": _metric(row, "near_collision"),
                        "reject": _metric(row, "reject"),
                        "correct_reject": _metric(row, "correct_reject"),
                        "false_reject": _metric(row, "false_reject"),
                        "mode_counts": mode_counts,
                    }
                )
    return records


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


def summarize_records(
    records: list[dict[str, object]],
    bin_width: float,
    margin_min: float,
    margin_max: float,
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
            "bin_left": float(edges[idx]),
            "bin_right": float(edges[idx + 1]),
            "bin_center": float((edges[idx] + edges[idx + 1]) / 2.0),
            "n": len(vals),
        }
        for metric_name in BOOL_KEYS:
            row[f"{metric_name}_rate"] = _mean_available(vals, metric_name)
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
        "n",
        "success_rate",
        "strict_success_rate",
        "collision_rate",
        "near_collision_rate",
        "reject_rate",
        "correct_reject_rate",
        "false_reject_rate",
        "commit_freq",
        "explore_freq",
        "recover_freq",
        "reject_freq",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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


def _plot_metric_lines(
    ax,
    rows: list[dict[str, object]],
    methods: list[str],
    metric_keys: list[tuple[str, str]],
    title: str,
) -> None:
    any_line = False
    for method in methods:
        for key, label in metric_keys:
            x, y = _series(rows, method, key)
            if len(x) == 0:
                continue
            any_line = True
            ax.plot(x, y, marker="o", linewidth=1.8, label=f"{method}: {label}")
    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Estimated margin D_hat - W_req_cons (m)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    if any_line:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No available data", ha="center", va="center", transform=ax.transAxes)


def plot_summary(rows: list[dict[str, object]], figures_dir: Path) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    _plot_metric_lines(
        axes[0, 0],
        rows,
        methods,
        [("success_rate", "success"), ("strict_success_rate", "strict")],
        "Task success by feasibility margin",
    )
    _plot_metric_lines(
        axes[0, 1],
        rows,
        methods,
        [("collision_rate", "collision"), ("near_collision_rate", "near")],
        "Safety events by feasibility margin",
    )
    _plot_metric_lines(
        axes[1, 0],
        rows,
        methods,
        [
            ("reject_rate", "reject"),
            ("correct_reject_rate", "correct"),
            ("false_reject_rate", "false"),
        ],
        "Reject decisions by feasibility margin",
    )
    _plot_metric_lines(
        axes[1, 1],
        rows,
        methods,
        [
            ("commit_freq", "commit"),
            ("explore_freq", "explore"),
            ("recover_freq", "recover"),
            ("reject_freq", "reject-mode"),
        ],
        "Mode frequency by feasibility margin",
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / "margin_phase_all_methods.png"
    pdf_path = figures_dir / "margin_phase_all_methods.pdf"
    fig.suptitle("Width-Margin Phase Diagram", fontsize=14)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[write] {png_path}")
    print(f"[write] {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate width-margin phase diagram from evaluation CSVs."
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional method labels, one per input CSV.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--margin-min", type=float, default=-0.10)
    parser.add_argument("--margin-max", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")
    if args.margin_max <= args.margin_min:
        raise ValueError("--margin-max must be greater than --margin-min")

    records = load_records(args.inputs, args.labels)
    if not records:
        raise RuntimeError(
            "No rows with usable margin data found. Expected delta_mean, "
            "final_delta_mean, or d_hat/w_req_cons columns."
        )

    summary, _edges = summarize_records(
        records=records,
        bin_width=float(args.bin_width),
        margin_min=float(args.margin_min),
        margin_max=float(args.margin_max),
    )
    if not summary:
        raise RuntimeError(
            "No rows fell inside the requested margin range. Adjust "
            "--margin-min/--margin-max."
        )

    output_dir = Path(args.output_dir)
    write_summary_csv(summary, output_dir / "tables" / "margin_phase_summary.csv")
    plot_summary(summary, output_dir / "figures")


if __name__ == "__main__":
    main()
