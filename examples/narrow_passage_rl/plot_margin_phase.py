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


def _validate_required_columns(path: Path, fieldnames: list[str] | None) -> None:
    fields = set(fieldnames or [])
    missing: list[str] = []
    if "method" not in fields:
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
        if len(labels) != len(inputs):
            raise ValueError("--labels must match --inputs when --methods is omitted")
        file_labels = {idx: label for idx, label in enumerate(labels)}

    for idx, path in enumerate(inputs):
        file_label = file_labels.get(idx)
        with Path(path).open(newline="") as f:
            reader = csv.DictReader(f)
            _validate_required_columns(Path(path), reader.fieldnames)
            for row_idx, row in enumerate(reader):
                stats["total_rows"] += 1
                raw_method = str(row.get("method") or Path(path).stem)
                if method_filter and raw_method not in method_filter:
                    stats["filtered_by_method"] += 1
                    continue
                delta = _delta(row)
                if delta is None:
                    stats["dropped_missing_margin"] += 1
                    continue
                method = method_labels.get(raw_method, file_label or raw_method)
                mode_counts = _mode_counts(row)
                timeout = _metric(row, "timeout")
                stuck = _metric(row, "stuck")
                records.append(
                    {
                        "source": str(path),
                        "row_idx": row_idx,
                        "raw_method": raw_method,
                        "method": str(method),
                        "delta": float(delta),
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
) -> None:
    any_line = False
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    markers = ["o", "s", "^", "D"]
    used_low_marker = False
    for method_idx, method in enumerate(methods):
        color = colors[method_idx % len(colors)] if colors else None
        for metric_idx, (key, label) in enumerate(metric_keys):
            x, y, low, ci_low, ci_high = _series_with_support(rows, method, key)
            if len(x) == 0:
                continue
            any_line = True
            marker = markers[method_idx % len(markers)]
            high = ~low
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
                            linestyle="-",
                            linewidth=1.8,
                            marker=marker,
                            markersize=4.5,
                            capsize=2.5,
                            alpha=0.85,
                            color=color,
                            label=f"{method}: {label}" if first_segment else None,
                        )
                        first_segment = False
                        start = None
            if np.any(low):
                ax.scatter(
                    x[low],
                    y[low],
                    marker=marker,
                    s=42,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.4,
                    label=(
                        f"low support (<{min_bin_count} eps/bin)"
                        if not used_low_marker
                        else None
                    ),
                )
                used_low_marker = True
    ax.axvspan(-0.05, 0.05, color="0.8", alpha=0.22, zorder=0)
    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(r"Estimated margin $\hat{D} - W_{\mathrm{req,cons}}$ (m)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    if any_line:
        ax.legend(fontsize=7)
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
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.2), sharex=True, constrained_layout=True)

    _plot_metric_lines(
        axes[0],
        rows,
        methods,
        [("success_rate", "success")],
        "Success rate",
        min_bin_count,
    )
    _plot_metric_lines(
        axes[1],
        rows,
        methods,
        [("collision_rate", "collision")],
        "Collision rate",
        min_bin_count,
    )
    _plot_metric_lines(
        axes[2],
        rows,
        methods,
        [("reject_rate", "explicit reject")],
        "Explicit reject rate",
        min_bin_count,
    )
    if xlim is not None:
        for ax in axes:
            ax.set_xlim(*xlim)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, fontsize=14)
    if write_png:
        png_path = figures_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=300)
        print(f"[write] {png_path}")
    if write_pdf:
        pdf_path = figures_dir / f"{stem}.pdf"
        fig.savefig(pdf_path)
        print(f"[write] {pdf_path}")
    plt.close(fig)


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
        ("Infeasible-side", lambda d: d < -0.05),
        ("Near-boundary", lambda d: -0.05 <= d <= 0.05),
        ("Feasible-side", lambda d: d > 0.05),
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
        "| Method | Regime | Episodes | Success | Collision | Near collision | Reject | Correct reject | Timeout/stuck |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {method} | {regime} | {episodes} | {success} | {collision} | {near_collision} | "
            "{reject} | {correct_reject} | {timeout_stuck} |".format(
                method=row["method"],
                regime=row["regime"],
                episodes=row["episodes"],
                success=_fmt_rate(row["success"]),
                collision=_fmt_rate(row["collision"]),
                near_collision=_fmt_rate(row["near_collision"]),
                reject=_fmt_rate(row["reject"]),
                correct_reject=_fmt_rate(row["correct_reject"]),
                timeout_stuck=_fmt_rate(row["timeout_stuck"]),
            )
        )
    md_lines.extend([
        "",
        "Notes:",
        "- Margin is computed at the shared pre-divergence decision snapshot.",
        "- Belief diagnostics for the rule baseline are used only for stratification and do not influence its actions.",
        f"- Bins with fewer than {min_bin_count} episodes are marked low support.",
        "- Reject denotes an explicit Reject mode, not generic failure.",
    ])
    md_path = table_dir / "paper_table_margin_phase_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[write] {md_path}")

    tex_lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Method & Regime & Episodes & Success & Collision & Near collision & Reject & Correct reject & Timeout/stuck \\",
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
    parser.add_argument("--margin-max", type=float, default=0.10)
    parser.add_argument("--min-bin-count", type=int, default=1)
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
    records, load_stats = load_records(args.inputs, labels, methods_filter)
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

    output_dir = Path(args.output_dir)
    if args.table_dir is None:
        figures_dir = output_dir / "figures"
        table_dir = output_dir / "tables"
        summary_name = "margin_phase_summary.csv"
    else:
        figures_dir = output_dir
        table_dir = Path(args.table_dir)
        summary_name = "margin_phase_rule_vs_degnav_rule.csv"

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
        title="Margin-Phase: Reactive Rule Baseline vs DEGNAV-Rule",
        xlim=(float(args.margin_min), float(args.margin_max)),
        write_png=True,
        write_pdf=True,
    )

    zoom_rows = [
        row for row in summary
        if -0.10 <= float(row["bin_center"]) <= 0.10
        and int(row.get("episode_count", 0)) >= int(args.min_bin_count)
    ]
    if zoom_rows:
        plot_summary(
            summary,
            figures_dir,
            methods,
            min_bin_count=int(args.min_bin_count),
            stem="margin_phase_near_boundary_zoom",
            title="Near-Boundary Margin Phase",
            xlim=(-0.10, 0.10),
            write_png=True,
            write_pdf=True,
        )
    else:
        print("[warn] near-boundary zoom skipped: no bins in [-0.10, 0.10] "
              f"with at least {args.min_bin_count} episodes")


if __name__ == "__main__":
    main()
