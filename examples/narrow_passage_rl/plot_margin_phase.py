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

DELTA_DIRECT_KEYS = ("delta_mean", "Delta_mean", "final_delta_mean", "delta", "margin")
D_HAT_KEYS = ("d_hat", "D_hat", "final_d_hat")
W_REQ_KEYS = ("w_req_cons", "W_req_cons", "final_w_req_cons")


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
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
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
                raw_method = str(row.get("method") or Path(path).stem)
                if method_filter and raw_method not in method_filter:
                    continue
                delta = _delta(row)
                if delta is None:
                    continue
                method = method_labels.get(raw_method, file_label or raw_method)
                mode_counts = _mode_counts(row)
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
            "n": len(vals),
            "low_support": len(vals) < min_bin_count,
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
        "low_support",
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = [
        (float(row["bin_center"]), row.get(key), bool(row.get("low_support", False)))
        for row in rows
        if row.get("method") == method and row.get(key) is not None
    ]
    if not vals:
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=bool),
        )
    vals.sort(key=lambda x: x[0])
    x = np.asarray([v[0] for v in vals], dtype=np.float32)
    y = np.asarray([float(v[1]) for v in vals], dtype=np.float32)
    low = np.asarray([bool(v[2]) for v in vals], dtype=bool)
    return x, y, low


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
    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D"]
    used_low_marker = False
    for method_idx, method in enumerate(methods):
        color = colors[method_idx % len(colors)] if colors else None
        for metric_idx, (key, label) in enumerate(metric_keys):
            x, y, low = _series_with_support(rows, method, key)
            if len(x) == 0:
                continue
            any_line = True
            style = linestyles[metric_idx % len(linestyles)]
            marker = markers[metric_idx % len(markers)]
            ax.plot(
                x,
                y,
                linestyle=style,
                linewidth=1.8,
                alpha=0.65,
                color=color,
                label=f"{method}: {label}",
            )
            high = ~low
            if np.any(high):
                ax.scatter(x[high], y[high], marker=marker, s=26, color=color)
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
    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(r"Estimated margin $\hat{D} - W_{\mathrm{req,cons}}$ (m)")
    ax.set_ylim(-0.03, 1.03)
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
        [("success_rate", "success"), ("strict_success_rate", "strict")],
        "Task success by feasibility margin",
        min_bin_count,
    )
    _plot_metric_lines(
        axes[1],
        rows,
        methods,
        [("collision_rate", "collision"), ("near_collision_rate", "near")],
        "Safety events by feasibility margin",
        min_bin_count,
    )
    _plot_metric_lines(
        axes[2],
        rows,
        methods,
        [
            ("reject_rate", "reject"),
            ("correct_reject_rate", "correct"),
            ("false_reject_rate", "false"),
        ],
        "Reject decisions by feasibility margin",
        min_bin_count,
    )
    if xlim is not None:
        for ax in axes:
            ax.set_xlim(*xlim)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, fontsize=14)
    if write_png:
        png_path = figures_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=240)
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
    for method in methods:
        method_records = [
            rec for rec in records
            if rec.get("method") == method
            and margin_min <= float(rec["delta"]) <= margin_max
        ]
        method_summary = [row for row in summary if row.get("method") == method]
        near_records = [
            rec for rec in method_records
            if -0.10 <= float(rec["delta"]) <= 0.10
        ]
        rows.append(
            {
                "method": method,
                "episodes": len(method_records),
                "success": _rate(method_records, "success"),
                "collision": _rate(method_records, "collision"),
                "near_collision": _rate(method_records, "near_collision"),
                "reject": _rate(method_records, "reject"),
                "near_boundary_episodes": len(near_records),
                "bins": len(method_summary),
                "low_support_bins": sum(
                    1 for row in method_summary
                    if bool(row.get("low_support", False))
                    or int(row.get("n", 0)) < min_bin_count
                ),
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
        "Binned by `delta_mean = d_hat - w_req_cons`. Episode counts are rows inside "
        f"the plotted margin range [{margin_min:.2f}, {margin_max:.2f}] m. "
        f"Low-support bins are bins with fewer than {min_bin_count} episodes.",
        "",
        "| Method | Episodes | Success | Collision | Near collision | Reject | Near-boundary episodes | Bins | Low-support bins |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {method} | {episodes} | {success} | {collision} | {near_collision} | "
            "{reject} | {near_boundary_episodes} | {bins} | {low_support_bins} |".format(
                method=row["method"],
                episodes=row["episodes"],
                success=_fmt_rate(row["success"]),
                collision=_fmt_rate(row["collision"]),
                near_collision=_fmt_rate(row["near_collision"]),
                reject=_fmt_rate(row["reject"]),
                near_boundary_episodes=row["near_boundary_episodes"],
                bins=row["bins"],
                low_support_bins=row["low_support_bins"],
            )
        )
    md_lines.extend([
        "",
        "Note: the reactive rule baseline does not consume a feasibility-belief state; its margin is logged as an environment-derived diagnostic for common x-axis binning.",
    ])
    md_path = table_dir / "paper_table_margin_phase_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[write] {md_path}")

    tex_lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & Episodes & Success & Collision & Near collision & Reject & Near-boundary eps. & Bins & Low-support bins \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{str(row['method']).replace('_', r'\\_')} & {row['episodes']} "
            f"& {_fmt_tex_rate(row['success'])} "
            f"& {_fmt_tex_rate(row['collision'])} "
            f"& {_fmt_tex_rate(row['near_collision'])} "
            f"& {_fmt_tex_rate(row['reject'])} "
            f"& {row['near_boundary_episodes']} "
            f"& {row['bins']} "
            f"& {row['low_support_bins']} \\\\"
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
    records = load_records(args.inputs, labels, methods_filter)
    if not records:
        raise RuntimeError(
            "No rows with usable margin data found. Expected delta_mean, "
            "final_delta_mean, or d_hat/w_req_cons columns."
        )
    methods = []
    seen = set()
    for rec in records:
        method = str(rec["method"])
        if method not in seen:
            methods.append(method)
            seen.add(method)

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
        and int(row.get("n", 0)) >= int(args.min_bin_count)
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
            write_png=False,
            write_pdf=True,
        )
    else:
        print("[warn] near-boundary zoom skipped: no bins in [-0.10, 0.10] "
              f"with at least {args.min_bin_count} episodes")


if __name__ == "__main__":
    main()
