#!/usr/bin/env python3
"""Plot memory transfer/interference diagnostics for the paper.

This figure is generated from the seed-level output of
``eval_memory_transfer_interference.py``.  It avoids using the saturated final
false-feasible reject rate as the central comparison and instead emphasizes:

1. passable success,
2. transfer to similar-but-new false-feasible passages,
3. interference on similar-but-feasible passages, and
4. normalized wasted attempts on false-feasible passages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "examples/narrow_passage_rl/results/narrow_passage_rl"
DEFAULT_INPUT = RESULTS_DIR / "memory_transfer_interference.csv"
DEFAULT_FIGURE_DIR = RESULTS_DIR / "figures"
DEFAULT_TABLE_DIR = RESULTS_DIR / "tables"

METHOD_ORDER = [
    "no_memory",
    "knn_failure_memory",
    "vanilla_episodic_memory",
    "geometry_guided_cross_episode_failure_memory",
]

DISPLAY_LABELS = {
    "no_memory": "No memory",
    "knn_failure_memory": "kNN memory",
    "vanilla_episodic_memory": "Episodic memory",
    "geometry_guided_cross_episode_failure_memory": "Geometry-guided",
}

SHORT_LABELS = {
    "no_memory": "No\nmemory",
    "knn_failure_memory": "kNN\nmemory",
    "vanilla_episodic_memory": "Episodic\nmemory",
    "geometry_guided_cross_episode_failure_memory": "Geometry-\nguided",
}

COLORS = {
    "no_memory": "#6f6f6f",
    "knn_failure_memory": "#4c78a8",
    "vanilla_episodic_memory": "#f58518",
    "geometry_guided_cross_episode_failure_memory": "#54a24b",
}


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    ylabel: str
    higher_is_better: bool
    value_getter: Callable[[dict[str, str]], float]
    value_format: str
    ylim: tuple[float, float] | None = None


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite for row={row}")
    return value


def _rate_pct(key: str) -> Callable[[dict[str, str]], float]:
    return lambda row: 100.0 * _float(row, key)


def _wasted_attempts_per_passage(row: dict[str, str]) -> float:
    attempts = _float(row, "wasted_false_feasible_attempts_round1")
    denom = _float(row, "n_same_false_feasible")
    if denom <= 0:
        raise ValueError(f"n_same_false_feasible must be positive for row={row}")
    return attempts / denom


PANEL_SPECS = [
    PanelSpec(
        key="passable_success_rate_pct",
        title="A. Passable success ↑",
        ylabel="Success (%)",
        higher_is_better=True,
        value_getter=_rate_pct("passable_success_rate"),
        value_format="{:.1f}",
        ylim=(0.0, 112.0),
    ),
    PanelSpec(
        key="transfer_reject_rate_pct",
        title="B. Transfer reject ↑",
        ylabel="Reject (%)",
        higher_is_better=True,
        value_getter=_rate_pct("transfer_reject_rate_on_similar_new_false_feasible"),
        value_format="{:.1f}",
        ylim=(0.0, 112.0),
    ),
    PanelSpec(
        key="interference_false_reject_rate_pct",
        title="C. Interference false reject ↓",
        ylabel="False reject (%)",
        higher_is_better=False,
        value_getter=_rate_pct("interference_false_reject_rate_on_similar_feasible"),
        value_format="{:.1f}",
        ylim=(0.0, 112.0),
    ),
    PanelSpec(
        key="wasted_ff_attempts_per_passage_round1",
        title="D. Wasted FF attempts/pass. ↓",
        ylabel="Attempts / passage",
        higher_is_better=False,
        value_getter=_wasted_attempts_per_passage,
        value_format="{:.2f}",
        ylim=(0.0, 1.12),
    ),
]


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"{path} has no rows")
    required = {
        "seed",
        "method",
        "n_rounds",
        "n_same_false_feasible",
        "n_transfer_false_feasible",
        "n_similar_feasible",
        "passable_success_rate",
        "wasted_false_feasible_attempts_round1",
        "transfer_reject_rate_on_similar_new_false_feasible",
        "interference_false_reject_rate_on_similar_feasible",
    }
    missing = sorted(required - set(reader.fieldnames or []))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    return rows


def summarize(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_method: dict[str, list[dict[str, str]]] = {method: [] for method in METHOD_ORDER}
    for row in rows:
        method = row["method"]
        if method in by_method:
            by_method[method].append(row)

    missing = [method for method, vals in by_method.items() if not vals]
    if missing:
        raise RuntimeError(f"Missing required methods in CSV: {missing}")

    summary: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        vals = by_method[method]
        seeds = sorted({int(float(row["seed"])) for row in vals})
        out: dict[str, object] = {
            "method": method,
            "display_label": DISPLAY_LABELS[method],
            "num_seeds": len(seeds),
            "seeds": ";".join(str(seed) for seed in seeds),
            "n_rounds": int(float(vals[0]["n_rounds"])),
            "n_same_false_feasible": int(float(vals[0]["n_same_false_feasible"])),
            "n_transfer_false_feasible": int(float(vals[0]["n_transfer_false_feasible"])),
            "n_similar_feasible": int(float(vals[0]["n_similar_feasible"])),
        }
        for spec in PANEL_SPECS:
            arr = np.asarray([spec.value_getter(row) for row in vals], dtype=np.float64)
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            out[f"{spec.key}_mean"] = float(np.mean(arr))
            out[f"{spec.key}_std"] = std
        summary.append(out)

    protocol = {
        "seeds": sorted({int(float(row["seed"])) for row in rows}),
        "n_rounds": sorted({int(float(row["n_rounds"])) for row in rows}),
        "n_same_false_feasible": sorted({int(float(row["n_same_false_feasible"])) for row in rows}),
        "n_transfer_false_feasible": sorted({int(float(row["n_transfer_false_feasible"])) for row in rows}),
        "n_similar_feasible": sorted({int(float(row["n_similar_feasible"])) for row in rows}),
        "preset": sorted({row.get("preset", "") for row in rows}),
    }
    return summary, protocol


def write_figure_data(summary: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "display_label",
        "num_seeds",
        "seeds",
        "n_rounds",
        "n_same_false_feasible",
        "n_transfer_false_feasible",
        "n_similar_feasible",
    ]
    for spec in PANEL_SPECS:
        fieldnames.extend([f"{spec.key}_mean", f"{spec.key}_std"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print(f"[write] {path}")


def plot(summary: list[dict[str, object]], figure_dir: Path) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / "memory_transfer_interference.pdf"
    png_path = figure_dir / "memory_transfer_interference.png"

    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.25), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.22, wspace=0.22, hspace=0.38)

    x = np.arange(len(METHOD_ORDER), dtype=np.float64)
    short_labels = [SHORT_LABELS[method] for method in METHOD_ORDER]
    handles = []
    for panel_idx, (ax, spec) in enumerate(zip(axes.reshape(-1), PANEL_SPECS)):
        means = np.asarray(
            [
                float(next(row for row in summary if row["method"] == method)[f"{spec.key}_mean"])
                for method in METHOD_ORDER
            ],
            dtype=np.float64,
        )
        stds = np.asarray(
            [
                float(next(row for row in summary if row["method"] == method)[f"{spec.key}_std"])
                for method in METHOD_ORDER
            ],
            dtype=np.float64,
        )
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3.0,
            color=[COLORS[method] for method in METHOD_ORDER],
            edgecolor="#222222",
            linewidth=0.5,
            width=0.72,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        if not handles:
            handles = list(bars)
        for bar, mean, std in zip(bars, means, stds):
            y = mean + std + (2.0 if "pct" in spec.key else 0.025)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                spec.value_format.format(mean),
                ha="center",
                va="bottom",
                fontsize=6.6,
            )
        ax.set_title(spec.title)
        ax.set_ylabel(spec.ylabel)
        ax.set_xticks(x)
        if panel_idx < 2:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xticklabels(short_labels)
        ax.grid(True, axis="y", alpha=0.25)
        if spec.ylim is not None:
            ax.set_ylim(*spec.ylim)
        elif np.any(means + stds > 0):
            ax.set_ylim(0.0, 1.12 * float(np.nanmax(means + stds)))

    fig.legend(
        handles,
        [DISPLAY_LABELS[method] for method in METHOD_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.3,
    )
    fig.text(
        0.5,
        0.045,
        "Transfer/interference are measured in the synthetic memory protocol; they do not establish broad real-world generalization.",
        ha="center",
        va="bottom",
        fontsize=7.0,
    )
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)
    print(f"[write] {pdf_path}")
    print(f"[write] {png_path}")
    return pdf_path, png_path


def write_metadata(
    *,
    path: Path,
    input_csv: Path,
    figure_data_csv: Path,
    pdf_path: Path,
    png_path: Path,
    protocol: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "figure": "memory transfer/interference comparison",
        "git_commit": _git_commit(),
        "raw_csv": _repo_relative(input_csv),
        "figure_data_csv": _repo_relative(figure_data_csv),
        "outputs": {
            "pdf": _repo_relative(pdf_path),
            "png": _repo_relative(png_path),
            "metadata": _repo_relative(path),
        },
        "methods": METHOD_ORDER,
        "display_labels": DISPLAY_LABELS,
        "seeds": protocol["seeds"],
        "rounds": protocol["n_rounds"],
        "denominators": {
            "passable_success_rate": "n_similar_feasible per seed",
            "transfer_reject_rate_on_similar_new_false_feasible": "n_transfer_false_feasible per seed",
            "interference_false_reject_rate_on_similar_feasible": "n_similar_feasible per seed",
            "wasted_false_feasible_attempts_per_passage_round1": "wasted_false_feasible_attempts_round1 / n_same_false_feasible per seed",
        },
        "protocol": {
            "n_same_false_feasible": protocol["n_same_false_feasible"],
            "n_transfer_false_feasible": protocol["n_transfer_false_feasible"],
            "n_similar_feasible": protocol["n_similar_feasible"],
            "preset": protocol["preset"],
        },
        "wasted_metric_definition": (
            "Panel D reports round-1 false-feasible traversal attempts divided by "
            "the number of same false-feasible passages for that seed. It is a "
            "wasted-attempt metric, not an unsafe-attempt metric."
        ),
        "aggregation_rule": (
            "Compute each metric per seed and method, then report mean bars with "
            "sample standard deviation error bars across seeds."
        ),
        "interpretation_note": (
            "Transfer and interference are measured in the synthetic memory protocol "
            "and do not establish broad real-world generalization."
        ),
        "excluded_metric_note": (
            "The saturated final false-feasible reject rate is intentionally not used "
            "as a central figure panel because memory baselines reach 100%."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[write] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the memory transfer/interference paper figure."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    summary, protocol = summarize(rows)
    figure_data_csv = args.table_dir / "memory_transfer_interference_figure_data.csv"
    write_figure_data(summary, figure_data_csv)
    pdf_path, png_path = plot(summary, args.figure_dir)
    write_metadata(
        path=args.figure_dir / "memory_transfer_interference.meta.json",
        input_csv=args.input_csv,
        figure_data_csv=figure_data_csv,
        pdf_path=pdf_path,
        png_path=png_path,
        protocol=protocol,
    )


if __name__ == "__main__":
    main()
