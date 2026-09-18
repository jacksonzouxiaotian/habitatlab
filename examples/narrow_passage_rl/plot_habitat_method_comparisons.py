#!/usr/bin/env python3
"""Plot paper-facing Habitat/MP3D method comparisons from saved results.

Official PointNav and the custom MP3D-derived narrow-passage protocol are
deliberately written to separate figures because their tasks, action spaces,
agent morphologies, and success definitions differ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
OUTPUT_DIR = DEFAULT_RESULTS / "narrow_passage_rl" / "figures"
OFFICIAL_DIR = DEFAULT_RESULTS / "official_pointnav_mp3d_v1_20260904"
NARROW_DIR = DEFAULT_RESULTS / "degnav_e2e_mp3d_narrow_v1_20260904"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#8E6CBA"
GRAY = "#8A8A8A"
LIGHT_GRAY = "#C7C7C7"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "png", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=320, bbox_inches="tight")
        outputs.append(path)
        print(f"[write] {path}")
    plt.close(fig)
    return outputs


def official_results(official_dir: Path) -> tuple[list[dict], list[Path]]:
    methods = [
        ("ForwardOnly", "forward_only", "Forward\nOnly", 1),
        ("Random Agent", "random", "Random\nAgent", 3),
        ("RandomForward", "random_forward", "Random\nForward", 3),
        ("GoalFollower", "goal_follower", "Goal\nFollower", 1),
        ("PointNav PPO", "pointnav_ppo", "PointNav\nPPO", 3),
        ("DD-PPO", "ddppo", "DD-PPO\ntransfer", 3),
        (
            "ShortestPathFollower",
            "shortest_path_follower",
            "ShortestPath\nOracle",
            1,
        ),
    ]
    outputs = []
    sources = []
    canonical_keys = None
    for name, directory, label, expected_seeds in methods:
        summary_path = official_dir / directory / "summary.csv"
        episode_path = official_dir / directory / "episodes.csv"
        rows = read_csv(summary_path)
        episodes = read_csv(episode_path)
        sources.extend((summary_path, episode_path))
        if len(rows) != expected_seeds or any(int(row["episodes"]) != 495 for row in rows):
            raise RuntimeError(f"incomplete official result for {name}")
        keys_by_seed: dict[int, set[tuple[str, str]]] = defaultdict(set)
        for row in episodes:
            keys_by_seed[int(row["eval_seed"])].add(
                (str(row["scene_id"]), str(row["episode_id"]))
            )
        if len(keys_by_seed) != expected_seeds or any(
            len(keys) != 495 for keys in keys_by_seed.values()
        ):
            raise RuntimeError(f"invalid official episode coverage for {name}")
        for keys in keys_by_seed.values():
            if canonical_keys is None:
                canonical_keys = keys
            elif keys != canonical_keys:
                raise RuntimeError(f"official episode-set mismatch for {name}")
        success = mean_std([float(row["success"]) for row in rows])
        spl = mean_std([float(row["spl"]) for row in rows])
        outputs.append(
            {
                "name": name,
                "label": label,
                "seeds": expected_seeds,
                "success": success,
                "spl": spl,
            }
        )
    return outputs, sources


def plot_official(rows: list[dict], output_dir: Path) -> list[Path]:
    labels = [row["label"] for row in rows]
    colors = [LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, GRAY, BLUE, GREEN, ORANGE]
    hatches = ["", "", "", "", "", "", "///"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.15))
    panels = (
        (axes[0], "success", "Success (%)", "(a) Goal-reaching success", 100.0),
        (axes[1], "spl", "SPL", "(b) Path-efficiency-weighted success", 1.0),
    )
    for ax, metric, ylabel, title, scale in panels:
        means = [scale * row[metric][0] for row in rows]
        errors = [scale * row[metric][1] for row in rows]
        bars = ax.bar(
            x,
            means,
            yerr=errors,
            capsize=3,
            color=colors,
            edgecolor="#444444",
            linewidth=0.65,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        for bar, value in zip(bars, means):
            label = f"{value:.1f}" if scale == 100.0 else f"{value:.2f}"
            offset = 1.6 if scale == 100.0 else 0.025
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=7.8,
            )
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", linewidth=0.45, alpha=0.30)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 109)
    axes[1].set_ylim(0, 1.09)
    fig.text(
        0.5,
        -0.015,
        "Official Habitat PointNav v1 / MP3D val (495 episodes per seed). "
        "DD-PPO uses a released Gibson-2+ checkpoint; ShortestPathFollower is a privileged NavMesh oracle.",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout(w_pad=2.0, rect=(0, 0.055, 1, 1))
    return save_figure(fig, output_dir / "habitat_official_pointnav_comparison")


def narrow_results(narrow_dir: Path) -> tuple[list[dict], list[Path]]:
    comparison_path = narrow_dir / "19d_comparison" / "formal_mp3d_comparison.csv"
    e2e_summary_path = narrow_dir / "summary_by_seed.csv"
    comparison = read_csv(comparison_path)
    e2e = read_csv(e2e_summary_path)
    sources = [comparison_path, e2e_summary_path]
    expected_19d = ("Geometry-19D", "kNN-memory-19D", "DEGNAV-memory-19D")
    by_name = {row["method"]: row for row in comparison}
    if tuple(by_name) != expected_19d or len(e2e) != 3:
        raise RuntimeError("incomplete MP3D-derived comparison")

    rows = []
    labels = {
        "Geometry-19D": "Geometry\n19D",
        "kNN-memory-19D": "kNN memory\n19D",
        "DEGNAV-memory-19D": "DEGNAV memory\n19D",
    }
    for method in expected_19d:
        row = by_name[method]
        if int(row["episodes"]) != 80:
            raise RuntimeError(f"invalid episode count for {method}")
        rows.append(
            {
                "name": method,
                "label": labels[method],
                "seeds": 1,
                "all_success": (float(row["success_rate"]), 0.0),
                "feasible_success": (float(row["feasible_success_rate"]), 0.0),
                "decision_accuracy": (float(row["decision_accuracy"]), 0.0),
                "correct_reject": (
                    float(row["false_feasible_correct_reject_rate"]),
                    0.0,
                ),
                "false_reject": (0.0, 0.0),
                "steps": (float(row["avg_steps"]), 0.0),
            }
        )
    rows.append(
        {
            "name": "DEGNAV-E2E",
            "label": "DEGNAV-E2E\nraw Depth",
            "seeds": 3,
            "all_success": mean_std([float(row["success_rate_all"]) for row in e2e]),
            "feasible_success": mean_std(
                [float(row["success_rate_feasible"]) for row in e2e]
            ),
            "decision_accuracy": mean_std(
                [float(row["decision_accuracy"]) for row in e2e]
            ),
            "correct_reject": mean_std(
                [float(row["correct_reject_rate_infeasible"]) for row in e2e]
            ),
            "false_reject": mean_std(
                [float(row["false_reject_rate_feasible"]) for row in e2e]
            ),
            "steps": mean_std([float(row["mean_steps"]) for row in e2e]),
        }
    )
    return rows, sources


def plot_narrow(rows: list[dict], output_dir: Path) -> list[Path]:
    x = np.arange(len(rows))
    labels = [row["label"] for row in rows]
    fig, (ax_metric, ax_steps) = plt.subplots(
        1, 2, figsize=(11.1, 4.25), gridspec_kw={"width_ratios": [1.72, 1.0]}
    )
    metrics = (
        ("all_success", "All Success", BLUE),
        ("feasible_success", "Feasible Success", GREEN),
        ("decision_accuracy", "Decision accuracy", PURPLE),
        ("correct_reject", "Candidate Correct Reject", ORANGE),
    )
    width = 0.19
    for index, (key, label, color) in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        means = [100.0 * row[key][0] for row in rows]
        errors = [100.0 * row[key][1] for row in rows]
        bars = ax_metric.bar(
            x + offset,
            means,
            width=width,
            yerr=errors,
            capsize=2.5,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        if key == "correct_reject":
            for bar, value in zip(bars, means):
                if value > 0:
                    ax_metric.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 3.2,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color="#4A3A00",
                    )
    ax_metric.set_title("(a) Navigation and decision quality", loc="left")
    ax_metric.set_ylabel("Rate (%)")
    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels(labels)
    ax_metric.set_ylim(0, 109)
    ax_metric.grid(True, axis="y", linewidth=0.45, alpha=0.30)
    ax_metric.set_axisbelow(True)
    ax_metric.legend(loc="lower left", ncol=2, frameon=True)
    e2e_false_reject = rows[-1]["false_reject"]
    ax_metric.text(
        0.99,
        0.02,
        f"Feasible False Reject (E2E):\n{100 * e2e_false_reject[0]:.2f}±{100 * e2e_false_reject[1]:.2f}%",
        transform=ax_metric.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.75"},
    )

    step_means = [row["steps"][0] for row in rows]
    step_errors = [row["steps"][1] for row in rows]
    step_colors = [LIGHT_GRAY, LIGHT_GRAY, GRAY, BLUE]
    y = np.arange(len(rows))
    bars = ax_steps.barh(
        y,
        step_means,
        xerr=step_errors,
        capsize=3,
        color=step_colors,
        edgecolor="#444444",
        linewidth=0.6,
    )
    for bar, value in zip(bars, step_means):
        ax_steps.text(
            value + 3,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=8,
        )
    ax_steps.set_title("(b) Execution efficiency", loc="left")
    ax_steps.set_xlabel("Mean episode steps (lower is better)")
    ax_steps.set_yticks(y)
    ax_steps.set_yticklabels(labels)
    ax_steps.set_xlim(0, max(step_means) * 1.24)
    ax_steps.invert_yaxis()
    ax_steps.grid(True, axis="x", linewidth=0.45, alpha=0.30)
    ax_steps.set_axisbelow(True)

    fig.text(
        0.5,
        -0.012,
        "Custom MP3D-derived narrow-passage validation (80 fixed episodes). "
        "E2E bars show mean ± population SD over three seeds; 19D rows are deterministic.",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout(w_pad=2.2, rect=(0, 0.055, 1, 1))
    return save_figure(fig, output_dir / "habitat_mp3d_narrow_method_comparison")


def candidate_outcomes(narrow_dir: Path) -> tuple[list[dict], list[Path]]:
    method_files = (
        ("Geometry 19D", narrow_dir / "geometry_19d_episodes.csv", "19d"),
        ("kNN memory 19D", narrow_dir / "knn_memory_19d_episodes.csv", "19d"),
        ("DEGNAV memory 19D", narrow_dir / "degnav_memory_19d_episodes.csv", "19d"),
    )
    outputs = []
    sources = []
    canonical_keys = None
    for label, path, kind in method_files:
        del kind
        rows = read_csv(path)
        sources.append(path)
        keys = {(row["scene_id"], row["episode_id"]) for row in rows}
        if len(rows) != 80 or len(keys) != 80:
            raise RuntimeError(f"invalid 19D episode file: {path}")
        if canonical_keys is None:
            canonical_keys = keys
        elif keys != canonical_keys:
            raise RuntimeError("19D episode-set mismatch")
        candidate = [row for row in rows if float(row["false_feasible"]) > 0.5]
        traversed = sum(float(row["success"]) > 0.5 for row in candidate)
        rejected = sum(float(row["rejected"]) > 0.5 for row in candidate)
        unresolved = len(candidate) - traversed - rejected
        if len(candidate) != 16 or unresolved < 0:
            raise RuntimeError(f"invalid candidate outcomes for {label}")
        outputs.append(
            {
                "label": label,
                "n": len(candidate),
                "traversed": traversed,
                "correct_reject": rejected,
                "unresolved": unresolved,
            }
        )

    e2e_rows = []
    e2e_sets = []
    for seed in (1701, 1702, 1703):
        path = narrow_dir / f"seed{seed}" / "eval" / "episodes.csv"
        rows = read_csv(path)
        sources.append(path)
        keys = {(row["scene_id"], row["episode_id"]) for row in rows}
        if len(rows) != 80 or len(keys) != 80:
            raise RuntimeError(f"invalid E2E episode file: {path}")
        e2e_sets.append(keys)
        e2e_rows.extend(
            row for row in rows if float(row["morphology_infeasible"]) > 0.5
        )
    if not all(keys == e2e_sets[0] for keys in e2e_sets):
        raise RuntimeError("E2E seed episode-set mismatch")
    traversed = sum(float(row["success"]) > 0.5 for row in e2e_rows)
    rejected = sum(float(row["correct_reject"]) > 0.5 for row in e2e_rows)
    unresolved = len(e2e_rows) - traversed - rejected
    if len(e2e_rows) != 48 or unresolved < 0:
        raise RuntimeError("invalid E2E candidate outcome coverage")
    outputs.append(
        {
            "label": "DEGNAV-E2E\nraw Depth",
            "n": len(e2e_rows),
            "traversed": traversed,
            "correct_reject": rejected,
            "unresolved": unresolved,
        }
    )
    return outputs, sources


def plot_candidate_outcomes(rows: list[dict], output_dir: Path) -> list[Path]:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    categories = (
        ("traversed", "Goal reached", GREEN),
        ("correct_reject", "Explicit correct reject", BLUE),
        ("unresolved", "Neither / unresolved failure", LIGHT_GRAY),
    )
    fig, ax = plt.subplots(figsize=(7.9, 4.45))
    bottom = np.zeros(len(rows), dtype=np.float64)
    for key, label, color in categories:
        values = np.asarray([100.0 * row[key] / row["n"] for row in rows])
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=label,
        )
        for bar, value, base in zip(bars, values, bottom):
            if value >= 4.5:
                text_color = "white" if color in (GREEN, BLUE) else "#333333"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=text_color,
                    fontweight="semibold",
                )
            elif value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() + 0.015,
                    base + value / 2,
                    f"{value:.1f}%",
                    ha="left",
                    va="center",
                    fontsize=7.2,
                    color="#333333",
                )
        bottom += values
    for index, row in enumerate(rows):
        suffix = " (3 seeds)" if row["n"] == 48 else ""
        ax.text(index, -7.0, f"n={row['n']}{suffix}", ha="center", va="top", fontsize=8)
    ax.set_title("Candidate-infeasible outcome decomposition", loc="left")
    ax.set_ylabel("Candidate outcomes (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-12, 104)
    ax.grid(True, axis="y", linewidth=0.45, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False)
    fig.text(
        0.5,
        -0.01,
        "The candidate-infeasible tag is a generator label, not a proof of physical impossibility; "
        "goal-reaching and explicit rejection are therefore reported separately.",
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return save_figure(fig, output_dir / "habitat_mp3d_candidate_outcomes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, default=OFFICIAL_DIR)
    parser.add_argument("--narrow-dir", type=Path, default=NARROW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    official, official_sources = official_results(args.official_dir)
    narrow, narrow_sources = narrow_results(args.narrow_dir)
    outcomes, outcome_sources = candidate_outcomes(args.narrow_dir)
    figure_paths = []
    figure_paths.extend(plot_official(official, args.output_dir))
    figure_paths.extend(plot_narrow(narrow, args.output_dir))
    figure_paths.extend(plot_candidate_outcomes(outcomes, args.output_dir))

    sources = sorted(set(official_sources + narrow_sources + outcome_sources))
    metadata = {
        "generated_figures": [str(path) for path in figure_paths],
        "source_sha256": {str(path): sha256_file(path) for path in sources},
        "official_pointnav": {
            "protocol": "official Habitat PointNav v1 / MP3D val",
            "episodes_per_seed": 495,
            "methods": [row["name"] for row in official],
            "comparison_boundary": "must not be numerically ranked against the custom narrow-passage figure",
        },
        "mp3d_derived_narrow_passage": {
            "protocol": "custom MP3D-derived narrow-passage validation",
            "episodes_per_method_seed": 80,
            "methods": [row["name"] for row in narrow],
            "e2e_seeds": [1701, 1702, 1703],
            "candidate_outcomes": outcomes,
        },
        "strict_success_plotted": False,
    }
    metadata_path = args.output_dir / "habitat_method_comparisons.meta.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[write] {metadata_path}")


if __name__ == "__main__":
    main()
