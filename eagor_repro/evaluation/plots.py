"""Post-hoc plots from saved confidence/error rows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eagor_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def reliability_diagram(
    rows: Iterable[Mapping[str, float]],
    output_path: str | Path,
    success_angle_deg: float = 15.0,
    bins: int = 10,
) -> Path:
    rows = list(rows)
    confidence = np.asarray([float(row["confidence"]) for row in rows])
    correct = np.asarray(
        [float(row["angular_error_deg"]) <= success_angle_deg for row in rows]
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers, accuracy = [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper + 1e-12)
        if mask.any():
            centers.append(float(confidence[mask].mean()))
            accuracy.append(float(correct[mask].mean()))
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="0.5")
    axis.plot(centers, accuracy, marker="o")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Angular success rate")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def baseline_comparison_plot(
    rows: Iterable[Mapping[str, float]], output_path: str | Path
) -> Path:
    """Render a compact matched-policy comparison from aggregate rows."""

    rows = list(rows)
    if not rows:
        raise ValueError("baseline comparison needs at least one row")
    labels = [str(row["policy_method"]) for row in rows]
    metrics = (
        ("mae_deg", "Mean angular error (deg)"),
        (
            "temporal_consistency_deg_per_step",
            "Temporal inconsistency (deg/step)",
        ),
        ("collision_count", "Mean collisions per episode"),
        (
            "average_inference_update_latency_ms",
            "Per-step perception + update latency (ms)",
        ),
    )
    colors = ("#6c757d", "#2a9d8f", "#e9c46a", "#c1292e")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        values = np.asarray([float(row[metric]) for row in rows], np.float64)
        bars = axis.bar(labels, values, color=colors[: len(rows)])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    episodes = int(float(rows[0].get("episodes", 0)))
    success = ", ".join(
        f"{label}={100.0 * float(row['success_rate']):.0f}%"
        for label, row in zip(labels, rows)
    )
    figure.suptitle(
        f"MP3D Oracle diagnostic, {episodes} unique scenes — SR: {success}\n"
        "Lower is better in all four panels; this is not a paper benchmark",
        fontsize=13,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path
