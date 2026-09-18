#!/usr/bin/env python3
"""Characterize two known weaknesses of the current SH belief update.

These are diagnostic/characterization experiments, not paper benchmark results:

1. repeated, perfectly correlated observations are treated as independent
   evidence, so coefficient magnitude grows and confidence saturates;
2. propagation models rotation but not translation, so bearing becomes stale
   when the target is occluded during lateral agent motion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

# The workstation's normal Matplotlib config directory is outside this
# isolated reproduction's writable roots.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/eagor_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eagor_repro.evaluation.metrics import angular_error_deg
from eagor_repro.spherical.belief_filter import SphericalHarmonicBeliefFilter
from eagor_repro.spherical.spherical_grid import SphericalGrid, sphere_to_erp


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_belief(
    axis: Any,
    belief: np.ndarray,
    title: str,
    predicted: np.ndarray | None = None,
    ground_truth: np.ndarray | None = None,
) -> None:
    height, width = belief.shape
    axis.imshow(
        belief,
        cmap="turbo",
        origin="upper",
        extent=(-180.0, 180.0, -90.0, 90.0),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    for direction, color, marker, label in (
        (predicted, "red", "+", "prediction"),
        (ground_truth, "lime", "x", "ground truth"),
    ):
        if direction is None:
            continue
        u, v = sphere_to_erp(direction, width, height)
        azimuth = float(u) / width * 360.0 - 180.0
        elevation = 90.0 - float(v) / height * 180.0
        axis.scatter(
            [azimuth],
            [elevation],
            c=color,
            marker=marker,
            s=90,
            linewidths=2,
            label=label,
        )
    axis.set(
        title=title,
        xlabel="azimuth (deg)",
        ylabel="elevation (deg)",
        xlim=(-180.0, 180.0),
        ylim=(-90.0, 90.0),
    )


def repeated_correlated_evidence(output_root: Path) -> Dict[str, Any]:
    """Repeat one unchanged observation and trace unjustified certainty."""

    grid = SphericalGrid(64, 128)
    target_azimuth_deg = 35.0
    target = np.asarray(
        [
            np.cos(np.deg2rad(target_azimuth_deg)),
            np.sin(np.deg2rad(target_azimuth_deg)),
            0.0,
        ],
        np.float64,
    )
    # A deliberately broad, plausible single-frame observation makes the
    # confidence change caused by duplicate evidence easy to see.
    likelihood = grid.gaussian_likelihood(target, sigma_deg=30.0)
    repeated = SphericalHarmonicBeliefFilter(
        grid, bandlimit=7, update_mode="paper", decode_mode="probability"
    )
    observe_once = SphericalHarmonicBeliefFilter(
        grid, bandlimit=7, update_mode="paper", decode_mode="probability"
    )

    rows: List[Dict[str, Any]] = []
    snapshots: Dict[int, np.ndarray] = {}
    snapshot_steps = (1, 5, 20, 60)
    for step in range(1, 61):
        repeated_estimate = repeated.update(likelihood, True, np.eye(3))
        once_estimate = observe_once.update(
            likelihood if step == 1 else None,
            step == 1,
            np.eye(3),
        )
        rows.append(
            {
                "step": step,
                "coefficient_l2": float(np.linalg.norm(repeated.coefficients)),
                "directional_coefficient_l2": float(
                    np.linalg.norm(repeated.coefficients[1:])
                ),
                "repeated_confidence": float(repeated_estimate.confidence),
                "observe_once_confidence": float(once_estimate.confidence),
                "angular_error_deg": float(
                    angular_error_deg(repeated_estimate.direction_xyz, target)
                ),
            }
        )
        if step in snapshot_steps:
            snapshots[step] = repeated_estimate.belief_heatmap.copy()

    csv_path = output_root / "weakness_1_repeated_evidence.csv"
    _write_csv(csv_path, rows)
    figure_path = output_root / "weakness_1_repeated_evidence.png"
    figure = plt.figure(figsize=(14, 7), constrained_layout=True)
    grid_spec = figure.add_gridspec(2, 4)
    norm_axis = figure.add_subplot(grid_spec[0, :2])
    confidence_axis = figure.add_subplot(grid_spec[0, 2:])
    steps = np.asarray([row["step"] for row in rows])
    norm_axis.plot(
        steps,
        [row["coefficient_l2"] for row in rows],
        color="#235789",
        linewidth=2,
        label=r"$\|c_t\|_2$",
    )
    norm_axis.plot(
        steps,
        [row["directional_coefficient_l2"] for row in rows],
        color="#d95f02",
        linewidth=2,
        label=r"$\|c_{t,\ell>0}\|_2$",
    )
    norm_axis.set(
        title="Duplicate frames make SH coefficient magnitude grow linearly",
        xlabel="repeated identical observations",
        ylabel="coefficient L2 norm",
    )
    norm_axis.grid(alpha=0.25)
    norm_axis.legend()

    confidence_axis.plot(
        steps,
        [row["repeated_confidence"] for row in rows],
        color="#c1292e",
        linewidth=2,
        label="update with the same frame every step",
    )
    confidence_axis.plot(
        steps,
        [row["observe_once_confidence"] for row in rows],
        "--",
        color="#2a9d8f",
        linewidth=2,
        label="observe once, then propagate",
    )
    confidence_axis.set(
        title="Confidence saturates although no independent evidence is added",
        xlabel="timestep",
        ylabel="resultant confidence",
        ylim=(0.0, 1.02),
    )
    confidence_axis.grid(alpha=0.25)
    confidence_axis.legend(loc="lower right")

    for column, step in enumerate(snapshot_steps):
        axis = figure.add_subplot(grid_spec[1, column])
        _plot_belief(axis, snapshots[step], f"posterior after {step} duplicate(s)")
        if column:
            axis.set_ylabel("")
    figure.suptitle(
        "Weakness 1 — temporally correlated observations are over-counted",
        fontsize=15,
    )
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)

    first, last = rows[0], rows[-1]
    return {
        "figure": str(figure_path),
        "csv": str(csv_path),
        "steps": len(rows),
        "coefficient_norm_step_1": first["coefficient_l2"],
        "coefficient_norm_step_60": last["coefficient_l2"],
        "coefficient_norm_ratio": last["coefficient_l2"]
        / first["coefficient_l2"],
        "confidence_step_1": first["repeated_confidence"],
        "confidence_step_60": last["repeated_confidence"],
        "observe_once_confidence_step_60": last["observe_once_confidence"],
        "interpretation": (
            "The paper-style additive update treats identical adjacent frames "
            "as independent evidence; direction stays correct but certainty is "
            "not calibrated and coefficient magnitude is unbounded."
        ),
    }


def translation_without_parallax(output_root: Path) -> Dict[str, Any]:
    """Move laterally under occlusion while the filter propagates rotation only."""

    grid = SphericalGrid(64, 128)
    belief_filter = SphericalHarmonicBeliefFilter(
        grid, bandlimit=7, update_mode="paper", decode_mode="probability"
    )
    target_world = np.asarray([4.0, 0.0, 0.0], np.float64)
    agent_y = np.linspace(0.0, 4.0, 41)
    rows: List[Dict[str, Any]] = []
    initial_coefficients: np.ndarray | None = None
    initial_belief: np.ndarray | None = None
    final_belief: np.ndarray | None = None
    final_prediction: np.ndarray | None = None
    final_truth: np.ndarray | None = None

    for step, y_position in enumerate(agent_y):
        agent_world = np.asarray([0.0, y_position, 0.0], np.float64)
        target_delta = target_world - agent_world
        truth = target_delta / np.linalg.norm(target_delta)
        if step == 0:
            likelihood = grid.gaussian_likelihood(truth, sigma_deg=12.0)
            estimate = belief_filter.update(likelihood, True, np.eye(3))
            initial_coefficients = belief_filter.coefficients.copy()
            initial_belief = estimate.belief_heatmap.copy()
        else:
            # Identity rotation is correct because the body heading does not
            # change.  There is intentionally no translation input in SH-BF.
            estimate = belief_filter.update(None, False, np.eye(3))
        error = angular_error_deg(estimate.direction_xyz, truth)
        rows.append(
            {
                "step": step,
                "lateral_translation_m": float(y_position),
                "true_azimuth_deg": float(
                    np.rad2deg(np.arctan2(truth[1], truth[0]))
                ),
                "predicted_azimuth_deg": float(np.rad2deg(estimate.azimuth)),
                "angular_error_deg": float(error),
                "confidence": float(estimate.confidence),
                "coefficient_change_l2": float(
                    np.linalg.norm(
                        belief_filter.coefficients - initial_coefficients
                    )
                ),
            }
        )
        final_belief = estimate.belief_heatmap.copy()
        final_prediction = estimate.direction_xyz.copy()
        final_truth = truth.copy()

    assert initial_belief is not None
    assert final_belief is not None
    assert final_prediction is not None
    assert final_truth is not None
    csv_path = output_root / "weakness_2_translation_parallax.csv"
    _write_csv(csv_path, rows)
    figure_path = output_root / "weakness_2_translation_parallax.png"
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    path_axis = axes[0, 0]
    path_axis.plot(np.zeros_like(agent_y), agent_y, "k-", linewidth=2, label="agent path")
    path_axis.scatter([0.0], [0.0], c="#2a9d8f", s=80, label="start")
    path_axis.scatter([0.0], [agent_y[-1]], c="#c1292e", s=80, label="end")
    path_axis.scatter(
        [target_world[0]], [target_world[1]], marker="*", c="#f4a261", s=180, label="target"
    )
    path_axis.plot([0.0, target_world[0]], [0.0, target_world[1]], "--", color="0.5")
    path_axis.plot(
        [0.0, target_world[0]],
        [agent_y[-1], target_world[1]],
        "--",
        color="0.5",
    )
    path_axis.set(
        title="Stationary target, lateral agent motion under occlusion",
        xlabel="world x (m)",
        ylabel="world y (m)",
        xlim=(-0.5, 4.5),
        ylim=(-0.5, 4.5),
        aspect="equal",
    )
    path_axis.grid(alpha=0.25)
    path_axis.legend(loc="upper right")

    error_axis = axes[0, 1]
    error_axis.plot(
        agent_y,
        [row["angular_error_deg"] for row in rows],
        color="#c1292e",
        linewidth=2,
        label="rotation-only SH-BF",
    )
    error_axis.plot(agent_y, np.zeros_like(agent_y), "--", color="0.4", label="translation-aware ideal")
    error_axis.set(
        title="Bearing error grows while coefficients remain unchanged",
        xlabel="lateral translation during occlusion (m)",
        ylabel="angular error (deg)",
        ylim=(-2.0, 50.0),
    )
    error_axis.grid(alpha=0.25)
    error_axis.legend()

    _plot_belief(
        axes[1, 0],
        initial_belief,
        "initial observation",
        predicted=np.asarray([1.0, 0.0, 0.0]),
        ground_truth=np.asarray([1.0, 0.0, 0.0]),
    )
    _plot_belief(
        axes[1, 1],
        final_belief,
        "after 4 m translation with no observation",
        predicted=final_prediction,
        ground_truth=final_truth,
    )
    axes[1, 1].legend(loc="lower right", fontsize=8)
    figure.suptitle(
        "Weakness 2 — rotation-equivariance does not compensate translational parallax",
        fontsize=15,
    )
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)

    return {
        "figure": str(figure_path),
        "csv": str(csv_path),
        "occluded_translation_m": float(agent_y[-1]),
        "final_true_azimuth_deg": rows[-1]["true_azimuth_deg"],
        "final_predicted_azimuth_deg": rows[-1]["predicted_azimuth_deg"],
        "final_angular_error_deg": rows[-1]["angular_error_deg"],
        "max_coefficient_change_l2": float(
            max(row["coefficient_change_l2"] for row in rows)
        ),
        "interpretation": (
            "With no body rotation and no observation, SH-BF leaves all "
            "coefficients unchanged even though translation changes the true "
            "agent-to-target bearing."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("eagor_outputs/belief_weaknesses"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "synthetic characterization; not a paper benchmark result",
        "weakness_1": repeated_correlated_evidence(args.output_root),
        "weakness_2": translation_without_parallax(args.output_root),
    }
    report_path = args.output_root / "belief_weaknesses_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
