#!/usr/bin/env python3
"""Draw a depth-to-19D-feature diagram for NarrowPassageNav."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_PATH = ROOT / "habitat-lab" / "habitat" / "tasks" / "narrow_passage" / "geometry.py"
_spec = importlib.util.spec_from_file_location("_narrow_passage_geometry", GEOMETRY_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load geometry module from {GEOMETRY_PATH}")
_geometry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _geometry
_spec.loader.exec_module(_geometry)

FEATURE_NAMES = _geometry.FEATURE_NAMES
NarrowPassageState = _geometry.NarrowPassageState
depth_to_passage_features = _geometry.depth_to_passage_features


def synthetic_depth(height: int = 180, width: int = 270) -> np.ndarray:
    """Create one corridor-like depth image in metres."""
    y = np.linspace(0.0, 1.0, height)[:, None]
    x = np.linspace(-1.0, 1.0, width)[None, :]
    depth = 4.2 - 2.0 * y + 0.25 * np.cos(np.pi * x)

    # Near side walls are closer than the center opening.
    side_wall = np.abs(x) > (0.36 + 0.38 * y)
    depth = np.where(side_wall, 0.34 + 0.65 * (1.0 - y), depth)

    # A slightly off-center closer obstacle makes left/right clearance differ.
    obstacle = ((x + 0.43) / 0.20) ** 2 + ((y - 0.66) / 0.18) ** 2 < 1.0
    depth = np.where(obstacle, 0.42, depth)

    # Keep a navigable-looking central slit.
    slit = np.abs(x - 0.05) < (0.18 + 0.05 * (1.0 - y))
    depth = np.where(slit, np.maximum(depth, 2.35 - 0.75 * y), depth)
    return np.clip(depth.astype(np.float32), 0.05, 5.0)


def roi_specs(height: int, width: int) -> list[tuple[str, slice, slice]]:
    near = slice(int(0.55 * height), int(0.9 * height))
    far = slice(int(0.25 * height), int(0.55 * height))
    left = slice(0, int(width / 3))
    center = slice(int(width / 3), int(2 * width / 3))
    right = slice(int(2 * width / 3), width)
    return [
        ("d_left_far", far, left),
        ("d_center_far", far, center),
        ("d_right_far", far, right),
        ("d_left_near", near, left),
        ("d_center_near", near, center),
        ("d_right_near", near, right),
    ]


def group_color(index: int) -> str:
    if index <= 5:
        return "#D9EAF7"
    if index <= 9:
        return "#E5F4DD"
    if index <= 12:
        return "#FFF0C9"
    return "#F1E5F7"


def group_name(index: int) -> str:
    if index <= 5:
        return "depth sectors"
    if index <= 9:
        return "passage geometry"
    if index <= 12:
        return "pose / goal state"
    return "motion / risk history"


def add_feature_table(ax, features: np.ndarray) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        1.02,
        "19-D narrow_passage_features",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.975,
        "depth sectors + geometry + robot state",
        fontsize=8.5,
        color="#4A4A4A",
        transform=ax.transAxes,
    )

    row_h = 0.045
    y0 = 0.905
    for i, (name, value) in enumerate(zip(FEATURE_NAMES, features)):
        y = y0 - i * row_h
        rect = patches.FancyBboxPatch(
            (0.0, y - row_h * 0.72),
            0.98,
            row_h * 0.72,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            linewidth=0.35,
            edgecolor="#A8A8A8",
            facecolor=group_color(i),
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            0.02,
            y - row_h * 0.45,
            f"x[{i:02d}] {name}",
            fontsize=7.6,
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.96,
            y - row_h * 0.45,
            f"{float(value):.2f}",
            fontsize=7.6,
            ha="right",
            va="center",
            family="monospace",
            transform=ax.transAxes,
        )


def plot(output: Path, pdf: bool = True) -> None:
    depth = synthetic_depth()
    state = NarrowPassageState(
        distance_to_local_goal=2.80,
        heading_error=0.24,
        lateral_offset=-0.08,
        current_vx=0.12,
        current_wz=0.18,
        stuck_score=0.08,
        collision_flag=0.0,
        previous_action_vx=0.10,
        previous_action_wz=0.12,
        robot_radius=0.18,
    )
    features = depth_to_passage_features(depth, state=state, max_depth=5.0)
    height, width = depth.shape

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 160,
        }
    )
    fig = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.28, 0.95, 1.25], height_ratios=[1, 0.58])
    ax_depth = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[:, 1])
    ax_vec = fig.add_subplot(gs[:, 2])
    ax_formula = fig.add_subplot(gs[1, 0])

    im = ax_depth.imshow(depth, cmap="viridis_r", vmin=0.0, vmax=5.0)
    ax_depth.set_title("Example depth map partition")
    ax_depth.set_xticks([])
    ax_depth.set_yticks([])
    for name, rows, cols in roi_specs(height, width):
        rect = patches.Rectangle(
            (cols.start, rows.start),
            cols.stop - cols.start,
            rows.stop - rows.start,
            fill=False,
            linewidth=1.8,
            edgecolor="white",
        )
        ax_depth.add_patch(rect)
        ax_depth.text(
            0.5 * (cols.start + cols.stop),
            0.5 * (rows.start + rows.stop),
            name.replace("_", "\n"),
            color="white",
            fontsize=8.2,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.22", "fc": "#00000088", "ec": "none"},
        )
    cbar = fig.colorbar(im, ax=ax_depth, shrink=0.82, pad=0.012)
    cbar.set_label("depth (m)")

    ax_formula.set_axis_off()
    ax_formula.text(0.0, 1.0, "Depth ROIs", fontsize=11, fontweight="bold", transform=ax_formula.transAxes)
    ax_formula.text(
        0.0,
        0.78,
        "Each sector uses the 10th percentile of valid pixels.",
        fontsize=9,
        color="#444444",
        transform=ax_formula.transAxes,
    )
    ax_formula.text(
        0.0,
        0.53,
        "near rows: 55%-90% image height\nfar rows: 25%-55% image height\ncolumns: left / center / right thirds",
        fontsize=8.6,
        transform=ax_formula.transAxes,
        linespacing=1.35,
    )

    ax_mid.set_axis_off()
    ax_mid.set_title("Feature construction", pad=8)
    formulas = [
        ("x[0:6]", "six depth-sector minima"),
        ("x[6]", "clearance_left = min(left_near, left_far)"),
        ("x[7]", "clearance_right = min(right_near, right_far)"),
        ("x[8]", "passage_width = clearance_left + clearance_right"),
        ("x[9]", "body_margin = min(clearance_left, clearance_right) - robot_radius"),
        ("x[10:13]", "heading error, lateral offset, local goal distance"),
        ("x[13:19]", "current velocity, stuck/collision flags, previous action"),
    ]
    for j, (idx, text) in enumerate(formulas):
        y = 0.91 - j * 0.12
        box = patches.FancyBboxPatch(
            (0.02, y - 0.065),
            0.96,
            0.085,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=0.65,
            edgecolor="#9B9B9B",
            facecolor="#F8F8F8",
            transform=ax_mid.transAxes,
        )
        ax_mid.add_patch(box)
        ax_mid.text(0.05, y - 0.022, idx, fontsize=8.4, fontweight="bold", transform=ax_mid.transAxes)
        ax_mid.text(0.30, y - 0.022, text, fontsize=7.4, transform=ax_mid.transAxes)
    ax_mid.annotate(
        "",
        xy=(0.0, 0.62),
        xytext=(-0.18, 0.62),
        xycoords=ax_mid.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"},
    )
    ax_mid.annotate(
        "",
        xy=(1.05, 0.62),
        xytext=(0.98, 0.62),
        xycoords=ax_mid.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"},
    )

    add_feature_table(ax_vec, features)
    fig.suptitle(
        "Depth Image to 19-D Narrow-Passage Observation",
        fontsize=15,
        fontweight="bold",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if pdf:
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/narrow_passage_rl/results/narrow_passage_rl/figures/depth_19d_feature_diagram.png"),
    )
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    plot(args.output, pdf=not args.no_pdf)
    print(args.output)
    if not args.no_pdf:
        print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
