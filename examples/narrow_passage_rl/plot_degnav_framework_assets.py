#!/usr/bin/env python3
"""Generate transparent vector assets for the DEGNav architecture figure.

The assets are method schematics, not experimental result plots.  Their labels
and geometry match the current strict implementation: an OBB morphology of
0.36 x 0.60 m with 0.03 m two-sided decision clearance, interval gating,
bounded geometry-memory evidence, and a calibration-only outcome posterior.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Arc,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)
from matplotlib.transforms import Affine2D  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "examples/narrow_passage_rl/results/narrow_passage_rl/figures/framework_assets_20260822"
)

BLUE = "#2C6E9B"
TEAL = "#2A9D8F"
GREEN = "#54A24B"
ORANGE = "#F58518"
RED = "#D64B3C"
PURPLE = "#7851A9"
GRAY = "#7F8C8D"
DARK = "#263238"
WALL = "#D9DEE3"


def _setup(figsize=(4.2, 3.0)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    return fig, ax


def _rounded(ax, xywh, text, *, fc, ec=DARK, fontsize=9, weight="bold"):
    x, y, w, h = xywh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=DARK,
    )


def _arrow(ax, start, end, *, color=BLUE, style="-", width=1.6):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=width,
            linestyle=style,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def draw_obb_yaw():
    fig, ax = _setup((4.0, 3.2))
    ax.add_patch(Rectangle((0.10, 0.05), 0.17, 0.90, facecolor=WALL, edgecolor="none"))
    ax.add_patch(Rectangle((0.73, 0.05), 0.17, 0.90, facecolor=WALL, edgecolor="none"))
    ax.plot([0.27, 0.27], [0.05, 0.95], color=DARK, linewidth=2.0)
    ax.plot([0.73, 0.73], [0.05, 0.95], color=DARK, linewidth=2.0)
    ax.annotate(
        "",
        xy=(0.73, 0.91),
        xytext=(0.27, 0.91),
        arrowprops=dict(arrowstyle="<->", color=BLUE, linewidth=1.8),
    )
    ax.text(0.50, 0.935, r"observed width $\widehat D_t$", ha="center", fontsize=9, color=BLUE)

    cx, cy = 0.50, 0.48
    body_w, body_l = 0.20, 0.42
    rect = Rectangle(
        (cx - body_w / 2, cy - body_l / 2),
        body_w,
        body_l,
        facecolor="#F7C9C3",
        edgecolor=RED,
        linewidth=2.0,
    )
    rect.set_transform(Affine2D().rotate_deg_around(cx, cy, -35) + ax.transData)
    ax.add_patch(rect)
    ax.arrow(cx, cy, 0.0, 0.25, width=0.005, head_width=0.035, color=RED, length_includes_head=True)
    ax.add_patch(Arc((cx, cy), 0.28, 0.28, angle=0, theta1=55, theta2=90, color=PURPLE, linewidth=1.6))
    ax.text(0.56, 0.61, r"$\psi_t$", fontsize=11, color=PURPLE)
    ax.annotate(
        "",
        xy=(0.68, 0.25),
        xytext=(0.32, 0.25),
        arrowprops=dict(arrowstyle="<->", color=PURPLE, linewidth=2.0),
    )
    ax.text(0.50, 0.18, r"$W_p(\psi)=A|\cos\psi|+B|\sin\psi|+2m$", ha="center", fontsize=9, color=PURPLE)
    ax.text(0.50, 0.075, r"$A=.36$ m, $B=.60$ m, $m=.03$ m", ha="center", fontsize=8.5, color=DARK)
    return fig


def draw_interval_belief():
    fig, ax = _setup((4.2, 3.0))
    x = np.linspace(-0.16, 0.20, 400)
    mu, sigma = 0.035, 0.045
    y = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    y /= y.max()
    xx = 0.10 + 0.80 * (x - x.min()) / (x.max() - x.min())
    yy = 0.18 + 0.58 * y
    ax.plot(xx, yy, color=TEAL, linewidth=2.5)
    ax.fill_between(xx, 0.18, yy, color="#BFE5DE", alpha=0.8)

    def map_x(value):
        return 0.10 + 0.80 * (value - x.min()) / (x.max() - x.min())

    boundary = map_x(0.0)
    lcb = map_x(mu - 1.645 * sigma)
    ucb = map_x(mu + 1.645 * sigma)
    ax.axvline(boundary, ymin=0.12, ymax=0.88, color=RED, linestyle="--", linewidth=1.7)
    ax.axvline(lcb, ymin=0.08, ymax=0.45, color=BLUE, linewidth=1.5)
    ax.axvline(ucb, ymin=0.08, ymax=0.45, color=BLUE, linewidth=1.5)
    ax.annotate("", xy=(ucb, 0.27), xytext=(lcb, 0.27), arrowprops=dict(arrowstyle="<->", color=BLUE, linewidth=1.6))
    ax.text((lcb + ucb) / 2, 0.30, r"$[LCB,UCB]$", ha="center", fontsize=9, color=BLUE)
    ax.text(boundary, 0.83, r"feasibility boundary $\delta=0$", ha="center", fontsize=8.5, color=RED)
    ax.text(map_x(mu), 0.78, r"$q_t(\delta)$", ha="center", fontsize=11, color=TEAL, fontweight="bold")
    ax.text(0.50, 0.06, "Interval overlaps boundary -> EXPLORE", ha="center", fontsize=9.5, color=ORANGE, fontweight="bold")
    return fig


def draw_fsm():
    fig, ax = _setup((5.0, 3.0))
    labels = ["APPROACH", "ALIGN", "ENTER", "TRAVERSE", "EXIT"]
    colors = ["#E8F1FA", "#FFF1DC", "#E5F5F1", "#DCEFE8", "#EAF4E2"]
    xs = [0.03, 0.22, 0.41, 0.60, 0.79]
    for x, label, color in zip(xs, labels, colors):
        _rounded(ax, (x, 0.63, 0.16, 0.16), label, fc=color, fontsize=7.2)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        _arrow(ax, (x0 + 0.16, 0.71), (x1, 0.71), color=BLUE)

    _rounded(ax, (0.25, 0.17, 0.19, 0.17), "RECOVER", fc="#FDE9E7", ec=RED, fontsize=9)
    _rounded(ax, (0.62, 0.17, 0.19, 0.17), "REJECT", fc="#F7D7D2", ec=RED, fontsize=9)
    _arrow(ax, (0.51, 0.63), (0.39, 0.34), color=RED)
    _arrow(ax, (0.44, 0.25), (0.30, 0.63), color=RED, style="--")
    _arrow(ax, (0.67, 0.63), (0.71, 0.34), color=RED)
    ax.text(0.43, 0.47, "contact / stuck\nfailed commitment", ha="center", fontsize=7.5, color=RED)
    ax.text(0.76, 0.45, "structural negative\nevidence", ha="center", fontsize=7.5, color=RED)
    ax.text(0.50, 0.91, "Passage-level controller", ha="center", fontsize=12, fontweight="bold", color=DARK)
    return fig


def draw_execution_history():
    fig, ax = _setup((4.8, 3.0))
    ax.plot([0.10, 0.90], [0.56, 0.56], color=DARK, linewidth=2.0)
    times = [0.12, 0.31, 0.50, 0.69, 0.88]
    labels = [r"$t-4$", r"$t-3$", r"$t-2$", r"$t-1$", r"$t$"]
    commands = [(0.05, 0.0), (0.05, 0.25), (0.0, 0.35), (-0.05, -0.2), (0.05, 0.0)]
    modes = ["EXPLORE", "ALIGN", "ALIGN", "RECOVER", "APPROACH"]
    mode_colors = [ORANGE, ORANGE, ORANGE, RED, BLUE]
    for x, label, command, mode, color in zip(times, labels, commands, modes, mode_colors):
        ax.plot([x, x], [0.52, 0.60], color=DARK, linewidth=1.5)
        ax.text(x, 0.64, label, ha="center", fontsize=9)
        ax.text(x, 0.42, rf"$({command[0]:.2f},{command[1]:.2f})$", ha="center", fontsize=7.1, color=BLUE)
        ax.text(x, 0.29, mode, ha="center", fontsize=7.7, color=color, fontweight="bold")
    ax.scatter([0.69], [0.56], s=180, facecolor="#F7D7D2", edgecolor=RED, linewidth=2, zorder=4)
    ax.text(0.69, 0.56, "!", ha="center", va="center", fontsize=13, color=RED, fontweight="bold", zorder=5)
    ax.text(0.69, 0.77, "contact / stuck", ha="center", fontsize=8.5, color=RED)
    ax.text(0.50, 0.88, r"Execution history $\mathcal{H}_{t-1}$", ha="center", fontsize=12, fontweight="bold", color=DARK)
    ax.text(0.50, 0.10, "raw steps.csv + compressed runtime counters", ha="center", fontsize=9, color=GRAY)
    ax.text(0.50, 0.015, r"command shown as $(v,\omega)$", ha="center", fontsize=8, color=GRAY)
    return fig


def draw_geometry_memory():
    fig, ax = _setup((4.8, 3.2))
    history = [
        (0.10, 0.68, "+", GREEN, "success"),
        (0.10, 0.39, "-", RED, "geometry fail"),
        (0.10, 0.10, "-", RED, "geometry fail"),
    ]
    for x, y, sign, color, label in history:
        _rounded(ax, (x, y, 0.22, 0.16), f"passage {sign}", fc="#FAFAFA", ec=color, fontsize=8.5)
        ax.text(x + 0.11, y - 0.035, label, ha="center", fontsize=7.5, color=color)
        _arrow(ax, (x + 0.22, y + 0.08), (0.56, 0.51), color=color, width=1.2)
    _rounded(ax, (0.56, 0.40, 0.30, 0.22), "query passage\nwidth + margin + yaw\n+ morphology + class", fc="#EFE8F7", ec=PURPLE, fontsize=8.5)
    ax.text(0.50, 0.92, "Geometry-guided memory", ha="center", fontsize=12, fontweight="bold", color=PURPLE)
    ax.text(0.50, 0.31, r"$a_i=\exp(-d_i/r)$", ha="center", fontsize=10, color=DARK)
    ax.text(0.50, 0.21, r"$\Delta_t^{mem}=g\log\frac{P+1}{N+1}$", ha="center", fontsize=11, color=PURPLE)
    _arrow(ax, (0.64, 0.16), (0.86, 0.16), color=PURPLE, width=2.0)
    ax.text(0.61, 0.07, "bounded logit correction", ha="center", fontsize=8.5, color=PURPLE)
    return fig


def draw_outcome_posterior():
    fig, ax = _setup((4.3, 3.0))
    x = np.linspace(0.20, 0.80, 400)
    prior = np.exp(-0.5 * ((x - 0.50) / 0.08) ** 2)
    post = np.exp(-0.5 * ((x - 0.42) / 0.035) ** 2)
    prior /= prior.max()
    post /= post.max()
    xx = 0.10 + 0.80 * (x - 0.20) / 0.60
    ax.plot(xx, 0.18 + 0.58 * prior, color=GRAY, linestyle="--", linewidth=2, label=r"$p_0(W)$")
    ax.plot(xx, 0.18 + 0.58 * post, color=BLUE, linewidth=2.6, label=r"$p_t(W)$")
    ax.fill_between(xx, 0.18, 0.18 + 0.58 * post, color="#CFE1F2", alpha=0.8)
    q75 = 0.444
    qx = 0.10 + 0.80 * (q75 - 0.20) / 0.60
    ax.axvline(qx, ymin=0.10, ymax=0.82, color=RED, linestyle="--", linewidth=1.8)
    ax.text(qx + 0.015, 0.69, r"$Q_{0.75}$", fontsize=9, color=RED)
    ax.text(0.50, 0.90, r"Outcome adaptation $p_t(W_{min})$", ha="center", fontsize=12, fontweight="bold", color=BLUE)
    ax.text(0.50, 0.08, "calibration / legacy branch only", ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.text(0.50, 0.015, r"not current strict $W_{req}^{cons}$", ha="center", fontsize=8.5, color=RED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(0.93, 0.84))
    return fig


ASSETS = {
    "asset_obb_yaw_geometry": draw_obb_yaw,
    "asset_interval_belief": draw_interval_belief,
    "asset_passage_fsm": draw_fsm,
    "asset_execution_history": draw_execution_history,
    "asset_geometry_memory": draw_geometry_memory,
    "asset_outcome_posterior": draw_outcome_posterior,
}


def save_asset(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    for path in (svg, png):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    fig.savefig(svg, transparent=True, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, transparent=True, dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return svg, png


def contact_sheet(output_dir: Path, generated: list[tuple[str, Path]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2))
    for ax, (stem, png) in zip(axes.flat, generated):
        image = plt.imread(png)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(stem.replace("asset_", "").replace("_", " "), fontsize=11, fontweight="bold")
    fig.suptitle("DEGNav framework insert assets (current strict semantics)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf = output_dir / "framework_asset_catalog.pdf"
    png = output_dir / "framework_asset_catalog.png"
    for path in (pdf, png):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    generated: list[tuple[str, Path]] = []
    for stem, draw in ASSETS.items():
        svg, png = save_asset(draw(), output_dir, stem)
        generated.append((stem, png))
        print(svg)
        print(png)
    for path in contact_sheet(output_dir, generated):
        print(path)


if __name__ == "__main__":
    main()
