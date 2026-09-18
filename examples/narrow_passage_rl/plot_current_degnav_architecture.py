#!/usr/bin/env python3
"""Render the paper-facing diagram of the current DEGNav strict architecture.

Solid arrows are active decision/control paths in the frozen strict evaluator.
Dashed arrows are audit, calibration, or legacy-only paths.  The distinction is
intentional: the outcome posterior quantile is not the selector-facing
``w_req_cons`` in the current strict evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "examples/narrow_passage_rl/results/narrow_passage_rl/figures"
)
DEFAULT_STEM = "degnav_current_architecture_20260822"

COLORS = {
    "input": "#E8F1FA",
    "belief": "#E5F5F1",
    "selector": "#FFF1DC",
    "controller": "#FDE9E7",
    "geometry": "#EEF4E5",
    "history": "#F1F2F4",
    "memory": "#EFE8F7",
    "posterior": "#E8F0FA",
    "edge": "#34495E",
    "active": "#2C6E9B",
    "memory_edge": "#7851A9",
    "audit": "#7F8C8D",
    "warning": "#B03A2E",
}


def _box(
    ax,
    xywh: tuple[float, float, float, float],
    title: str,
    lines: list[str],
    *,
    facecolor: str,
    edgecolor: str | None = None,
    linestyle: str = "-",
    title_size: float = 10.5,
    body_size: float = 8.7,
    linewidth: float = 1.25,
    title_color: str = "#1F2933",
    zorder: int = 2,
) -> None:
    x, y, w, h = xywh
    edgecolor = edgecolor or COLORS["edge"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.5 * w,
        y + h - 0.030,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=zorder + 1,
    )
    body_top = y + h - 0.082
    gap = min(0.033, (h - 0.105) / max(len(lines), 1))
    for index, line in enumerate(lines):
        ax.text(
            x + 0.018,
            body_top - index * gap,
            line,
            ha="left",
            va="top",
            fontsize=body_size,
            color="#263238",
            zorder=zorder + 1,
        )


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "",
    color: str | None = None,
    linestyle: str = "-",
    connectionstyle: str = "arc3",
    linewidth: float = 1.6,
    label_offset: tuple[float, float] = (0.0, 0.0),
    zorder: int = 5,
) -> None:
    color = color or COLORS["active"]
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=3,
        shrinkB=3,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    if label:
        mx = 0.5 * (start[0] + end[0]) + label_offset[0]
        my = 0.5 * (start[1] + end[1]) + label_offset[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=7.7,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.9, alpha=0.92),
            zorder=zorder + 1,
        )


def _path_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    label: str = "",
    label_xy: tuple[float, float] | None = None,
    color: str | None = None,
    linestyle: str = "-",
    linewidth: float = 1.6,
    zorder: int = 5,
) -> None:
    """Draw an orthogonal multi-segment arrow with an optional label."""

    color = color or COLORS["active"]
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    if label and label_xy is not None:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=7.7,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.9, alpha=0.92),
            zorder=zorder + 1,
        )


def render(output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}.{suffix}" for suffix in ("svg", "pdf", "png")]
    outputs.append(output_dir / f"{stem}.meta.json")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing architecture artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(15.6, 8.7))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    fig.text(
        0.5,
        0.970,
        "DEGNav strict feasibility: current execution, belief, and memory paths",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#17202A",
    )
    fig.text(
        0.5,
        0.938,
        "Solid arrows affect the frozen strict controller; dashed arrows are audit / calibration / legacy only",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#52616B",
    )

    _box(
        ax,
        (0.025, 0.635, 0.190, 0.245),
        "Observation + execution signals",
        [
            r"Depth rays/sectors, $\widehat D_t$, clearance",
            r"yaw $\psi_t$, lateral offset, pose",
            r"current / previous command $(v,\omega)$",
            "contact flag, stuck score and counters",
            "19-D geometry feature vector",
        ],
        facecolor=COLORS["input"],
    )

    _box(
        ax,
        (0.255, 0.595, 0.300, 0.300),
        "Dynamic feasibility estimator",
        [
            r"Structural: $\mu_{\delta,s}=\widehat D-W_s$",
            r"Pose-conditioned: $\mu_{\delta,p}=\widehat D-W_p(\psi)$",
            r"Engineering $\sigma_\delta$: rays + dropout + fit residual",
            r"+ temporal width variation + yaw / pose uncertainty",
            r"$p_{feas}=\Phi(\mu_\delta/\sigma_\delta)$; LCB / UCB",
            r"Parallel $q_t(\delta)$: propagate $\Gamma_{\Delta\psi}$, fuse $\oplus$",
            "posterior variance -> information-gain audit / scan exit",
        ],
        facecolor=COLORS["belief"],
    )

    _box(
        ax,
        (0.590, 0.635, 0.175, 0.245),
        "Strict interval selector",
        [
            r"Commit: both LCBs $>\tau_c$",
            r"Reject: structural UCB $<-\tau_r$",
            "Explore: interval overlaps boundary",
            "or pose-conditioned readiness is unmet",
            "Recover: contact, stuck, failed commitment",
            r"then bounded $\Delta_t^{mem}$ correction",
        ],
        facecolor=COLORS["selector"],
    )

    _box(
        ax,
        (0.805, 0.635, 0.170, 0.245),
        "Passage FSM + OBB safety",
        [
            "APPROACH -> ALIGN -> ENTER",
            "TRAVERSE -> EXIT",
            "bounded RECOVER / terminal REJECT",
            "posterior variance bounds",
            "active scan duration",
            "swept-OBB action projection",
            r"requested vs executed $(v_t,\omega_t)$",
        ],
        facecolor=COLORS["controller"],
    )

    _box(
        ax,
        (0.255, 0.455, 0.300, 0.115),
        "Shared morphology and yaw-aware required width",
        [
            r"$W_s=A+2m$;  $W_p(\psi)=A|\cos\psi|+B|\sin\psi|+2m$    ($A=.36$, $B=.60$, $m=.03$ m)",
        ],
        facecolor=COLORS["geometry"],
        title_size=9.5,
        body_size=8.4,
    )

    _box(
        ax,
        (0.025, 0.100, 0.210, 0.285),
        r"Execution history $\mathcal{H}_{t-1}$",
        [
            r"outcomes $y_{1:t-1}$: categorical records",
            r"commands: per-step requested / executed $(v,\omega)$",
            "contact / stuckness: flags, scores, counters",
            "steps.csv: raw one-row-per-step audit trail",
            "runtime: EWMA, deque, counters, previous mode",
            "No single learned history embedding",
        ],
        facecolor=COLORS["history"],
    )

    _box(
        ax,
        (0.275, 0.135, 0.150, 0.215),
        "Outcome evidence router",
        [
            "Positive: successful traversal",
            "Negative: confirmed geometric failure",
            "Audit only: unknown collision, timeout,",
            "stuck/control failure, censored Reject",
        ],
        facecolor="#FAFAFA",
    )

    _box(
        ax,
        (0.465, 0.080, 0.245, 0.330),
        "Geometry-guided cross-episode memory",
        [
            "EpisodeRecord list (RAM; reset per method / seed)",
            "similarity: width + structural margin + yaw",
            "+ morphology + corridor class",
            r"weights $a_i=\exp(-d_i/r)$ -> positive $P$, negative $N$",
            r"$\Delta_t^{mem}=\mathrm{clip}[g\log((P+1)/(N+1))]$",
            r"$\mathrm{logit}(p_{corr})=\mathrm{logit}(p_{base})+\Delta_t^{mem}$",
            "soft: Commit -> Explore; strong support: Reject",
        ],
        facecolor=COLORS["memory"],
        edgecolor=COLORS["memory_edge"],
    )

    _box(
        ax,
        (0.750, 0.080, 0.225, 0.330),
        r"Outcome posterior $p_t(W_{min})$",
        [
            "200-point grid over 0.20--0.80 m",
            r"$p_t(W)\propto p_{t-1}(W)L(y_t|D_t,W)$",
            "updated only by attempted, valid geometry evidence",
            r"mean $\widehat W$; conservative $Q_{0.75}[p_t(W)]$",
            "used by calibration / legacy should_attempt()",
            r"NOT the strict selector's $W_{req}^{cons}$",
        ],
        facecolor=COLORS["posterior"],
        edgecolor=COLORS["audit"],
        linestyle="--",
        title_color=COLORS["warning"],
        body_size=8.1,
    )

    # Active top-row computation and control path.
    _arrow(ax, (0.215, 0.755), (0.255, 0.755), label="19-D features")
    _arrow(
        ax,
        (0.555, 0.755),
        (0.590, 0.755),
        label="belief + recovery signals",
    )
    _arrow(ax, (0.765, 0.755), (0.805, 0.755), label="mode")

    # Shared geometry feeds both decision model and physical safety.
    _arrow(ax, (0.405, 0.570), (0.405, 0.595), color="#5C7A29")
    _path_arrow(
        ax,
        [(0.555, 0.515), (0.575, 0.515), (0.575, 0.585), (0.890, 0.585), (0.890, 0.635)],
        label="same OBB morphology",
        label_xy=(0.735, 0.585),
        color="#5C7A29",
    )

    # Executed actions and observations close the control loop.
    _arrow(
        ax,
        (0.890, 0.880),
        (0.120, 0.880),
        label="robot / simulator feedback",
        color=COLORS["active"],
        connectionstyle="arc3,rad=0.16",
        label_offset=(0.0, 0.035),
    )

    # The observation/execution stream produces the persistent raw audit trail.
    _arrow(
        ax,
        (0.120, 0.635),
        (0.120, 0.385),
        label="per-step trace",
        color=COLORS["audit"],
        linestyle="--",
        label_offset=(0.045, 0.0),
    )
    _arrow(ax, (0.235, 0.235), (0.275, 0.235), label="episode end", color=COLORS["edge"])
    _arrow(ax, (0.425, 0.255), (0.465, 0.255), label="valid geometry", color=COLORS["memory_edge"])
    _path_arrow(
        ax,
        [(0.400, 0.350), (0.400, 0.435), (0.860, 0.435), (0.860, 0.410)],
        label="same valid outcome",
        label_xy=(0.720, 0.435),
        color=COLORS["audit"],
        linestyle="--",
    )

    # Cross-episode memory actively corrects the base selector.
    _arrow(
        ax,
        (0.590, 0.410),
        (0.675, 0.635),
        label=r"$\Delta_t^{mem}$ / terminal support",
        color=COLORS["memory_edge"],
        connectionstyle="arc3,rad=0.10",
        label_offset=(0.035, 0.0),
        linewidth=2.0,
    )

    # Make the absent strict connection explicit instead of drawing a false arrow.
    ax.text(
        0.862,
        0.045,
        r"No current path: $Q_{0.75}[p_t(W)] \nrightarrow W_{req}^{cons}$ in strict evaluation",
        ha="center",
        va="center",
        fontsize=8.4,
        color=COLORS["warning"],
        fontweight="bold",
    )

    # Legend.
    _arrow(ax, (0.030, 0.035), (0.085, 0.035), color=COLORS["active"], linewidth=1.8)
    ax.text(0.090, 0.035, "active decision / control path", va="center", fontsize=8.2)
    _arrow(
        ax,
        (0.250, 0.035),
        (0.305, 0.035),
        color=COLORS["audit"],
        linestyle="--",
        linewidth=1.5,
    )
    ax.text(0.310, 0.035, "audit, calibration, or legacy-only path", va="center", fontsize=8.2)
    _arrow(ax, (0.535, 0.035), (0.590, 0.035), color=COLORS["memory_edge"], linewidth=2.0)
    ax.text(0.595, 0.035, "geometry-memory intervention", va="center", fontsize=8.2)

    fig.subplots_adjust(left=0.012, right=0.988, top=0.925, bottom=0.025)
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=260, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": "current DEGNav strict feasibility execution-belief-memory architecture",
        "scope": "current procedural strict evaluator; not a proposed unimplemented architecture",
        "solid_arrow_semantics": "active selector/control path",
        "dashed_arrow_semantics": "audit, calibration, or legacy-only path",
        "important_boundary": (
            "DMinCalibrator Q0.75 posterior quantile is not selector-facing "
            "w_req_cons in the frozen strict evaluation"
        ),
        "selector_w_req_cons": (
            "RobotMorphology pose projection: width*abs(cos(yaw)) + "
            "length*abs(sin(yaw)) + 2*safety_margin"
        ),
        "outputs": {path.suffix.lstrip("."): str(path) for path in outputs[:3]},
        "source": str(Path(__file__).resolve()),
    }
    outputs[3].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args()
    for path in render(args.output_dir.resolve(), str(args.stem)):
        print(path)


if __name__ == "__main__":
    main()
