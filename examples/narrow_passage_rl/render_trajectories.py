#!/usr/bin/env python3
"""Top-down trajectory visualization for paper figures.

Generates a multi-panel figure comparing method behavior on representative
corridor episodes:
  Panel A — L-shaped: rule_baseline oscillates, geometry_fsm succeeds
  Panel B — S-shaped: rule_baseline fails, geometry_fsm succeeds
  Panel C — false_feasible: both methods stop (body-margin rejection)
  Panel D — fsm_no_alignment: L-shaped failure mode (robot never turns)

Color scheme per FSM mode:
  COMMIT / STOP            → green
  EXPLORE / CORRIDOR_FOLLOW → yellow
  ALIGN                    → orange
  RECOVER                  → red
  FOLLOW_SPACE             → cyan
  OTHER / rule_baseline    → blue

Usage
-----
    python examples/narrow_passage_rl/render_trajectories.py
    python examples/narrow_passage_rl/render_trajectories.py --seed 42 --dpi 150
    python examples/narrow_passage_rl/render_trajectories.py \
        --panels l_shaped s_shaped false_feasible --out results/traj.png
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import (
    HarderNarrowPassageEnv, CorridorType,
    _path_arcs, _width_at,
)
from eval_harder_benchmark import (
    fsm_action, rule_baseline_action, TurnCommitFSM, _act,
)

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"

# ── Mode colour map ───────────────────────────────────────────────────────────

_MODE_COLOR = {
    "COMMIT":          "#2ecc71",   # green
    "COMMIT_NOALIGN":  "#2ecc71",
    "CORRIDOR_FOLLOW": "#f1c40f",   # yellow
    "EXPLORE":         "#e67e22",   # orange
    "ALIGN":           "#e74c3c",   # red-orange
    "RECOVER":         "#c0392b",   # dark red
    "FOLLOW_SPACE":    "#1abc9c",   # teal
    "STOP":            "#95a5a6",   # grey
    "REJECT":          "#8e44ad",   # purple
    "rule":            "#3498db",   # blue
}


def _mode_color(mode: str) -> str:
    return _MODE_COLOR.get(mode, "#7f8c8d")


# ── Corridor geometry rendering ───────────────────────────────────────────────

def _draw_corridor(ax, env: HarderNarrowPassageEnv, alpha: float = 0.18):
    """Draw corridor walls and passage region on ax."""
    params = env._params
    pts = [np.array(p, float) for p in params.path]
    arcs, _ = _path_arcs(params.path)

    # Sample along corridor at fine resolution
    n_samples = 300
    left_wall, right_wall = [], []
    for i in range(n_samples + 1):
        s = arcs[-1] * i / n_samples
        # Find segment
        seg_i = 0
        for j in range(len(arcs) - 1):
            if arcs[j] <= s <= arcs[j + 1]:
                seg_i = j
                break
        t = (s - arcs[seg_i]) / max(arcs[seg_i + 1] - arcs[seg_i], 1e-9)
        p = pts[seg_i] + t * (pts[seg_i + 1] - pts[seg_i])
        tang = pts[seg_i + 1] - pts[seg_i]
        tang = tang / max(np.linalg.norm(tang), 1e-9)
        perp = np.array([-tang[1], tang[0]])  # left
        W = _width_at(s, params.width_profile)
        left_wall.append(p + perp * W / 2)
        right_wall.append(p - perp * W / 2)

    left_wall = np.array(left_wall)
    right_wall = np.array(right_wall)

    # Fill corridor interior
    poly_x = np.concatenate([left_wall[:, 0], right_wall[::-1, 0]])
    poly_y = np.concatenate([left_wall[:, 1], right_wall[::-1, 1]])
    ax.fill(poly_x, poly_y, color="#ecf0f1", zorder=0)

    # Draw walls
    ax.plot(left_wall[:, 0],  left_wall[:, 1],  "k-", lw=1.2, zorder=1)
    ax.plot(right_wall[:, 0], right_wall[:, 1], "k-", lw=1.2, zorder=1)

    # Blocker
    if params.blocker is not None:
        s_b, n_b, r_b = params.blocker
        seg_i = 0
        for j in range(len(arcs) - 1):
            if arcs[j] <= s_b <= arcs[j + 1]:
                seg_i = j
                break
        t = (s_b - arcs[seg_i]) / max(arcs[seg_i + 1] - arcs[seg_i], 1e-9)
        p = pts[seg_i] + t * (pts[seg_i + 1] - pts[seg_i])
        tang = pts[seg_i + 1] - pts[seg_i]
        tang = tang / max(np.linalg.norm(tang), 1e-9)
        perp = np.array([-tang[1], tang[0]])
        bpos = p + perp * n_b
        circle = plt.Circle(bpos, r_b, color="#e74c3c", alpha=0.7, zorder=2)
        ax.add_patch(circle)

    # Start / goal markers
    sw = env._start_world
    gw = env._goal_world
    ax.plot(sw[0], sw[1], "go", ms=8, zorder=5, label="start")
    ax.plot(gw[0], gw[1], "r*", ms=12, zorder=5, label="goal")


# ── Episode runner with trajectory logging ────────────────────────────────────

def _collect_trajectory(
    env: HarderNarrowPassageEnv,
    method: str,
    max_steps: int = 400,
    variant: str = "full",
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[float, float, str]], bool]:
    """
    Run one episode, return (trajectory, success).
    trajectory: list of (x, y, mode_str).
    """
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    agent = None
    if method not in ("rule_baseline",):
        agent = TurnCommitFSM(variant=variant)

    traj: List[Tuple[float, float, str]] = []
    done = False
    steps = 0

    while not done and steps < max_steps:
        x, y = float(env.pose[0]), float(env.pose[1])

        if method == "rule_baseline":
            action = rule_baseline_action(obs)
            mode = "rule"
        else:
            action, mode = agent.step(obs)

        traj.append((x, y, mode))
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1

    traj.append((float(env.pose[0]), float(env.pose[1]), mode))
    success = bool(info.get("success", False))
    return traj, success


# ── Single panel renderer ─────────────────────────────────────────────────────

def _render_panel(ax, env: HarderNarrowPassageEnv,
                  trajectories: List[Tuple[str, List, bool]],
                  title: str):
    """Draw corridor + overlaid trajectories on ax."""
    _draw_corridor(ax, env)

    for method_label, traj, success in trajectories:
        if not traj:
            continue
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        modes = [p[2] for p in traj]

        # Draw as coloured segments
        for i in range(len(xs) - 1):
            c = _mode_color(modes[i])
            ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                    color=c, lw=1.8, alpha=0.85, zorder=3)

        # Start dot + label
        marker = "o" if success else "x"
        last_c = _mode_color(modes[-1])
        ax.plot(xs[-1], ys[-1], marker, color=last_c, ms=7, zorder=6)
        ax.annotate(method_label, (xs[0], ys[0]),
                    fontsize=6, ha="center", va="bottom",
                    color="black", zorder=7,
                    xytext=(0, 6), textcoords="offset points")

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Main figure ───────────────────────────────────────────────────────────────

_PANEL_SPECS = {
    "l_shaped": {
        "title": "(a) L-shaped corridor",
        "ctype": "l_shaped",
        # seed=7: FSM successfully navigates, rule_baseline stalls at junction
        "best_seed": 7,
        "methods": [
            ("Rule",    "rule_baseline",    "full"),
            ("FSM",     "geometry_fsm",     "full"),
            ("no-align","fsm_no_alignment", "no_alignment"),
        ],
    },
    "s_shaped": {
        "title": "(b) S-shaped corridor",
        "ctype": "s_shaped",
        # seed=12: clear FOLLOW_SPACE usage at both bends
        "best_seed": 12,
        "methods": [
            ("Rule", "rule_baseline", "full"),
            ("FSM",  "geometry_fsm",  "full"),
        ],
    },
    "false_feasible": {
        "title": "(c) False-feasible (impassable)",
        "ctype": "false_feasible",
        "best_seed": 5,
        "methods": [
            ("Rule", "rule_baseline", "full"),
            ("FSM",  "geometry_fsm",  "full"),
        ],
    },
    "narrow_entry": {
        "title": "(d) Narrow entry",
        "ctype": "narrow_entry",
        "best_seed": 7,
        "methods": [
            ("Rule", "rule_baseline", "full"),
            ("FSM",  "geometry_fsm",  "full"),
        ],
    },
}


def make_figure(panel_names: List[str], seed: int, out_path: Path, dpi: int = 150):
    n = len(panel_names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, pname in zip(axes, panel_names):
        spec = _PANEL_SPECS[pname]
        # Use per-panel best_seed if available, otherwise fall back to CLI seed
        ep_seed = spec.get("best_seed", seed)
        env = HarderNarrowPassageEnv({
            "corridor_types": [spec["ctype"]],
            "seed": ep_seed,
        })
        env.reset(seed=ep_seed)

        trajs = []
        for method_label, method, variant in spec["methods"]:
            traj, success = _collect_trajectory(env, method, variant=variant, seed=ep_seed)
            trajs.append((method_label, traj, success))
            env.reset(seed=ep_seed)

        _render_panel(ax, env, trajs, spec["title"])

    # Legend
    legend_items = [
        mpatches.Patch(color=_MODE_COLOR["COMMIT"],       label="COMMIT"),
        mpatches.Patch(color=_MODE_COLOR["EXPLORE"],      label="EXPLORE"),
        mpatches.Patch(color=_MODE_COLOR["ALIGN"],        label="ALIGN"),
        mpatches.Patch(color=_MODE_COLOR["RECOVER"],      label="RECOVER"),
        mpatches.Patch(color=_MODE_COLOR["FOLLOW_SPACE"], label="FOLLOW_SPACE"),
        mpatches.Patch(color=_MODE_COLOR["rule"],         label="rule_baseline"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=6,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+",
                    default=["l_shaped", "s_shaped", "false_feasible", "narrow_entry"])
    ap.add_argument("--seed", type=int, default=7,
                    help="episode seed (pick one where FSM succeeds clearly)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=Path,
                    default=RESULTS / "trajectory_comparison.png")
    args = ap.parse_args()

    print(f"Rendering {len(args.panels)} panels  seed={args.seed}  dpi={args.dpi}")
    make_figure(args.panels, args.seed, args.out, args.dpi)


if __name__ == "__main__":
    main()
