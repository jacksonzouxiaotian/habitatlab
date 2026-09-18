#!/usr/bin/env python3
"""Plot the yaw-aware required-width model used by DEGNAV.

The figure is intended for the method section of the narrow-passage paper.  It
shows the yaw-dependent body envelope

    w_body(theta) = A |cos(theta)| + B |sin(theta)|

where A is the frontal body width and B is the sagittal body length/envelope.
The repository currently documents A through the narrow-passage belief config
and training protocol.  If B is not documented in code/config, the script uses a
clearly labelled illustrative value and records that choice in metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from examples.narrow_passage_rl.narrow_passage.models.belief_state import (
    BeliefStateConfig,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "examples/narrow_passage_rl/results/narrow_passage_rl/figures"
)

CONFIG_CANDIDATES = (
    REPO_ROOT / "examples/narrow_passage_rl/configs/train_ppo.yaml",
    REPO_ROOT / "examples/narrow_passage_rl/configs/sim2real.yaml",
    REPO_ROOT / "examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.meta.json",
)


@dataclass(frozen=True)
class ScalarParam:
    value: float
    source: str
    illustrative: bool = False


@dataclass(frozen=True)
class FigureParams:
    A_body_width_m: ScalarParam
    B_body_length_m: ScalarParam
    leg_envelope_margin_m: ScalarParam
    sensor_margin_m: ScalarParam
    roll_margin_m: ScalarParam
    execution_margin_m: ScalarParam
    alignable_yaw_deg: float
    forced_oblique_yaw_deg: float
    passage_width_example_m: float | None

    @property
    def additive_margin_m(self) -> float:
        return (
            self.leg_envelope_margin_m.value
            + self.sensor_margin_m.value
            + self.roll_margin_m.value
            + self.execution_margin_m.value
        )


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


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _find_json_scalar(path: Path, keys: Iterable[str]) -> ScalarParam | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in keys and isinstance(value, (int, float)):
                    return float(value), key
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = walk(item)
                if found is not None:
                    return found
        return None

    found = walk(data)
    if found is None:
        return None
    value, key = found
    return ScalarParam(value=value, source=f"{_repo_relative(path)}:{key}")


def _find_text_scalar(path: Path, keys: Iterable[str]) -> ScalarParam | None:
    text = _read_text(path)
    if not text:
        return None
    for key in keys:
        pattern = re.compile(
            rf"(?m)^\s*{re.escape(key)}\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$"
        )
        match = pattern.search(text)
        if match:
            return ScalarParam(
                value=float(match.group(1)),
                source=f"{_repo_relative(path)}:{key}",
            )
    return None


def _find_config_scalar(keys: Iterable[str]) -> ScalarParam | None:
    for path in CONFIG_CANDIDATES:
        found = _find_json_scalar(path, keys)
        if found is not None:
            return found
        found = _find_text_scalar(path, keys)
        if found is not None:
            return found
    return None


def _coalesce_param(
    override: float | None,
    override_name: str,
    config_keys: Iterable[str],
    fallback: ScalarParam,
) -> ScalarParam:
    if override is not None:
        return ScalarParam(float(override), f"CLI --{override_name}")
    found = _find_config_scalar(config_keys)
    if found is not None:
        return found
    return fallback


def _build_params(args: argparse.Namespace) -> FigureParams:
    belief_cfg = BeliefStateConfig()
    body_width_fallback = ScalarParam(
        value=float(belief_cfg.default_w_req_prior),
        source="BeliefStateConfig.default_w_req_prior",
    )
    body_length_fallback = ScalarParam(
        value=float(args.illustrative_body_length),
        source="CLI --illustrative-body-length default",
        illustrative=True,
    )
    zero_fallback = ScalarParam(value=0.0, source="not configured; default 0.0")

    return FigureParams(
        A_body_width_m=_coalesce_param(
            args.body_width,
            "body-width",
            ("body_width_m", "robot_body_width", "default_w_req_prior"),
            body_width_fallback,
        ),
        B_body_length_m=_coalesce_param(
            args.body_length,
            "body-length",
            ("body_length_m", "robot_body_length", "body_length", "legged_body_length_m"),
            body_length_fallback,
        ),
        leg_envelope_margin_m=_coalesce_param(
            args.leg_envelope_margin,
            "leg-envelope-margin",
            ("leg_envelope_margin_m", "leg_envelope", "leg_envelope_margin"),
            zero_fallback,
        ),
        sensor_margin_m=_coalesce_param(
            args.sensor_margin,
            "sensor-margin",
            ("sensor_margin_m", "sensor_margin"),
            zero_fallback,
        ),
        roll_margin_m=_coalesce_param(
            args.roll_margin,
            "roll-margin",
            ("roll_margin_m", "roll_bound_margin_m", "roll_bound"),
            zero_fallback,
        ),
        execution_margin_m=_coalesce_param(
            args.execution_margin,
            "execution-margin",
            ("execution_margin_m", "execution_margin"),
            zero_fallback,
        ),
        alignable_yaw_deg=float(args.alignable_yaw_deg),
        forced_oblique_yaw_deg=float(args.forced_oblique_yaw_deg),
        passage_width_example_m=args.passage_width_example,
    )


def _w_body(theta_rad: np.ndarray | float, A: float, B: float) -> np.ndarray | float:
    return A * np.abs(np.cos(theta_rad)) + B * np.abs(np.sin(theta_rad))


def _plot(params: FigureParams, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig_yaw_required_width.pdf"
    png_path = output_dir / "fig_yaw_required_width.png"
    meta_path = output_dir / "fig_yaw_required_width.meta.json"

    A = params.A_body_width_m.value
    B = params.B_body_length_m.value
    theta_deg = np.linspace(0.0, 90.0, 361)
    theta_rad = np.deg2rad(theta_deg)
    width = _w_body(theta_rad, A, B)

    align_deg = float(np.clip(params.alignable_yaw_deg, 0.0, 90.0))
    forced_deg = float(np.clip(params.forced_oblique_yaw_deg, 0.0, 90.0))
    theta_star_deg = math.degrees(math.atan2(B, A))
    align_width = float(_w_body(math.radians(align_deg), A, B))
    forced_width = float(_w_body(math.radians(forced_deg), A, B))
    theta_star_width = float(_w_body(math.radians(theta_star_deg), A, B))
    extra_width = forced_width - align_width

    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.94, bottom=0.33)

    ax.plot(
        theta_deg,
        width,
        color="#1f4e79",
        linewidth=2.2,
        label=r"$w_{\mathrm{body}}(\theta)=A|\cos\theta|+B|\sin\theta|$",
    )

    if params.passage_width_example_m is not None:
        passage = float(params.passage_width_example_m)
        infeasible = width > passage
        if np.any(infeasible):
            ax.fill_between(
                theta_deg,
                passage,
                width,
                where=infeasible,
                color="#d08c60",
                alpha=0.18,
                linewidth=0,
                label="example constrained region",
            )
            ax.axhline(passage, color="#a05d36", linewidth=0.8, linestyle="--")

    ax.scatter(
        [align_deg, forced_deg, theta_star_deg],
        [align_width, forced_width, theta_star_width],
        s=[22, 24, 22],
        color=["#2f7d32", "#b13b2e", "#5f4b8b"],
        zorder=5,
    )
    ax.axvline(align_deg, color="#2f7d32", linewidth=0.8, alpha=0.55)
    ax.axvline(forced_deg, color="#b13b2e", linewidth=0.8, alpha=0.55)
    ax.axvline(theta_star_deg, color="#5f4b8b", linewidth=0.8, alpha=0.55)

    y_pad = 0.035 * max(width)
    ax.annotate(
        r"$\theta$",
        xy=(align_deg, align_width),
        xytext=(align_deg + 5.0, align_width + y_pad),
        arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "#2f7d32"},
        color="#1f1f1f",
        fontsize=7.2,
    )
    ax.annotate(
        r"$\theta_{\min}+\delta\theta$",
        xy=(forced_deg, forced_width),
        xytext=(max(6.0, forced_deg - 26.0), forced_width + 1.7 * y_pad),
        arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "#b13b2e"},
        color="#1f1f1f",
        fontsize=7.2,
    )
    ax.annotate(
        r"$\theta^*=\tan^{-1}(B/A)$",
        xy=(theta_star_deg, theta_star_width),
        xytext=(max(3.0, theta_star_deg - 38.0), theta_star_width - 4.0 * y_pad),
        arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "#5f4b8b"},
        color="#1f1f1f",
        fontsize=7.2,
    )
    arrow_x = min(86.0, forced_deg + 9.0)
    ax.annotate(
        "",
        xy=(arrow_x, forced_width),
        xytext=(arrow_x, align_width),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.8, "color": "#4a4a4a"},
    )
    ax.text(
        arrow_x + 1.7,
        0.5 * (align_width + forced_width),
        "extra\nwidth",
        va="center",
        ha="left",
        fontsize=7.0,
        color="#2f2f2f",
    )

    ax.set_xlim(0.0, 90.0)
    y_upper = max(width.max(), forced_width, theta_star_width)
    if params.passage_width_example_m is not None:
        y_upper = max(y_upper, float(params.passage_width_example_m))
    ax.set_ylim(max(0.0, width.min() - 0.05), y_upper + 0.08)
    ax.set_xlabel(r"entry yaw $\theta$ (deg)")
    ax.set_ylabel("required width (m)")
    ax.grid(True, color="#d0d0d0", linewidth=0.45, alpha=0.55)
    ax.legend(loc="lower right", frameon=False, handlelength=2.3)

    illustrative_note = (
        " B is illustrative because no configured body length was found."
        if params.B_body_length_m.illustrative
        else ""
    )
    footer = "Additive margins: vertical shift only; yaw shape unchanged."
    if illustrative_note:
        footer += "\nB illustrative: no configured body length found."
    fig.text(0.16, 0.055, footer, ha="left", va="bottom", fontsize=5.6, color="#222222")

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)

    metadata = {
        "figure": "yaw-aware required-width model",
        "formula": "w_body(theta) = A |cos(theta)| + B |sin(theta)|",
        "git_commit": _git_commit(),
        "parameters": {
            "A_body_width_m": asdict(params.A_body_width_m),
            "B_body_length_m": asdict(params.B_body_length_m),
            "leg_envelope_margin_m": asdict(params.leg_envelope_margin_m),
            "sensor_margin_m": asdict(params.sensor_margin_m),
            "roll_margin_m": asdict(params.roll_margin_m),
            "execution_margin_m": asdict(params.execution_margin_m),
            "additive_margin_m": params.additive_margin_m,
            "alignable_yaw_deg": align_deg,
            "forced_oblique_yaw_deg": forced_deg,
            "theta_star_deg": theta_star_deg,
            "passage_width_example_m": params.passage_width_example_m,
        },
        "derived": {
            "alignable_width_m": align_width,
            "forced_oblique_width_m": forced_width,
            "extra_width_m": extra_width,
            "theta_star_width_m": theta_star_width,
        },
        "notes": [
            "Main curve excludes additive sensing, roll, and execution margins.",
            "Those margins shift the curve vertically without changing its yaw-dependent shape.",
            "A constrained-entry region is drawn only when --passage-width-example is supplied.",
        ],
        "outputs": {
            "pdf": _repo_relative(pdf_path),
            "png": _repo_relative(png_path),
            "metadata": _repo_relative(meta_path),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the paper Figure 1 yaw-aware required-width plot."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--body-width", type=float, default=None, help="A in meters.")
    parser.add_argument("--body-length", type=float, default=None, help="B in meters.")
    parser.add_argument(
        "--illustrative-body-length",
        type=float,
        default=0.60,
        help="Used for B only when no body length is configured.",
    )
    parser.add_argument("--leg-envelope-margin", type=float, default=None)
    parser.add_argument("--sensor-margin", type=float, default=None)
    parser.add_argument("--roll-margin", type=float, default=None)
    parser.add_argument("--execution-margin", type=float, default=None)
    parser.add_argument("--alignable-yaw-deg", type=float, default=8.0)
    parser.add_argument("--forced-oblique-yaw-deg", type=float, default=45.0)
    parser.add_argument(
        "--passage-width-example",
        type=float,
        default=None,
        help="Optional documented passage-width boundary for constrained-region shading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = _build_params(args)
    metadata = _plot(params, args.output_dir)
    print(f"[write] {metadata['outputs']['pdf']}")
    print(f"[write] {metadata['outputs']['png']}")
    print(f"[write] {metadata['outputs']['metadata']}")
    print(
        "[params] "
        f"A={metadata['parameters']['A_body_width_m']['value']:.3f}m "
        f"({metadata['parameters']['A_body_width_m']['source']}), "
        f"B={metadata['parameters']['B_body_length_m']['value']:.3f}m "
        f"({metadata['parameters']['B_body_length_m']['source']})"
    )


if __name__ == "__main__":
    main()
