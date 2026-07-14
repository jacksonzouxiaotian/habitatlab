#!/usr/bin/env python3
"""Extended D_min / required-width calibration experiment.

This script evaluates whether a Bayesian required-width belief can recover from
both over-conservative and under-conservative priors, and whether the resulting
``p_feas = P(D_min <= passage_width)`` is directionally calibrated.

The experiment is intentionally lightweight and policy-independent: the
ground-truth traversal boundary is ``true_width`` and an attempted passage
succeeds iff ``passage_width >= true_width``.  This keeps the calibration result
about the required-width belief rather than about a particular PPO checkpoint.

Required paper run:

    python examples/narrow_passage_rl/eval_dmin_calibration.py \
      --priors 0.26 0.31 0.36 0.46 0.56 \
      --true-width 0.36 \
      --episodes 300 \
      --seeds 0 1 2 \
      --output-dir examples/narrow_passage_rl/results/narrow_passage_rl
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dmin_calibrator import CalibConfig, DMinCalibrator


RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"
DEFAULT_TRUE_WIDTH = 0.36
DEFAULT_PRIORS = [0.26, 0.31, 0.36, 0.46, 0.56]
RELIABILITY_BINS = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
]
COMMAND = (
    "python examples/narrow_passage_rl/eval_dmin_calibration.py "
    "--priors 0.26 0.31 0.36 0.46 0.56 --true-width 0.36 "
    "--episodes 300 --seeds 0 1 2 "
    "--output-dir examples/narrow_passage_rl/results/narrow_passage_rl"
)


def _fmt_float(value: float | None, ndigits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "not run"
    return f"{float(value):.{ndigits}f}"


def _fmt_percent(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "not run"
    return f"{100.0 * float(value):.1f}%"


def _fmt_percent_latex(value: float | None) -> str:
    return _fmt_percent(value).replace("%", r"\%")


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _std(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _mean_std_text(values: list[float], *, percent: bool = False) -> str:
    mean = _mean(values)
    std = _std(values)
    if mean is None:
        return "not run"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{scale * mean:.1f}±{scale * (std or 0.0):.1f}{suffix}"


def _interpret_prior(prior: float, true_width: float) -> str:
    if abs(prior - true_width) < 1e-6:
        return "oracle"
    if prior > true_width:
        return "over-conservative"
    return "under-conservative"


def _sample_width(rng: np.random.Generator, true_width: float,
                  width_lo: float, width_hi: float) -> float:
    """Sample widths with extra mass near the feasibility boundary."""

    if rng.random() < 0.70:
        low = max(width_lo, true_width - 0.14)
        high = min(width_hi, true_width + 0.18)
        return float(rng.uniform(low, high))
    return float(rng.uniform(width_lo, width_hi))


def _run_prior_seed(
    *,
    prior_width: float,
    true_width: float,
    seed: int,
    episodes: int,
    width_lo: float,
    width_hi: float,
    p_feas_threshold: float,
    explore_prob: float,
    noise_sigma: float,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + int(round(1000.0 * prior_width)))
    cfg = CalibConfig(
        d_min_lo=min(0.15, true_width - 0.20, prior_width - 0.20),
        d_min_hi=max(0.85, true_width + 0.40, prior_width + 0.30),
        n_bins=320,
        noise_sigma=noise_sigma,
        explore_prob=explore_prob,
        conservative_pct=0.05,
    )
    calibrator = DMinCalibrator(d_init=prior_width, cfg=cfg)
    rows: list[dict[str, Any]] = []

    for ep_idx in range(episodes):
        passage_width = _sample_width(rng, true_width, width_lo, width_hi)
        posterior_mean = calibrator.d_hat
        posterior_q95 = calibrator.quantile(0.95)
        p_feas = calibrator.prob_feasible(passage_width)
        predicted_feasible = bool(p_feas >= p_feas_threshold)
        explore_attempt = bool((not predicted_feasible) and rng.random() < explore_prob)
        attempted = bool(predicted_feasible or explore_attempt)
        reject = not attempted

        physically_feasible = bool(passage_width >= true_width)
        attempt_success = bool(attempted and physically_feasible)
        collision = bool(attempted and not physically_feasible)
        unsafe_attempt = collision

        if attempted:
            calibrator.update(passage_width, success=attempt_success, attempted=True)

        posterior_mean_post = calibrator.d_hat
        posterior_q95_post = calibrator.quantile(0.95)

        rows.append(
            {
                "seed": seed,
                "episode_id": f"prior_{prior_width:.2f}_seed_{seed}_ep_{ep_idx:06d}",
                "episode": ep_idx,
                "prior_width": round(prior_width, 4),
                "true_width": round(true_width, 4),
                "passage_width": round(passage_width, 4),
                "posterior_mean": round(posterior_mean, 6),
                "posterior_q95": round(posterior_q95, 6),
                "posterior_mean_post": round(posterior_mean_post, 6),
                "posterior_q95_post": round(posterior_q95_post, 6),
                "p_feas": round(float(p_feas), 6),
                "predicted_feasible": float(predicted_feasible),
                "outcome_success": float(physically_feasible),
                "attempted": float(attempted),
                "attempt_success": float(attempt_success),
                "explore_attempt": float(explore_attempt),
                "reject": float(reject),
                "collision": float(collision),
                "unsafe_attempt": float(unsafe_attempt),
                "brier": round((float(p_feas) - float(physically_feasible)) ** 2, 6),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def _summary_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    priors = sorted({float(row["prior_width"]) for row in episode_rows})
    for prior in priors:
        rows = [row for row in episode_rows if float(row["prior_width"]) == prior]
        seeds = sorted({int(row["seed"]) for row in rows})
        per_seed: dict[str, list[float]] = {
            "reject": [],
            "collision": [],
            "unsafe_attempt": [],
            "outcome_success": [],
            "predicted_feasible": [],
            "attempt_success": [],
            "brier": [],
            "mean_p_feas": [],
            "final_posterior_mean": [],
            "final_posterior_q95": [],
        }
        ece_values = []
        for seed in seeds:
            seed_rows = [row for row in rows if int(row["seed"]) == seed]
            if not seed_rows:
                continue
            per_seed["reject"].append(_mean([float(row["reject"]) for row in seed_rows]) or 0.0)
            per_seed["collision"].append(_mean([float(row["collision"]) for row in seed_rows]) or 0.0)
            per_seed["unsafe_attempt"].append(_mean([float(row["unsafe_attempt"]) for row in seed_rows]) or 0.0)
            per_seed["outcome_success"].append(_mean([float(row["outcome_success"]) for row in seed_rows]) or 0.0)
            per_seed["predicted_feasible"].append(_mean([float(row["predicted_feasible"]) for row in seed_rows]) or 0.0)
            per_seed["attempt_success"].append(_mean([float(row["attempt_success"]) for row in seed_rows]) or 0.0)
            per_seed["brier"].append(_mean([float(row["brier"]) for row in seed_rows]) or 0.0)
            per_seed["mean_p_feas"].append(_mean([float(row["p_feas"]) for row in seed_rows]) or 0.0)
            final = sorted(seed_rows, key=lambda row: int(row["episode"]))[-1]
            per_seed["final_posterior_mean"].append(float(final["posterior_mean_post"]))
            per_seed["final_posterior_q95"].append(float(final["posterior_q95_post"]))
            ece_values.append(_ece(seed_rows))

        out.append(
            {
                "prior_width": round(prior, 4),
                "true_width": rows[0]["true_width"],
                "interpretation": _interpret_prior(prior, float(rows[0]["true_width"])),
                "seeds": len(seeds),
                "episodes": len(rows),
                "final_posterior_mean": _fmt_float(_mean(per_seed["final_posterior_mean"]), 4),
                "final_posterior_mean_std": _fmt_float(_std(per_seed["final_posterior_mean"]), 4),
                "final_posterior_q95": _fmt_float(_mean(per_seed["final_posterior_q95"]), 4),
                "final_posterior_q95_std": _fmt_float(_std(per_seed["final_posterior_q95"]), 4),
                "mean_p_feas": _fmt_float(_mean(per_seed["mean_p_feas"]), 4),
                "empirical_feasible_rate": _fmt_float(_mean(per_seed["outcome_success"]), 4),
                "predicted_feasible_rate": _fmt_float(_mean(per_seed["predicted_feasible"]), 4),
                "attempt_success_rate": _fmt_float(_mean(per_seed["attempt_success"]), 4),
                "reject_rate": _fmt_float(_mean(per_seed["reject"]), 4),
                "collision_rate": _fmt_float(_mean(per_seed["collision"]), 4),
                "unsafe_attempt_rate": _fmt_float(_mean(per_seed["unsafe_attempt"]), 4),
                "brier": _fmt_float(_mean(per_seed["brier"]), 5),
                "ece": _fmt_float(_mean(ece_values), 5),
            }
        )
    return out


def _bin_for(p_feas: float) -> tuple[float, float] | None:
    for lo, hi in RELIABILITY_BINS:
        if lo <= p_feas < hi or (hi == 1.0 and p_feas <= 1.0):
            return lo, hi
    return None


def _reliability_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    total_by_prior = {
        prior: len([row for row in episode_rows if float(row["prior_width"]) == prior])
        for prior in sorted({float(row["prior_width"]) for row in episode_rows})
    }
    for prior in sorted(total_by_prior):
        prior_rows = [row for row in episode_rows if float(row["prior_width"]) == prior]
        for lo, hi in RELIABILITY_BINS:
            rows = [
                row for row in prior_rows
                if _bin_for(float(row["p_feas"])) == (lo, hi)
            ]
            sq_err = [
                (float(row["p_feas"]) - float(row["outcome_success"])) ** 2
                for row in rows
            ]
            out.append(
                {
                    "prior_width": round(prior, 4),
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "bin": f"[{lo:.1f}, {hi:.1f}{']' if hi == 1.0 else ')'}",
                    "count": len(rows),
                    "mean_p_feas": "" if not rows else f"{_mean([float(row['p_feas']) for row in rows]):.6f}",
                    "empirical_success": "" if not rows else f"{_mean([float(row['outcome_success']) for row in rows]):.6f}",
                    "brier_contribution": (
                        "" if not rows
                        else f"{sum(sq_err) / max(total_by_prior[prior], 1):.6f}"
                    ),
                }
            )
    return out


def _ece(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    total = len(rows)
    err = 0.0
    for lo, hi in RELIABILITY_BINS:
        bin_rows = [
            row for row in rows
            if _bin_for(float(row["p_feas"])) == (lo, hi)
        ]
        if not bin_rows:
            continue
        mean_p = _mean([float(row["p_feas"]) for row in bin_rows]) or 0.0
        empirical = _mean([float(row["outcome_success"]) for row in bin_rows]) or 0.0
        err += (len(bin_rows) / total) * abs(mean_p - empirical)
    return float(err)


def _write_extended_tables(table_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    md = table_dir / "paper_table_calibration_extended.md"
    tex = table_dir / "paper_table_calibration_extended.tex"

    lines = [
        "# Table: Extended Required-Width Calibration",
        "",
        "| Prior W_hat | Type | Final posterior mean | Final q95 | Mean p_feas | Empirical feasible | Reject | Unsafe attempt | Brier | ECE |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {prior_width:.2f} | {interpretation} | {mean}±{mean_std} | {q95}±{q95_std} | "
            "{mean_p} | {emp} | {reject} | {unsafe} | {brier} | {ece} |".format(
                prior_width=float(row["prior_width"]),
                interpretation=row["interpretation"],
                mean=row["final_posterior_mean"],
                mean_std=row["final_posterior_mean_std"],
                q95=row["final_posterior_q95"],
                q95_std=row["final_posterior_q95_std"],
                mean_p=row["mean_p_feas"],
                emp=_fmt_percent(float(row["empirical_feasible_rate"])),
                reject=_fmt_percent(float(row["reject_rate"])),
                unsafe=_fmt_percent(float(row["unsafe_attempt_rate"])),
                brier=row["brier"],
                ece=row["ece"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `outcome_success` in the raw CSV is the oracle physical feasibility label under `true_width`, used for p_feas reliability analysis.",
            "- `unsafe_attempt` is an attempted passage with `passage_width < true_width`, exposing under-conservative priors.",
            "- Over-conservative priors mainly increase rejection; under-conservative priors mainly increase unsafe attempts until the posterior updates.",
            f"- Exact command: `{COMMAND}`.",
        ]
    )
    md.write_text("\n".join(lines) + "\n")

    latex_lines = [
        r"\begin{tabular}{rlrrrrrrrr}",
        r"\toprule",
        r"$\hat{W}$ & Type & Final mean & Final q95 & Mean $p_{feas}$ & Emp. feasible & Reject & Unsafe attempt & Brier & ECE \\",
        r"\midrule",
    ]
    for row in summary_rows:
        latex_lines.append(
            f"{float(row['prior_width']):.2f} & {row['interpretation']} "
            f"& {row['final_posterior_mean']} $\\pm$ {row['final_posterior_mean_std']} "
            f"& {row['final_posterior_q95']} $\\pm$ {row['final_posterior_q95_std']} "
            f"& {row['mean_p_feas']} "
            f"& {_fmt_percent_latex(float(row['empirical_feasible_rate']))} "
            f"& {_fmt_percent_latex(float(row['reject_rate']))} "
            f"& {_fmt_percent_latex(float(row['unsafe_attempt_rate']))} "
            f"& {row['brier']} & {row['ece']} \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex.write_text("\n".join(latex_lines) + "\n")
    print(f"[write] {md}")
    print(f"[write] {tex}")


def _plot_figures(
    *,
    episode_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    fig_dir: Path,
    true_width: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for prior in sorted({float(row["prior_width"]) for row in reliability_rows}):
        rows = [
            row for row in reliability_rows
            if float(row["prior_width"]) == prior and str(row["count"]) not in {"", "0"}
        ]
        if not rows:
            continue
        xs = [float(row["mean_p_feas"]) for row in rows]
        ys = [float(row["empirical_success"]) for row in rows]
        ax.plot(xs, ys, marker="o", label=f"W_hat={prior:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.5", label="ideal")
    ax.set_xlabel("Mean predicted feasibility")
    ax.set_ylabel("Empirical feasibility")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = fig_dir / f"calibration_reliability_curve.{ext}"
        fig.savefig(path, dpi=180)
        print(f"[write] {path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for prior in sorted({float(row["prior_width"]) for row in episode_rows}):
        rows = [row for row in episode_rows if float(row["prior_width"]) == prior]
        episodes = sorted({int(row["episode"]) for row in rows})
        ys = []
        for ep in episodes:
            ep_rows = [row for row in rows if int(row["episode"]) == ep]
            ys.append(_mean([float(row["posterior_mean_post"]) for row in ep_rows]) or float("nan"))
        ax.plot(episodes, ys, label=f"W_hat={prior:.2f}")
    ax.axhline(true_width, linestyle="--", linewidth=1.0, color="0.4", label="true width")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Posterior mean required width (m)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = fig_dir / "calibration_posterior_convergence.pdf"
    fig.savefig(path)
    print(f"[write] {path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", nargs="+", type=float, default=DEFAULT_PRIORS)
    ap.add_argument("--true-width", type=float, default=DEFAULT_TRUE_WIDTH)
    ap.add_argument("--episodes", "--n-episodes", dest="episodes", type=int, default=300)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=None,
                    help="legacy single-seed alias; ignored when --seeds is provided")
    ap.add_argument("--output-dir", type=Path, default=RESULTS)
    ap.add_argument("--width-lo", type=float, default=None)
    ap.add_argument("--width-hi", type=float, default=None)
    ap.add_argument("--p-feas-threshold", type=float, default=0.50)
    ap.add_argument("--explore-prob", type=float, default=0.15)
    ap.add_argument("--noise-sigma", type=float, default=0.03)
    args = ap.parse_args()

    seeds = list(args.seeds)
    if args.seed is not None and args.seeds == [0, 1, 2]:
        seeds = [args.seed]
    width_lo = args.width_lo if args.width_lo is not None else max(0.16, args.true_width - 0.16)
    width_hi = args.width_hi if args.width_hi is not None else args.true_width + 0.34

    print(
        f"[calibration] priors={args.priors} true_width={args.true_width:.3f} "
        f"episodes={args.episodes} seeds={seeds} width_range=({width_lo:.3f}, {width_hi:.3f})"
    )

    episode_rows: list[dict[str, Any]] = []
    for prior in args.priors:
        for seed in seeds:
            rows = _run_prior_seed(
                prior_width=float(prior),
                true_width=float(args.true_width),
                seed=int(seed),
                episodes=int(args.episodes),
                width_lo=float(width_lo),
                width_hi=float(width_hi),
                p_feas_threshold=float(args.p_feas_threshold),
                explore_prob=float(args.explore_prob),
                noise_sigma=float(args.noise_sigma),
            )
            episode_rows.extend(rows)
            final = rows[-1]
            print(
                f"  prior={prior:.2f} seed={seed} final_mean={float(final['posterior_mean_post']):.3f} "
                f"final_q95={float(final['posterior_q95_post']):.3f}"
            )

    raw_dir = args.output_dir / "raw"
    table_dir = args.output_dir / "tables"
    fig_dir = args.output_dir / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _summary_rows(episode_rows)
    reliability_rows = _reliability_rows(episode_rows)

    _write_csv(raw_dir / "calibration_episode_predictions.csv", episode_rows)
    _write_csv(raw_dir / "calibration_prior_sweep.csv", summary_rows)
    _write_csv(table_dir / "calibration_reliability_bins.csv", reliability_rows)
    _write_extended_tables(table_dir, summary_rows)
    _plot_figures(
        episode_rows=episode_rows,
        reliability_rows=reliability_rows,
        fig_dir=fig_dir,
        true_width=float(args.true_width),
    )


if __name__ == "__main__":
    main()
