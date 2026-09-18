#!/usr/bin/env python3
"""Create a paper-facing calibration convergence/error figure from saved CSVs."""

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
DEFAULT_RESULTS = ROOT / "results" / "narrow_passage_rl"
COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output-stem", default="calibration_posterior_convergence_paper"
    )
    args = parser.parse_args()

    episode_path = args.results_dir / "raw" / "calibration_episode_predictions.csv"
    sweep_path = args.results_dir / "raw" / "calibration_prior_sweep.csv"
    if not episode_path.is_file() or not sweep_path.is_file():
        parser.error("saved calibration CSVs are incomplete")

    episode_rows = read_rows(episode_path)
    sweep_rows = read_rows(sweep_path)
    priors = sorted({float(row["prior_width"]) for row in episode_rows})
    true_widths = {float(row["true_width"]) for row in episode_rows}
    if len(true_widths) != 1:
        parser.error(f"expected one true width, got {sorted(true_widths)}")
    true_width = true_widths.pop()

    trajectories: dict[float, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seeds_by_prior: dict[float, set[int]] = defaultdict(set)
    for row in episode_rows:
        prior = float(row["prior_width"])
        episode = int(row["episode"])
        trajectories[prior][episode].append(float(row["posterior_mean_post"]))
        seeds_by_prior[prior].add(int(row["seed"]))

    expected_episodes = sorted(trajectories[priors[0]])
    if any(sorted(trajectories[prior]) != expected_episodes for prior in priors):
        parser.error("priors do not share the same episode indices")
    if any(len(values) != len(seeds_by_prior[prior])
           for prior in priors for values in trajectories[prior].values()):
        parser.error("incomplete seed coverage in convergence trajectories")

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
    fig, (ax_curve, ax_error) = plt.subplots(
        1, 2, figsize=(10.2, 3.85), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    x = np.asarray(expected_episodes, dtype=np.int32)
    for color, prior in zip(COLORS, priors):
        values = np.asarray(
            [trajectories[prior][episode] for episode in expected_episodes],
            dtype=np.float64,
        )
        mean = values.mean(axis=1)
        std = values.std(axis=1, ddof=1)
        ax_curve.plot(
            x,
            mean,
            color=color,
            linewidth=1.65,
            label=rf"Initial $\hat{{W}}_0={prior:.2f}$ m",
        )
        ax_curve.fill_between(
            x, mean - std, mean + std, color=color, alpha=0.11, linewidth=0
        )

    ax_curve.axhline(
        true_width,
        color="#4D4D4D",
        linestyle="--",
        linewidth=1.25,
        label=rf"True threshold $D_{{\min}}={true_width:.2f}$ m",
    )
    ax_curve.set_title("(a) Posterior convergence across initial priors", loc="left")
    ax_curve.set_xlabel("Calibration episode")
    ax_curve.set_ylabel(r"Posterior mean of $D_{\min}$ (m)")
    ax_curve.set_xlim(expected_episodes[0], expected_episodes[-1])
    ax_curve.set_ylim(0.245, 0.54)
    ax_curve.grid(True, linewidth=0.45, alpha=0.30)
    ax_curve.legend(loc="upper right", frameon=True, ncol=1)

    sweep = {float(row["prior_width"]): row for row in sweep_rows}
    if set(sweep) != set(priors):
        parser.error("prior sweep and episode predictions use different priors")
    prior_x = np.asarray(priors)
    mean_error_cm = 100.0 * np.asarray(
        [float(sweep[p]["final_posterior_mean"]) - true_width for p in priors]
    )
    mean_error_std_cm = 100.0 * np.asarray(
        [float(sweep[p]["final_posterior_mean_std"]) for p in priors]
    )
    q95_error_cm = 100.0 * np.asarray(
        [float(sweep[p]["final_posterior_q95"]) - true_width for p in priors]
    )
    q95_error_std_cm = 100.0 * np.asarray(
        [float(sweep[p]["final_posterior_q95_std"]) for p in priors]
    )

    ax_error.axhline(0.0, color="#4D4D4D", linestyle="--", linewidth=1.25)
    ax_error.axvline(true_width, color="0.75", linestyle=":", linewidth=1.0)
    ax_error.errorbar(
        prior_x - 0.004,
        mean_error_cm,
        yerr=mean_error_std_cm,
        color="#0072B2",
        marker="o",
        markersize=5.0,
        linewidth=1.5,
        capsize=3,
        label="Posterior mean",
    )
    ax_error.errorbar(
        prior_x + 0.004,
        q95_error_cm,
        yerr=q95_error_std_cm,
        color="#D55E00",
        marker="s",
        markersize=4.7,
        linewidth=1.5,
        capsize=3,
        label="Posterior q95",
    )
    ax_error.set_title("(b) Residual error after 300 episodes", loc="left")
    ax_error.set_xlabel(r"Initial prior $\hat{W}_0$ (m)")
    ax_error.set_ylabel("Final estimate error (cm)")
    ax_error.set_xticks(prior_x)
    ax_error.set_xlim(min(priors) - 0.03, max(priors) + 0.03)
    ax_error.set_ylim(-2.7, 0.35)
    ax_error.grid(True, axis="y", linewidth=0.45, alpha=0.30)
    ax_error.legend(loc="lower left", frameon=True)

    fig.tight_layout(w_pad=2.1)
    output_dir = args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / args.output_stem
    for suffix in ("pdf", "png", "svg"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=320, bbox_inches="tight")
        print(f"[write] {path}")
    plt.close(fig)

    metadata = {
        "figure": str(output_stem),
        "source_files": {
            str(episode_path): sha256_file(episode_path),
            str(sweep_path): sha256_file(sweep_path),
        },
        "priors_m": priors,
        "true_width_m": true_width,
        "seeds_by_prior": {
            f"{prior:.2f}": sorted(seeds_by_prior[prior]) for prior in priors
        },
        "episodes_per_seed": len(expected_episodes),
        "left_panel": "posterior mean after each attempted-outcome update; mean +/- one sample standard deviation across seeds",
        "right_panel": "final posterior mean and q95 error relative to the synthetic true threshold; mean +/- one sample standard deviation",
        "scope": "policy-independent synthetic threshold calibration; not a Habitat closed-loop navigation result",
    }
    metadata_path = output_stem.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[write] {metadata_path}")


if __name__ == "__main__":
    main()
