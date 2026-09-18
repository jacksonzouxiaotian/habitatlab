#!/usr/bin/env python3
"""Plot the frozen margin-phase CSV against its actual estimated margin."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = (
    "reactive_rule_baseline", "mean_only", "fixed_uncertainty",
    "no_yaw_aware_readiness", "no_memory", "full_dynamic_uncertainty",
)
NAMES = {
    "reactive_rule_baseline": "Reactive rule baseline",
    "mean_only": "Mean only",
    "fixed_uncertainty": "Fixed uncertainty",
    "no_yaw_aware_readiness": "DEGNav w/o yaw-aware readiness",
    "no_memory": "DEGNav w/o memory",
    "full_dynamic_uncertainty": "DEGNav full",
}
COLORS = {
    "reactive_rule_baseline": "#7f7f7f", "mean_only": "#ff7f0e",
    "fixed_uncertainty": "#2ca02c", "no_yaw_aware_readiness": "#9467bd",
    "no_memory": "#8c564b", "full_dynamic_uncertainty": "#1f77b4",
}
METRICS = ("F-SR", "CR", "FR", "Collision", "Timeout/stuck")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with (run_dir / "margin_phase.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for axis, metric in zip(axes.flat, METRICS):
        for method in METHODS:
            points = []
            for row in rows:
                if row["method"] != method:
                    continue
                x = float(row["mean_D_hat_minus_W_req_cons"])
                y = float(row[metric])
                if math.isfinite(x) and math.isfinite(y):
                    points.append((x, 100.0 * y))
            points.sort()
            if points:
                axis.plot(
                    [item[0] for item in points],
                    [item[1] for item in points],
                    marker="o", linewidth=1.5, color=COLORS[method],
                    label=NAMES[method],
                )
        axis.axvline(0.0, color="#222222", linestyle="--", linewidth=0.9)
        axis.set_title(metric)
        axis.set_ylabel("Percent")
        axis.grid(alpha=0.25)
    axes.flat[-1].set_visible(False)
    for axis in axes[1, :2]:
        axis.set_xlabel(r"Mean estimated margin $\widehat D-\widehat W_{\mathrm{req}}^{\mathrm{cons}}$ (m)")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Margin phase on the actual estimated-margin axis")
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {})):
        output = run_dir / f"margin_phase_estimated.{suffix}"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        fig.savefig(output, bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
