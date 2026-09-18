#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from eagor_repro.config import load_config
from eagor_repro.evaluation.habitat_evaluator import evaluate_objectnav
from eagor_repro.evaluation.plots import baseline_comparison_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched EAGOR baseline evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/eagor/base.yaml"))
    parser.add_argument("--overlay", type=Path, default=None)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["centroid", "circular_centroid", "grid", "eagor"],
    )
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.overlay:
        config = OmegaConf.merge(config, OmegaConf.load(args.overlay))
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    OmegaConf.resolve(config)
    summaries = {
        method: evaluate_objectnav(config, method) for method in args.policies
    }
    metric_names = [
        "success_rate",
        "spl",
        "steps",
        "mae_deg",
        "seam_error_deg",
        "occlusion_error_deg",
        "temporal_consistency_deg_per_step",
        "collision_count",
        "average_inference_update_latency_ms",
    ]
    aggregate = []
    for method, episodes in summaries.items():
        row = {"policy_method": method, "episodes": len(episodes)}
        for metric in metric_names:
            values = [
                float(episode[metric])
                for episode in episodes
                if metric in episode and np.isfinite(float(episode[metric]))
            ]
            row[metric] = float(np.mean(values)) if values else float("nan")
        aggregate.append(row)
    summary_dir = Path(str(config.output.root)) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    table_path = summary_dir / "baseline_comparison.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)
    markdown_path = summary_dir / "baseline_comparison.md"
    headers = list(aggregate[0].keys())
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in aggregate:
            handle.write(
                "| "
                + " | ".join(
                    f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key])
                    for key in headers
                )
                + " |\n"
            )
    plot_path = baseline_comparison_plot(
        aggregate, summary_dir / "baseline_comparison.png"
    )
    summaries["aggregate"] = aggregate
    summaries["comparison_csv"] = str(table_path)
    summaries["comparison_markdown"] = str(markdown_path)
    summaries["comparison_plot"] = str(plot_path)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
