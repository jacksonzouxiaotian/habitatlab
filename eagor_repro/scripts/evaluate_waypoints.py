#!/usr/bin/env python3
"""Bundled-scene waypoint-following direction/closed-loop stress evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eagor_repro.scripts.run_smoke_test import habitat_sim_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/eagor_results/waypoints"))
    parser.add_argument(
        "--initial-headings-deg",
        nargs="+",
        type=float,
        default=[0.0, 45.0, 90.0, 135.0, 179.0],
    )
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()
    results = []
    for heading in args.initial_headings_deg:
        label = f"waypoint_heading_{heading:g}".replace("-", "neg_").replace(".", "p")
        results.append(
            habitat_sim_smoke(
                args.output_root,
                max_steps=args.max_steps,
                initial_heading_deg=heading,
                artifact_stem=label,
            )
        )
    summary = {
        "fixture": "bundled scene + synthetic waypoint-oracle likelihood",
        "not_an_objectnav_result": True,
        "initial_headings_deg": args.initial_headings_deg,
        "success_rate": float(np.mean([result["success"] for result in results])),
        "mean_steps": float(np.mean([result["steps"] for result in results])),
        "mean_angular_error_deg": float(
            np.mean([result["mean_angular_error_deg"] for result in results])
        ),
        "episodes": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "waypoint_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

