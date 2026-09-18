#!/usr/bin/env python3
"""Build a comparison table only from evaluator-generated metrics.json files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.metrics:
        metrics = json.loads(path.read_text())
        row = {
            "method": metrics["method"],
            "variant": path.parent.name,
            "episodes": metrics["episodes"],
        }
        for key in ("feasible_success", "correct_rejection", "false_rejection", "collision", "timeout", "stuck", "recovery_success"):
            value = metrics[key]
            row[f"{key}_num"] = value["numerator"]
            row[f"{key}_den"] = value["denominator"]
            row[f"{key}_rate"] = value["rate"]
        for mode, value in metrics["mode_usage"].items():
            row[f"{mode.lower()}_num"] = value["numerator"]
            row[f"{mode.lower()}_den"] = value["denominator"]
            row[f"{mode.lower()}_rate"] = value["rate"]
        row["mean_episode_return"] = metrics["mean_episode_return"]
        row["mean_spl"] = metrics["mean_spl"]
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Four-mode selector comparison",
        "",
        "All values below were read from evaluator-produced `metrics.json` files.",
        "",
        "| Method | Feasible success | Correct rejection | False rejection | Collision | Timeout | Recovery success |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = []
        for key in ("feasible_success", "correct_rejection", "false_rejection", "collision", "timeout", "recovery_success"):
            cells.append(f"{row[f'{key}_num']}/{row[f'{key}_den']} = {row[f'{key}_rate']:.2%}")
        lines.append(
            f"| {row['method']} ({row['variant']}) | "
            + " | ".join(cells) + " |"
        )
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"[write] {args.output_dir}")


if __name__ == "__main__":
    main()
