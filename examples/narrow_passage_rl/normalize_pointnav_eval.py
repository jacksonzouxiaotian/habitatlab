#!/usr/bin/env python3
"""Join Habitat PointNav episode metrics with narrow-passage labels."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path


def _number(row: dict[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name not in row or row[name] == "":
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    with gzip.open(args.dataset, "rt", encoding="utf-8") as handle:
        dataset = json.load(handle)
    metadata = {
        str(episode["episode_id"]): dict(episode.get("info") or {})
        for episode in dataset["episodes"]
    }

    with args.raw_csv.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"empty PointNav evaluation CSV: {args.raw_csv}")

    rows = []
    for raw in raw_rows:
        episode_id = str(raw["episode_id"])
        if episode_id not in metadata:
            raise KeyError(f"episode {episode_id!r} is absent from {args.dataset}")
        info = metadata[episode_id]
        success = _number(raw, "success")
        steps = int(round(_number(raw, "num_steps", "steps")))
        collision_count = _number(
            raw,
            "collisions.count",
            "collisions_count",
            "collision_count",
        )
        rows.append(
            {
                "episode_id": episode_id,
                "scene_id": raw.get("scene_id", ""),
                "split": "val",
                "seed": "checkpoint_deterministic_eval",
                "method": "PointNav PPO (pretrained transfer)",
                "difficulty": str(info.get("difficulty", "unknown")),
                "body_margin_label": info.get("body_margin", ""),
                "false_feasible": float(bool(info.get("false_feasible", False))),
                "success": success,
                "spl": _number(raw, "spl"),
                "distance_to_goal": _number(raw, "distance_to_goal"),
                "collision": float(collision_count > 0.0),
                "collision_count": collision_count,
                "stuck": float(steps >= args.max_steps and success <= 0.5),
                "rejected": 0.0,
                "near_collision": "",
                "steps": steps,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {args.output_csv} ({len(rows)} episodes)")


if __name__ == "__main__":
    main()
