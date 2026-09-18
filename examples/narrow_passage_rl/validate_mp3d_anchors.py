#!/usr/bin/env python3
"""Validate the formal scene-disjoint MP3D narrow-passage anchor split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty anchors file: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=480)
    parser.add_argument("--expected-val", type=int, default=120)
    parser.add_argument("--robot-radius", type=float, default=0.18)
    parser.add_argument("--max-per-scene", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    splits = {"train": _read(args.train), "val": _read(args.val)}
    expected = {"train": args.expected_train, "val": args.expected_val}
    errors = []
    seen_ids = set()
    scene_sets = {}
    summaries = {}

    for split, rows in splits.items():
        if len(rows) != expected[split]:
            errors.append(
                f"{split} row count {len(rows)} != {expected[split]}"
            )
        ids = [row["episode_id"] for row in rows]
        duplicates = [eid for eid, count in Counter(ids).items() if count > 1]
        if duplicates:
            errors.append(f"{split} duplicate episode IDs: {duplicates[:5]}")
        overlap_ids = seen_ids.intersection(ids)
        if overlap_ids:
            errors.append(f"cross-split episode ID overlap: {sorted(overlap_ids)[:5]}")
        seen_ids.update(ids)

        scene_counts = Counter(row["scene_id"] for row in rows)
        scene_sets[split] = set(scene_counts)
        if scene_counts and max(scene_counts.values()) > args.max_per_scene:
            errors.append(
                f"{split} max episodes per scene {max(scene_counts.values())} "
                f"> {args.max_per_scene}"
            )
        missing_scenes = [scene for scene in scene_counts if not Path(scene).is_file()]
        if missing_scenes:
            errors.append(f"{split} missing scene files: {missing_scenes[:5]}")

        difficulty = Counter(row["difficulty"] for row in rows)
        for row in rows:
            radius = float(row["robot_radius"])
            if abs(radius - args.robot_radius) > 1e-6:
                errors.append(
                    f"{split}/{row['episode_id']} robot_radius={radius}"
                )
                break
            false_feasible = int(float(row["false_feasible"]))
            margin = float(row["body_margin"])
            expected_false = row["difficulty"] == "false_feasible"
            if bool(false_feasible) != expected_false:
                errors.append(
                    f"{split}/{row['episode_id']} inconsistent false-feasible label"
                )
                break
            if expected_false != (margin < 0.0):
                errors.append(
                    f"{split}/{row['episode_id']} inconsistent body-margin sign"
                )
                break

        summaries[split] = {
            "path": str((args.train if split == "train" else args.val).resolve()),
            "sha256": _sha256(args.train if split == "train" else args.val),
            "episodes": len(rows),
            "scenes": len(scene_counts),
            "difficulty_counts": dict(sorted(difficulty.items())),
            "max_episodes_per_scene": max(scene_counts.values(), default=0),
        }

    overlap_scenes = scene_sets["train"].intersection(scene_sets["val"])
    if overlap_scenes:
        errors.append(f"train/val scene overlap: {sorted(overlap_scenes)[:5]}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "valid": not errors,
        "errors": errors,
        "robot_radius_m": args.robot_radius,
        "scene_disjoint": not overlap_scenes,
        "splits": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("; ".join(errors))
    print(f"[valid] {args.output}")


if __name__ == "__main__":
    main()
