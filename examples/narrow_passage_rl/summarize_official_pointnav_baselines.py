#!/usr/bin/env python3
"""Build the paper-facing summary for official MP3D PointNav baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METHODS = (
    ("forward_only", "ForwardOnly", "official example lower bound"),
    ("random", "Random Agent", "official random lower bound"),
    ("random_forward", "RandomForward", "official weak reactive lower bound"),
    ("goal_follower", "GoalFollower", "official PointGoal reactive agent"),
    ("pointnav_ppo", "PointNav PPO", "official learned baseline"),
    ("ddppo", "DD-PPO", "official recurrent learned baseline"),
    (
        "shortest_path_follower",
        "ShortestPathFollower",
        "official NavMesh oracle upper bound",
    ),
)

EXPECTED_SEEDS = {
    "forward_only": 1,
    "random": 3,
    "random_forward": 3,
    "goal_follower": 1,
    "pointnav_ppo": 3,
    "ddppo": 3,
    "shortest_path_follower": 1,
}
EXPECTED_EPISODES_PER_SEED = 495


def _episode_sequences(path: Path) -> dict[int, list[tuple[str, str]]]:
    """Return ordered (scene, episode) keys grouped by evaluation seed."""

    if not path.is_file():
        return {}
    grouped: dict[int, list[tuple[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["eval_seed"])
            grouped.setdefault(seed, []).append(
                (str(row["scene_id"]), str(row["episode_id"]))
            )
    return grouped


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), statistics.pstdev(values)


def _format(mean: float, std: float, *, percent: bool = False) -> str:
    if not math.isfinite(mean):
        return "pending"
    scale = 100.0 if percent else 1.0
    if math.isfinite(std) and std > 0.0:
        return f"{mean * scale:.2f}±{std * scale:.2f}{'%' if percent else ''}"
    return f"{mean * scale:.2f}{'%' if percent else ''}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    args.input_dir.mkdir(parents=True, exist_ok=True)

    episode_sequences = {
        method: _episode_sequences(args.input_dir / method / "episodes.csv")
        for method, _display_name, _role in METHODS
    }
    canonical_method = next(
        (method for method, _name, _role in METHODS if episode_sequences[method]),
        None,
    )
    canonical_sequence = (
        next(iter(episode_sequences[canonical_method].values()))
        if canonical_method is not None
        else []
    )
    canonical_episode_set = set(canonical_sequence)

    output_rows = []
    for method, display_name, role in METHODS:
        summary_path = args.input_dir / method / "summary.csv"
        if not summary_path.is_file():
            output_rows.append(
                {
                    "method": display_name,
                    "official_role": role,
                    "episodes_per_seed": "pending",
                    "eval_seeds": 0,
                    "success": "pending",
                    "spl": "pending",
                    "distance_to_goal": "pending",
                    "collision_rate": "pending",
                    "mean_steps": "pending",
                    "status": "pending",
                }
            )
            continue

        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        sequences = episode_sequences[method]
        episode_protocol_complete = (
            len(sequences) == EXPECTED_SEEDS[method]
            and all(
                len(sequence) == EXPECTED_EPISODES_PER_SEED
                and len(set(sequence)) == EXPECTED_EPISODES_PER_SEED
                and set(sequence) == canonical_episode_set
                for sequence in sequences.values()
            )
        )
        complete = episode_protocol_complete and (
            len(rows) == EXPECTED_SEEDS[method]
            and all(
                int(row["episodes"]) == EXPECTED_EPISODES_PER_SEED
                for row in rows
            )
        )
        metrics = {}
        for key in (
            "success",
            "spl",
            "distance_to_goal",
            "collision_rate",
            "mean_steps",
        ):
            metrics[key] = _mean_std([float(row[key]) for row in rows])
        episode_counts = sorted({int(row["episodes"]) for row in rows})
        output_rows.append(
            {
                "method": display_name,
                "official_role": role,
                "episodes_per_seed": "/".join(map(str, episode_counts)),
                "eval_seeds": len(rows),
                "success": _format(*metrics["success"], percent=True),
                "spl": _format(*metrics["spl"]),
                "distance_to_goal": _format(*metrics["distance_to_goal"]),
                "collision_rate": _format(
                    *metrics["collision_rate"], percent=True
                ),
                "mean_steps": _format(*metrics["mean_steps"]),
                "status": (
                    "complete"
                    if complete
                    else f"incomplete ({len(rows)}/{EXPECTED_SEEDS[method]} seeds)"
                ),
            }
        )

    csv_path = args.input_dir / "summary_all_methods.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "# Official Habitat PointNav Baselines — MP3D v1 val",
        "",
        "All completed rows use the official MP3D PointNav v1 validation "
        "episodes, stock Habitat PointNav task defaults, and episode-level "
        "metric logging. No strict-success column is reported.",
        "",
        "| Method | Official role | Episodes/seed | Eval seeds | Success | SPL | Distance to goal | Episodes with contact | Steps | Status |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in output_rows:
        lines.append(
            "| {method} | {official_role} | {episodes_per_seed} | "
            "{eval_seeds} | {success} | {spl} | {distance_to_goal} | "
            "{collision_rate} | {mean_steps} | {status} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Protocol notes",
            "",
            "- PointNav PPO uses the released MP3D RGB-D checkpoint.",
            "- DD-PPO uses the released Gibson-2+ Depth ResNet50-LSTM "
            "checkpoint and is therefore labeled Gibson-to-MP3D transfer.",
            "- Released learned policies use sampled actions, matching the "
            "official DD-PPO model documentation.",
            "- `Episodes with contact` is the fraction of episodes containing "
            "at least one simulator collision; it is not collisions per step. "
            "Mean collision counts remain available in each method's raw "
            "`summary.csv`.",
            "- ShortestPathFollower uses privileged NavMesh access and is an "
            "oracle ceiling, not a deployable policy.",
            "- DEGNAV rows belong in a separate method table until evaluated "
            "on exactly these PointNav episode IDs and task defaults.",
            "",
        ]
    )
    markdown_path = args.input_dir / "paper_table_official_pointnav.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "dataset": "official MP3D PointNav v1",
        "split": "val",
        "table": str(markdown_path),
        "summary_csv": str(csv_path),
        "complete_methods": [
            row["method"] for row in output_rows if row["status"] == "complete"
        ],
        "pending_methods": [
            row["method"] for row in output_rows if row["status"] == "pending"
        ],
        "incomplete_methods": [
            row["method"]
            for row in output_rows
            if row["status"].startswith("incomplete")
        ],
        "episode_protocol_audit": {
            "key": ["scene_id", "episode_id"],
            "expected_unique_episodes_per_seed": EXPECTED_EPISODES_PER_SEED,
            "canonical_method": canonical_method,
            "all_available_seeds_match_canonical_episode_set": all(
                not episode_sequences[method]
                or all(
                    set(sequence) == canonical_episode_set
                    for sequence in sequences.values()
                )
                for method, sequences in episode_sequences.items()
            ),
        },
    }
    (args.input_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {csv_path}")
    print(f"[write] {markdown_path}")


if __name__ == "__main__":
    main()
