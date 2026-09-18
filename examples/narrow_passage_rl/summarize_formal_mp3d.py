#!/usr/bin/env python3
"""Build the paper-facing MP3D comparison without a strict-SR column."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _mean(rows: list[dict[str, str]], key: str) -> float:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else math.nan


def _rate(rows: list[dict[str, str]], predicate) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else math.nan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pct(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def _decimal(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.1f}"


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be METHOD=CSV_PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--input must be METHOD=CSV_PATH")
    return label.strip(), Path(raw_path)


def _episode_keys(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            (str(row["scene_id"]), str(row["episode_id"]))
            for row in csv.DictReader(handle)
        ]


def summarize(label: str, path: Path) -> tuple[dict, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty evaluation CSV: {path}")

    feasible = [row for row in rows if _float(row, "false_feasible") <= 0.5]
    infeasible = [row for row in rows if _float(row, "false_feasible") > 0.5]
    feasible_successes = sum(_float(row, "success") > 0.5 for row in feasible)
    correct_rejects = sum(_float(row, "rejected") > 0.5 for row in infeasible)
    decision_accuracy = (feasible_successes + correct_rejects) / len(rows)
    result = {
        "method": label,
        "episodes": len(rows),
        "feasible_episodes": len(feasible),
        "false_feasible_episodes": len(infeasible),
        "success_rate": _mean(rows, "success"),
        "feasible_success_rate": _mean(feasible, "success"),
        "false_feasible_correct_reject_rate": _rate(
            infeasible, lambda row: _float(row, "rejected") > 0.5
        ),
        "decision_accuracy": decision_accuracy,
        "collision_rate": _mean(rows, "collision"),
        "stuck_rate": _mean(rows, "stuck"),
        "reject_rate": _mean(rows, "rejected"),
        "near_collision_rate": _mean(rows, "near_collision"),
        "avg_steps": _mean(rows, "steps"),
    }
    provenance = {
        "method": label,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "rows": len(rows),
    }
    return result, provenance


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Formal MP3D Narrow-Passage Comparison",
        "",
        (
            "All methods use the same scene-disjoint validation episodes. "
            "The table intentionally omits strict success rate."
        ),
        "",
        "| Method | N | Success | Feasible SR | Correct reject | Decision accuracy | Collision | Stuck | Reject | Near collision | Avg steps |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {episodes} | {success} | {feasible} | {correct_reject} | "
            "{accuracy} | {collision} | {stuck} | {reject} | {near} | {steps} |".format(
                method=row["method"],
                episodes=row["episodes"],
                success=_pct(row["success_rate"]),
                feasible=_pct(row["feasible_success_rate"]),
                correct_reject=_pct(row["false_feasible_correct_reject_rate"]),
                accuracy=_pct(row["decision_accuracy"]),
                collision=_pct(row["collision_rate"]),
                stuck=_pct(row["stuck_rate"]),
                reject=_pct(row["reject_rate"]),
                near=_pct(row["near_collision_rate"]),
                steps=_decimal(row["avg_steps"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=_parse_input, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    provenance = []
    episode_keys: dict[str, list[tuple[str, str]]] = {}
    for label, path in args.input:
        if not path.is_file():
            raise FileNotFoundError(path)
        summary, source = summarize(label, path)
        summaries.append(summary)
        provenance.append(source)
        episode_keys[label] = _episode_keys(path)

    canonical_label = next(iter(episode_keys))
    canonical_keys = episode_keys[canonical_label]
    canonical_set = set(canonical_keys)
    protocol_matches = {
        label: len(keys) == len(canonical_keys)
        and len(set(keys)) == len(keys)
        and set(keys) == canonical_set
        for label, keys in episode_keys.items()
    }
    if not all(protocol_matches.values()):
        raise ValueError(
            "input methods do not contain the same unique held-out episodes: "
            f"{protocol_matches}"
        )

    write_csv(args.output_dir / "formal_mp3d_comparison.csv", summaries)
    write_markdown(args.output_dir / "formal_mp3d_comparison.md", summaries)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "inputs": provenance,
        "strict_success_rate_in_table": False,
        "episode_protocol_audit": {
            "key": ["scene_id", "episode_id"],
            "canonical_method": canonical_label,
            "unique_episodes": len(canonical_set),
            "all_methods_match_episode_set": True,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "formal_mp3d_comparison.md")


if __name__ == "__main__":
    main()
