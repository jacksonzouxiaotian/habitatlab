#!/usr/bin/env python3
"""Paired summary for baseline NaVILA and online NaVILA+DEGNAV rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = (
    ("success", "higher"),
    ("spl", "higher"),
    ("ndtw", "higher"),
    ("distance_to_goal", "lower"),
    ("path_length", "lower"),
    ("steps_taken", "lower"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_chunks(folder: Path, split: str, chunks: int) -> tuple[dict, list[dict]]:
    episodes: dict[str, dict] = {}
    sources = []
    for index in range(chunks):
        path = folder / f"{split}_{chunks}-{index}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        overlap = set(episodes).intersection(data)
        if overlap:
            raise ValueError(f"duplicate episode IDs in {path}: {sorted(overlap)[:5]}")
        episodes.update(data)
        sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    return episodes, sources


def paired_ci(delta: np.ndarray, seed: int = 20260901) -> tuple[float, float]:
    if delta.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(4000, delta.size))
    means = delta[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def summarize_metrics(baseline: dict, adapted: dict) -> tuple[list[dict], list[dict]]:
    if set(baseline) != set(adapted):
        missing = sorted(set(baseline) - set(adapted))[:10]
        extra = sorted(set(adapted) - set(baseline))[:10]
        raise ValueError(f"episode mismatch: missing={missing}, extra={extra}")
    paired_rows = []
    for episode_id in sorted(baseline, key=lambda value: (len(value), value)):
        row = {"episode_id": episode_id}
        for metric, _ in METRICS:
            base = float(baseline[episode_id][metric])
            adapted_value = float(adapted[episode_id][metric])
            row[f"baseline_{metric}"] = base
            row[f"adapted_{metric}"] = adapted_value
            row[f"delta_{metric}"] = adapted_value - base
        adapter = adapted[episode_id].get("degnav_adapter", {})
        row["adapter_interventions"] = int(adapter.get("interventions", 0))
        row["adapter_memory_writes"] = int(adapter.get("memory_writes", 0))
        paired_rows.append(row)

    summary = []
    for metric, direction in METRICS:
        base_values = np.asarray(
            [row[f"baseline_{metric}"] for row in paired_rows], dtype=float
        )
        adapted_values = np.asarray(
            [row[f"adapted_{metric}"] for row in paired_rows], dtype=float
        )
        delta = adapted_values - base_values
        ci_low, ci_high = paired_ci(delta)
        summary.append(
            {
                "metric": metric,
                "direction": direction,
                "episodes": len(paired_rows),
                "baseline": float(base_values.mean()),
                "adapted": float(adapted_values.mean()),
                "delta": float(delta.mean()),
                "paired_delta_ci95_low": ci_low,
                "paired_delta_ci95_high": ci_high,
            }
        )
    return summary, paired_rows


def adapter_summary(adapted: dict) -> dict:
    modes: Counter[str] = Counter()
    proposals: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    totals = Counter()
    intervened_episodes = 0
    for episode in adapted.values():
        stats = episode.get("degnav_adapter", {})
        modes.update(stats.get("mode_counts", {}))
        proposals.update(stats.get("proposal_counts", {}))
        actions.update(stats.get("action_counts", {}))
        interventions = int(stats.get("interventions", 0))
        intervened_episodes += int(interventions > 0)
        for key in ("interventions", "collisions_observed", "memory_writes"):
            totals[key] += int(stats.get(key, 0))
    decisions = sum(modes.values())
    return {
        "episodes": len(adapted),
        "intervened_episodes": intervened_episodes,
        "intervened_episode_rate": intervened_episodes / max(1, len(adapted)),
        "decisions": decisions,
        "interventions": totals["interventions"],
        "intervention_rate_per_decision": totals["interventions"] / max(1, decisions),
        "collisions_observed": totals["collisions_observed"],
        "memory_writes": totals["memory_writes"],
        "mode_counts": dict(modes),
        "proposal_counts": dict(proposals),
        "action_counts": dict(actions),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict], adapter: dict) -> None:
    lines = [
        "# Paired NaVILA vs NaVILA + Online DEGNAV",
        "",
        "All rows pair the exact same R2R–MP3D episode IDs.",
        "",
        "| Metric | Direction | N | NaVILA | + DEGNAV | Delta | Paired 95% CI |",
        "|:---|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {metric} | {direction} | {episodes} | {baseline:.4f} | "
            "{adapted:.4f} | {delta:+.4f} | [{low:+.4f}, {high:+.4f}] |".format(
                metric=row["metric"],
                direction=row["direction"],
                episodes=row["episodes"],
                baseline=row["baseline"],
                adapted=row["adapted"],
                delta=row["delta"],
                low=row["paired_delta_ci95_low"],
                high=row["paired_delta_ci95_high"],
            )
        )
    lines.extend(
        [
            "",
            "## Adapter activity",
            "",
            f"- Episodes with at least one intervention: {adapter['intervened_episodes']}/{adapter['episodes']} ({100.0 * adapter['intervened_episode_rate']:.1f}%).",
            f"- Interventions: {adapter['interventions']}/{adapter['decisions']} decisions ({100.0 * adapter['intervention_rate_per_decision']:.1f}%).",
            f"- Collision observations / memory writes: {adapter['collisions_observed']} / {adapter['memory_writes']}.",
            f"- Mode counts: `{json.dumps(adapter['mode_counts'], sort_keys=True)}`.",
            "- Failure memory persists within each sequential evaluation chunk; it is reset between chunks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--adapted-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline, baseline_sources = load_chunks(
        args.baseline_dir, args.split, args.chunks
    )
    adapted, adapted_sources = load_chunks(
        args.adapted_dir, args.split, args.chunks
    )
    summary, paired_rows = summarize_metrics(baseline, adapted)
    activity = adapter_summary(adapted)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metric_comparison.csv", summary)
    write_csv(args.output_dir / "paired_episodes.csv", paired_rows)
    write_markdown(args.output_dir / "comparison.md", summary, activity)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "chunks": args.chunks,
        "episodes": len(paired_rows),
        "baseline_sources": baseline_sources,
        "adapted_sources": adapted_sources,
        "adapter_activity": activity,
        "memory_scope": "persistent within each sequential chunk; reset between chunks",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "comparison.md")


if __name__ == "__main__":
    main()
