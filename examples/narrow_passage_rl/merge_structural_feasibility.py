#!/usr/bin/env python3
"""Merge three completed structural-feasibility seed shards with fail-fast audit."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_structural_feasibility import (
    EVALUATION_METHODS,
    LATERAL,
    MARGIN_BINS,
    NOISE,
    SCENES,
    YAWS,
    _gate_report,
    _summary,
    _write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output_root = args.output_root or args.root
    if output_root != args.root and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    shards = [args.root / f"seed{seed}" for seed in (0, 1, 2)]
    for shard in shards:
        for name in ("episodes.csv", "steps.csv", "run_metadata.json", "gate_report.json"):
            if not (shard / name).is_file():
                raise FileNotFoundError(f"Incomplete shard: {shard / name}")

    rows: list[dict[str, Any]] = []
    for shard in shards:
        with (shard / "episodes.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    for row in rows:
        raw_passable = str(row.get("passable", row.get("passable_label", ""))).lower()
        row["passable"] = raw_passable in {"1", "1.0", "true"}
        row["passable_label"] = float(bool(row["passable"]))
    shard_metadata = [
        json.loads((shard / "run_metadata.json").read_text(encoding="utf-8"))
        for shard in shards
    ]
    method_lists = {tuple(item["methods"]) for item in shard_metadata}
    if len(method_lists) != 1:
        raise RuntimeError(f"Shard method mismatch: {method_lists}")
    methods = next(iter(method_lists))
    if set(methods) != set(EVALUATION_METHODS):
        raise RuntimeError(f"Expected frozen six-method protocol, got {methods}")
    factor_designs = {item["factor_design"] for item in shard_metadata}
    if factor_designs != {"publication"}:
        raise RuntimeError(f"Expected publication factor design: {factor_designs}")
    scenarios_per_seed = len(MARGIN_BINS) * len(SCENES) * len(YAWS) * len(NOISE)
    expected_rows = 3 * scenarios_per_seed * len(methods)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} episode rows, got {len(rows)}")

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario_id"])].append(row)
    expected_methods = set(methods)
    for scenario_id, paired in by_scenario.items():
        if {str(row["method"]) for row in paired} != expected_methods:
            raise RuntimeError(f"Method pairing failure: {scenario_id}")
        if len({row["observation_sha256"] for row in paired}) != 1:
            raise RuntimeError(f"Observation hash mismatch: {scenario_id}")
        if len({row["random_draw_sha256"] for row in paired}) != 1:
            raise RuntimeError(f"Random hash mismatch: {scenario_id}")

    unique = [paired[0] for paired in by_scenario.values()]
    cell_counts = Counter(
        (row["seed"], row["margin_bin"], row["corridor_type"]) for row in unique
    )
    if set(cell_counts.values()) != {len(YAWS) * len(NOISE)}:
        raise RuntimeError(f"Unbalanced margin×scene×seed cells: {cell_counts}")
    axes = {
        "yaw": Counter(int(float(row["yaw_deg"])) for row in unique),
        "noise": Counter(row["noise_level"] for row in unique),
        "lateral": Counter(float(row["lateral_offset"]) for row in unique),
    }
    for name, counts in axes.items():
        expected_axis_count = len(unique) // len(counts)
        if len(set(counts.values())) != 1 or next(iter(counts.values())) != expected_axis_count:
            raise RuntimeError(f"Unbalanced {name} axis: {counts}")

    episodes_out = output_root / "episodes.csv"
    _write_csv(rows, episodes_out)
    steps_out = output_root / "steps.csv"
    if steps_out.exists():
        raise FileExistsError(f"Refusing to overwrite {steps_out}")
    header: str | None = None
    with steps_out.open("w", encoding="utf-8", newline="") as destination:
        for shard in shards:
            with (shard / "steps.csv").open(encoding="utf-8", newline="") as source:
                shard_header = source.readline()
                if header is None:
                    header = shard_header
                    destination.write(header)
                elif shard_header != header:
                    raise RuntimeError(f"Step schema mismatch in {shard}")
                shutil.copyfileobj(source, destination, length=1024 * 1024)

    _write_csv(_summary(rows, tuple(methods)), output_root / "subset_metrics.csv")
    gate = _gate_report(rows, tuple(methods))
    gate["three_seed_merge_validated"] = True
    gate["paired_scenario_count"] = len(unique)
    gate["episode_row_count"] = len(rows)
    gate["axis_counts"] = {name: dict(counts) for name, counts in axes.items()}
    (output_root / "gate_report.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "merged_three_seed_structural_feasibility_evaluation",
        "shards": [str(shard.resolve()) for shard in shards],
        "paired_scenario_count": len(unique),
        "episode_row_count": len(rows),
        "methods": methods,
        "balanced_cell_size": len(YAWS) * len(NOISE),
        "axis_counts": {name: dict(counts) for name, counts in axes.items()},
        "protected_legacy_result": "paper_dynamic_20260818_170356",
        "protocol": json.loads(
            (shards[0] / "run_metadata.json").read_text(encoding="utf-8")
        ),
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
