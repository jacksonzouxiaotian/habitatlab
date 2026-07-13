#!/usr/bin/env python3
"""Validate harder_benchmark_episodes.csv against the current logging schema."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.logging_schema import CANONICAL_CSV_FIELDS, BELIEF_KEYS  # noqa: E402


REQUIRED_FIELDS = [
    "d_hat",
    "w_req_cons",
    "delta_mean",
    "delta_var",
    "p_feas",
    "risk",
    "commit_count",
    "explore_count",
    "recover_count",
    "reject_count",
]

BOOL_FIELDS = [
    "success",
    "strict_success",
    "collision",
    "near_collision",
    "belief_available",
    "mode_interface_available",
]

MODE_COUNT_FIELDS = ["commit_count", "explore_count", "recover_count", "reject_count"]
MODE_RATIO_FIELDS = ["commit_ratio", "explore_ratio", "recover_ratio", "reject_ratio"]


def _float(value: object) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out


def _finite(value: object) -> float | None:
    out = _float(value)
    if out is None or not math.isfinite(out):
        return None
    return out


def _boolish(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return True
    val = _float(value)
    return val in (0.0, 1.0)


def _truthy(value: object) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    val = _float(value)
    return bool(val and val > 0.5)


def _nanish(value: object) -> bool:
    val = _float(value)
    return val is not None and math.isnan(val)


def validate_rows(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["CSV contains no rows."]

    header = set(rows[0].keys())
    for field in CANONICAL_CSV_FIELDS + REQUIRED_FIELDS:
        if field not in header:
            errors.append(f"Missing required field: {field}")

    seen_episode_ids: set[str] = set()
    duplicate_ids = 0
    belief_by_method: dict[str, list[bool]] = defaultdict(list)
    belief_non_nan_by_method: dict[str, int] = defaultdict(int)

    for idx, row in enumerate(rows, start=2):
        eid = str(row.get("episode_id", "")).strip()
        if not eid:
            errors.append(f"Line {idx}: episode_id is empty.")
        elif eid in seen_episode_ids:
            duplicate_ids += 1
        else:
            seen_episode_ids.add(eid)

        for field in MODE_COUNT_FIELDS:
            value = _finite(row.get(field))
            if _truthy(row.get("mode_interface_available")) and value is not None and value < 0:
                errors.append(f"Line {idx}: {field} is negative.")

        for field in MODE_RATIO_FIELDS:
            value = _finite(row.get(field))
            if value is not None and not (0.0 <= value <= 1.0):
                errors.append(f"Line {idx}: {field}={value} outside [0, 1].")

        p_feas = _finite(row.get("p_feas"))
        if p_feas is not None and not (0.0 <= p_feas <= 1.0):
            errors.append(f"Line {idx}: p_feas={p_feas} outside [0, 1].")

        delta_var = _finite(row.get("delta_var"))
        if delta_var is not None and delta_var < 0.0:
            errors.append(f"Line {idx}: delta_var={delta_var} is negative.")

        for field in BOOL_FIELDS:
            if field in row and str(row.get(field, "")).strip() and not _boolish(row[field]):
                errors.append(f"Line {idx}: {field}={row[field]!r} is not bool-like.")

        method = str(row.get("method", "")).strip() or "(missing method)"
        belief_available = _truthy(row.get("belief_available"))
        belief_by_method[method].append(belief_available)
        if belief_available:
            if any(not _nanish(row.get(key)) for key in BELIEF_KEYS):
                belief_non_nan_by_method[method] += 1
        else:
            for key in BELIEF_KEYS:
                if key in row and not _nanish(row.get(key)):
                    errors.append(
                        f"Line {idx}: {method} has belief_available=false but {key} is not NaN."
                    )

    if duplicate_ids:
        errors.append(f"Found {duplicate_ids} duplicate episode_id values.")

    for method, flags in sorted(belief_by_method.items()):
        if any(flags) and belief_non_nan_by_method[method] == 0:
            errors.append(f"{method}: belief_available=true but all belief values are NaN.")

    return errors


def _coverage(rows: Iterable[dict[str, str]], fields: Iterable[str]) -> dict[str, float]:
    rows = list(rows)
    out = {}
    for field in fields:
        good = sum(1 for row in rows if _finite(row.get(field)) is not None)
        out[field] = good / max(len(rows), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "results" / "narrow_passage_rl" / "harder_benchmark_episodes.csv",
    )
    args = parser.parse_args()

    with args.csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    errors = validate_rows(rows)
    print(f"[validate] {args.csv}")
    print(f"  rows: {len(rows)}")
    print(f"  columns: {len(reader.fieldnames or [])}")
    print("  required coverage:")
    for field, frac in _coverage(rows, REQUIRED_FIELDS).items():
        print(f"    {field:16s}: {frac:.3f}")

    if errors:
        print("  status: FAIL")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("  status: PASS")


if __name__ == "__main__":
    main()
