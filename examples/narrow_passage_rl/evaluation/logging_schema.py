"""Canonical per-episode logging schema for narrow-passage evaluators.

The helpers in this file standardize CSV rows without pretending that every
baseline exposes every internal signal.  Unsupported belief-state quantities are
written as NaN, and methods without the four-mode interface are explicitly marked
with ``mode_interface_available=False``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


MODE_NAMES = ("commit", "explore", "recover", "reject")
BELIEF_KEYS = (
    "d_hat",
    "w_req_prior",
    "w_req_cons",
    "delta_mean",
    "delta_var",
    "p_feas",
    "risk",
    "memory_risk",
)
BELIEF_AGG_STATS = ("mean", "min", "max", "final")

EPISODE_LOG_FIELDS = [
    "episode_id",
    "scene_id",
    "split",
    "seed",
    "method",
    "success",
    "strict_success",
    "collision",
    "near_collision",
    "steps",
    "path_length",
    "min_clearance",
    "body_margin",
    "d_hat",
    "w_req_prior",
    "w_req_cons",
    "delta_mean",
    "delta_var",
    "p_feas",
    "risk",
    "memory_risk",
    "commit_count",
    "explore_count",
    "recover_count",
    "reject_count",
    "total_mode_count",
    "commit_ratio",
    "explore_ratio",
    "recover_ratio",
    "reject_ratio",
    "final_mode",
    "belief_available",
    "mode_interface_available",
]

BELIEF_AGG_FIELDS = [
    f"{key}_{stat}" for key in BELIEF_KEYS for stat in BELIEF_AGG_STATS
]

CANONICAL_CSV_FIELDS = EPISODE_LOG_FIELDS + BELIEF_AGG_FIELDS


def nan() -> float:
    return float("nan")


def is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def mode_bucket(mode: str) -> str:
    """Map implementation-specific mode strings to the four paper modes."""

    m = str(mode or "").lower()
    if "reject" in m:
        return "reject"
    if "recover" in m:
        return "recover"
    if "explore" in m or "align" in m or "follow" in m or "corridor" in m:
        return "explore"
    if "commit" in m or "stop" in m:
        return "commit"
    return "commit"


def mode_count_ratio_fields(
    mode_counts: Mapping[str, Any] | None,
    *,
    mode_interface_available: bool,
) -> dict[str, Any]:
    """Return canonical mode counts and ratios.

    When no four-mode interface exists, the count fields are NaN instead of 0 so
    downstream analysis does not mistake a direct-velocity/APF policy for a mode
    policy that never used Recover.
    """

    out: dict[str, Any] = {}
    if not mode_interface_available:
        for mode in MODE_NAMES:
            out[f"{mode}_count"] = nan()
            out[f"{mode}_ratio"] = nan()
        out["total_mode_count"] = nan()
        return out

    counts = {
        mode: max(0, int(float((mode_counts or {}).get(mode, 0))))
        for mode in MODE_NAMES
    }
    total = int(sum(counts.values()))
    denom = max(total, 1)
    for mode in MODE_NAMES:
        out[f"{mode}_count"] = counts[mode]
    out["total_mode_count"] = total
    for mode in MODE_NAMES:
        out[f"{mode}_ratio"] = float(counts[mode] / denom)
    return out


def aggregate_belief(
    samples: Iterable[Mapping[str, Any]] | None,
) -> dict[str, float]:
    """Aggregate belief/risk signals over an episode.

    The returned keys always include ``<name>_mean``, ``<name>_min``,
    ``<name>_max``, and ``<name>_final`` for each belief key.  Empty or missing
    signals become NaN.
    """

    series = list(samples or [])
    out: dict[str, float] = {}
    for key in BELIEF_KEYS:
        values = [
            value
            for value in (_finite_float(sample.get(key)) for sample in series)
            if value is not None
        ]
        final = values[-1] if values else nan()
        out[f"{key}_mean"] = float(np.mean(values)) if values else nan()
        out[f"{key}_min"] = float(np.min(values)) if values else nan()
        out[f"{key}_max"] = float(np.max(values)) if values else nan()
        out[f"{key}_final"] = float(final)
    return out


def belief_final_fields(
    belief: Mapping[str, Any] | None,
    aggregates: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in BELIEF_KEYS:
        value = None
        if belief is not None:
            value = _finite_float(belief.get(key))
        if value is None and aggregates is not None:
            value = _finite_float(aggregates.get(f"{key}_final"))
        out[key] = float(value) if value is not None else nan()
    return out


def finalize_episode_row(
    row: Mapping[str, Any],
    *,
    belief: Mapping[str, Any] | None = None,
    belief_samples: Iterable[Mapping[str, Any]] | None = None,
    belief_available: bool | None = None,
    mode_counts: Mapping[str, Any] | None = None,
    mode_interface_available: bool | None = None,
    final_mode: str | None = None,
) -> dict[str, Any]:
    """Return a row with canonical fields filled and extra fields preserved."""

    base = dict(row)
    samples = list(belief_samples or [])
    if belief_available is None:
        belief_available = bool(belief is not None or samples)
    if mode_interface_available is None:
        mode_interface_available = mode_counts is not None

    aggregates = aggregate_belief(samples if belief_available else [])
    belief_fields = (
        belief_final_fields(belief, aggregates)
        if belief_available
        else {key: nan() for key in BELIEF_KEYS}
    )
    if not belief_available:
        aggregates = {field: nan() for field in BELIEF_AGG_FIELDS}

    mode_fields = mode_count_ratio_fields(
        mode_counts,
        mode_interface_available=bool(mode_interface_available),
    )
    final_mode_value = (
        final_mode
        if final_mode is not None
        else base.get("final_mode", base.get("mode", "unavailable"))
    )

    canonical = {
        "episode_id": base.get("episode_id", base.get("episode", "")),
        "scene_id": base.get("scene_id", ""),
        "split": base.get("split", ""),
        "seed": base.get("seed", nan()),
        "method": base.get("method", ""),
        "success": base.get("success", nan()),
        "strict_success": base.get("strict_success", nan()),
        "collision": base.get("collision", nan()),
        "near_collision": base.get("near_collision", nan()),
        "steps": base.get("steps", nan()),
        "path_length": base.get("path_length", nan()),
        "min_clearance": base.get(
            "min_clearance",
            base.get("min_body_margin", nan()),
        ),
        "body_margin": base.get(
            "body_margin",
            base.get("min_body_margin", nan()),
        ),
        **belief_fields,
        **mode_fields,
        "final_mode": str(final_mode_value or "unavailable"),
        "belief_available": bool(belief_available),
        "mode_interface_available": bool(mode_interface_available),
        **aggregates,
    }

    out = {field: canonical.get(field, nan()) for field in CANONICAL_CSV_FIELDS}
    for key, value in base.items():
        if key not in out:
            out[key] = value
    return out


def fieldnames_for_rows(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return stable CSV header: canonical schema first, extras in row order."""

    fields = list(CANONICAL_CSV_FIELDS)
    seen = set(fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def validate_csv_row(row: Mapping[str, Any]) -> None:
    missing = [field for field in CANONICAL_CSV_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Episode row is missing canonical fields: {missing}")
