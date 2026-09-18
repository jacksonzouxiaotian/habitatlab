"""Auditable observation and action contract for the learned mode selector.

The ranges below describe values produced by the current Habitat sensor and
procedural-v2 environment.  They are documentation/validation metadata, not
hard clipping bounds.  Policy normalization is learned from training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


OBSERVATION_DIM = 19
MEMORY_DIM = 4


class Mode(IntEnum):
    COMMIT = 0
    EXPLORE = 1
    RECOVER = 2
    REJECT = 3


MODE_NAMES = tuple(mode.name for mode in Mode)
MEMORY_CONTEXT_NAMES = (
    "geometry_similarity",
    "supported_failure_count",
    "supported_success_count",
    "recurrence_risk",
)


@dataclass(frozen=True)
class FeatureSpec:
    index: int
    name: str
    source: str
    unit: str
    raw_range: str
    normalized_range: str = "running mean/std (unbounded)"


FEATURE_SPECS = (
    FeatureSpec(0, "d_left_near", "depth near-left ROI p10", "m", "[0, 5]"),
    FeatureSpec(1, "d_center_near", "depth near-center ROI p10", "m", "[0, 5]"),
    FeatureSpec(2, "d_right_near", "depth near-right ROI p10", "m", "[0, 5]"),
    FeatureSpec(3, "d_left_far", "depth far-left ROI p10", "m", "[0, 5]"),
    FeatureSpec(4, "d_center_far", "depth far-center ROI p10", "m", "[0, 5]"),
    FeatureSpec(5, "d_right_far", "depth far-right ROI p10", "m", "[0, 5]"),
    FeatureSpec(6, "clearance_left", "min(left near, left far)", "m", "Habitat [0,5]; procedural [-1,5]"),
    FeatureSpec(7, "clearance_right", "min(right near, right far)", "m", "Habitat [0,5]; procedural [-1,5]"),
    FeatureSpec(8, "passage_width", "clearance_left + clearance_right", "m", "Habitat [0,10]; procedural typically [0,10]"),
    FeatureSpec(9, "body_margin", "min(clearances) - robot radius", "m", "Habitat [-r,5-r]; procedural unbounded geometry then finite"),
    FeatureSpec(10, "heading_error", "goal/path heading minus robot yaw", "rad", "[-pi, pi]"),
    FeatureSpec(11, "lateral_offset", "signed offset from start-goal/path line", "m", "unbounded"),
    FeatureSpec(12, "distance_to_local_goal", "planar Euclidean goal distance", "m", "[0, +inf)"),
    FeatureSpec(13, "current_vx", "Habitat planar displacement per task step; procedural commanded vx", "m/step or m/s", "Habitat [0,+inf); procedural [-0.15,0.35]"),
    FeatureSpec(14, "current_wz", "Habitat currently constant 0; procedural commanded wz", "rad/s", "Habitat {0}; procedural [-0.8,0.8]"),
    FeatureSpec(15, "stuck_score", "translation stuck counter / threshold", "ratio", "[0,1]"),
    FeatureSpec(16, "collision_flag", "Habitat-Sim collided / procedural OBB collision", "bool", "{0,1}"),
    FeatureSpec(17, "previous_action_vx", "Habitat normalized requested action; procedural previous vx", "dimensionless or m/s", "Habitat [-1,1]; procedural [-0.15,0.35]"),
    FeatureSpec(18, "previous_action_wz", "Habitat normalized requested action; procedural previous wz", "dimensionless or rad/s", "Habitat [-1,1]; procedural [-0.8,0.8]"),
)


def coerce_observation(observation: Any) -> np.ndarray:
    """Return the 19-D actor observation and enforce the startup contract."""

    if isinstance(observation, dict):
        if "narrow_passage_features" not in observation:
            raise AssertionError(
                "observation dict is missing 'narrow_passage_features'"
            )
        observation = observation["narrow_passage_features"]
    array = np.asarray(observation, dtype=np.float32).reshape(-1)
    assert array.shape[-1] == OBSERVATION_DIM, (
        f"expected {OBSERVATION_DIM}-D observation, got {array.shape}"
    )
    assert np.isfinite(array).all(), "observation contains NaN or Inf"
    return array


def coerce_memory_context(memory_context: Any | None) -> np.ndarray:
    if memory_context is None:
        array = np.zeros(MEMORY_DIM, dtype=np.float32)
    else:
        array = np.asarray(memory_context, dtype=np.float32).reshape(-1)
    assert array.shape[-1] == MEMORY_DIM, (
        f"expected {MEMORY_DIM}-D memory context, got {array.shape}"
    )
    assert np.isfinite(array).all(), "memory context contains NaN or Inf"
    return array


def assert_selector_contract(observation: Any, action_space: Any) -> None:
    """Required fail-fast assertions from the experiment protocol."""

    checked = coerce_observation(observation)
    assert checked.shape[-1] == 19
    assert np.isfinite(checked).all()
    assert getattr(action_space, "n", None) == 4, (
        f"expected Discrete(4), got {action_space!r}"
    )
