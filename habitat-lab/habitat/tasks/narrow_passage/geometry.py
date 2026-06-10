#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


FEATURE_NAMES = (
    "d_left_near",
    "d_center_near",
    "d_right_near",
    "d_left_far",
    "d_center_far",
    "d_right_far",
    "clearance_left",
    "clearance_right",
    "passage_width",
    "body_margin",
    "heading_error",
    "lateral_offset",
    "distance_to_local_goal",
    "current_vx",
    "current_wz",
    "stuck_score",
    "collision_flag",
    "previous_action_vx",
    "previous_action_wz",
)

FEATURE_DIM = len(FEATURE_NAMES)

MEMORY_FEATURE_NAMES = (
    "failure_count",
    "failed_state_count",
    "should_recover",
    "should_reject",
)
MEMORY_FEATURE_DIM = len(MEMORY_FEATURE_NAMES)


@dataclass
class NarrowPassageState:
    distance_to_local_goal: float = 0.0
    heading_error: float = 0.0
    lateral_offset: float = 0.0
    current_vx: float = 0.0
    current_wz: float = 0.0
    stuck_score: float = 0.0
    collision_flag: float = 0.0
    previous_action_vx: float = 0.0
    previous_action_wz: float = 0.0
    robot_radius: float = 0.18


@dataclass
class NarrowPassageMemoryState:
    failure_count: float = 0.0
    failed_state_count: float = 0.0
    should_recover: float = 0.0
    should_reject: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.failure_count,
                self.failed_state_count,
                self.should_recover,
                self.should_reject,
            ],
            dtype=np.float32,
        )


def _safe_depth(depth: Optional[np.ndarray], max_depth: float) -> np.ndarray:
    if depth is None:
        return np.full((64, 64), max_depth, dtype=np.float32)

    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.nan_to_num(arr, nan=max_depth, posinf=max_depth, neginf=0.0)
    return np.clip(arr, 0.0, max_depth)


def _roi_min(depth: np.ndarray, row_slice: slice, col_slice: slice) -> float:
    roi = depth[row_slice, col_slice]
    if roi.size == 0:
        return 0.0
    valid = roi[roi > 1e-4]
    if valid.size == 0:
        return 0.0
    return float(np.percentile(valid, 10))


def depth_to_passage_features(
    depth: Optional[np.ndarray],
    state: Optional[NarrowPassageState] = None,
    max_depth: float = 5.0,
) -> np.ndarray:
    """Extract a compact, interpretable narrow-passage observation vector.

    The features are intentionally simple: sector depths from the depth image
    plus geometric/risk state provided by the task or wrapper.
    """

    state = state or NarrowPassageState()
    depth_arr = _safe_depth(depth, max_depth)
    h, w = depth_arr.shape

    near = slice(int(0.55 * h), int(0.9 * h))
    far = slice(int(0.25 * h), int(0.55 * h))
    left = slice(0, int(w / 3))
    center = slice(int(w / 3), int(2 * w / 3))
    right = slice(int(2 * w / 3), w)

    d_left_near = _roi_min(depth_arr, near, left)
    d_center_near = _roi_min(depth_arr, near, center)
    d_right_near = _roi_min(depth_arr, near, right)
    d_left_far = _roi_min(depth_arr, far, left)
    d_center_far = _roi_min(depth_arr, far, center)
    d_right_far = _roi_min(depth_arr, far, right)

    clearance_left = min(d_left_near, d_left_far)
    clearance_right = min(d_right_near, d_right_far)
    passage_width = clearance_left + clearance_right
    body_margin = 0.5 * passage_width - state.robot_radius

    return np.asarray(
        [
            d_left_near,
            d_center_near,
            d_right_near,
            d_left_far,
            d_center_far,
            d_right_far,
            clearance_left,
            clearance_right,
            passage_width,
            body_margin,
            state.heading_error,
            state.lateral_offset,
            state.distance_to_local_goal,
            state.current_vx,
            state.current_wz,
            state.stuck_score,
            state.collision_flag,
            state.previous_action_vx,
            state.previous_action_wz,
        ],
        dtype=np.float32,
    )


def features_to_dict(features: np.ndarray) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(FEATURE_NAMES, features)}
