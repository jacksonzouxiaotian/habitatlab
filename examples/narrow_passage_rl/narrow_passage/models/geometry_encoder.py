"""Explicit passage geometry encoder.

The encoder is intentionally lightweight: it standardizes the interpretable
geometry vector used by both the FSM and learned baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GEOMETRY_FEATURES = [
    "passage_width",
    "robot_body_width",
    "clearance_left",
    "clearance_right",
    "min_clearance",
    "width_body_ratio",
    "entrance_angle",
    "obstacle_asymmetry",
    "local_goal_angle",
    "velocity_history",
    "stuck_score",
]


@dataclass
class PassageGeometry:
    passage_width: float
    robot_body_width: float
    clearance_left: float
    clearance_right: float
    min_clearance: float
    width_body_ratio: float
    entrance_angle: float
    obstacle_asymmetry: float
    local_goal_angle: float
    velocity_history: float
    stuck_score: float

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in GEOMETRY_FEATURES], dtype=np.float32)


class GeometryEncoder:
    """Normalize an interpretable geometry vector into `z_g`."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> None:
        self.mean = np.zeros(len(GEOMETRY_FEATURES), dtype=np.float32) if mean is None else mean
        self.std = np.ones(len(GEOMETRY_FEATURES), dtype=np.float32) if std is None else std

    def __call__(self, geometry: PassageGeometry | np.ndarray) -> np.ndarray:
        x = geometry.as_array() if isinstance(geometry, PassageGeometry) else np.asarray(geometry, dtype=np.float32)
        return (x - self.mean) / np.maximum(self.std, 1e-6)
