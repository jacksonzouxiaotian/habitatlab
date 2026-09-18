"""Per-frame circular ERP centroid baseline."""

from __future__ import annotations

import numpy as np

from eagor_repro.policies.common import DirectionPrediction
from eagor_repro.spherical.spherical_grid import SphericalGrid, direction_to_angles


class CircularCentroidPolicy:
    method = "circular_centroid"

    def __init__(self, grid: SphericalGrid, epsilon: float = 1e-6) -> None:
        self.grid = grid
        self.epsilon = epsilon

    def reset(self) -> None:
        pass

    def update(
        self,
        likelihood: np.ndarray | None,
        target_visible: bool,
        rotation_current_from_previous: np.ndarray | None = None,
    ) -> DirectionPrediction:
        values = (
            np.zeros(self.grid.shape, np.float64)
            if likelihood is None
            else np.nan_to_num(np.asarray(likelihood, np.float64))
        )
        weights = values * self.grid.area_weights
        total = float(weights.sum())
        moment = np.sum(
            weights[..., None] * self.grid.direction_xyz, axis=(0, 1)
        )
        norm = float(np.linalg.norm(moment))
        valid = bool(target_visible and total > self.epsilon and norm > self.epsilon)
        direction = moment / norm if valid else np.asarray([1.0, 0.0, 0.0])
        azimuth, elevation = direction_to_angles(direction)
        concentration = norm / total if valid else 0.0
        heatmap = values / max(float(values.max(initial=0.0)), self.epsilon)
        return DirectionPrediction(
            direction_xyz=direction,
            azimuth=azimuth,
            elevation=elevation,
            confidence=float(np.clip(concentration, 0.0, 1.0)),
            belief_heatmap=heatmap.astype(np.float32),
            valid=valid,
            method=self.method,
        )

