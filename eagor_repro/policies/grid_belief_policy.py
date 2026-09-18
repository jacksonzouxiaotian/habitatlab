"""ERP-grid temporal accumulation baseline with circular yaw interpolation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as ndimage_shift

from eagor_repro.policies.common import DirectionPrediction
from eagor_repro.spherical.spherical_grid import SphericalGrid, direction_to_angles


class GridBeliefPolicy:
    method = "grid"

    def __init__(
        self,
        grid: SphericalGrid,
        decay: float = 1.0,
        interpolation_order: int = 1,
        epsilon: float = 1e-6,
    ) -> None:
        self.grid = grid
        self.decay = float(decay)
        self.interpolation_order = int(interpolation_order)
        self.epsilon = epsilon
        self.accumulator: np.ndarray | None = None

    def reset(self) -> None:
        self.accumulator = None

    def _propagate(self, rotation: np.ndarray | None) -> None:
        if self.accumulator is None or rotation is None:
            return
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        column_shift = yaw / (2.0 * np.pi) * self.grid.width
        # ERP engineering baseline: bilinear interpolation, vertical nearest,
        # and explicit circular wrapping in azimuth.
        self.accumulator = ndimage_shift(
            self.accumulator,
            shift=(0.0, column_shift),
            order=self.interpolation_order,
            mode="grid-wrap",
            prefilter=self.interpolation_order > 1,
        )

    def update(
        self,
        likelihood: np.ndarray | None,
        target_visible: bool,
        rotation_current_from_previous: np.ndarray | None = None,
    ) -> DirectionPrediction:
        self._propagate(rotation_current_from_previous)
        if self.accumulator is not None:
            self.accumulator *= self.decay
        if target_visible and likelihood is not None:
            observation = np.clip(
                np.nan_to_num(np.asarray(likelihood, np.float64)), 0.0, 1.0
            )
            self.accumulator = (
                observation.copy()
                if self.accumulator is None
                else self.accumulator + observation
            )
        if self.accumulator is None:
            self.accumulator = np.zeros(self.grid.shape, np.float64)

        weights = self.accumulator * self.grid.area_weights
        total = float(weights.sum())
        moment = np.sum(
            weights[..., None] * self.grid.direction_xyz, axis=(0, 1)
        )
        norm = float(np.linalg.norm(moment))
        valid = total > self.epsilon and norm > self.epsilon
        direction = moment / norm if valid else np.asarray([1.0, 0.0, 0.0])
        azimuth, elevation = direction_to_angles(direction)
        heatmap = self.accumulator / max(
            float(self.accumulator.max(initial=0.0)), self.epsilon
        )
        return DirectionPrediction(
            direction_xyz=direction,
            azimuth=azimuth,
            elevation=elevation,
            confidence=float(np.clip(norm / max(total, self.epsilon), 0.0, 1.0)),
            belief_heatmap=heatmap.astype(np.float32),
            valid=valid,
            method=self.method,
        )

