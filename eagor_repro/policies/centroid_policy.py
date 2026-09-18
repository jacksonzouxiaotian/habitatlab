"""Naive ERP pixel centroid baseline (deliberately seam-incorrect)."""

from __future__ import annotations

import numpy as np

from eagor_repro.policies.common import DirectionPrediction
from eagor_repro.spherical.spherical_grid import (
    SphericalGrid,
    direction_to_angles,
    erp_pixel_to_sphere,
)


class CentroidPolicy:
    method = "centroid"

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
        total = float(values.sum())
        valid = bool(target_visible and total > self.epsilon)
        if valid:
            yy, xx = np.mgrid[: self.grid.height, : self.grid.width]
            u = float((values * (xx + 0.5)).sum() / total)
            v = float((values * (yy + 0.5)).sum() / total)
            direction = erp_pixel_to_sphere(
                u, v, self.grid.width, self.grid.height
            )
            concentration = float(values.max(initial=0.0))
        else:
            direction = np.asarray([1.0, 0.0, 0.0])
            concentration = 0.0
        azimuth, elevation = direction_to_angles(direction)
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

