"""Ground-truth target-direction policy for diagnostic upper bounds only."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from eagor_repro.policies.common import DirectionPrediction
from eagor_repro.spherical.spherical_grid import SphericalGrid, direction_to_angles


class OracleDirectionPolicy:
    """Return the egocentric bearing to the nearest valid target instance.

    This policy deliberately bypasses perception and belief estimation.  It is
    only valid for failure attribution and every evaluator summary using it is
    marked ``oracle_upper_bound=true``.
    """

    method = "oracle_direction"

    def __init__(self, grid: SphericalGrid, heatmap_sigma_deg: float = 5.0) -> None:
        self.grid = grid
        self.heatmap_sigma_deg = float(heatmap_sigma_deg)

    def reset(self) -> None:
        pass

    @staticmethod
    def nearest_egocentric_direction(
        agent_position: Sequence[float],
        rotation_world_from_body: np.ndarray,
        target_positions: Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, float, int]:
        """Return body-frame direction, Euclidean distance, and target index."""

        position = np.asarray(agent_position, dtype=np.float64)
        rotation = np.asarray(rotation_world_from_body, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation_world_from_body must have shape (3, 3)")
        targets = np.asarray(target_positions, dtype=np.float64).reshape(-1, 3)
        if len(targets) == 0:
            raise ValueError("target_positions must contain at least one target")
        deltas = targets - position
        distances = np.linalg.norm(deltas, axis=1)
        index = int(np.argmin(distances))
        distance = float(distances[index])
        if distance <= 1e-12:
            direction_body = np.asarray([1.0, 0.0, 0.0], np.float64)
        else:
            direction_body = rotation.T @ (deltas[index] / distance)
            direction_body /= max(float(np.linalg.norm(direction_body)), 1e-12)
        return direction_body, distance, index

    def predict_direction(self, direction_xyz: np.ndarray) -> DirectionPrediction:
        direction = np.asarray(direction_xyz, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("Oracle direction must be non-zero")
        direction = direction / norm
        azimuth, elevation = direction_to_angles(direction)
        return DirectionPrediction(
            direction_xyz=direction,
            azimuth=azimuth,
            elevation=elevation,
            confidence=1.0,
            belief_heatmap=self.grid.gaussian_likelihood(
                direction, sigma_deg=self.heatmap_sigma_deg
            ),
            valid=True,
            method=self.method,
        )

    def predict_nearest(
        self,
        agent_position: Sequence[float],
        rotation_world_from_body: np.ndarray,
        target_positions: Sequence[Sequence[float]],
    ) -> tuple[DirectionPrediction, float, int]:
        direction, distance, index = self.nearest_egocentric_direction(
            agent_position, rotation_world_from_body, target_positions
        )
        return self.predict_direction(direction), distance, index
