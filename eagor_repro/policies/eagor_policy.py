"""Policy adapter around the SH-BF estimator."""

from __future__ import annotations

import numpy as np

from eagor_repro.policies.common import DirectionPrediction
from eagor_repro.spherical.belief_filter import SphericalHarmonicBeliefFilter


class EAGORPolicy:
    method = "eagor"

    def __init__(self, belief_filter: SphericalHarmonicBeliefFilter) -> None:
        self.belief_filter = belief_filter

    def reset(self) -> None:
        self.belief_filter.reset()

    def update(
        self,
        likelihood: np.ndarray | None,
        target_visible: bool,
        rotation_current_from_previous: np.ndarray | None = None,
    ) -> DirectionPrediction:
        estimate = self.belief_filter.update(
            likelihood,
            target_visible,
            rotation_current_from_previous,
        )
        return DirectionPrediction(
            direction_xyz=estimate.direction_xyz,
            azimuth=estimate.azimuth,
            elevation=estimate.elevation,
            confidence=estimate.confidence,
            belief_heatmap=estimate.belief_heatmap,
            valid=estimate.valid,
            method=self.method,
        )

