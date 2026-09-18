from __future__ import annotations

import numpy as np

from eagor_repro.policies.centroid_policy import CentroidPolicy
from eagor_repro.policies.circular_centroid_policy import CircularCentroidPolicy
from eagor_repro.spherical.spherical_grid import SphericalGrid, wrap_to_pi


def _seam_likelihood(grid: SphericalGrid) -> np.ndarray:
    first = grid.gaussian_likelihood(
        [np.cos(np.deg2rad(179)), np.sin(np.deg2rad(179)), 0], 5
    )
    second = grid.gaussian_likelihood(
        [np.cos(np.deg2rad(-179)), np.sin(np.deg2rad(-179)), 0], 5
    )
    return np.maximum(first, second)


def test_centroid_seam_failure() -> None:
    grid = SphericalGrid(48, 96)
    estimate = CentroidPolicy(grid).update(_seam_likelihood(grid), True)
    error = abs(np.rad2deg(wrap_to_pi(estimate.azimuth - np.pi)))
    assert error > 120.0


def test_circular_centroid_seam_success() -> None:
    grid = SphericalGrid(48, 96)
    estimate = CircularCentroidPolicy(grid).update(_seam_likelihood(grid), True)
    error = abs(np.rad2deg(wrap_to_pi(estimate.azimuth - np.pi)))
    assert error < 5.0

