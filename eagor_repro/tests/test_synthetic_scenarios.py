from __future__ import annotations

import numpy as np

from eagor_repro.evaluation.metrics import angular_error_deg
from eagor_repro.spherical.belief_filter import SphericalHarmonicBeliefFilter
from eagor_repro.spherical.rotation import yaw_rotation
from eagor_repro.spherical.spherical_grid import SphericalGrid


def _direction(azimuth_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
    azimuth, elevation = np.deg2rad([azimuth_deg, elevation_deg])
    return np.asarray(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def _filter(grid: SphericalGrid) -> SphericalHarmonicBeliefFilter:
    return SphericalHarmonicBeliefFilter(
        grid, bandlimit=7, decode_mode="probability"
    )


def test_dual_peak_preserves_spherical_mean() -> None:
    grid = SphericalGrid(40, 80)
    first, second = _direction(-25), _direction(25)
    likelihood = np.maximum(
        grid.gaussian_likelihood(first, 8), grid.gaussian_likelihood(second, 8)
    )
    estimate = _filter(grid).update(likelihood, True)
    assert angular_error_deg(estimate.direction_xyz, _direction(0)) < 2.0


def test_polar_peak_not_erp_overweighted() -> None:
    grid = SphericalGrid(48, 96)
    target = _direction(35, 75)
    estimate = _filter(grid).update(grid.gaussian_likelihood(target, 12), True)
    assert angular_error_deg(estimate.direction_xyz, target) < 4.0


def test_noisy_peak_remains_stable() -> None:
    rng = np.random.default_rng(0)
    grid = SphericalGrid(48, 96)
    target = _direction(123, -15)
    likelihood = grid.gaussian_likelihood(target, 10)
    likelihood = np.clip(likelihood + rng.uniform(0, 0.03, grid.shape), 0, 1)
    estimate = _filter(grid).update(likelihood, True)
    assert angular_error_deg(estimate.direction_xyz, target) < 6.0


def test_occlusion_then_reacquisition() -> None:
    grid = SphericalGrid(40, 80)
    filt = _filter(grid)
    filt.update(grid.gaussian_likelihood(_direction(0), 10), True)
    hidden = filt.update(None, False, yaw_rotation(-np.pi / 2))
    assert angular_error_deg(hidden.direction_xyz, _direction(-90)) < 2.0
    recovered = filt.update(grid.gaussian_likelihood(_direction(-90), 10), True)
    assert angular_error_deg(recovered.direction_xyz, _direction(-90)) < 2.0


def test_repeated_correct_evidence_recovers_false_observation() -> None:
    grid = SphericalGrid(40, 80)
    filt = _filter(grid)
    correct = _direction(0)
    filt.update(grid.gaussian_likelihood(_direction(90), 12), True)
    estimate = None
    for _ in range(3):
        estimate = filt.update(grid.gaussian_likelihood(correct, 12), True)
    assert estimate is not None
    assert angular_error_deg(estimate.direction_xyz, correct) < 3.0
    assert estimate.confidence > 0.8

