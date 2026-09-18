from __future__ import annotations

import numpy as np

from eagor_repro.spherical.belief_filter import SphericalHarmonicBeliefFilter
from eagor_repro.spherical.rotation import rotate_real_sh_coefficients, yaw_rotation
from eagor_repro.spherical.spherical_grid import (
    SphericalGrid,
    erp_pixel_to_sphere,
    sphere_to_erp,
    wrap_to_pi,
)
from eagor_repro.spherical.spherical_harmonics import RealSphericalHarmonics


def angular_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.rad2deg(
            np.arccos(
                np.clip(
                    np.dot(first, second)
                    / (np.linalg.norm(first) * np.linalg.norm(second)),
                    -1.0,
                    1.0,
                )
            )
        )
    )


def test_erp_pixel_to_sphere() -> None:
    directions = erp_pixel_to_sphere(
        np.asarray([32.0, 48.0, 32.0]),
        np.asarray([16.0, 16.0, 0.0]),
        64,
        32,
    )
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.allclose(directions[0], [1.0, 0.0, 0.0], atol=1e-7)
    assert np.allclose(directions[1], [0.0, 1.0, 0.0], atol=1e-7)
    assert np.allclose(directions[2], [0.0, 0.0, 1.0], atol=1e-7)


def test_sphere_to_erp_roundtrip() -> None:
    u = np.asarray([0.2, 17.5, 32.0, 63.8])
    v = np.asarray([3.5, 15.0, 24.5, 30.0])
    direction = erp_pixel_to_sphere(u, v, 64, 32)
    recovered_u, recovered_v = sphere_to_erp(direction, 64, 32)
    circular_u_error = np.abs((recovered_u - u + 32.0) % 64.0 - 32.0)
    assert np.max(circular_u_error) < 1e-8
    assert np.allclose(recovered_v, v, atol=1e-8)


def test_area_weights_sum_to_4pi() -> None:
    grid = SphericalGrid(31, 62)
    assert np.isclose(grid.area_weights.sum(), 4.0 * np.pi, atol=1e-12)
    assert np.allclose(np.linalg.norm(grid.direction_xyz, axis=-1), 1.0)
    assert grid.area_weights[0, 0] < grid.area_weights[15, 0]


def test_front_direction_mapping() -> None:
    assert np.allclose(erp_pixel_to_sphere(32, 16, 64, 32), [1, 0, 0])


def test_left_right_direction_mapping() -> None:
    left = erp_pixel_to_sphere(48, 16, 64, 32)
    right = erp_pixel_to_sphere(16, 16, 64, 32)
    assert np.allclose(left, [0, 1, 0], atol=1e-7)
    assert np.allclose(right, [0, -1, 0], atol=1e-7)


def test_seam_continuity() -> None:
    grid = SphericalGrid(32, 64)
    left = grid.direction_xyz[16, 0]
    right = grid.direction_xyz[16, -1]
    assert angular_error(left, right) < 360.0 / 64.0 + 1e-6


def test_single_peak_sh_projection() -> None:
    grid = SphericalGrid(48, 96)
    target = np.asarray([np.cos(0.7), np.sin(0.7), 0.0])
    filt = SphericalHarmonicBeliefFilter(
        grid, bandlimit=7, decode_mode="probability"
    )
    estimate = filt.update(grid.gaussian_likelihood(target, 12.0), True)
    assert estimate.valid
    assert angular_error(estimate.direction_xyz, target) < 5.0


def test_paper_decode_single_peak() -> None:
    grid = SphericalGrid(48, 96)
    target = np.asarray([np.cos(-0.8), np.sin(-0.8), 0.0])
    filt = SphericalHarmonicBeliefFilter(grid, bandlimit=7, decode_mode="paper")
    estimate = filt.update(grid.gaussian_likelihood(target, 12.0), True)
    assert angular_error(estimate.direction_xyz, target) < 5.0
    assert 0.0 <= estimate.confidence <= 1.0


def test_no_motion_propagation() -> None:
    grid = SphericalGrid(32, 64)
    sh = RealSphericalHarmonics(grid, 5)
    coeff = sh.project(grid.gaussian_likelihood([1, 0, 0], 15))
    rotated = rotate_real_sh_coefficients(coeff, np.eye(3), sh)
    assert np.allclose(rotated, coeff, atol=1e-12)


def _rotation_filter() -> tuple[SphericalHarmonicBeliefFilter, np.ndarray]:
    grid = SphericalGrid(48, 96)
    target = np.asarray([1.0, 0.0, 0.0])
    filt = SphericalHarmonicBeliefFilter(
        grid, bandlimit=7, decode_mode="probability"
    )
    filt.update(grid.gaussian_likelihood(target, 12), True)
    return filt, target


def test_positive_yaw_rotation_sign() -> None:
    # Agent turns left: previous front appears at current right (-y).
    filt, _ = _rotation_filter()
    estimate = filt.update(None, False, yaw_rotation(-np.pi / 2))
    assert angular_error(estimate.direction_xyz, [0, -1, 0]) < 3.0


def test_negative_yaw_rotation_sign() -> None:
    # Agent turns right: previous front appears at current left (+y).
    filt, _ = _rotation_filter()
    estimate = filt.update(None, False, yaw_rotation(np.pi / 2))
    assert angular_error(estimate.direction_xyz, [0, 1, 0]) < 3.0


def test_four_quarter_turns() -> None:
    filt, target = _rotation_filter()
    for _ in range(4):
        estimate = filt.update(None, False, yaw_rotation(-np.pi / 2))
    assert angular_error(estimate.direction_xyz, target) < 1e-6


def test_general_so3_rotation_interface() -> None:
    grid = SphericalGrid(64, 128)
    sh = RealSphericalHarmonics(grid, 7)
    target = np.asarray([1.0, 0.0, 0.0])
    coeff = sh.project(grid.gaussian_likelihood(target, 15))
    pitch = np.deg2rad(35.0)
    rotation = np.asarray(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    rotated = rotate_real_sh_coefficients(coeff, rotation, sh)
    field = sh.reconstruct(rotated)
    moment = np.sum(
        (field - field.min())[..., None]
        * grid.area_weights[..., None]
        * grid.direction_xyz,
        axis=(0, 1),
    )
    moment /= np.linalg.norm(moment)
    assert angular_error(moment, rotation @ target) < 3.0


def test_missing_observation_propagate_only() -> None:
    filt, _ = _rotation_filter()
    before = filt.coefficients.copy()
    filt.update(None, False, np.eye(3))
    assert np.allclose(filt.coefficients, before)


def test_no_nan_with_empty_likelihood() -> None:
    grid = SphericalGrid(16, 32)
    filt = SphericalHarmonicBeliefFilter(grid, bandlimit=3)
    estimate = filt.update(np.zeros(grid.shape), True)
    assert np.all(np.isfinite(filt.coefficients))
    assert np.all(np.isfinite(estimate.belief_heatmap))


def test_probability_normalization() -> None:
    grid = SphericalGrid(24, 48)
    filt = SphericalHarmonicBeliefFilter(
        grid, bandlimit=5, decode_mode="probability"
    )
    filt.update(grid.gaussian_likelihood([0, 1, 0], 10), True)
    field = filt.harmonics.reconstruct(filt.coefficients)
    density = np.exp(field - field.max())
    probability = density * grid.area_weights
    probability /= probability.sum()
    assert np.isclose(probability.sum(), 1.0)
    assert np.all(np.isfinite(probability))


def test_wrap_to_pi_across_seam() -> None:
    delta = wrap_to_pi(np.deg2rad(-179.0) - np.deg2rad(179.0))
    assert np.isclose(np.rad2deg(delta), 2.0)
