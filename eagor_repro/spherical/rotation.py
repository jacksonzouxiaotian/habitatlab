"""Viewer-frame pose conversion and SH belief propagation."""

from __future__ import annotations

import numpy as np

from eagor_repro.spherical.spherical_harmonics import RealSphericalHarmonics


def yaw_rotation(yaw_rad: float) -> np.ndarray:
    """Active right-handed rotation about EAGOR ``+z``.

    Positive yaw rotates ``+x`` (front) toward ``+y`` (left).
    """

    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def relative_view_rotation(
    rotation_world_from_body_previous: np.ndarray,
    rotation_world_from_body_current: np.ndarray,
) -> np.ndarray:
    """Return ``R_current_from_previous = R_WB,t.T @ R_WB,t-1``."""

    previous = np.asarray(rotation_world_from_body_previous, dtype=np.float64)
    current = np.asarray(rotation_world_from_body_current, dtype=np.float64)
    return current.T @ previous


def habitat_rotation_world_from_eagor_body(
    habitat_quaternion: object,
) -> np.ndarray:
    """Convert a Habitat agent quaternion to the EAGOR body convention.

    Habitat local camera axes are ``-z`` forward, ``-x`` left, ``+y`` up.
    EAGOR uses ``+x`` forward, ``+y`` left, ``+z`` up.  Habitat quaternions
    actively map Habitat-local vectors into the world frame.
    """

    try:
        import quaternion
    except ImportError as exc:  # pragma: no cover - Habitat environment only
        raise RuntimeError("numpy-quaternion is required for Habitat poses") from exc

    rotation_world_from_habitat = quaternion.as_rotation_matrix(
        habitat_quaternion
    )
    rotation_habitat_from_eagor = np.asarray(
        ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0))
    )
    return rotation_world_from_habitat @ rotation_habitat_from_eagor


def _extract_planar_yaw(rotation: np.ndarray, tolerance: float = 1e-8) -> float | None:
    candidate = yaw_rotation(np.arctan2(rotation[1, 0], rotation[0, 0]))
    if np.allclose(rotation, candidate, atol=tolerance, rtol=0.0):
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return None


def rotate_real_sh_coefficients(
    coefficients: np.ndarray,
    rotation_current_from_previous: np.ndarray,
    harmonics: RealSphericalHarmonics,
) -> np.ndarray:
    """Propagate coefficients from the previous to current viewer frame.

    Planar yaw uses the exact real-SH block rotation (the Wigner-D restriction
    to z rotations).  General SO(3) is supported with rotation-equivariant
    spherical resampling followed by SH projection.  The latter is a documented
    engineering completion because the paper omits Wigner-D conventions and
    this repository does not require e3nn.
    """

    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    rotation = np.asarray(rotation_current_from_previous, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_current_from_previous must be 3x3")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rotation_current_from_previous must be orthonormal")

    yaw = _extract_planar_yaw(rotation)
    if yaw is not None:
        result = coefficients.copy()
        for ell in range(1, harmonics.bandlimit + 1):
            for m in range(1, ell + 1):
                positive = harmonics.index(ell, m)
                negative = harmonics.index(ell, -m)
                c, s = np.cos(m * yaw), np.sin(m * yaw)
                cosine_coeff = coefficients[positive]
                sine_coeff = coefficients[negative]
                result[positive] = c * cosine_coeff - s * sine_coeff
                result[negative] = s * cosine_coeff + c * sine_coeff
        return result

    # f_current(w) = f_previous(R_current_from_previous.T @ w).
    previous_frame_directions = harmonics.grid.flat_directions @ rotation
    rotated_field = harmonics.evaluate(
        coefficients, previous_frame_directions
    ).reshape(harmonics.grid.shape)
    return harmonics.project(rotated_field)

