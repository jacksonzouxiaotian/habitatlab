"""ERP pixel/sphere mappings with solid-angle quadrature weights.

The EAGOR viewer frame used throughout this package is right handed:

* ``+x``: forward (azimuth 0)
* ``+y``: left (azimuth +pi/2)
* ``+z``: up (elevation +pi/2)

Public pixel conversion follows Eq. (1) in the paper exactly.  Pixel arrays are
sampled at pixel centres, while each row's area weight is computed from its
exact latitude-band boundaries.  The latter makes the discrete area exactly
``4*pi`` and prevents ERP polar oversampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to ``[-pi, pi)`` without special-casing scalar inputs."""

    wrapped = (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if np.ndim(angle) == 0 else wrapped


def erp_pixel_to_sphere(
    u: np.ndarray | float,
    v: np.ndarray | float,
    width: int,
    height: int,
) -> np.ndarray:
    """Map continuous ERP pixel coordinates to EAGOR unit directions.

    This is the literal paper mapping ``theta=2*pi*u/W-pi`` and
    ``phi=pi/2-pi*v/H``.  Callers that represent pixel samples should pass
    ``u+0.5, v+0.5``; exact boundary/direction tests may pass integer values.
    """

    theta = 2.0 * np.pi * np.asarray(u, dtype=np.float64) / width - np.pi
    phi = np.pi / 2.0 - np.pi * np.asarray(v, dtype=np.float64) / height
    cos_phi = np.cos(phi)
    return np.stack(
        (cos_phi * np.cos(theta), cos_phi * np.sin(theta), np.sin(phi)),
        axis=-1,
    )


def sphere_to_erp(
    direction_xyz: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map unit directions to continuous ERP coordinates with circular ``u``."""

    direction = np.asarray(direction_xyz, dtype=np.float64)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float64).eps):
        raise ValueError("direction_xyz contains a zero-length vector")
    direction = direction / norm
    theta = np.arctan2(direction[..., 1], direction[..., 0])
    phi = np.arcsin(np.clip(direction[..., 2], -1.0, 1.0))
    u = ((theta + np.pi) * width / (2.0 * np.pi)) % width
    v = (np.pi / 2.0 - phi) * height / np.pi
    return u, v


def direction_to_angles(direction_xyz: np.ndarray) -> Tuple[float, float]:
    """Return ``(azimuth, elevation)`` in radians for one direction."""

    direction = np.asarray(direction_xyz, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return (
        float(np.arctan2(direction[1], direction[0])),
        float(np.arcsin(np.clip(direction[2], -1.0, 1.0))),
    )


@dataclass
class SphericalGrid:
    """A cached, centre-sampled ERP discretization of :math:`S^2`."""

    height: int
    width: int
    dtype: np.dtype = np.float64
    theta_grid: np.ndarray = field(init=False, repr=False)
    phi_grid: np.ndarray = field(init=False, repr=False)
    direction_xyz: np.ndarray = field(init=False, repr=False)
    area_weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.height < 2 or self.width < 4:
            raise ValueError("A spherical ERP grid requires height>=2 and width>=4")

        u = np.arange(self.width, dtype=np.float64) + 0.5
        v = np.arange(self.height, dtype=np.float64) + 0.5
        theta = 2.0 * np.pi * u / self.width - np.pi
        phi = np.pi / 2.0 - np.pi * v / self.height
        self.theta_grid, self.phi_grid = np.meshgrid(theta, phi)
        cos_phi = np.cos(self.phi_grid)
        self.direction_xyz = np.stack(
            (
                cos_phi * np.cos(self.theta_grid),
                cos_phi * np.sin(self.theta_grid),
                np.sin(self.phi_grid),
            ),
            axis=-1,
        ).astype(self.dtype, copy=False)

        # Exact solid angle of each latitude band divided equally over W cells.
        phi_upper = np.pi / 2.0 - np.pi * np.arange(self.height) / self.height
        phi_lower = np.pi / 2.0 - np.pi * (
            np.arange(self.height) + 1.0
        ) / self.height
        row_solid_angle = (2.0 * np.pi / self.width) * (
            np.sin(phi_upper) - np.sin(phi_lower)
        )
        self.area_weights = np.broadcast_to(
            row_solid_angle[:, None], (self.height, self.width)
        ).copy().astype(self.dtype, copy=False)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def flat_directions(self) -> np.ndarray:
        return self.direction_xyz.reshape(-1, 3)

    @property
    def flat_area_weights(self) -> np.ndarray:
        return self.area_weights.reshape(-1)

    def gaussian_likelihood(
        self,
        direction_xyz: np.ndarray,
        sigma_deg: float = 8.0,
        floor: float = 0.0,
    ) -> np.ndarray:
        """Create an isotropic spherical Gaussian-like likelihood.

        Angular distance, rather than ERP pixel distance, keeps this synthetic
        helper seam- and latitude-correct.
        """

        direction = np.asarray(direction_xyz, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        cos_distance = np.clip(self.direction_xyz @ direction, -1.0, 1.0)
        distance = np.arccos(cos_distance)
        sigma = np.deg2rad(max(float(sigma_deg), 1e-3))
        result = np.exp(-0.5 * (distance / sigma) ** 2)
        return np.maximum(result, floor).astype(np.float32)

