"""Real spherical harmonics used by the EAGOR reproduction.

Convention (made explicit because the paper does not specify it):

* orthonormal SciPy complex SH, including the Condon--Shortley phase;
* ``theta`` is azimuth and SciPy's polar angle is ``pi/2-elevation``;
* real basis: ``m<0`` sine, ``m=0`` zonal, ``m>0`` cosine;
* real conversion multiplies the complex component by ``sqrt(2)*(-1)^m``;
* coefficient order is ``(l,m)=(0,0),(1,-1),(1,0),(1,1),...``.

With this convention the positive degree-1 coefficients align with ``+x``
(``m=+1``), ``+y`` (``m=-1``), and ``+z`` (``m=0``), up to the common
normalization constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.special import sph_harm

from eagor_repro.spherical.spherical_grid import SphericalGrid


def coefficient_layout(bandlimit: int) -> List[Tuple[int, int]]:
    return [(ell, m) for ell in range(bandlimit + 1) for m in range(-ell, ell + 1)]


def real_sh_matrix(directions: np.ndarray, bandlimit: int) -> np.ndarray:
    """Evaluate the package's real SH convention at arbitrary directions."""

    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    directions = directions / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )
    azimuth = np.arctan2(directions[:, 1], directions[:, 0])
    polar = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    columns = []
    for ell, m in coefficient_layout(bandlimit):
        if m < 0:
            complex_y = sph_harm(-m, ell, azimuth, polar)
            value = np.sqrt(2.0) * ((-1.0) ** m) * complex_y.imag
        elif m == 0:
            value = sph_harm(0, ell, azimuth, polar).real
        else:
            complex_y = sph_harm(m, ell, azimuth, polar)
            value = np.sqrt(2.0) * ((-1.0) ** m) * complex_y.real
        columns.append(value)
    return np.stack(columns, axis=1)


@dataclass
class RealSphericalHarmonics:
    """Precomputed real-SH analysis/synthesis matrices for one ERP grid."""

    grid: SphericalGrid
    bandlimit: int = 7
    basis_matrix: np.ndarray = field(init=False, repr=False)
    layout: List[Tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        if self.bandlimit < 0:
            raise ValueError("bandlimit must be non-negative")
        self.layout = coefficient_layout(self.bandlimit)
        self.basis_matrix = real_sh_matrix(
            self.grid.flat_directions, self.bandlimit
        )

    @property
    def num_coefficients(self) -> int:
        return (self.bandlimit + 1) ** 2

    def index(self, ell: int, m: int) -> int:
        if ell < 0 or ell > self.bandlimit or abs(m) > ell:
            raise IndexError(f"Invalid SH coefficient ({ell}, {m})")
        return ell * ell + ell + m

    def project(self, field: np.ndarray) -> np.ndarray:
        """Compute ``Y.T @ (field * dOmega)`` as in paper Eq. (3)."""

        values = np.asarray(field, dtype=np.float64)
        if values.shape != self.grid.shape:
            raise ValueError(
                f"Expected field shape {self.grid.shape}, got {values.shape}"
            )
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return self.basis_matrix.T @ (
            values.reshape(-1) * self.grid.flat_area_weights
        )

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if coefficients.size != self.num_coefficients:
            raise ValueError(
                f"Expected {self.num_coefficients} coefficients, "
                f"got {coefficients.size}"
            )
        values = self.basis_matrix @ coefficients
        return values.reshape(self.grid.shape)

    def evaluate(
        self, coefficients: np.ndarray, directions: np.ndarray
    ) -> np.ndarray:
        matrix = real_sh_matrix(directions, self.bandlimit)
        return matrix @ np.asarray(coefficients, dtype=np.float64)

