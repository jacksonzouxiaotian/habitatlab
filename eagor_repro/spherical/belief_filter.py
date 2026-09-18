"""Recursive Spherical Harmonic Belief Field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from eagor_repro.spherical.rotation import rotate_real_sh_coefficients
from eagor_repro.spherical.spherical_grid import (
    SphericalGrid,
    direction_to_angles,
)
from eagor_repro.spherical.spherical_harmonics import RealSphericalHarmonics


@dataclass(frozen=True)
class BeliefEstimate:
    direction_xyz: np.ndarray
    azimuth: float
    elevation: float
    confidence: float
    resultant_length: float
    valid: bool
    belief_heatmap: np.ndarray


class SphericalHarmonicBeliefFilter:
    """Episode-local EAGOR SH-BF with paper and stability modes."""

    def __init__(
        self,
        grid: SphericalGrid,
        bandlimit: int = 7,
        epsilon: float = 1e-6,
        update_mode: Literal["paper", "decayed"] = "paper",
        belief_decay: float = 1.0,
        observation_weight: float = 1.0,
        missing_observation_mode: Literal[
            "propagate_only", "weak_uniform"
        ] = "propagate_only",
        decode_mode: Literal["paper", "probability"] = "paper",
        max_belief_age: Optional[int] = 50,
    ) -> None:
        if update_mode not in ("paper", "decayed"):
            raise ValueError(f"Unsupported update_mode: {update_mode}")
        if decode_mode not in ("paper", "probability"):
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")
        self.grid = grid
        self.harmonics = RealSphericalHarmonics(grid, bandlimit)
        self.epsilon = float(epsilon)
        self.update_mode = update_mode
        self.belief_decay = float(belief_decay)
        self.observation_weight = float(observation_weight)
        self.missing_observation_mode = missing_observation_mode
        self.decode_mode = decode_mode
        self.max_belief_age = max_belief_age
        self.coefficients: Optional[np.ndarray] = None
        self.belief_age = 0

    def reset(self) -> None:
        self.coefficients = None
        self.belief_age = 0

    def sanitize_likelihood(self, likelihood: np.ndarray) -> np.ndarray:
        values = np.asarray(likelihood, dtype=np.float64)
        if values.shape != self.grid.shape:
            raise ValueError(
                f"Expected likelihood shape {self.grid.shape}, got {values.shape}"
            )
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        values = np.clip(values, 0.0, 1.0)
        maximum = float(values.max(initial=0.0))
        if maximum > self.epsilon:
            values = values / maximum
        else:
            values.fill(self.epsilon)
        return np.clip(values, self.epsilon, 1.0)

    def observation_coefficients(self, likelihood: np.ndarray) -> np.ndarray:
        sanitized = self.sanitize_likelihood(likelihood)
        log_likelihood = np.log(sanitized)
        return self.harmonics.project(log_likelihood)

    def update(
        self,
        likelihood: Optional[np.ndarray],
        target_visible: bool,
        rotation_current_from_previous: Optional[np.ndarray] = None,
    ) -> BeliefEstimate:
        """Propagate the prior and fuse one current observation."""

        if self.coefficients is not None:
            rotation = (
                np.eye(3)
                if rotation_current_from_previous is None
                else rotation_current_from_previous
            )
            prior = rotate_real_sh_coefficients(
                self.coefficients, rotation, self.harmonics
            )
            if self.update_mode == "decayed":
                prior *= self.belief_decay
        else:
            prior = None

        use_observation = target_visible and likelihood is not None
        if not use_observation and self.missing_observation_mode == "weak_uniform":
            likelihood = np.full(self.grid.shape, self.epsilon)
            use_observation = True

        observation = (
            self.observation_coefficients(likelihood) * self.observation_weight
            if use_observation and likelihood is not None
            else None
        )

        if prior is None and observation is None:
            self.coefficients = np.zeros(self.harmonics.num_coefficients)
        elif prior is None:
            self.coefficients = observation
        elif observation is None:
            self.coefficients = prior
        else:
            self.coefficients = prior + observation

        self.coefficients = np.nan_to_num(
            self.coefficients, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.belief_age = 0 if use_observation else self.belief_age + 1
        if (
            self.max_belief_age is not None
            and self.belief_age > self.max_belief_age
            and self.update_mode == "decayed"
        ):
            self.coefficients *= self.belief_decay
        return self.decode()

    def decode(self, mode: Optional[str] = None) -> BeliefEstimate:
        mode = self.decode_mode if mode is None else mode
        if self.coefficients is None:
            field = np.zeros(self.grid.shape)
        else:
            field = self.harmonics.reconstruct(self.coefficients)
        field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

        if mode == "probability":
            # ``field`` is a log-belief and is commonly entirely negative.
            # Using max(initial=0) would fail to shift it and can underflow the
            # total mass after recursive updates.  The grid is non-empty, so
            # always subtract its actual maximum (log-sum-exp stabilization).
            stabilized = field - float(np.max(field))
            density = np.exp(np.clip(stabilized, -700.0, 0.0))
            mass = density * self.grid.area_weights
            normalizer = float(mass.sum())
            if not np.isfinite(normalizer) or normalizer <= self.epsilon:
                mass = self.grid.area_weights.copy()
                normalizer = float(mass.sum())
            probability_mass = mass / normalizer
            moment = np.sum(
                probability_mass[..., None] * self.grid.direction_xyz,
                axis=(0, 1),
            )
            heatmap = probability_mass / max(
                float(probability_mass.max(initial=0.0)), self.epsilon
            )
        elif mode == "paper":
            # This grid integral is equivalent to using only degree-1
            # coefficients.  It implements Eq. (6) literally on the signed
            # log-belief field.  The paper's [0,1] resultant claim is
            # under-specified for signed log scores, hence abs(total) + clamp.
            weighted = field * self.grid.area_weights
            moment = np.sum(
                weighted[..., None] * self.grid.direction_xyz, axis=(0, 1)
            )
            # Eq. (6) calls f a signed log-posterior yet claims a [0,1]
            # resultant, which is impossible with its signed denominator.
            # A constant shift does not change the spherical moment, so use
            # the minimally shifted non-negative field only for normalization.
            # This stability completion is documented in the reproduction README.
            heatmap = field - float(field.min(initial=0.0))
            denominator = float((heatmap * self.grid.area_weights).sum())
            heatmap /= max(float(heatmap.max(initial=0.0)), self.epsilon)
        else:
            raise ValueError(f"Unsupported decode mode: {mode}")

        norm = float(np.linalg.norm(moment))
        valid = bool(np.isfinite(norm) and norm > self.epsilon)
        direction = moment / norm if valid else np.asarray((1.0, 0.0, 0.0))
        if mode == "probability":
            resultant = float(np.clip(norm, 0.0, 1.0))
        else:
            resultant = float(
                np.clip(norm / max(denominator, self.epsilon), 0.0, 1.0)
            )
        azimuth, elevation = direction_to_angles(direction)
        return BeliefEstimate(
            direction_xyz=direction.astype(np.float64),
            azimuth=azimuth,
            elevation=elevation,
            confidence=resultant,
            resultant_length=resultant,
            valid=valid,
            belief_heatmap=heatmap.astype(np.float32),
        )
