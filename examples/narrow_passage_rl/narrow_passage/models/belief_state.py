"""Feasibility-belief state for DEGNAV-RL.

This module converts the existing 19-D ``narrow_passage_features`` observation
into the compact belief state used by the paper-facing DEGNAV-RL formulation.
It intentionally has no Habitat/SB3 dependency so it can be reused by synthetic
and Habitat evaluators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


# Source of truth: habitat-lab/habitat/tasks/narrow_passage/geometry.py
# FEATURE_NAMES = (
#   0 d_left_near, 1 d_center_near, 2 d_right_near,
#   3 d_left_far,  4 d_center_far,  5 d_right_far,
#   6 clearance_left, 7 clearance_right, 8 passage_width, 9 body_margin,
#   10 heading_error, 11 lateral_offset, 12 distance_to_local_goal,
#   13 current_vx, 14 current_wz, 15 stuck_score, 16 collision_flag,
#   17 previous_action_vx, 18 previous_action_wz,
# )
FEATURE_DIM = 19
FEATURE_KEY = "narrow_passage_features"

IDX_CLEARANCE_LEFT = 6
IDX_CLEARANCE_RIGHT = 7
IDX_PASSAGE_WIDTH = 8
IDX_BODY_MARGIN = 9
IDX_HEADING_ERROR = 10
IDX_LATERAL_OFFSET = 11
IDX_STUCK_SCORE = 15
IDX_COLLISION_FLAG = 16

BELIEF_FEATURE_NAMES = (
    "p_feas",
    "delta_mean",
    "delta_var",
    "heading_error",
    "lateral_error",
    "stuck_score",
    "collision_flag",
    "memory_risk",
    "d_hat",
    "w_req_prior",
    "w_req_cons",
    "clearance_left",
    "clearance_right",
    "body_margin",
)


@dataclass(frozen=True)
class BeliefStateConfig:
    """Configuration for converting geometry observations into belief states.

    ``d_hat`` is the estimated available passage width from the geometry sensor.
    ``w_req_prior`` is the prior effective robot width, and ``w_req_cons`` is a
    conservative required width used for feasibility.  The Gaussian feasibility
    model is intentionally lightweight:

        delta = d_hat - w_req_cons
        p_feas = Phi(delta / sqrt(sigma_d^2 + sigma_w^2))
    """

    feature_dim: int = FEATURE_DIM
    default_w_req_prior: float = 0.36
    conservative_margin: float = 0.06
    sigma_d: float = 0.04
    sigma_w: float = 0.03
    min_variance: float = 1e-8
    use_yaw_prior: bool = True
    max_yaw_prior_angle: float = math.radians(60.0)
    min_yaw_cos: float = 0.5

    @property
    def default_w_req_cons(self) -> float:
        return self.default_w_req_prior + self.conservative_margin


@dataclass(frozen=True)
class BeliefState:
    """Compact feasibility belief plus diagnostics for DEGNAV-RL."""

    p_feas: float
    delta_mean: float
    delta_var: float
    heading_error: float
    lateral_error: float
    stuck_score: float
    collision_flag: float
    memory_risk: float
    d_hat: float
    w_req_prior: float
    w_req_cons: float
    clearance_left: float
    clearance_right: float
    body_margin: float

    def as_array(self) -> np.ndarray:
        """Return the belief vector as finite ``float32`` values."""

        return np.asarray(
            [
                self.p_feas,
                self.delta_mean,
                self.delta_var,
                self.heading_error,
                self.lateral_error,
                self.stuck_score,
                self.collision_flag,
                self.memory_risk,
                self.d_hat,
                self.w_req_prior,
                self.w_req_cons,
                self.clearance_left,
                self.clearance_right,
                self.body_margin,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def feature_names() -> list[str]:
        return list(BELIEF_FEATURE_NAMES)

    @classmethod
    def from_obs(
        cls,
        obs: Sequence[float] | np.ndarray | Mapping[str, object],
        memory_risk: float = 0.0,
        w_req_prior: float | None = None,
        w_req_cons: float | None = None,
        cfg: BeliefStateConfig | None = None,
    ) -> "BeliefState":
        """Build a belief state from a 19-D narrow-passage observation.

        ``obs`` may be either the feature vector itself or an observation dict
        containing ``"narrow_passage_features"``.  A clear ``ValueError`` is
        raised when the feature vector is missing, has the wrong length, or
        contains non-finite values.
        """

        cfg = cfg or BeliefStateConfig()
        features = _coerce_features(obs, cfg.feature_dim)

        memory_risk = _finite_float("memory_risk", memory_risk)
        w_req_prior_value = (
            _yaw_projected_width_prior(features, cfg)
            if w_req_prior is None
            else _finite_float("w_req_prior", w_req_prior)
        )
        w_req_cons_value = (
            w_req_prior_value + float(cfg.conservative_margin)
            if w_req_cons is None
            else _finite_float("w_req_cons", w_req_cons)
        )

        d_hat = float(features[IDX_PASSAGE_WIDTH])
        delta_mean = d_hat - w_req_cons_value
        delta_var = max(
            cfg.min_variance,
            float(cfg.sigma_d) ** 2 + float(cfg.sigma_w) ** 2,
        )
        p_feas = _normal_cdf(delta_mean / math.sqrt(delta_var))

        belief = cls(
            p_feas=float(min(1.0, max(0.0, p_feas))),
            delta_mean=float(delta_mean),
            delta_var=float(delta_var),
            heading_error=float(features[IDX_HEADING_ERROR]),
            lateral_error=float(features[IDX_LATERAL_OFFSET]),
            stuck_score=float(features[IDX_STUCK_SCORE]),
            collision_flag=float(features[IDX_COLLISION_FLAG]),
            memory_risk=float(min(1.0, max(0.0, memory_risk))),
            d_hat=float(d_hat),
            w_req_prior=float(w_req_prior_value),
            w_req_cons=float(w_req_cons_value),
            clearance_left=float(features[IDX_CLEARANCE_LEFT]),
            clearance_right=float(features[IDX_CLEARANCE_RIGHT]),
            body_margin=float(features[IDX_BODY_MARGIN]),
        )
        _assert_finite_belief(belief)
        return belief


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _yaw_projected_width_prior(features: np.ndarray, cfg: BeliefStateConfig) -> float:
    """Return the required body-width prior after yaw projection.

    When the robot enters a passage at a heading error, its effective lateral
    envelope is larger than its frontal width.  We use a lightweight
    ``width / cos(|heading_error|)`` approximation, capped by
    ``max_yaw_prior_angle`` and ``min_yaw_cos`` so the prior stays finite.  The
    ``no_yaw_prior`` ablation disables this and uses the fixed frontal envelope.
    """

    base_width = float(cfg.default_w_req_prior)
    if not cfg.use_yaw_prior:
        return base_width
    yaw = min(abs(float(features[IDX_HEADING_ERROR])), float(cfg.max_yaw_prior_angle))
    cos_yaw = max(float(cfg.min_yaw_cos), math.cos(yaw))
    return float(base_width / cos_yaw)


def _coerce_features(
    obs: Sequence[float] | np.ndarray | Mapping[str, object],
    feature_dim: int,
) -> np.ndarray:
    if isinstance(obs, Mapping):
        if FEATURE_KEY not in obs:
            raise ValueError(
                f"Observation mapping must contain {FEATURE_KEY!r}; "
                f"got keys={list(obs.keys())}"
            )
        obs = obs[FEATURE_KEY]  # type: ignore[index]

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.shape[0] != feature_dim:
        raise ValueError(
            f"Expected {feature_dim}-D {FEATURE_KEY}, got shape={arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{FEATURE_KEY} contains non-finite values")
    return arr


def _finite_float(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _assert_finite_belief(belief: BeliefState) -> None:
    arr = belief.as_array()
    if arr.dtype != np.float32 or not np.all(np.isfinite(arr)):
        raise ValueError("BeliefState produced non-finite float32 values")
