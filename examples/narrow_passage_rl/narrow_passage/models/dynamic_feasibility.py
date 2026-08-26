"""Stateful, auditable feasibility and uncertainty estimation for DEGNav.

This module is used only by the strict feasibility-ablation protocol.  The
legacy controller keeps its historical :mod:`belief_state` implementation so
old results remain reproducible.

The available-width uncertainty combines five observable terms: ray/sector
dispersion, missing/max-range depth returns, a sector boundary-fit residual,
temporal variation of the reported passage width, and yaw/pose uncertainty.
All terms are logged separately; none is learned or tuned per ablation.

In addition, the estimator carries an explicit recursive Gaussian margin
belief.  Its posterior moments are audit-only parallel outputs: the published
strict selector continues to consume the legacy ``mu_delta_*`` and engineering
``sigma_delta_*`` fields.  Keeping this information barrier makes the new
belief confidence measurable without silently changing an existing ablation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .belief_state import (
    FEATURE_DIM,
    FEATURE_KEY,
    IDX_BODY_MARGIN,
    IDX_CLEARANCE_LEFT,
    IDX_CLEARANCE_RIGHT,
    IDX_COLLISION_FLAG,
    IDX_HEADING_ERROR,
    IDX_LATERAL_OFFSET,
    IDX_PASSAGE_WIDTH,
    IDX_STUCK_SCORE,
)
from .robot_morphology import RobotMorphology


@dataclass(frozen=True)
class DynamicFeasibilityConfig:
    """Declared morphology and uncertainty model for strict ablations.

    ``body_length`` is explicit because a circular collision proxy cannot
    identify a yaw-dependent rectangular envelope.  It is a decision-model
    morphology parameter, shared unchanged by all strict variants.
    """

    feature_dim: int = FEATURE_DIM
    morphology: RobotMorphology = field(default_factory=RobotMorphology)
    max_yaw: float = math.radians(90.0)
    max_depth: float = 5.0
    temporal_window: int = 6
    temporal_mean_alpha: float = 0.55
    min_sigma: float = 1e-4
    base_depth_sigma: float = 0.008
    base_width_sigma: float = 0.008
    ray_dispersion_gain: float = 0.020
    dropout_gain: float = 0.080
    boundary_residual_gain: float = 0.20
    temporal_gain: float = 1.0
    base_yaw_sigma: float = math.radians(1.0)
    yaw_dropout_gain: float = math.radians(5.0)
    yaw_magnitude_gain: float = 0.04
    base_lateral_sigma: float = 0.005
    lateral_dropout_gain: float = 0.025
    lateral_uncertainty_gain: float = 0.10
    fixed_sigma_delta: float = 0.050
    enable_belief_propagation: bool = True
    enable_belief_fusion: bool = True
    belief_process_sigma_struct: float = 0.003
    belief_yaw_process_gain: float = 0.50

    def __post_init__(self) -> None:
        if self.max_depth <= 0.0 or self.temporal_window < 2:
            raise ValueError("max_depth must be positive and temporal_window >= 2")
        if not 0.0 < self.temporal_mean_alpha <= 1.0:
            raise ValueError("temporal_mean_alpha must be in (0, 1]")
        if self.min_sigma <= 0.0 or self.fixed_sigma_delta <= 0.0:
            raise ValueError("uncertainty scales must be positive")
        if self.belief_process_sigma_struct < 0.0:
            raise ValueError("belief_process_sigma_struct must be non-negative")
        if self.belief_yaw_process_gain < 0.0:
            raise ValueError("belief_yaw_process_gain must be non-negative")

    @property
    def body_width(self) -> float:
        return self.morphology.width

    @property
    def body_length(self) -> float:
        return self.morphology.length

    @property
    def safety_margin(self) -> float:
        return self.morphology.safety_margin


def required_width_without_yaw(cfg: DynamicFeasibilityConfig) -> float:
    """Frontal rectangular envelope including two-sided safety clearance."""

    return float(cfg.morphology.structural_required_width)


def required_width_with_yaw(
    yaw_error: float,
    cfg: DynamicFeasibilityConfig,
) -> float:
    """Projected lateral envelope of a rectangle at ``yaw_error``.

    Geometry requires addition of both non-negative projections and addition of
    safety clearance::

        A * |cos(yaw)| + B * |sin(yaw)| + 2 * safety_margin

    Subtracting ``B*|sin|`` or the safety margin can make the required width
    shrink as the robot turns and is therefore not a conservative width model.
    """

    yaw = min(abs(float(yaw_error)), float(cfg.max_yaw))
    return float(cfg.morphology.pose_required_width(yaw))


def _normal_cdf(x: float) -> float:
    return float(0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0))))


def _coerce_features(
    obs: Sequence[float] | np.ndarray | Mapping[str, object],
    feature_dim: int,
) -> np.ndarray:
    if isinstance(obs, Mapping):
        if FEATURE_KEY not in obs:
            raise ValueError(f"Observation must contain {FEATURE_KEY!r}")
        obs = obs[FEATURE_KEY]  # type: ignore[index]
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if arr.shape != (feature_dim,):
        raise ValueError(f"Expected {feature_dim}-D {FEATURE_KEY}, got {arr.shape}")
    # Depth rays may use NaN for a true invalid sensor return.  Non-depth state
    # must remain finite so invalid geometry cannot silently enter the filter.
    if not np.all(np.isfinite(arr[6:])):
        raise ValueError(f"{FEATURE_KEY} non-depth state contains non-finite values")
    return arr


def _boundary_fit_residual(rays: np.ndarray, valid: np.ndarray) -> float:
    """RMSE of a first-order boundary fit across near/far ray triplets."""

    x = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    residuals: list[float] = []
    for group in (slice(0, 3), slice(3, 6)):
        group_valid = valid[group]
        if int(group_valid.sum()) < 3:
            continue
        y = rays[group]
        design = np.column_stack((np.ones(3, dtype=np.float64), x))
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        residuals.extend((y - fitted).tolist())
    if not residuals:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(residuals)))))


@dataclass(frozen=True)
class GaussianMarginBelief:
    """Scalar Gaussian approximation ``q(delta)=N(mu, variance)``."""

    mu: float
    variance: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu):
            raise ValueError("belief mean must be finite")
        if not math.isfinite(self.variance) or self.variance < 0.0:
            raise ValueError("belief variance must be finite and non-negative")


def propagate_margin_belief(
    posterior: GaussianMarginBelief,
    *,
    required_width_delta: float,
    process_variance: float,
    enabled: bool = True,
) -> GaussianMarginBelief:
    r"""Apply the yaw-change prior propagation operator.

    The margin state is :math:`\delta=D-W(\psi)`.  For
    :math:`\Delta\psi=\psi_t-\psi_{t-1}`, the explicit operator is

    .. math::

       q_t^-(\delta)=\Gamma_{\Delta\psi}[q_{t-1}](\delta),\qquad
       \mu_t^-=\mu_{t-1}-(W_t-W_{t-1}),\quad
       P_t^-=P_{t-1}+Q_t.

    ``enabled=False`` is the propagate ablation: it returns the prior
    unchanged, while the observation-fusion branch remains available.
    """

    if not enabled:
        return posterior
    if process_variance < 0.0:
        raise ValueError("process_variance must be non-negative")
    return GaussianMarginBelief(
        mu=float(posterior.mu - required_width_delta),
        variance=float(posterior.variance + process_variance),
    )


def fuse_margin_observation(
    propagated: GaussianMarginBelief,
    *,
    observation_mu: float,
    observation_variance: float,
    gain: float,
    enabled: bool = True,
) -> GaussianMarginBelief:
    r"""Apply the observation-fusion operator ``oplus``.

    For an independent Gaussian observation
    :math:`r_t(\delta)=\mathcal N(z_t,R_t)`, the declared linear fusion is

    .. math::

       q_t=q_t^-\oplus r_t,\qquad
       \mu_t=(1-K_t)\mu_t^-+K_t z_t,\quad
       P_t=(1-K_t)^2P_t^-+K_t^2R_t.

    The default :math:`K_t=\alpha=0.55` is exactly the historical width-EWMA
    mean update.  ``enabled=False`` is the fuse ablation: it returns the
    propagated prior unchanged.
    """

    if not enabled:
        return propagated
    if not 0.0 <= gain <= 1.0:
        raise ValueError("fusion gain must be in [0, 1]")
    if observation_variance < 0.0:
        raise ValueError("observation_variance must be non-negative")
    complement = 1.0 - float(gain)
    return GaussianMarginBelief(
        mu=float(complement * propagated.mu + gain * observation_mu),
        variance=float(
            complement**2 * propagated.variance
            + float(gain) ** 2 * observation_variance
        ),
    )


class DynamicFeasibilityEstimator:
    r"""Rolling estimator shared by every strict feasibility variant.

    Its explicit two-step recursion is

    .. math::

       q_t^-=\Gamma_{\Delta\psi}[q_{t-1}],\qquad
       q_t=q_t^-\oplus r_t.

    Structural and pose-conditioned beliefs use the same observed width.  The
    structural propagation has :math:`W_t-W_{t-1}=0`; the pose path propagates
    the exact OBB yaw projection, with
    :math:`Q_t^{pose}=Q_t^{struct}+(g_\psi |\partial W/\partial\psi|
    \operatorname{wrap}(\Delta\psi))^2`.  Posterior moments are logged
    separately from the established engineering uncertainty used by the
    selector.
    """

    def __init__(self, cfg: DynamicFeasibilityConfig | None = None):
        self.cfg = cfg or DynamicFeasibilityConfig()
        self.reset()

    def reset(self) -> None:
        self._width_history: deque[float] = deque(maxlen=self.cfg.temporal_window)
        self._mu_width: float | None = None
        self._last_in_passage: bool | None = None
        self._belief_struct: GaussianMarginBelief | None = None
        self._belief_pose: GaussianMarginBelief | None = None
        self._belief_yaw: float | None = None
        self._belief_pose_required_width: float | None = None
        self._belief_update_count = 0

    def _reset_belief(self) -> None:
        """Clear posterior state when the observation regime changes.

        Mathematically this starts a new sequence at :math:`q_0=r_0`; it avoids
        propagating the procedural open-space sentinel into a passage belief.
        """

        self._belief_struct = None
        self._belief_pose = None
        self._belief_yaw = None
        self._belief_pose_required_width = None
        self._belief_update_count = 0

    def update(
        self,
        obs: Sequence[float] | np.ndarray | Mapping[str, object],
        *,
        use_yaw_prior: bool = True,
        fixed_uncertainty: bool = False,
        memory_risk: float = 0.0,
        enable_belief_propagation: bool | None = None,
        enable_belief_fusion: bool | None = None,
    ) -> dict[str, float | bool | str]:
        r"""Update engineering diagnostics and the parallel recursive belief.

        With :math:`z_t=D_t-W(\psi_t)`, this method first applies
        :math:`\Gamma_{\Delta\psi}` via :func:`propagate_margin_belief`, then
        applies :math:`\oplus` via :func:`fuse_margin_observation`.  The two
        optional flags override the config independently for causal isolation;
        omitting them preserves the historical public call signature.
        """

        features = _coerce_features(obs, self.cfg.feature_dim)
        cfg = self.cfg

        raw_width = float(features[IDX_PASSAGE_WIDTH])
        in_passage = bool(
            float(features[IDX_BODY_MARGIN]) <= 0.50
            and raw_width <= cfg.max_depth
        )
        if self._last_in_passage is not None and in_passage != self._last_in_passage:
            # Procedural-v2 reports 10 m in open space.  It is a regime sentinel,
            # not a noisy sample of the upcoming 0.3-0.7 m passage; mixing the
            # two creates metre-scale fake temporal uncertainty at entry.
            self._width_history.clear()
            self._mu_width = None
            self._reset_belief()
        self._last_in_passage = in_passage
        if self._mu_width is None:
            self._mu_width = raw_width
        else:
            alpha = float(cfg.temporal_mean_alpha)
            self._mu_width = alpha * raw_width + (1.0 - alpha) * self._mu_width
        self._width_history.append(raw_width)
        mu_d = float(self._mu_width)

        rays = features[:6].copy()
        valid_hit = (
            np.isfinite(rays)
            & (rays > 0.0)
            & (rays < cfg.max_depth * 0.995)
        )
        valid_no_return = (
            np.isfinite(rays)
            & (rays >= cfg.max_depth * 0.995)
        )
        invalid = ~(valid_hit | valid_no_return)
        valid_hit_ratio = float(np.mean(valid_hit))
        valid_no_return_ratio = float(np.mean(valid_no_return))
        invalid_ratio = float(np.mean(invalid))
        valid_depth_ratio = float(valid_hit_ratio + valid_no_return_ratio)
        # Only a real invalid encoding is dropout.  Reaching max range without
        # a hit is positive free-space evidence, not missing sensor data.
        dropout_ratio = invalid_ratio
        valid_rays = rays[valid_hit]
        ray_dispersion = (
            float(np.std(valid_rays, ddof=0)) if valid_rays.size >= 2 else 0.0
        )
        boundary_residual = _boundary_fit_residual(rays, valid_hit)
        temporal_width_variation = (
            float(np.std(np.asarray(self._width_history), ddof=0))
            if len(self._width_history) >= 2
            else 0.0
        )

        ray_sigma = cfg.ray_dispersion_gain * math.tanh(
            ray_dispersion / max(cfg.max_depth * 0.25, 1e-9)
        )
        dropout_sigma = cfg.dropout_gain * dropout_ratio
        boundary_sigma = cfg.boundary_residual_gain * min(boundary_residual, 0.50)
        temporal_sigma = cfg.temporal_gain * temporal_width_variation
        sigma_d = math.sqrt(
            cfg.base_depth_sigma**2
            + ray_sigma**2
            + dropout_sigma**2
            + boundary_sigma**2
            + temporal_sigma**2
        )

        yaw_error = float(features[IDX_HEADING_ERROR])
        lateral_error = float(features[IDX_LATERAL_OFFSET])
        w_pose_with_yaw = required_width_with_yaw(yaw_error, cfg)
        w_structural = required_width_without_yaw(cfg)
        selected_pose_w = w_pose_with_yaw if use_yaw_prior else w_structural

        yaw = min(abs(yaw_error), cfg.max_yaw)
        d_width_d_yaw = abs(
            -cfg.body_width * math.sin(yaw)
            + cfg.body_length * math.cos(yaw)
        )
        yaw_sigma = (
            cfg.base_yaw_sigma
            + cfg.yaw_dropout_gain * dropout_ratio
            + cfg.yaw_magnitude_gain * yaw
        )
        lateral_sigma = (
            cfg.base_lateral_sigma
            + cfg.lateral_dropout_gain * dropout_ratio
            + cfg.lateral_uncertainty_gain * min(abs(lateral_error), 0.50)
        )
        yaw_pose_sigma = math.hypot(d_width_d_yaw * yaw_sigma, lateral_sigma)
        sigma_w_pose = math.hypot(cfg.base_width_sigma, yaw_pose_sigma)
        dynamic_sigma_struct = max(
            cfg.min_sigma, math.hypot(sigma_d, cfg.base_width_sigma)
        )
        dynamic_sigma_pose = max(cfg.min_sigma, math.hypot(sigma_d, sigma_w_pose))
        sigma_struct = (
            float(cfg.fixed_sigma_delta)
            if fixed_uncertainty
            else float(dynamic_sigma_struct)
        )
        sigma_pose = (
            float(cfg.fixed_sigma_delta)
            if fixed_uncertainty
            else float(dynamic_sigma_pose)
        )

        mu_struct = float(mu_d - w_structural)
        mu_pose = float(mu_d - selected_pose_w)
        mu_delta_with_yaw = float(mu_d - w_pose_with_yaw)
        mu_delta_without_yaw = mu_struct
        p_struct = _normal_cdf(mu_struct / sigma_struct)
        p_pose = _normal_cdf(mu_pose / sigma_pose)

        left_boundary_supported = bool(
            math.isfinite(float(features[IDX_CLEARANCE_LEFT]))
            and float(features[IDX_CLEARANCE_LEFT]) < cfg.max_depth * 0.995
        )
        right_boundary_supported = bool(
            math.isfinite(float(features[IDX_CLEARANCE_RIGHT]))
            and float(features[IDX_CLEARANCE_RIGHT]) < cfg.max_depth * 0.995
        )
        boundary_support_count = int(left_boundary_supported) + int(
            right_boundary_supported
        )
        passage_aperture_observable = bool(
            raw_width < cfg.max_depth * 0.995
            and boundary_support_count >= 2
        )
        if passage_aperture_observable and in_passage:
            observation_regime = "passage_boundary_supported"
        elif passage_aperture_observable:
            observation_regime = "approach_boundary_supported"
        elif valid_no_return_ratio > 0.5:
            observation_regime = "open_free_space_unobservable_width"
        else:
            observation_regime = "invalid_or_partial_unobservable_width"

        propagate_enabled = (
            cfg.enable_belief_propagation
            if enable_belief_propagation is None
            else bool(enable_belief_propagation)
        )
        fuse_enabled = (
            cfg.enable_belief_fusion
            if enable_belief_fusion is None
            else bool(enable_belief_fusion)
        )
        observation_struct = float(raw_width - w_structural)
        observation_pose = float(raw_width - selected_pose_w)
        observation_var_struct = float(dynamic_sigma_struct**2)
        observation_var_pose = float(dynamic_sigma_pose**2)
        initialized_from_observation = self._belief_struct is None
        if initialized_from_observation:
            propagated_struct = GaussianMarginBelief(
                observation_struct, observation_var_struct
            )
            propagated_pose = GaussianMarginBelief(
                observation_pose, observation_var_pose
            )
            posterior_struct = propagated_struct
            posterior_pose = propagated_pose
        else:
            assert self._belief_pose is not None
            assert self._belief_yaw is not None
            assert self._belief_pose_required_width is not None
            raw_delta_yaw = float(yaw_error - self._belief_yaw)
            delta_yaw = float(
                math.atan2(math.sin(raw_delta_yaw), math.cos(raw_delta_yaw))
            )
            required_width_delta = float(
                selected_pose_w - self._belief_pose_required_width
            )
            q_struct = float(cfg.belief_process_sigma_struct**2)
            q_pose = float(
                q_struct
                + (
                    cfg.belief_yaw_process_gain
                    * d_width_d_yaw
                    * delta_yaw
                ) ** 2
            )
            propagated_struct = propagate_margin_belief(
                self._belief_struct,
                required_width_delta=0.0,
                process_variance=q_struct,
                enabled=propagate_enabled,
            )
            propagated_pose = propagate_margin_belief(
                self._belief_pose,
                required_width_delta=required_width_delta,
                process_variance=q_pose,
                enabled=propagate_enabled,
            )
            posterior_struct = fuse_margin_observation(
                propagated_struct,
                observation_mu=observation_struct,
                observation_variance=observation_var_struct,
                gain=cfg.temporal_mean_alpha,
                enabled=fuse_enabled,
            )
            posterior_pose = fuse_margin_observation(
                propagated_pose,
                observation_mu=observation_pose,
                observation_variance=observation_var_pose,
                gain=cfg.temporal_mean_alpha,
                enabled=fuse_enabled,
            )
        self._belief_struct = posterior_struct
        self._belief_pose = posterior_pose
        self._belief_yaw = yaw_error
        self._belief_pose_required_width = selected_pose_w
        self._belief_update_count += 1

        posterior_sigma_struct = math.sqrt(posterior_struct.variance)
        posterior_sigma_pose = math.sqrt(posterior_pose.variance)
        posterior_p_struct = _normal_cdf(
            posterior_struct.mu / max(posterior_sigma_struct, cfg.min_sigma)
        )
        posterior_p_pose = _normal_cdf(
            posterior_pose.mu / max(posterior_sigma_pose, cfg.min_sigma)
        )
        # Bounded distribution concentration.  It is computed solely from the
        # recursive posterior variance; the fixed reference only makes it
        # dimensionless and comparable across rows.
        posterior_concentration_struct = 1.0 / (
            1.0 + posterior_sigma_struct / cfg.fixed_sigma_delta
        )
        posterior_concentration_pose = 1.0 / (
            1.0 + posterior_sigma_pose / cfg.fixed_sigma_delta
        )
        equivalence_error_struct = float(posterior_struct.mu - mu_struct)
        equivalence_error_pose = float(posterior_pose.mu - mu_pose)
        out: dict[str, float | bool | str] = {
            "mu_D": mu_d,
            "var_D": float(sigma_d**2),
            "mu_W": float(selected_pose_w),
            "var_W": float(sigma_w_pose**2),
            "mu_delta_struct": mu_struct,
            "sigma_delta_struct": sigma_struct,
            "dynamic_sigma_delta_struct": float(dynamic_sigma_struct),
            "var_delta_struct": float(sigma_struct**2),
            "p_feas_struct": float(min(1.0, max(0.0, p_struct))),
            "mu_delta_pose": mu_pose,
            "sigma_delta_pose": sigma_pose,
            "dynamic_sigma_delta_pose": float(dynamic_sigma_pose),
            "var_delta_pose": float(sigma_pose**2),
            "p_feas_pose": float(min(1.0, max(0.0, p_pose))),
            # Recursive belief distribution.  These fields deliberately do not
            # replace the engineering sigma or selector-facing compatibility
            # aliases above/below.
            "posterior_mu_delta_struct": float(posterior_struct.mu),
            "posterior_var_delta_struct": float(posterior_struct.variance),
            "posterior_sigma_delta_struct": float(posterior_sigma_struct),
            "posterior_p_feas_struct": float(
                min(1.0, max(0.0, posterior_p_struct))
            ),
            "posterior_concentration_struct": float(
                posterior_concentration_struct
            ),
            "posterior_mu_delta_pose": float(posterior_pose.mu),
            "posterior_var_delta_pose": float(posterior_pose.variance),
            "posterior_sigma_delta_pose": float(posterior_sigma_pose),
            "posterior_p_feas_pose": float(
                min(1.0, max(0.0, posterior_p_pose))
            ),
            "posterior_concentration_pose": float(posterior_concentration_pose),
            "propagated_mu_delta_struct": float(propagated_struct.mu),
            "propagated_var_delta_struct": float(propagated_struct.variance),
            "propagated_mu_delta_pose": float(propagated_pose.mu),
            "propagated_var_delta_pose": float(propagated_pose.variance),
            "belief_mean_equivalence_error_struct": equivalence_error_struct,
            "belief_mean_equivalence_error_pose": equivalence_error_pose,
            "belief_propagation_enabled": bool(propagate_enabled),
            "belief_fusion_enabled": bool(fuse_enabled),
            "belief_initialized_from_observation": bool(
                initialized_from_observation
            ),
            "belief_update_count": int(self._belief_update_count),
            # Compatibility fields explicitly alias the pose-conditioned path.
            "mu_delta": mu_pose,
            "sigma_delta": sigma_pose,
            "dynamic_sigma_delta": float(dynamic_sigma_pose),
            "var_delta": float(sigma_pose**2),
            "p_feas": float(min(1.0, max(0.0, p_pose))),
            "delta_mean": mu_pose,
            "delta_var": float(sigma_pose**2),
            "d_hat": mu_d,
            "w_req_prior": float(selected_pose_w),
            "w_req_cons": float(selected_pose_w),
            "yaw_error": yaw_error,
            "lateral_error": lateral_error,
            "stuck_score": float(features[IDX_STUCK_SCORE]),
            "collision_flag": float(features[IDX_COLLISION_FLAG]),
            "body_margin": float(features[IDX_BODY_MARGIN]),
            "clearance_left": float(features[IDX_CLEARANCE_LEFT]),
            "clearance_right": float(features[IDX_CLEARANCE_RIGHT]),
            "memory_risk": float(min(1.0, max(0.0, memory_risk))),
            "risk": float(
                min(1.0, max(0.0, 1.0 - p_pose + float(memory_risk)))
            ),
            "W_required_with_yaw": float(w_pose_with_yaw),
            "W_required_without_yaw": float(w_structural),
            "mu_delta_with_yaw": mu_delta_with_yaw,
            "mu_delta_without_yaw": mu_delta_without_yaw,
            "valid_depth_ratio": valid_depth_ratio,
            "dropout_ratio": dropout_ratio,
            "valid_hit_ratio": valid_hit_ratio,
            "valid_no_return_ratio": valid_no_return_ratio,
            "invalid_ratio": invalid_ratio,
            "left_boundary_supported": left_boundary_supported,
            "right_boundary_supported": right_boundary_supported,
            "boundary_support_count": boundary_support_count,
            "width_observable": passage_aperture_observable,
            "passage_aperture_observable": passage_aperture_observable,
            "observation_regime": observation_regime,
            "structural_width_ci_gate_enabled": passage_aperture_observable,
            "ray_dispersion": ray_dispersion,
            "boundary_fitting_residual": boundary_residual,
            "temporal_width_variation": temporal_width_variation,
            "yaw_sigma": float(yaw_sigma),
            "lateral_pose_sigma": float(lateral_sigma),
            "yaw_pose_sigma": float(yaw_pose_sigma),
            "sigma_D": float(sigma_d),
            "sigma_W": float(sigma_w_pose),
            "uncertainty_source": (
                "fixed_global" if fixed_uncertainty else "dynamic_scene_dependent"
            ),
            "use_yaw_prior": bool(use_yaw_prior),
            "in_passage_regime": in_passage,
        }
        if not all(
            math.isfinite(float(value))
            for value in out.values()
            if isinstance(value, (int, float, np.floating))
        ):
            raise ValueError("DynamicFeasibilityEstimator produced non-finite values")
        return out

    def config_record(self) -> dict[str, float | int]:
        return asdict(self.cfg)
