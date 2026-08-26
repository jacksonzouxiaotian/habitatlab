"""Strict, auditable structural/pose feasibility selectors for DEGNav."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class FeasibilityAblation(str, Enum):
    FULL_DYNAMIC_UNCERTAINTY = "full_dynamic_uncertainty"
    MEAN_ONLY = "mean_only"
    FIXED_UNCERTAINTY = "fixed_uncertainty"
    NO_YAW_AWARE_READINESS = "no_yaw_aware_readiness"
    NO_MEMORY = "no_memory"
    # Accepted compatibility spellings. They are never independent paper rows.
    FULL = "full"
    POINT_ESTIMATE = "point_estimate"
    NO_UNCERTAINTY = "no_uncertainty"
    NO_YAW_PRIOR = "no_yaw_prior"


MAIN_ABLATIONS = (
    FeasibilityAblation.FULL_DYNAMIC_UNCERTAINTY.value,
    FeasibilityAblation.MEAN_ONLY.value,
    FeasibilityAblation.FIXED_UNCERTAINTY.value,
    FeasibilityAblation.NO_YAW_AWARE_READINESS.value,
    FeasibilityAblation.NO_MEMORY.value,
)
COMPATIBILITY_ALIASES = {
    FeasibilityAblation.FULL.value: FeasibilityAblation.FULL_DYNAMIC_UNCERTAINTY.value,
    FeasibilityAblation.POINT_ESTIMATE.value: FeasibilityAblation.MEAN_ONLY.value,
    FeasibilityAblation.NO_UNCERTAINTY.value: FeasibilityAblation.MEAN_ONLY.value,
    FeasibilityAblation.NO_YAW_PRIOR.value: FeasibilityAblation.NO_YAW_AWARE_READINESS.value,
}
ABLATIONS = MAIN_ABLATIONS + tuple(COMPATIBILITY_ALIASES)

PAPER_METHOD_NAMES = {
    FeasibilityAblation.FULL_DYNAMIC_UNCERTAINTY.value: "DEGNav full dynamic uncertainty",
    FeasibilityAblation.MEAN_ONLY.value: "DEGNav mean only",
    FeasibilityAblation.FIXED_UNCERTAINTY.value: "DEGNav fixed uncertainty",
    FeasibilityAblation.NO_YAW_AWARE_READINESS.value: "DEGNav w/o yaw-aware readiness",
    FeasibilityAblation.NO_MEMORY.value: "DEGNav w/o failure memory",
}


def canonical_ablation(ablation: str) -> str:
    if ablation in MAIN_ABLATIONS:
        return ablation
    try:
        return COMPATIBILITY_ALIASES[ablation]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feasibility ablation {ablation!r}; expected {ABLATIONS}"
        ) from exc


class AblationMode(str, Enum):
    COMMIT = "COMMIT"
    EXPLORE = "EXPLORE"
    RECOVER = "RECOVER"
    REJECT = "REJECT"


@dataclass(frozen=True)
class SelectorThresholds:
    """One shared set of interval and deterministic boundaries."""

    kappa: float = 1.645
    tau_commit: float = 0.020
    tau_reject: float = 0.020
    recover_stuck_score: float = 0.80
    reject_memory_risk: float = 0.80

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if self.tau_commit < 0.0 or self.tau_reject < 0.0:
            raise ValueError("tau_commit and tau_reject must be non-negative")


@dataclass(frozen=True)
class SharedDecisionState:
    """Non-margin signals identically available to paired variants."""

    collision: bool = False
    stuck_score: float = 0.0
    memory_risk: float = 0.0
    prior_commitment_failed: bool = False
    explicit_blocker: bool = False


@dataclass(frozen=True)
class ModeSelection:
    mode: AblationMode
    reason: str
    decision_rule: str
    uncertainty_triggered: bool = False


def interval_bounds(
    mu_delta: float,
    sigma_delta: float,
    kappa: float,
) -> tuple[float, float]:
    if sigma_delta < 0.0:
        raise ValueError("sigma_delta must be non-negative")
    radius = float(kappa) * float(sigma_delta)
    return float(mu_delta - radius), float(mu_delta + radius)


def _shared_override(
    shared: SharedDecisionState,
    cfg: SelectorThresholds,
) -> ModeSelection | None:
    if (
        shared.collision
        or shared.stuck_score > cfg.recover_stuck_score
        or shared.prior_commitment_failed
    ):
        return ModeSelection(
            AblationMode.RECOVER,
            "prior commitment failed/contact/stuck override",
            "shared_recovery",
        )
    if shared.explicit_blocker:
        return ModeSelection(
            AblationMode.REJECT,
            "explicit observed blocker",
            "shared_explicit_blocker",
        )
    if shared.memory_risk >= cfg.reject_memory_risk:
        return ModeSelection(
            AblationMode.REJECT,
            "repeated failure memory",
            "shared_memory",
        )
    return None


def _deterministic_mode(
    mu_struct: float,
    mu_pose: float,
    cfg: SelectorThresholds,
) -> AblationMode:
    if mu_struct < -cfg.tau_reject:
        return AblationMode.REJECT
    if mu_struct > cfg.tau_commit and mu_pose > cfg.tau_commit:
        return AblationMode.COMMIT
    return AblationMode.EXPLORE


def select_interval_gate(
    *,
    mu_struct: float,
    sigma_struct: float,
    mu_pose: float,
    sigma_pose: float,
    shared: SharedDecisionState,
    cfg: SelectorThresholds,
    rule_name: str,
) -> ModeSelection:
    """LCB/UCB gate with structural Reject and pose-only readiness gating."""

    override = _shared_override(shared, cfg)
    if override is not None:
        return override
    struct_lcb, struct_ucb = interval_bounds(mu_struct, sigma_struct, cfg.kappa)
    pose_lcb, _ = interval_bounds(mu_pose, sigma_pose, cfg.kappa)
    deterministic = _deterministic_mode(mu_struct, mu_pose, cfg)
    if struct_ucb < -cfg.tau_reject:
        mode = AblationMode.REJECT
        reason = "structural UCB below reject boundary"
    elif struct_lcb <= cfg.tau_commit:
        mode = AblationMode.EXPLORE
        reason = "structural interval overlaps feasibility boundary"
    elif pose_lcb > cfg.tau_commit:
        mode = AblationMode.COMMIT
        reason = "structural and pose-conditioned LCBs above commit boundary"
    else:
        mode = AblationMode.EXPLORE
        reason = "structurally feasible but pose-conditioned readiness uncertain"
    return ModeSelection(
        mode,
        reason,
        rule_name,
        uncertainty_triggered=(mode is not deterministic),
    )


def select_mean_only(
    *,
    mu_struct: float,
    mu_pose: float,
    shared: SharedDecisionState,
    cfg: SelectorThresholds,
    rule_name: str = "mean_only_dual_margin",
) -> ModeSelection:
    """Deterministic dual-margin gate; sigma cannot enter this function."""

    override = _shared_override(shared, cfg)
    if override is not None:
        return override
    mode = _deterministic_mode(mu_struct, mu_pose, cfg)
    if mode is AblationMode.REJECT:
        reason = "structural mean below reject boundary"
    elif mode is AblationMode.COMMIT:
        reason = "structural and pose-conditioned means above commit boundary"
    elif mu_struct <= cfg.tau_commit:
        reason = "structural mean inside feasibility deadband"
    else:
        reason = "structurally feasible but pose-conditioned mean not ready"
    return ModeSelection(mode, reason, rule_name)


# Backward import compatibility; behavior is exactly mean_only.
def select_point_estimate(**kwargs) -> ModeSelection:
    if "mu_delta" in kwargs:
        mu = float(kwargs.pop("mu_delta"))
        kwargs.setdefault("mu_struct", mu)
        kwargs.setdefault("mu_pose", mu)
    return select_mean_only(**kwargs)


def select_ablation_mode(
    ablation: str,
    metrics: Mapping[str, float],
    shared: SharedDecisionState,
    cfg: SelectorThresholds | None = None,
) -> ModeSelection:
    """Dispatch while reading only signals permitted by the ablation."""

    cfg = cfg or SelectorThresholds()
    canonical = canonical_ablation(ablation)
    if canonical == FeasibilityAblation.MEAN_ONLY.value:
        rule = (
            "compatibility_alias_behaviorally_equivalent_to_mean_only"
            if ablation in {"point_estimate", "no_uncertainty"}
            else "mean_only_dual_margin"
        )
        return select_mean_only(
            mu_struct=float(metrics["mu_delta_struct"]),
            mu_pose=float(metrics["mu_delta_pose"]),
            shared=shared,
            cfg=cfg,
            rule_name=rule,
        )
    rule = (
        "fixed_sigma_dual_interval"
        if canonical == FeasibilityAblation.FIXED_UNCERTAINTY.value
        else "dynamic_sigma_dual_interval"
    )
    return select_interval_gate(
        mu_struct=float(metrics["mu_delta_struct"]),
        sigma_struct=float(metrics["sigma_delta_struct"]),
        mu_pose=float(metrics["mu_delta_pose"]),
        sigma_pose=float(metrics["sigma_delta_pose"]),
        shared=shared,
        cfg=cfg,
        rule_name=rule,
    )


def uses_yaw_prior(ablation: str) -> bool:
    return canonical_ablation(ablation) != FeasibilityAblation.NO_YAW_AWARE_READINESS.value


def uses_fixed_uncertainty(ablation: str) -> bool:
    return canonical_ablation(ablation) == FeasibilityAblation.FIXED_UNCERTAINTY.value


def uses_memory(ablation: str) -> bool:
    return canonical_ablation(ablation) != FeasibilityAblation.NO_MEMORY.value


def capability_record(ablation: str) -> dict[str, bool | str]:
    """Machine-readable experimental contract written beside every run."""

    canonical = canonical_ablation(ablation)
    interval = canonical != FeasibilityAblation.MEAN_ONLY.value
    return {
        "canonical_method": canonical,
        "compatibility_alias": ablation != canonical,
        "behaviorally_equivalent_alias": (
            "mean_only" if ablation in {"point_estimate", "no_uncertainty"} else ""
        ),
        "dynamic_mu_estimator": True,
        "dynamic_uncertainty_estimator_executed": True,
        "uncertainty_logged_for_audit": True,
        "uncertainty_is_audit_only": not interval,
        "selector_reads_sigma_delta": interval,
        "dual_interval_gate": interval,
        "fixed_sigma": canonical == FeasibilityAblation.FIXED_UNCERTAINTY.value,
        "yaw_aware_readiness": uses_yaw_prior(ablation),
        "geometry_indexed_memory": uses_memory(ablation),
        "structural_only_reject": True,
        "pose_conditioned_commit": True,
        "heading_alignment": True,
        "recovery": True,
    }
