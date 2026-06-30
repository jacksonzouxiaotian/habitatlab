"""Risk-aware traversability estimator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskWeights:
    alpha_network: float = 0.25
    beta_memory: float = 0.35
    gamma_clearance: float = 0.30
    delta_stuck: float = 0.10


@dataclass
class RiskEstimate:
    p_pass: float
    risk: float
    network_risk: float
    memory_risk: float
    clearance_risk: float
    stuck_risk: float


def fuse_risk(
    network_risk: float,
    memory_risk: float,
    clearance_risk: float,
    stuck_risk: float,
    weights: RiskWeights | None = None,
) -> RiskEstimate:
    weights = weights or RiskWeights()
    risk = (
        weights.alpha_network * network_risk
        + weights.beta_memory * memory_risk
        + weights.gamma_clearance * clearance_risk
        + weights.delta_stuck * stuck_risk
    )
    risk = max(0.0, min(1.0, risk))
    return RiskEstimate(
        p_pass=1.0 - risk,
        risk=risk,
        network_risk=network_risk,
        memory_risk=memory_risk,
        clearance_risk=clearance_risk,
        stuck_risk=stuck_risk,
    )
