#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict


@dataclass
class RiskConfig:
    clearance_low: float = 0.08
    clearance_high: float = 0.14
    lateral_low: float = 0.24
    lateral_high: float = 0.34
    heading_low: float = 0.45
    heading_high: float = 0.75
    stuck_high: float = 0.50
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.45
    reject_after_failures: int = 3


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return float(value >= high)
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


class GeometryRiskEstimator:
    """Interpretable geometry/failure risk score for mode decisions."""

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.failure_count = 0

    def reset(self):
        self.failure_count = 0

    def mark_failure(self):
        self.failure_count += 1

    def score(self, obs) -> Dict[str, float]:
        cfg = self.config
        min_clearance = min(float(obs[6]), float(obs[7]))
        lateral_offset = abs(float(obs[11]))
        heading_error = abs(float(obs[10]))
        stuck_score = float(obs[15])
        collision_flag = float(obs[16])

        clearance_risk = 1.0 - _ramp(
            min_clearance, cfg.clearance_low, cfg.clearance_high
        )
        lateral_risk = _ramp(lateral_offset, cfg.lateral_low, cfg.lateral_high)
        heading_risk = _ramp(heading_error, cfg.heading_low, cfg.heading_high)
        stuck_risk = _ramp(stuck_score, 0.0, cfg.stuck_high)
        memory_risk = min(1.0, self.failure_count / max(1, cfg.reject_after_failures))

        risk = max(
            clearance_risk,
            0.8 * lateral_risk,
            0.8 * heading_risk,
            stuck_risk,
            collision_flag,
            0.6 * memory_risk,
        )

        return {
            "risk": float(risk),
            "clearance_risk": float(clearance_risk),
            "lateral_risk": float(lateral_risk),
            "heading_risk": float(heading_risk),
            "stuck_risk": float(stuck_risk),
            "memory_risk": float(memory_risk),
            "min_clearance": float(min_clearance),
        }

    def mode(self, obs) -> str:
        cfg = self.config
        risk = self.score(obs)["risk"]
        if self.failure_count >= cfg.reject_after_failures:
            return "REJECT"
        if risk >= cfg.high_risk_threshold:
            return "RECOVER"
        if risk >= cfg.medium_risk_threshold:
            return "ALIGN"
        return "COMMIT"
