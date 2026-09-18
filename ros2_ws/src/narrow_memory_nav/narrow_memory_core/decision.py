from dataclasses import dataclass
from typing import Dict

from narrow_memory_core.features import PassageFeatures
from narrow_memory_core.memory import FailureMemoryConfig, PassageFailureMemory


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return float(value >= high)
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


@dataclass(frozen=True)
class DecisionOutput:
    mode: str
    can_pass: bool
    risk: float
    memory_count: int
    failed_state_count: int


class FailureAwareNavigator:
    def __init__(
        self,
        memory_config: FailureMemoryConfig | None = None,
        clearance_low: float = 0.06,
        clearance_high: float = 0.13,
        lateral_low: float = 0.24,
        lateral_high: float = 0.34,
        heading_low: float = 0.45,
        heading_high: float = 0.75,
        stuck_high: float = 0.50,
        high_risk_threshold: float = 0.75,
        medium_risk_threshold: float = 0.50,
        memory_recovery_risk: float = 0.45,
    ) -> None:
        self.memory = PassageFailureMemory(memory_config)
        self.clearance_low = clearance_low
        self.clearance_high = clearance_high
        self.lateral_low = lateral_low
        self.lateral_high = lateral_high
        self.heading_low = heading_low
        self.heading_high = heading_high
        self.stuck_high = stuck_high
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        self.memory_recovery_risk = memory_recovery_risk

    def reset(self) -> None:
        self.memory.reset()

    def mark_failure(self, features: PassageFeatures) -> None:
        self.memory.add_failure(features)

    def score(self, features: PassageFeatures) -> Dict[str, float]:
        clearance_risk = 1.0 - _ramp(
            features.min_clearance, self.clearance_low, self.clearance_high
        )
        lateral_risk = _ramp(
            abs(features.lateral_offset), self.lateral_low, self.lateral_high
        )
        heading_risk = _ramp(
            abs(features.heading_error), self.heading_low, self.heading_high
        )
        stuck_risk = _ramp(features.stuck_score, 0.0, self.stuck_high)
        memory_risk = min(
            1.0,
            self.memory.count(features) / max(1, self.memory.config.reject_count),
        )
        risk = max(
            clearance_risk,
            0.8 * lateral_risk,
            0.8 * heading_risk,
            stuck_risk,
            features.collision_flag,
            0.6 * memory_risk,
        )
        return {
            "risk": float(risk),
            "clearance_risk": float(clearance_risk),
            "lateral_risk": float(lateral_risk),
            "heading_risk": float(heading_risk),
            "stuck_risk": float(stuck_risk),
            "memory_risk": float(memory_risk),
        }

    def decide(self, features: PassageFeatures) -> DecisionOutput:
        if features.collision_flag > 0.5 or features.stuck_score >= 0.7:
            self.mark_failure(features)

        risk = self.score(features)["risk"]
        if self.memory.should_reject(features):
            mode = "REJECT"
        elif self.memory.should_recover(features) and risk >= self.memory_recovery_risk:
            mode = "RECOVER"
        elif risk >= self.high_risk_threshold:
            mode = "RECOVER"
        elif risk >= self.medium_risk_threshold:
            mode = "ALIGN"
        else:
            mode = "COMMIT"

        return DecisionOutput(
            mode=mode,
            can_pass=mode not in {"REJECT", "RECOVER"},
            risk=float(risk),
            memory_count=self.memory.count(features),
            failed_state_count=self.memory.num_failed_states,
        )
