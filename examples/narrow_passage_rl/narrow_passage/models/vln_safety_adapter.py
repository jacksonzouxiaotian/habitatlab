"""Safety-aware adapter between a VLN action proposal and DEGNAV control.

NaVILA remains responsible for language-conditioned semantic action proposals.
This adapter does not modify or retrain the model.  It gives trusted runtime
directives and measured geometry/proprioceptive risk precedence over a proposed
forward action, then exposes the existing Commit/Explore/Recover/Reject modes.

Route instructions are never parsed as immediate commands unless
``trusted_control_directive`` is explicitly set.  This prevents a future phrase
such as "turn left after the sofa" from overriding the current VLN decision.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from .policy import DecisionMode


VALID_VLN_ACTIONS = ("move_forward", "turn_left", "turn_right", "stop")


@dataclass(frozen=True)
class VLNSafetyAdapterConfig:
    """Thresholds for the deterministic system-level safety interface."""

    reject_risk: float = 0.80
    explore_risk: float = 0.40
    reject_p_feas: float = 0.20
    commit_p_feas: float = 0.70
    reject_memory_risk: float = 0.80
    recover_stuck_score: float = 0.80
    reject_min_clearance_m: float = 0.08
    reject_body_margin_m: float = 0.02
    cautious_forward_cm: float = 20.0
    default_forward_cm: float = 25.0
    default_turn_deg: float = 30.0
    max_forward_cm: float = 75.0
    max_turn_deg: float = 90.0


@dataclass(frozen=True)
class VLNAdapterObservation:
    """Runtime evidence available below the VLN policy."""

    instruction: str
    trusted_control_directive: bool = False
    p_feas: Optional[float] = None
    risk: Optional[float] = None
    memory_risk: float = 0.0
    stuck_score: float = 0.0
    collision_flag: bool = False
    min_clearance_m: Optional[float] = None
    body_margin_m: Optional[float] = None


@dataclass(frozen=True)
class VLNAdapterDecision:
    """Result of applying trusted directives and measured safety evidence."""

    model_action: str
    model_value: Optional[float]
    adapted_action: str
    adapted_value: Optional[float]
    mode: DecisionMode
    intervention: str
    reason: str
    overridden: bool
    terminal_stop: bool
    effective_risk: float


@dataclass(frozen=True)
class ParsedDirective:
    action: str
    value: Optional[float]


_STOP_PATTERNS = (
    re.compile(r"\bstop\b", re.IGNORECASE),
    re.compile(r"\bdo not (?:move|continue|proceed|advance)\b", re.IGNORECASE),
    re.compile(r"\bremain at (?:the )?current position\b", re.IGNORECASE),
    re.compile(r"\bstay (?:here|still)\b", re.IGNORECASE),
)
_LEFT_PATTERN = re.compile(r"\b(?:turn|rotate|face|opening).*?\bleft\b", re.IGNORECASE)
_RIGHT_PATTERN = re.compile(r"\b(?:turn|rotate|face|opening).*?\bright\b", re.IGNORECASE)
_FORWARD_PATTERN = re.compile(
    r"\b(?:move|proceed|continue|advance|walk).*?"
    r"\b(?:forward|straight|ahead|through)\b",
    re.IGNORECASE,
)


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _extract_value(text: str, action: str, cfg: VLNSafetyAdapterConfig) -> float:
    if action == "move_forward":
        centimeters = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:cm|centimeter|centimeters)\b",
            text,
            re.IGNORECASE,
        )
        if centimeters:
            return float(centimeters.group(1))
        meters = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b",
            text,
            re.IGNORECASE,
        )
        if meters:
            return 100.0 * float(meters.group(1))
        return cfg.default_forward_cm
    degrees = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:degree|degrees|deg)\b",
        text,
        re.IGNORECASE,
    )
    return float(degrees.group(1)) if degrees else cfg.default_turn_deg


def parse_trusted_control_directive(
    instruction: str,
    cfg: VLNSafetyAdapterConfig | None = None,
) -> Optional[ParsedDirective]:
    """Parse only an explicitly trusted immediate-control instruction."""

    cfg = cfg or VLNSafetyAdapterConfig()
    text = str(instruction).strip()
    marker = re.search(r"instruction update\s*:", text, re.IGNORECASE)
    if marker:
        text = text[marker.end() :].strip()
    if not text:
        return None
    if any(pattern.search(text) for pattern in _STOP_PATTERNS):
        return ParsedDirective("stop", None)
    if _LEFT_PATTERN.search(text):
        return ParsedDirective("turn_left", _extract_value(text, "turn_left", cfg))
    if _RIGHT_PATTERN.search(text):
        return ParsedDirective(
            "turn_right", _extract_value(text, "turn_right", cfg)
        )
    if _FORWARD_PATTERN.search(text):
        return ParsedDirective(
            "move_forward", _extract_value(text, "move_forward", cfg)
        )
    return None


def _clamp_action_value(
    action: str,
    value: Optional[float],
    cfg: VLNSafetyAdapterConfig,
) -> Optional[float]:
    if action == "stop":
        return None
    number = _finite_or_none(value)
    if action == "move_forward":
        number = cfg.default_forward_cm if number is None else number
        return max(0.0, min(cfg.max_forward_cm, number))
    number = cfg.default_turn_deg if number is None else number
    return max(0.0, min(cfg.max_turn_deg, number))


class VLNSafetyAdapter:
    """Apply trusted control and DEGNAV safety evidence to a VLN proposal."""

    def __init__(self, cfg: VLNSafetyAdapterConfig | None = None) -> None:
        self.cfg = cfg or VLNSafetyAdapterConfig()

    def adapt(
        self,
        model_action: str,
        model_value: Optional[float],
        observation: VLNAdapterObservation,
    ) -> VLNAdapterDecision:
        cfg = self.cfg
        proposal = (
            model_action if model_action in VALID_VLN_ACTIONS else "stop"
        )
        proposal_value = _clamp_action_value(proposal, model_value, cfg)

        directive = (
            parse_trusted_control_directive(observation.instruction, cfg)
            if observation.trusted_control_directive
            else None
        )
        if directive is not None and directive.action == "stop":
            return self._decision(
                proposal,
                proposal_value,
                "stop",
                None,
                DecisionMode.REJECT,
                "trusted_directive",
                "trusted terminal stop directive",
                terminal_stop=True,
                effective_risk=self._effective_risk(observation),
            )

        if observation.collision_flag:
            return self._decision(
                proposal,
                proposal_value,
                "stop",
                None,
                DecisionMode.RECOVER,
                "safety",
                "collision flag requires low-level recovery",
                effective_risk=1.0,
            )
        if float(observation.stuck_score) >= cfg.recover_stuck_score:
            return self._decision(
                proposal,
                proposal_value,
                "stop",
                None,
                DecisionMode.RECOVER,
                "safety",
                "stuck score requires low-level recovery",
                effective_risk=self._effective_risk(observation),
            )

        effective_risk = self._effective_risk(observation)
        p_feas = _finite_or_none(observation.p_feas)
        clearance = _finite_or_none(observation.min_clearance_m)
        body_margin = _finite_or_none(observation.body_margin_m)
        reject_reasons = []
        if float(observation.memory_risk) >= cfg.reject_memory_risk:
            reject_reasons.append("failure-memory risk")
        if effective_risk >= cfg.reject_risk:
            reject_reasons.append("fused risk")
        if p_feas is not None and p_feas <= cfg.reject_p_feas:
            reject_reasons.append("low feasibility probability")
        if clearance is not None and clearance <= cfg.reject_min_clearance_m:
            reject_reasons.append("insufficient measured clearance")
        if body_margin is not None and body_margin <= cfg.reject_body_margin_m:
            reject_reasons.append("insufficient body margin")
        if reject_reasons:
            return self._decision(
                proposal,
                proposal_value,
                "stop",
                None,
                DecisionMode.REJECT,
                "safety",
                ", ".join(reject_reasons),
                effective_risk=effective_risk,
            )

        selected_action = directive.action if directive is not None else proposal
        selected_value = (
            directive.value if directive is not None else proposal_value
        )
        selected_value = _clamp_action_value(selected_action, selected_value, cfg)
        uncertain = effective_risk >= cfg.explore_risk or (
            p_feas is not None and p_feas < cfg.commit_p_feas
        )
        if uncertain:
            if selected_action == "move_forward":
                selected_value = min(
                    cfg.cautious_forward_cm,
                    selected_value or cfg.cautious_forward_cm,
                )
            mode = DecisionMode.EXPLORE
            intervention = "caution" if directive is None else "directive_and_caution"
            reason = "uncertain geometry; use cautious realization"
        else:
            mode = DecisionMode.COMMIT
            intervention = "trusted_directive" if directive is not None else "none"
            reason = (
                "trusted immediate directive"
                if directive is not None
                else "VLN proposal accepted"
            )
        return self._decision(
            proposal,
            proposal_value,
            selected_action,
            selected_value,
            mode,
            intervention,
            reason,
            effective_risk=effective_risk,
        )

    def _effective_risk(self, observation: VLNAdapterObservation) -> float:
        cfg = self.cfg
        candidates = [
            max(0.0, min(1.0, float(observation.memory_risk))),
            max(0.0, min(1.0, float(observation.stuck_score))),
        ]
        risk = _finite_or_none(observation.risk)
        if risk is not None:
            candidates.append(max(0.0, min(1.0, risk)))
        clearance = _finite_or_none(observation.min_clearance_m)
        if clearance is not None:
            candidates.append(
                max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - clearance
                        / max(2.0 * cfg.reject_min_clearance_m, 1e-6),
                    ),
                )
            )
        body_margin = _finite_or_none(observation.body_margin_m)
        if body_margin is not None:
            candidates.append(
                max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - body_margin
                        / max(2.0 * cfg.reject_body_margin_m, 1e-6),
                    ),
                )
            )
        return max(candidates)

    @staticmethod
    def _decision(
        model_action: str,
        model_value: Optional[float],
        adapted_action: str,
        adapted_value: Optional[float],
        mode: DecisionMode,
        intervention: str,
        reason: str,
        terminal_stop: bool = False,
        effective_risk: float = 0.0,
    ) -> VLNAdapterDecision:
        value_changed = (
            model_value is not None
            and adapted_value is not None
            and not math.isclose(model_value, adapted_value)
        )
        overridden = adapted_action != model_action or value_changed
        return VLNAdapterDecision(
            model_action=model_action,
            model_value=model_value,
            adapted_action=adapted_action,
            adapted_value=adapted_value,
            mode=mode,
            intervention=intervention,
            reason=reason,
            overridden=overridden,
            terminal_stop=terminal_stop,
            effective_risk=float(effective_risk),
        )


__all__ = [
    "ParsedDirective",
    "VALID_VLN_ACTIONS",
    "VLNAdapterDecision",
    "VLNAdapterObservation",
    "VLNSafetyAdapter",
    "VLNSafetyAdapterConfig",
    "parse_trusted_control_directive",
]
