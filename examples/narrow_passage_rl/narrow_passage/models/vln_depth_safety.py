"""Online depth safety for discrete VLN actions in Habitat.

The adapter is deliberately independent of Habitat internals: it consumes a
depth image, a discrete action, and the previous transition metrics.  This
makes its intervention logic unit-testable and allows NaVILA to call it for
both fresh model proposals and queued macro-action steps.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3


@dataclass(frozen=True)
class DepthSafetyConfig:
    robot_radius_m: float = 0.18
    depth_max_m: float = 10.0
    hard_front_m: float = 0.35
    cautious_front_m: float = 0.70
    reject_body_margin_m: float = 0.02
    commit_body_margin_m: float = 0.12
    memory_reject_count: int = 3
    signature_front_bin_m: float = 0.20
    signature_side_bin_m: float = 0.10


@dataclass(frozen=True)
class DepthEvidence:
    front_clearance_m: float
    left_clearance_m: float
    right_clearance_m: float
    body_margin_m: float
    valid_fraction: float


@dataclass(frozen=True)
class DepthSafetyDecision:
    proposed_action: int
    adapted_action: int
    mode: str
    intervention: str
    reason: str
    allow_macro_queue: bool
    evidence: DepthEvidence
    memory_count: int


def _quantile(values: np.ndarray, q: float, default: float) -> float:
    finite = values[np.isfinite(values) & (values > 0.0)]
    return float(np.quantile(finite, q)) if finite.size else float(default)


def depth_evidence(
    depth: np.ndarray,
    cfg: DepthSafetyConfig | None = None,
) -> DepthEvidence:
    cfg = cfg or DepthSafetyConfig()
    image = np.asarray(depth, dtype=np.float32).squeeze()
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D depth image after squeeze, got {image.shape}")

    finite = np.isfinite(image) & (image > 0.0)
    valid_fraction = float(np.mean(finite))
    physical = image.copy()
    finite_values = physical[finite]
    if finite_values.size and float(np.nanmax(finite_values)) <= 1.5:
        physical *= cfg.depth_max_m
    physical[~finite] = np.nan

    height, width = physical.shape
    row_lo, row_hi = int(0.42 * height), max(int(0.68 * height), 1)
    scan = np.nanmedian(physical[row_lo:row_hi], axis=0)
    angles = np.linspace(-math.pi / 4.0, math.pi / 4.0, width)
    front = scan[np.abs(angles) <= math.radians(12.0)]

    left_mask = (angles >= math.radians(22.0)) & (
        angles <= math.radians(44.0)
    )
    right_mask = (angles <= -math.radians(22.0)) & (
        angles >= -math.radians(44.0)
    )
    left_lateral = scan[left_mask] * np.abs(np.sin(angles[left_mask]))
    right_lateral = scan[right_mask] * np.abs(np.sin(angles[right_mask]))
    front_clearance = _quantile(front, 0.15, cfg.depth_max_m)
    left_clearance = _quantile(left_lateral, 0.25, cfg.depth_max_m)
    right_clearance = _quantile(right_lateral, 0.25, cfg.depth_max_m)
    body_margin = min(left_clearance, right_clearance) - cfg.robot_radius_m
    return DepthEvidence(
        front_clearance_m=front_clearance,
        left_clearance_m=left_clearance,
        right_clearance_m=right_clearance,
        body_margin_m=body_margin,
        valid_fraction=valid_fraction,
    )


class OnlineDepthSafetyAdapter:
    """Stateful forward-action gate with collision-indexed episodic memory."""

    def __init__(self, cfg: DepthSafetyConfig | None = None) -> None:
        self.cfg = cfg or DepthSafetyConfig()
        self.failure_memory: dict[tuple[int, int, int], int] = defaultdict(int)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.last_collision = False
        self.last_signature: tuple[int, int, int] | None = None
        self.mode_counts: Counter[str] = Counter()
        self.proposal_counts: Counter[int] = Counter()
        self.action_counts: Counter[int] = Counter()
        self.interventions = 0
        self.memory_writes = 0
        self.collisions_observed = 0
        self.min_front_clearance_m = float("inf")
        self.min_body_margin_m = float("inf")

    def _signature(self, evidence: DepthEvidence) -> tuple[int, int, int]:
        cfg = self.cfg
        return (
            int(evidence.front_clearance_m / cfg.signature_front_bin_m),
            int(evidence.left_clearance_m / cfg.signature_side_bin_m),
            int(evidence.right_clearance_m / cfg.signature_side_bin_m),
        )

    @staticmethod
    def _open_side_turn(evidence: DepthEvidence) -> int:
        return (
            TURN_LEFT
            if evidence.left_clearance_m >= evidence.right_clearance_m
            else TURN_RIGHT
        )

    def decide(self, proposed_action: int, depth: np.ndarray) -> DepthSafetyDecision:
        proposed_action = int(proposed_action)
        evidence = depth_evidence(depth, self.cfg)
        signature = self._signature(evidence)
        memory_count = int(self.failure_memory.get(signature, 0))
        self.last_signature = signature
        self.proposal_counts[proposed_action] += 1
        self.min_front_clearance_m = min(
            self.min_front_clearance_m, evidence.front_clearance_m
        )
        self.min_body_margin_m = min(
            self.min_body_margin_m, evidence.body_margin_m
        )

        decision = self._decide(
            proposed_action, evidence, memory_count
        )
        self.mode_counts[decision.mode] += 1
        self.action_counts[decision.adapted_action] += 1
        self.interventions += int(decision.adapted_action != proposed_action)
        return decision

    def _decide(
        self,
        proposed_action: int,
        evidence: DepthEvidence,
        memory_count: int,
    ) -> DepthSafetyDecision:
        if proposed_action != MOVE_FORWARD:
            return DepthSafetyDecision(
                proposed_action,
                proposed_action,
                "COMMIT",
                "none",
                "non-forward VLN action accepted",
                True,
                evidence,
                memory_count,
            )

        safe_turn = self._open_side_turn(evidence)
        if self.last_collision:
            return DepthSafetyDecision(
                proposed_action,
                safe_turn,
                "RECOVER",
                "collision_recovery",
                "previous forward transition collided",
                False,
                evidence,
                memory_count,
            )
        if memory_count >= self.cfg.memory_reject_count:
            return DepthSafetyDecision(
                proposed_action,
                safe_turn,
                "REJECT",
                "failure_memory",
                "repeated collision signature rejects forward realization",
                False,
                evidence,
                memory_count,
            )
        if evidence.front_clearance_m <= self.cfg.hard_front_m:
            return DepthSafetyDecision(
                proposed_action,
                safe_turn,
                "RECOVER",
                "front_clearance",
                "hard frontal clearance requires recovery turn",
                False,
                evidence,
                memory_count,
            )
        if evidence.body_margin_m <= self.cfg.reject_body_margin_m:
            return DepthSafetyDecision(
                proposed_action,
                safe_turn,
                "REJECT",
                "body_margin",
                "estimated body margin rejects forward realization",
                False,
                evidence,
                memory_count,
            )
        if (
            evidence.front_clearance_m < self.cfg.cautious_front_m
            or evidence.body_margin_m < self.cfg.commit_body_margin_m
        ):
            return DepthSafetyDecision(
                proposed_action,
                proposed_action,
                "EXPLORE",
                "macro_cap",
                "uncertain clearance permits one cautious forward step",
                False,
                evidence,
                memory_count,
            )
        return DepthSafetyDecision(
            proposed_action,
            proposed_action,
            "COMMIT",
            "none",
            "forward realization accepted",
            True,
            evidence,
            memory_count,
        )

    def observe_transition(self, info: dict[str, Any]) -> None:
        collisions = info.get("collisions", {}) if isinstance(info, dict) else {}
        collided = bool(
            collisions.get("is_collision", False)
            if isinstance(collisions, dict)
            else False
        )
        self.last_collision = collided
        if not collided:
            return
        self.collisions_observed += 1
        if self.last_signature is not None:
            self.failure_memory[self.last_signature] += 1
            self.memory_writes += 1

    def episode_stats(self) -> dict[str, Any]:
        def finite_or_none(value: float) -> float | None:
            return float(value) if math.isfinite(value) else None

        return {
            "enabled": True,
            "config": asdict(self.cfg),
            "interventions": int(self.interventions),
            "mode_counts": dict(self.mode_counts),
            "proposal_counts": {str(k): int(v) for k, v in self.proposal_counts.items()},
            "action_counts": {str(k): int(v) for k, v in self.action_counts.items()},
            "collisions_observed": int(self.collisions_observed),
            "memory_writes": int(self.memory_writes),
            "memory_entries": int(len(self.failure_memory)),
            "min_front_clearance_m": finite_or_none(
                self.min_front_clearance_m
            ),
            "min_body_margin_m": finite_or_none(self.min_body_margin_m),
        }
