"""Training-free depth-aware local candidate-heading planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from eagor_repro.planning.local_traversability import sector_clearance
from eagor_repro.spherical.spherical_grid import wrap_to_pi


@dataclass(frozen=True)
class LocalHeadingPrediction:
    target_heading: float
    selected_heading: float
    clearance_m: float
    planner_status: str
    blocked_candidates: int
    recovery_triggered: bool
    selected_score: float


class CandidateHeadingPlanner:
    VALID_MODES = {"direct", "depth_traversability", "oracle_nav"}

    def __init__(
        self,
        mode: str = "direct",
        candidate_offsets_deg: Sequence[float] = (
            0,
            -15,
            15,
            -30,
            30,
            -45,
            45,
            -60,
            60,
        ),
        sector_width_deg: float = 10.0,
        clearance_percentile: float = 25.0,
        vertical_band_deg: float = 35.0,
        obstacle_distance_m: float = 0.5,
        safety_distance_m: float = 0.7,
        goal_weight: float = 1.0,
        clearance_weight: float = 0.7,
        obstacle_weight: float = 2.0,
        turn_weight: float = 0.15,
    ) -> None:
        normalized = str(mode).lower()
        if normalized not in self.VALID_MODES:
            raise ValueError(f"Unknown planning mode {mode!r}")
        if not candidate_offsets_deg:
            raise ValueError("candidate_offsets_deg must not be empty")
        self.mode = normalized
        self.offsets_rad = np.deg2rad(np.asarray(candidate_offsets_deg, np.float64))
        self.sector_width_deg = float(sector_width_deg)
        self.clearance_percentile = float(clearance_percentile)
        self.vertical_band_deg = float(vertical_band_deg)
        self.obstacle_distance_m = float(obstacle_distance_m)
        self.safety_distance_m = float(safety_distance_m)
        self.goal_weight = float(goal_weight)
        self.clearance_weight = float(clearance_weight)
        self.obstacle_weight = float(obstacle_weight)
        self.turn_weight = float(turn_weight)

    @property
    def oracle_upper_bound(self) -> bool:
        return self.mode == "oracle_nav"

    def reset(self) -> None:
        pass

    def plan(
        self,
        target_heading: float,
        depth_erp: np.ndarray | None = None,
        oracle_heading: float | None = None,
    ) -> LocalHeadingPrediction:
        target = float(wrap_to_pi(target_heading))
        if self.mode == "direct":
            return LocalHeadingPrediction(
                target, target, float("nan"), "direct", 0, False, 1.0
            )
        if self.mode == "oracle_nav":
            if oracle_heading is None:
                raise ValueError(
                    "oracle_nav requires an explicit Habitat shortest-path heading"
                )
            selected = float(wrap_to_pi(oracle_heading))
            return LocalHeadingPrediction(
                target,
                selected,
                float("nan"),
                "oracle_nav_shortest_path",
                0,
                False,
                1.0,
            )
        if depth_erp is None:
            # Missing depth must never silently degrade to unsafe forward motion.
            recovery = float(wrap_to_pi(target + np.pi / 2.0))
            return LocalHeadingPrediction(
                target,
                recovery,
                0.0,
                "missing_depth_recovery",
                len(self.offsets_rad),
                True,
                float("-inf"),
            )

        candidates = np.asarray(
            [wrap_to_pi(target + offset) for offset in self.offsets_rad],
            dtype=np.float64,
        )
        clearances = np.asarray(
            [
                sector_clearance(
                    depth_erp,
                    heading,
                    self.sector_width_deg,
                    self.clearance_percentile,
                    self.vertical_band_deg,
                )[0]
                for heading in candidates
            ],
            dtype=np.float64,
        )
        blocked = clearances < self.safety_distance_m
        alignment = np.cos(
            np.asarray([wrap_to_pi(value - target) for value in candidates])
        )
        clearance_score = np.clip(
            clearances / max(self.safety_distance_m, 1e-6), 0.0, 2.0
        ) / 2.0
        obstacle_risk = (clearances < self.obstacle_distance_m).astype(np.float64)
        turning_cost = np.abs(candidates) / np.pi
        scores = (
            self.goal_weight * alignment
            + self.clearance_weight * clearance_score
            - self.obstacle_weight * obstacle_risk
            - self.turn_weight * turning_cost
        )

        feasible = np.flatnonzero(~blocked)
        recovery = feasible.size == 0
        if not recovery:
            index = int(feasible[np.argmax(scores[feasible])])
            status = "target_free" if index == 0 else "detour"
        else:
            # Prefer the clearest side, then a large-magnitude turn.  This
            # prevents an all-blocked state from issuing MOVE_FORWARD.
            nonzero = np.flatnonzero(np.abs(candidates) > np.deg2rad(1.0))
            pool = nonzero if nonzero.size else np.arange(len(candidates))
            index = int(
                max(pool, key=lambda item: (clearances[item], abs(candidates[item])))
            )
            status = "all_blocked_recovery"
        return LocalHeadingPrediction(
            target_heading=target,
            selected_heading=float(candidates[index]),
            clearance_m=float(clearances[index]),
            planner_status=status,
            blocked_candidates=int(blocked.sum()),
            recovery_triggered=bool(recovery),
            selected_score=float(scores[index]),
        )
