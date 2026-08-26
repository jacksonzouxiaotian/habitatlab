#!/usr/bin/env python3
"""Cross-episode passage memory for narrow-passage navigation.

Three functions integrated into one module:

  1. D_min calibration  — Bayesian posterior p(D_min | traversal history)
     across ALL past episodes.  Provides d_hat (estimated minimum navigable
     width) and d_hat_conservative (pessimistic upper percentile).

  2. Passage fingerprinting  — Compact descriptor of each passage attempt:
     (width_bucket, approach_difficulty, corridor_class_signal).  Stored with
     outcomes so similar past experiences can be retrieved for new passages.

  3. Decision policy  — Given retrieved similar episodes, sets FSM mode:
       cautious   (prior SR < 0.4 and ≥ 3 similar attempts)  →  slow speed,
                   max alignment gain, skip COMMIT → always EXPLORE
       reject     (prior SR < 0.1 and ≥ 5 similar attempts)  →  don't enter
       standard   otherwise                                   →  normal FSM

The 4-dim observation vector output matches the layout of MEMORY_FEATURE_NAMES
(failure_memory.py) so the existing RL policy and FSM code need no changes:
  [0] d_hat          ← posterior mean of D_min (replaces failure_count)
  [1] n_similar_norm ← # similar past episodes / 20 (replaces failed_state_count)
  [2] prior_sr       ← success rate among similar episodes (replaces should_recover)
  [3] cautious_flag  ← 1 if cautious / reject mode (replaces should_reject)

Usage (cross-episode, shared across env resets):
    mem = CrossEpisodeMemory()

    for episode_idx in range(N):
        obs, _ = env.reset()
        passage_width = float(obs[8])

        if not mem.should_attempt(passage_width):
            # memory says reject — skip this passage type
            info = {"success": False}
        else:
            # run FSM with memory-adjusted mode
            mode = mem.fsm_mode(obs)
            done = False
            while not done:
                action = fsm_step(obs, mode, mem.local)
                obs, _, done, _, info = env.step(action)
                mem.local.update(obs, info.get("collision", False))

        mem.record_episode(obs_at_entry, info["success"], passage_width,
                           corridor_type=info.get("corridor_type", "unknown"))

    print(f"D_hat = {mem.d_hat:.3f}  (true robot diameter ~ 0.36 m)")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from dmin_calibrator import CalibConfig, DMinCalibrator
from failure_memory import FailureMemoryConfig, PassageFailureMemory
from narrow_passage.models.robot_morphology import RobotMorphology


# ── Fingerprint / retrieval ───────────────────────────────────────────────────

class EpisodeOutcome(str, Enum):
    """Auditable episode outcomes used by geometry memory.

    Control failures and censored decisions are deliberately distinct from
    evidence that the robot footprint cannot fit through the passage.
    """

    SUCCESS = "success"
    GEOMETRIC_INFEASIBLE = "geometric_infeasible"
    COLLISION_GEOMETRY = "collision_geometry"
    COLLISION_CONTROL = "collision_control"
    COLLISION_UNKNOWN = "collision_unknown"
    TIMEOUT_CONTROL = "timeout_control"
    STUCK_CONTROL = "stuck_control"
    REJECT_CENSORED = "reject_censored"


class MemoryWriteType(str, Enum):
    POSITIVE_GEOMETRY = "positive_geometry"
    NEGATIVE_GEOMETRY = "negative_geometry"
    AUDIT_ONLY = "audit_only"
    DISABLED = "disabled"


GEOMETRY_POSITIVE_OUTCOMES = frozenset({EpisodeOutcome.SUCCESS})
GEOMETRY_NEGATIVE_OUTCOMES = frozenset({
    EpisodeOutcome.GEOMETRIC_INFEASIBLE,
    EpisodeOutcome.COLLISION_GEOMETRY,
})


@dataclass
class EpisodeRecord:
    """One stored episode outcome."""
    fingerprint: Tuple        # (width_bkt, heading_bkt, stuck_signal)
    success: bool
    passage_width: float
    steps: int = 0
    corridor_type: str = "unknown"
    episode_idx: int = 0
    outcome: EpisodeOutcome = EpisodeOutcome.COLLISION_UNKNOWN
    attempted: bool = False
    env_step_count: int = 0
    geometry_caused: bool = False
    memory_write_type: MemoryWriteType = MemoryWriteType.AUDIT_ONLY
    memory_write_reason: str = "legacy audit-only record"
    structural_margin: float = 0.0
    yaw_error: float = 0.0
    body_width: float = 0.36
    body_length: float = 0.60
    safety_margin: float = 0.03

    @property
    def geometry_memory_write(self) -> bool:
        return self.memory_write_type in {
            MemoryWriteType.POSITIVE_GEOMETRY,
            MemoryWriteType.NEGATIVE_GEOMETRY,
        }

    @property
    def geometry_sign(self) -> int:
        if self.memory_write_type is MemoryWriteType.POSITIVE_GEOMETRY:
            return 1
        if self.memory_write_type is MemoryWriteType.NEGATIVE_GEOMETRY:
            return -1
        return 0


def _fingerprint(obs: np.ndarray, corridor_type: str = "unknown") -> Tuple:
    """Compact descriptor: (passage_width_bucket, heading_difficulty, type_class).

    Designed to be robust to sensor noise: uses coarse buckets so that similar
    passages map to the same key even with depth measurement variance.
    """
    passage_width = float(obs[8])
    heading_err = abs(float(obs[10]))
    lateral_off = abs(float(obs[11]))

    # 8cm width buckets
    w_bkt = int(passage_width / 0.08)
    # combined approach difficulty (heading + lateral)
    diff = heading_err / 0.4 + lateral_off / 0.2  # normalise
    d_bkt = int(min(diff, 3.0) / 0.5)
    # rough corridor class from type string
    type_class = 0
    if "l_shaped" in corridor_type or "s_shaped" in corridor_type:
        type_class = 1
    elif "false" in corridor_type:
        type_class = 2
    elif "narrow_exit" in corridor_type or "narrow_entry" in corridor_type:
        type_class = 3

    return (w_bkt, d_bkt, type_class)


def _fp_distance(a: Tuple, b: Tuple) -> int:
    """L-inf distance between two fingerprints."""
    return max(abs(a[i] - b[i]) for i in range(min(len(a), len(b))))


# ── Decision policy ───────────────────────────────────────────────────────────

class FSMMode:
    STANDARD = "standard"    # normal speed and gains
    CAUTIOUS = "cautious"    # slower + higher alignment gain
    REJECT = "reject"        # do not enter passage


@dataclass
class MemoryConfig:
    # D_min calibration
    calib: CalibConfig = field(default_factory=CalibConfig)
    # Retrieval
    fp_radius: int = 1          # fingerprint L-inf ball radius for "similar"
    min_similar_for_cautious: int = 3
    min_similar_for_reject: int = 5
    sr_cautious_threshold: float = 0.40
    sr_reject_threshold: float = 0.10
    # Episode-local failure memory config
    local_cfg: FailureMemoryConfig = field(default_factory=FailureMemoryConfig)
    # Max stored records (discard oldest on overflow)
    max_records: int = 500
    # Repaired geometry-memory retrieval.  Coarse fingerprints remain logged
    # and can be re-enabled only for historical reproduction.
    use_continuous_similarity: bool = True
    continuous_similarity_radius: float = 0.50
    width_weight: float = 2.0
    margin_weight: float = 4.0
    yaw_weight: float = 0.50
    body_width_weight: float = 8.0
    body_length_weight: float = 8.0
    margin_sign_mismatch_penalty: float = 1.0
    corridor_class_mismatch_penalty: float = 0.75
    require_same_morphology: bool = True
    morphology_tolerance: float = 1e-6
    # Bounded logit correction; memory is soft evidence unless the explicit
    # terminal-support rule below is satisfied.
    delta_max: float = 0.75
    memory_logit_gain: float = 0.50
    memory_explore_probability: float = 0.50
    terminal_negative_support: int = 3
    terminal_confidence: float = 0.90
    terminal_reject_margin: float = 0.02
    confidence_prior_strength: float = 0.25
    frozen: bool = False

    def __post_init__(self) -> None:
        if self.continuous_similarity_radius <= 0.0:
            raise ValueError("continuous_similarity_radius must be positive")
        if self.delta_max <= 0.0:
            raise ValueError("delta_max must be positive")
        if self.terminal_negative_support < 3:
            raise ValueError("terminal_negative_support must be at least three")
        if not 0.0 <= self.terminal_confidence <= 1.0:
            raise ValueError("terminal_confidence must be in [0, 1]")


# ── Main class ────────────────────────────────────────────────────────────────

class CrossEpisodeMemory:
    """Cross-episode passage memory.  One instance shared across all episodes.

    At episode reset: call reset_local() to clear the episode-local state.
    At episode end:   call record_episode() with the outcome.
    During episode:   use local (PassageFailureMemory) for within-episode
                      recovery triggers, and fsm_mode() / should_attempt()
                      for cross-episode decisions.
    """

    def __init__(self, cfg: MemoryConfig = None):
        self.cfg = cfg or MemoryConfig()
        self._calibrator = DMinCalibrator(cfg=self.cfg.calib if cfg else CalibConfig())
        self._records: List[EpisodeRecord] = []
        self._episode_idx = 0
        self.local = PassageFailureMemory(self.cfg.local_cfg)

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def reset_local(self):
        """Call at the start of each episode."""
        self.local.reset()

    def reset(self) -> None:
        """Reset both episode-local and cross-episode state.

        The strict evaluator calls this explicitly at every method/seed
        boundary so online evidence cannot leak across paired runs.
        """

        self._calibrator = DMinCalibrator(cfg=self.cfg.calib)
        self._records.clear()
        self._episode_idx = 0
        self.local.reset()

    def record_episode(
        self,
        entry_obs: np.ndarray,
        success: bool | None,
        passage_width: float,
        steps: int = 0,
        corridor_type: str = "unknown",
        *,
        outcome: EpisodeOutcome | str | None = None,
        attempted: bool | None = None,
        env_step_count: int | None = None,
        geometry_caused: bool = False,
        structural_margin: float | None = None,
        yaw_error: float | None = None,
        morphology: RobotMorphology | None = None,
    ) -> Dict[str, object]:
        """Store one structured outcome and update geometry belief if valid.

        Compatibility callers may still pass ``success`` positionally.  A
        legacy boolean failure is conservatively classified as
        :class:`EpisodeOutcome.COLLISION_UNKNOWN`, never as negative geometry.
        ``attempted`` is derived from the real number of environment steps and
        cannot be asserted independently.
        """

        env_steps = int(steps if env_step_count is None else env_step_count)
        if env_steps < 0:
            raise ValueError("env_step_count must be non-negative")
        inferred_attempted = env_steps > 0
        if attempted is not None and bool(attempted) != inferred_attempted:
            raise ValueError("attempted must equal env_step_count > 0")
        attempted = inferred_attempted

        if outcome is None:
            if bool(success):
                outcome_value = EpisodeOutcome.SUCCESS
                # Old successful callers sometimes omitted steps.  Preserve
                # their API while keeping the invariant for the stored record.
                if env_steps == 0:
                    env_steps = max(1, int(steps))
                    attempted = True
            else:
                outcome_value = EpisodeOutcome.COLLISION_UNKNOWN
        else:
            outcome_value = EpisodeOutcome(outcome)
        if outcome_value is EpisodeOutcome.REJECT_CENSORED:
            if env_steps != 0:
                # A later policy rejection is still censored geometry evidence,
                # but it was an attempted episode.  Keep the true attempt bit.
                attempted = True
            else:
                attempted = False

        morphology = morphology or RobotMorphology()
        margin = (
            float(passage_width) - morphology.structural_required_width
            if structural_margin is None
            else float(structural_margin)
        )
        yaw = float(entry_obs[10]) if yaw_error is None else float(yaw_error)

        if self.cfg.frozen:
            write_type = MemoryWriteType.DISABLED
            write_reason = "frozen memory: audit record only"
        elif not attempted:
            write_type = MemoryWriteType.AUDIT_ONLY
            write_reason = "no env.step executed; outcome is censored"
        elif outcome_value in GEOMETRY_POSITIVE_OUTCOMES:
            write_type = MemoryWriteType.POSITIVE_GEOMETRY
            write_reason = "successful traversal is positive geometry evidence"
        elif outcome_value in GEOMETRY_NEGATIVE_OUTCOMES and (
            outcome_value is EpisodeOutcome.GEOMETRIC_INFEASIBLE
            or bool(geometry_caused)
        ):
            write_type = MemoryWriteType.NEGATIVE_GEOMETRY
            write_reason = "confirmed geometry-caused failure"
        else:
            write_type = MemoryWriteType.AUDIT_ONLY
            write_reason = f"{outcome_value.value} is not geometry-negative evidence"

        fp = _fingerprint(entry_obs, corridor_type)
        rec = EpisodeRecord(
            fingerprint=fp,
            success=bool(success),
            passage_width=passage_width,
            steps=steps,
            corridor_type=corridor_type,
            episode_idx=self._episode_idx,
            outcome=outcome_value,
            attempted=bool(attempted),
            env_step_count=env_steps,
            geometry_caused=bool(geometry_caused),
            memory_write_type=write_type,
            memory_write_reason=write_reason,
            structural_margin=margin,
            yaw_error=yaw,
            body_width=morphology.width,
            body_length=morphology.length,
            safety_margin=morphology.safety_margin,
        )
        self._records.append(rec)
        if len(self._records) > self.cfg.max_records:
            self._records.pop(0)

        if write_type is MemoryWriteType.POSITIVE_GEOMETRY:
            self._calibrator.update(passage_width, True, attempted=True)
        elif write_type is MemoryWriteType.NEGATIVE_GEOMETRY:
            self._calibrator.update(passage_width, False, attempted=True)
        self._episode_idx += 1
        assert not (
            outcome_value is EpisodeOutcome.REJECT_CENSORED
            and rec.geometry_memory_write
        )
        return {
            "episode_outcome": outcome_value.value,
            "attempted": bool(attempted),
            "env_step_count": env_steps,
            "geometry_caused": bool(geometry_caused),
            "geometry_memory_write": rec.geometry_memory_write,
            "memory_write_type": write_type.value,
            "memory_write_reason": write_reason,
        }

    # ── D_min interface ───────────────────────────────────────────────────────

    @property
    def d_hat(self) -> float:
        return self._calibrator.d_hat

    @property
    def d_hat_conservative(self) -> float:
        return self._calibrator.d_hat_conservative

    @property
    def d_hat_std(self) -> float:
        return self._calibrator.std

    def should_attempt(
        self, passage_width: float, corridor_type: str = "unknown",
        entry_obs: np.ndarray = None,
        rng: np.random.Generator = None
    ) -> bool:
        """Combine D_min gating and retrieval-based reject to decide entry.

        Pass entry_obs (the full reset observation) for accurate fingerprint
        matching — omitting it uses a zero-obs which zeroes d_bkt and may miss
        stored records that had non-zero heading/lateral at approach.
        """
        if entry_obs is not None:
            obs_q = entry_obs
        else:
            obs_q = np.zeros(19, dtype=np.float32)
            obs_q[8] = passage_width
        info = self.retrieval_stats(obs_q, corridor_type)
        if (info["n_similar"] >= self.cfg.min_similar_for_reject and
                info["success_rate"] < self.cfg.sr_reject_threshold):
            return False
        # D_min calibration gate
        return self._calibrator.should_attempt(passage_width, rng)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    @staticmethod
    def _type_class(corridor_type: str) -> int:
        return int(_fingerprint(np.zeros(19, dtype=np.float32), corridor_type)[2])

    def _continuous_distance(
        self,
        record: EpisodeRecord,
        *,
        passage_width: float,
        structural_margin: float,
        yaw_error: float,
        morphology: RobotMorphology,
        corridor_type: str,
    ) -> float:
        c = self.cfg
        if c.require_same_morphology and (
            abs(record.body_width - morphology.width) > c.morphology_tolerance
            or abs(record.body_length - morphology.length) > c.morphology_tolerance
            or abs(record.safety_margin - morphology.safety_margin) > c.morphology_tolerance
        ):
            return float("inf")
        delta_yaw = abs(math.atan2(
            math.sin(record.yaw_error - yaw_error),
            math.cos(record.yaw_error - yaw_error),
        ))
        distance = (
            c.width_weight * abs(record.passage_width - passage_width)
            + c.margin_weight * abs(record.structural_margin - structural_margin)
            + c.yaw_weight * delta_yaw
            + c.body_width_weight * abs(record.body_width - morphology.width)
            + c.body_length_weight * abs(record.body_length - morphology.length)
        )
        if record.structural_margin * structural_margin < 0.0:
            distance += c.margin_sign_mismatch_penalty
        if self._type_class(record.corridor_type) != self._type_class(corridor_type):
            distance += c.corridor_class_mismatch_penalty
        return float(distance)

    def retrieval_stats(
        self,
        obs: np.ndarray,
        corridor_type: str = "unknown",
        *,
        morphology: RobotMorphology | None = None,
        structural_margin: float | None = None,
    ) -> Dict:
        """Return weighted positive/negative geometry evidence and audit data."""

        fp = _fingerprint(obs, corridor_type)
        morphology = morphology or RobotMorphology()
        width = float(obs[8])
        margin = (
            width - morphology.structural_required_width
            if structural_margin is None
            else float(structural_margin)
        )
        yaw = float(obs[10])
        candidates = [r for r in self._records if r.geometry_memory_write]
        distances: list[float] = []
        similar: list[EpisodeRecord] = []
        weights: list[float] = []
        for record in candidates:
            if self.cfg.use_continuous_similarity:
                distance = self._continuous_distance(
                    record,
                    passage_width=width,
                    structural_margin=margin,
                    yaw_error=yaw,
                    morphology=morphology,
                    corridor_type=corridor_type,
                )
                if distance > self.cfg.continuous_similarity_radius:
                    continue
                weight = math.exp(-distance / self.cfg.continuous_similarity_radius)
            else:
                distance = float(_fp_distance(record.fingerprint, fp))
                if distance > self.cfg.fp_radius:
                    continue
                weight = 1.0
            similar.append(record)
            distances.append(distance)
            weights.append(weight)
        n = len(similar)
        if n == 0:
            return {
                "n_similar": 0,
                "success_rate": 0.5,   # uniform prior
                "fsm_mode": FSMMode.STANDARD,
                "cautious": False,
                "positive_geometry_support": 0,
                "negative_geometry_support": 0,
                "positive_geometry_weight": 0.0,
                "negative_geometry_weight": 0.0,
                "evidence_conflict": False,
                "memory_confidence": 0.0,
                "memory_delta": 0.0,
                "similarity_weight_sum": 0.0,
                "min_similarity_distance": float("nan"),
                "mean_similarity_distance": float("nan"),
                "similarity_mode": (
                    "continuous_morphology_aware"
                    if self.cfg.use_continuous_similarity else "coarse_fingerprint"
                ),
            }
        positive_support = sum(r.geometry_sign > 0 for r in similar)
        negative_support = sum(r.geometry_sign < 0 for r in similar)
        positive_weight = sum(
            weight for record, weight in zip(similar, weights)
            if record.geometry_sign > 0
        )
        negative_weight = sum(
            weight for record, weight in zip(similar, weights)
            if record.geometry_sign < 0
        )
        total_weight = positive_weight + negative_weight
        sr = float(positive_weight / total_weight) if total_weight > 0.0 else 0.5
        raw_delta = self.cfg.memory_logit_gain * math.log(
            (positive_weight + 1.0) / (negative_weight + 1.0)
        )
        memory_delta = float(np.clip(raw_delta, -self.cfg.delta_max, self.cfg.delta_max))
        memory_confidence = float(
            total_weight / (total_weight + self.cfg.confidence_prior_strength)
        )
        conflict = positive_support > 0 and negative_support > 0
        if negative_support >= self.cfg.min_similar_for_cautious and not conflict:
            mode = FSMMode.CAUTIOUS
        else:
            mode = FSMMode.STANDARD
        return {
            "n_similar": n,
            "success_rate": sr,
            "fsm_mode": mode,
            "cautious": mode != FSMMode.STANDARD,
            "positive_geometry_support": positive_support,
            "negative_geometry_support": negative_support,
            "positive_geometry_weight": float(positive_weight),
            "negative_geometry_weight": float(negative_weight),
            "evidence_conflict": conflict,
            "memory_confidence": memory_confidence,
            "memory_delta": memory_delta,
            "similarity_weight_sum": float(total_weight),
            "min_similarity_distance": float(min(distances)),
            "mean_similarity_distance": float(np.mean(distances)),
            "similarity_mode": (
                "continuous_morphology_aware"
                if self.cfg.use_continuous_similarity else "coarse_fingerprint"
            ),
        }

    def correct_feasibility(
        self,
        obs: np.ndarray,
        *,
        base_p_feas: float,
        base_structural_margin: float,
        corridor_type: str = "unknown",
        morphology: RobotMorphology | None = None,
    ) -> Dict[str, object]:
        """Apply bounded memory evidence in logit space.

        ``corrected_logit = logit(base_p_feas) + clip(memory_delta,
        -delta_max, delta_max)``.  Terminal rejection additionally requires at
        least three uncontested negative geometry records, confidence >= 0.90,
        and a negative base structural margin.
        """

        stats = self.retrieval_stats(
            obs,
            corridor_type,
            morphology=morphology,
            structural_margin=base_structural_margin,
        )
        p = float(np.clip(base_p_feas, 1e-6, 1.0 - 1e-6))
        base_logit = math.log(p / (1.0 - p))
        memory_delta = float(stats["memory_delta"])
        corrected_logit = base_logit + memory_delta
        corrected_p = 1.0 / (1.0 + math.exp(-corrected_logit))
        terminal_reject = bool(
            int(stats["negative_geometry_support"])
            >= self.cfg.terminal_negative_support
            and float(stats["memory_confidence"]) >= self.cfg.terminal_confidence
            and float(base_structural_margin) < -self.cfg.terminal_reject_margin
            and int(stats["positive_geometry_support"]) == 0
        )
        return {
            **stats,
            "base_p_feas": float(base_p_feas),
            "base_logit": float(base_logit),
            "memory_delta": memory_delta,
            "corrected_logit": float(corrected_logit),
            "corrected_p_feas": float(corrected_p),
            "memory_terminal_reject": terminal_reject,
            "memory_cautious": bool(
                memory_delta < 0.0 and not bool(stats["evidence_conflict"])
            ),
        }

    def fsm_mode(self, obs: np.ndarray, corridor_type: str = "unknown") -> str:
        """Return a legacy-compatible *non-terminal* memory mode.

        Terminal geometry rejection now needs the base structural belief and is
        available only through :meth:`correct_feasibility`.
        """
        return self.retrieval_stats(obs, corridor_type)["fsm_mode"]

    # ── Memory observation vector (4-dim) ─────────────────────────────────────

    def as_array(self, obs: np.ndarray, corridor_type: str = "unknown") -> np.ndarray:
        """4-dim vector compatible with MEMORY_FEATURE_NAMES / NarrowPassageMemoryState.

        Layout:
          [0] d_hat          — posterior mean D_min (calibrated width threshold)
          [1] n_similar_norm — # similar past episodes / 20 (exploration signal)
          [2] prior_sr       — success rate in similar past episodes
          [3] cautious_flag  — 1.0 if cautious or reject mode active
        """
        info = self.retrieval_stats(obs, corridor_type)
        return np.array([
            self.d_hat,
            min(1.0, info["n_similar"] / 20.0),
            float(info["success_rate"]),
            float(info["cautious"]),
        ], dtype=np.float32)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        n = len(self._records)
        if n == 0:
            return {"n_episodes": 0, "overall_sr": 0.0, "d_hat": self.d_hat}
        by_type: Dict[str, List[bool]] = {}
        for r in self._records:
            by_type.setdefault(r.corridor_type, []).append(r.success)
        sr_by_type = {t: sum(v) / len(v) for t, v in by_type.items()}
        return {
            "n_episodes": n,
            "overall_sr": sum(r.success for r in self._records) / n,
            "d_hat": self.d_hat,
            "d_hat_std": self.d_hat_std,
            "sr_by_type": sr_by_type,
            "positive_geometry_records": sum(r.geometry_sign > 0 for r in self._records),
            "negative_geometry_records": sum(r.geometry_sign < 0 for r in self._records),
            "audit_only_records": sum(r.geometry_sign == 0 for r in self._records),
        }

    def __repr__(self):
        s = self.summary()
        return (f"CrossEpisodeMemory(n={s['n_episodes']}, "
                f"sr={s['overall_sr']:.2f}, "
                f"d_hat={s['d_hat']:.3f}±{s.get('d_hat_std', 0):.3f})")
