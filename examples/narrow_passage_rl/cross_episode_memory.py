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

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dmin_calibrator import CalibConfig, DMinCalibrator
from failure_memory import FailureMemoryConfig, PassageFailureMemory


# ── Fingerprint / retrieval ───────────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    """One stored episode outcome."""
    fingerprint: Tuple        # (width_bkt, heading_bkt, stuck_signal)
    success: bool
    passage_width: float
    steps: int = 0
    corridor_type: str = "unknown"
    episode_idx: int = 0


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

    def record_episode(
        self,
        entry_obs: np.ndarray,
        success: bool,
        passage_width: float,
        steps: int = 0,
        corridor_type: str = "unknown",
    ):
        """Call at the end of each episode with the outcome.

        Updates the D_min posterior and stores the fingerprinted record.
        """
        fp = _fingerprint(entry_obs, corridor_type)
        rec = EpisodeRecord(
            fingerprint=fp,
            success=success,
            passage_width=passage_width,
            steps=steps,
            corridor_type=corridor_type,
            episode_idx=self._episode_idx,
        )
        self._records.append(rec)
        if len(self._records) > self.cfg.max_records:
            self._records.pop(0)

        self._calibrator.update(passage_width, success, attempted=True)
        self._episode_idx += 1

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

    def retrieval_stats(self, obs: np.ndarray, corridor_type: str = "unknown") -> Dict:
        """Return statistics from similar past episodes."""
        fp = _fingerprint(obs, corridor_type)
        similar = [
            r for r in self._records
            if _fp_distance(r.fingerprint, fp) <= self.cfg.fp_radius
        ]
        n = len(similar)
        if n == 0:
            return {
                "n_similar": 0,
                "success_rate": 0.5,   # uniform prior
                "fsm_mode": FSMMode.STANDARD,
                "cautious": False,
            }
        sr = float(sum(r.success for r in similar)) / n
        if n >= self.cfg.min_similar_for_reject and sr < self.cfg.sr_reject_threshold:
            mode = FSMMode.REJECT
        elif n >= self.cfg.min_similar_for_cautious and sr < self.cfg.sr_cautious_threshold:
            mode = FSMMode.CAUTIOUS
        else:
            mode = FSMMode.STANDARD
        return {
            "n_similar": n,
            "success_rate": sr,
            "fsm_mode": mode,
            "cautious": mode != FSMMode.STANDARD,
        }

    def fsm_mode(self, obs: np.ndarray, corridor_type: str = "unknown") -> str:
        """Return FSMMode string for the current passage based on memory."""
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
        }

    def __repr__(self):
        s = self.summary()
        return (f"CrossEpisodeMemory(n={s['n_episodes']}, "
                f"sr={s['overall_sr']:.2f}, "
                f"d_hat={s['d_hat']:.3f}±{s.get('d_hat_std', 0):.3f})")
