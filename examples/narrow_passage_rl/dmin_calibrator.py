#!/usr/bin/env python3
"""Bayesian D_min self-calibration module.

The robot's minimum navigable passage width D_min = 2 × robot_radius is a
physical constant — but in practice it may be uncertain (unknown payload,
irregular shape, sensor bias).  This module maintains a posterior distribution
p(D_min | traversal history) and updates it from passage outcomes.

Update rule (soft / noisy):
  Success at width W  →  D_min is likely ≤ W:
      likelihood  ∝  sigmoid((W - D_min) / σ)
  Failure at width W  →  D_min is likely ≥ W:
      likelihood  ∝  sigmoid((D_min - W) / σ)

The posterior is represented as a discrete distribution over a fine grid of
D_min values in [d_min_lo, d_min_hi].

Usage
-----
    cal = DMinCalibrator(d_true=0.36, d_init=0.55)   # wrong initial guess
    for episode in episodes:
        attempt = cal.should_attempt(obs_passage_width)
        if attempt:
            success = run_fsm(env)
        else:
            success = False   # rejected without attempt
        cal.update(obs_passage_width, success, attempted=attempt)
    print(cal.d_hat)   # converges toward 0.36
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CalibConfig:
    d_min_lo:  float = 0.20   # grid lower bound (m)
    d_min_hi:  float = 0.80   # grid upper bound (m)
    n_bins:    int   = 200    # number of grid points
    noise_sigma: float = 0.03  # observation noise (m)
    # exploration: probability of attempting even when D_hat > passage_width
    explore_prob: float = 0.20
    # conservative percentile used as D_hat for reject decisions
    conservative_pct: float = 0.25   # 25th percentile (pessimistic)


class DMinCalibrator:
    """Bayesian estimator for D_min (minimum navigable passage width)."""

    def __init__(self, d_init: float = 0.50, cfg: CalibConfig = None):
        self.cfg = cfg or CalibConfig()
        c = self.cfg
        self.d_vals = np.linspace(c.d_min_lo, c.d_min_hi, c.n_bins)

        # Gaussian prior centered on d_init
        prior_sigma = 0.08
        self.probs = np.exp(-0.5 * ((self.d_vals - d_init) / prior_sigma) ** 2)
        self.probs /= self.probs.sum()

        self.history: List[Tuple[float, bool]] = []   # (width, success)
        self.n_updates = 0

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def d_hat(self) -> float:
        """Posterior mean — best point estimate of D_min."""
        return float(np.dot(self.probs, self.d_vals))

    @property
    def d_hat_conservative(self) -> float:
        """Upper-percentile estimate for conservative commit/reject decisions."""
        return self.quantile(1.0 - self.cfg.conservative_pct)

    @property
    def std(self) -> float:
        """Posterior standard deviation."""
        mu = self.d_hat
        return float(np.sqrt(np.dot(self.probs, (self.d_vals - mu) ** 2)))

    def should_attempt(self, passage_width: float, rng: np.random.Generator = None) -> bool:
        """Return True if the robot should attempt traversal of a passage.

        Uses d_hat_conservative (pessimistic) to decide.  With probability
        explore_prob, attempts even if the conservative estimate suggests
        the passage is too narrow (exploration for calibration).
        """
        if rng is None:
            rng = np.random.default_rng()
        margin_hat = passage_width - self.d_hat_conservative
        if margin_hat >= 0:
            return True   # confident the passage is wide enough
        # Passage appears too narrow — explore with some probability
        return bool(rng.random() < self.cfg.explore_prob)

    def quantile(self, pct: float) -> float:
        """Return the posterior quantile for ``pct`` in [0, 1]."""

        pct = float(np.clip(pct, 0.0, 1.0))
        cumsum = np.cumsum(self.probs)
        idx = int(np.searchsorted(cumsum, pct))
        return float(self.d_vals[min(idx, len(self.d_vals) - 1)])

    def prob_feasible(self, passage_width: float) -> float:
        """Return ``P(D_min <= passage_width)`` under the posterior."""

        width = float(passage_width)
        return float(self.probs[self.d_vals <= width].sum())

    def update(self, passage_width: float, success: bool, attempted: bool = True):
        """Update posterior from one traversal outcome.

        If not attempted (rejected without trial), no information gained
        about D_min from that episode.
        """
        if not attempted:
            return

        W = float(passage_width)
        sigma = self.cfg.noise_sigma

        if success:
            # D_min is likely ≤ W
            likelihood = 1.0 / (1.0 + np.exp(-(W - self.d_vals) / sigma))
        else:
            # D_min is likely ≥ W  (or the passage was navigable but FSM failed)
            likelihood = 1.0 / (1.0 + np.exp(-(self.d_vals - W) / sigma))

        self.probs = self.probs * likelihood
        s = self.probs.sum()
        if s > 1e-300:
            self.probs /= s
        else:
            # Numerical underflow — reset to uniform (shouldn't happen with σ>0)
            self.probs = np.ones_like(self.probs) / len(self.probs)

        self.history.append((W, success))
        self.n_updates += 1

    def posterior_interval(self, alpha: float = 0.90) -> Tuple[float, float]:
        """Return (lo, hi) credible interval containing `alpha` probability mass."""
        cumsum = np.cumsum(self.probs)
        lo_idx = int(np.searchsorted(cumsum, (1 - alpha) / 2))
        hi_idx = int(np.searchsorted(cumsum, 1 - (1 - alpha) / 2))
        return float(self.d_vals[lo_idx]), float(self.d_vals[min(hi_idx, len(self.d_vals)-1)])
