#!/usr/bin/env python3
"""Observation wrapper for the Replay Memory Policy baseline.

This is intentionally weaker than Geometry-Guided Failure Memory: it appends a
small generic historical embedding to the policy input, but does not perform
explicit geometry-aware retrieval or high-level Reject/Recover mode selection.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - legacy fallback
    import gym
    from gym import spaces


@dataclass
class ReplayStats:
    attempts: int = 0
    failures: int = 0
    successes: int = 0
    last_outcome: float = 0.0
    last_min_clearance: float = 0.0
    last_stuck: float = 0.0
    last_collision: float = 0.0


def replay_key(obs: np.ndarray) -> Tuple[int, int, int]:
    width_bucket = int(float(obs[8]) / 0.15)
    heading_bucket = int(abs(float(obs[10])) / 0.35)
    lateral_bucket = int(abs(float(obs[11])) / 0.15)
    return width_bucket, heading_bucket, lateral_bucket


class ReplayMemoryObservationWrapper(gym.Wrapper):
    """Append an 8-D generic replay-memory embedding to the 19-D geometry obs."""

    memory_dim = 8

    def __init__(self, env):
        super().__init__(env)
        self.memory: Dict[Tuple[int, int, int], ReplayStats] = defaultdict(ReplayStats)
        low = np.concatenate(
            [
                np.asarray(env.observation_space.low, dtype=np.float32),
                np.full(self.memory_dim, -np.inf, dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.asarray(env.observation_space.high, dtype=np.float32),
                np.full(self.memory_dim, np.inf, dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._entry_key = None
        self._min_clearance = 0.0
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._entry_key = replay_key(obs)
        self._min_clearance = float(obs[9])
        self._last_obs = obs
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._min_clearance = min(self._min_clearance, float(obs[9]))
        self._last_obs = obs
        done = terminated or truncated
        if done and self._entry_key is not None:
            stats = self.memory[self._entry_key]
            success = float(info.get("success", 0.0)) > 0.5
            collision = float(info.get("collision", 0.0)) > 0.5
            stuck = float(info.get("stuck", 0.0)) > 0.5
            stats.attempts += 1
            stats.successes += int(success)
            stats.failures += int(not success)
            stats.last_outcome = 1.0 if success else -1.0
            stats.last_min_clearance = self._min_clearance
            stats.last_stuck = float(stuck)
            stats.last_collision = float(collision)
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        stats = self.memory[replay_key(obs)]
        attempts = float(stats.attempts)
        emb = np.array(
            [
                min(attempts / 20.0, 1.0),
                min(stats.failures / 20.0, 1.0),
                min(stats.successes / 20.0, 1.0),
                stats.failures / max(attempts, 1.0),
                stats.last_outcome,
                np.clip(stats.last_min_clearance, -1.0, 1.0),
                stats.last_stuck,
                stats.last_collision,
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs.astype(np.float32), emb], axis=0)
