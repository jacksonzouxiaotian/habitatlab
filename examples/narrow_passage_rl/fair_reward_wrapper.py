#!/usr/bin/env python3
"""Fair RL reward wrapper for narrow-passage learning baselines.

The native v2 reward intentionally rewards clearance, but the raw body-margin
signal is very large in open space (~4.82 m).  A recurrent policy can exploit
that by staying outside the passage until timeout.  This wrapper replaces the
reward with a success/progress/safety objective that clips clearance bonuses to
the narrow-passage scale and penalizes timeout, no-progress, and outside-idle
behavior.
"""

from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - legacy fallback
    import gym


@dataclass
class FairRewardConfig:
    progress_weight: float = 3.0
    success_reward: float = 50.0
    collision_penalty: float = 25.0
    timeout_penalty: float = 20.0
    step_penalty: float = 0.03
    action_smooth_penalty: float = 0.03
    heading_penalty: float = 0.15
    lateral_penalty: float = 0.20
    stuck_penalty: float = 5.0
    near_collision_penalty: float = 1.5
    safe_clearance_bonus: float = 0.15
    clearance_clip: float = 0.30
    outside_margin_threshold: float = 1.0
    outside_idle_penalty: float = 0.20
    min_progress_for_idle: float = 0.002


class FairNarrowPassageRewardWrapper(gym.Wrapper):
    """Replace env reward with a fair dense reward for learning baselines."""

    def __init__(self, env, cfg: FairRewardConfig = None):
        super().__init__(env)
        self.cfg = cfg or FairRewardConfig()
        self._prev_dist = None
        self._prev_action = np.zeros(2, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_dist = float(obs[12])
        self._prev_action = np.zeros(2, dtype=np.float32)
        return obs, info

    @property
    def max_steps(self):
        return self.env.max_steps

    def step(self, action):
        obs, _native_reward, terminated, truncated, info = self.env.step(action)
        reward = self._fair_reward(obs, action, terminated or truncated, info)
        self._prev_dist = float(obs[12])
        self._prev_action = np.asarray(action, dtype=np.float32).copy()
        return obs, reward, terminated, truncated, info

    def _fair_reward(self, obs, action, done, info):
        cfg = self.cfg
        dist = float(obs[12])
        prev_dist = dist if self._prev_dist is None else self._prev_dist
        progress = prev_dist - dist
        success = float(info.get("success", 0.0)) > 0.5
        collision = float(info.get("collision", 0.0)) > 0.5
        timeout = bool(done and not success and not collision)
        body_margin = float(obs[9])
        heading = abs(float(obs[10]))
        lateral = abs(float(obs[11]))
        stuck = float(obs[15])

        reward = 0.0
        reward += cfg.progress_weight * progress
        reward += cfg.safe_clearance_bonus * float(
            np.clip(body_margin, -cfg.clearance_clip, cfg.clearance_clip)
        )
        reward -= cfg.heading_penalty * heading
        reward -= cfg.lateral_penalty * lateral
        reward -= cfg.stuck_penalty * stuck
        reward -= cfg.step_penalty

        if body_margin < 0.05:
            reward -= cfg.near_collision_penalty * (0.05 - body_margin) / 0.05

        if body_margin > cfg.outside_margin_threshold and progress < cfg.min_progress_for_idle:
            reward -= cfg.outside_idle_penalty

        action_arr = np.asarray(action, dtype=np.float32)
        reward -= cfg.action_smooth_penalty * float(np.abs(action_arr - self._prev_action).sum())
        reward += cfg.success_reward * float(success)
        reward -= cfg.collision_penalty * float(collision)
        reward -= cfg.timeout_penalty * float(timeout)
        return float(reward)
