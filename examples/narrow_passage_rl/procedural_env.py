#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        class _FallbackEnv:
            pass

        class _FallbackBox:
            def __init__(self, low, high, shape=None, dtype=np.float32):
                self.low = (
                    np.full(shape, low, dtype=dtype)
                    if shape
                    else np.asarray(low, dtype=dtype)
                )
                self.high = (
                    np.full(shape, high, dtype=dtype)
                    if shape
                    else np.asarray(high, dtype=dtype)
                )
                self.shape = shape or self.low.shape
                self.dtype = dtype

            def sample(self):
                low = np.where(np.isfinite(self.low), self.low, -1.0)
                high = np.where(np.isfinite(self.high), self.high, 1.0)
                return np.random.uniform(low, high).astype(self.dtype)

        class _FallbackSpaces:
            Box = _FallbackBox

        class _FallbackGym:
            Env = _FallbackEnv

        gym = _FallbackGym()
        spaces = _FallbackSpaces()


@dataclass
class PassageParams:
    width: float
    length: float
    start_x: float
    start_y: float
    start_yaw: float
    obstacle_x: float
    obstacle_y: float
    obstacle_radius: float
    false_feasible: bool


class ProceduralNarrowPassageEnv(gym.Env):
    """Fast 2D local passage traversal/recovery task.

    State is robot pose in a corridor-aligned frame. The robot starts before the
    passage, must align with the centerline, pass through, and reach the exit.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Dict = None):
        super().__init__()
        config = config or {}
        self.max_steps = int(config.get("max_steps", 300))
        self.dt = float(config.get("dt", 0.25))
        self.robot_radius = float(config.get("robot_radius", 0.18))
        self.max_vx = float(config.get("max_vx", 0.35))
        self.min_vx = float(config.get("min_vx", -0.15))
        self.max_wz = float(config.get("max_wz", 0.8))
        self.width_range = tuple(config.get("width_range", (0.45, 1.2)))
        self.yaw_range = tuple(config.get("yaw_range", (-0.75, 0.75)))
        self.start_x_range = tuple(config.get("start_x_range", (-0.35, 0.35)))
        self.false_feasible_prob = float(config.get("false_feasible_prob", 0.15))
        self.rng = np.random.default_rng(config.get("seed", None))
        self.reward_weights = {
            "progress": float(config.get("reward_progress_weight", 4.0)),
            "heading_alignment": float(
                config.get("reward_heading_alignment_weight", 0.6)
            ),
            "lateral_centering": float(
                config.get("reward_lateral_centering_weight", 0.3)
            ),
            "clearance": float(config.get("reward_clearance_weight", 0.4)),
            "body_margin": float(config.get("reward_body_margin_weight", 0.8)),
            "collision": float(config.get("reward_collision_penalty", 25.0)),
            "stuck": float(config.get("reward_stuck_penalty", 6.0)),
            "oscillation": float(config.get("reward_oscillation_penalty", 0.35)),
            "action_smoothness": float(
                config.get("reward_action_smoothness_penalty", 0.15)
            ),
            "unsafe_margin": float(config.get("reward_unsafe_margin_penalty", 8.0)),
            "success": float(config.get("reward_success", 20.0)),
            "timeout": float(config.get("reward_timeout_penalty", 1.0)),
            "slack": float(config.get("reward_slack_penalty", 0.01)),
        }
        self.safe_body_margin = float(config.get("safe_body_margin", 0.12))
        self.safe_width = float(
            config.get("safe_passage_width", 2.0 * self.robot_radius + 0.12)
        )
        self._last_reward_terms = {}

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([self.min_vx, -self.max_wz], dtype=np.float32),
            high=np.array([self.max_vx, self.max_wz], dtype=np.float32),
            dtype=np.float32,
        )

        self.params = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_dist = 0.0
        self.stuck_steps = 0
        self.step_count = 0
        self.collision = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        width = float(self.rng.uniform(*self.width_range))
        length = float(self.rng.uniform(2.0, 4.0))
        false_feasible = bool(self.rng.random() < self.false_feasible_prob)
        obstacle_radius = float(self.rng.uniform(0.08, 0.18))
        obstacle_x = float(self.rng.uniform(-0.25, 0.25))
        obstacle_y = float(self.rng.uniform(0.3, length - 0.3))
        if not false_feasible:
            obstacle_radius = 0.0

        self.params = PassageParams(
            width=width,
            length=length,
            start_x=float(self.rng.uniform(*self.start_x_range)),
            start_y=float(self.rng.uniform(-1.2, -0.6)),
            start_yaw=float(self.rng.uniform(*self.yaw_range)),
            obstacle_x=obstacle_x,
            obstacle_y=obstacle_y,
            obstacle_radius=obstacle_radius,
            false_feasible=false_feasible,
        )
        self.pose[:] = [
            self.params.start_x,
            self.params.start_y,
            self.params.start_yaw,
        ]
        self.prev_action[:] = 0.0
        self.prev_dist = self._distance_to_goal()
        self.stuck_steps = 0
        self.step_count = 0
        self.collision = False
        self._last_reward_terms = {}
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_pose = self.pose.copy()
        prev_dist = self._distance_to_goal()

        vx, wz = float(action[0]), float(action[1])
        self.pose[2] = self._wrap_angle(self.pose[2] + wz * self.dt)
        self.pose[0] += vx * math.sin(self.pose[2]) * self.dt
        self.pose[1] += vx * math.cos(self.pose[2]) * self.dt

        self.step_count += 1
        self.collision = self._is_collision()
        moved = float(np.linalg.norm(self.pose[:2] - prev_pose[:2]))
        progress = prev_dist - self._distance_to_goal()
        if moved < 1e-3 and progress < 1e-3:
            self.stuck_steps += 1
        else:
            self.stuck_steps = max(0, self.stuck_steps - 1)

        obs = self._obs(action)
        success = self._is_success()
        timeout = self.step_count >= self.max_steps
        done = success or self.collision or timeout
        reward = self._reward(prev_dist, obs, action, success, timeout, moved=moved)
        self.prev_action = action.astype(np.float32)
        self.prev_dist = self._distance_to_goal()

        info = {
            "success": float(success),
            "collision": float(self.collision),
            "stuck": float(self.stuck_steps >= 20),
            "passage_width": self.params.width,
            "false_feasible": float(self.params.false_feasible),
            "min_clearance": min(obs[6], obs[7]),
        }
        info.update(self._last_reward_terms)
        return obs, reward, done, False, info

    def _obs(self, action=None):
        action = self.prev_action if action is None else action
        p = self.params
        half_free = max(0.0, p.width * 0.5 - self.robot_radius)
        clearance_left = max(0.0, half_free - self.pose[0])
        clearance_right = max(0.0, half_free + self.pose[0])
        dist_to_front = max(0.0, p.length - self.pose[1])
        near_center = min(5.0, dist_to_front)
        far_center = min(5.0, dist_to_front + 1.0)
        stuck_score = min(1.0, self.stuck_steps / 20.0)
        body_margin = min(clearance_left, clearance_right)
        passage_width = clearance_left + clearance_right + 2.0 * self.robot_radius

        return np.array(
            [
                clearance_left,
                near_center,
                clearance_right,
                clearance_left,
                far_center,
                clearance_right,
                clearance_left,
                clearance_right,
                passage_width,
                body_margin,
                self._wrap_angle(self.pose[2]),
                self.pose[0],
                self._distance_to_goal(),
                action[0],
                action[1],
                stuck_score,
                float(self.collision),
                self.prev_action[0],
                self.prev_action[1],
            ],
            dtype=np.float32,
        )

    def _reward(self, prev_dist, obs, action, success, timeout, moved=None):
        progress = prev_dist - self._distance_to_goal()
        terms = self._dense_reward_terms(
            progress=progress,
            obs=obs,
            action=action,
            success=success,
            timeout=timeout,
            collision=self.collision,
            moved=moved,
        )
        self._last_reward_terms = terms
        return float(sum(terms.values()))

    def _dense_reward_terms(
        self,
        progress,
        obs,
        action,
        success,
        timeout,
        collision,
        moved=None,
    ):
        left = float(obs[6])
        right = float(obs[7])
        passage_width = float(obs[8])
        body_margin = float(obs[9])
        heading_error = abs(float(obs[10]))
        stuck_score = float(obs[15])
        action_delta = float(np.abs(action - self.prev_action).sum())
        oscillating = (
            abs(float(action[1])) > 0.05
            and abs(float(self.prev_action[1])) > 0.05
            and np.sign(float(action[1])) != np.sign(float(self.prev_action[1]))
        )
        ineffective_motion = 0.0
        if moved is not None and abs(float(action[0])) > 0.05 and float(moved) < 1e-3:
            ineffective_motion = 1.0

        clearance_balance = min(left, right) - 0.5 * abs(left - right)
        safe_margin_deficit = max(0.0, self.safe_body_margin - body_margin)
        safe_width_deficit = max(0.0, self.safe_width - passage_width)

        w = self.reward_weights
        return {
            "reward/progress": w["progress"] * float(progress),
            "reward/heading_alignment": w["heading_alignment"]
            * math.cos(heading_error),
            "reward/lateral_centering": -w["lateral_centering"]
            * abs(float(obs[11])),
            "reward/clearance": w["clearance"] * clearance_balance,
            "reward/body_margin": w["body_margin"]
            * min(body_margin, self.safe_body_margin),
            "reward/collision": -w["collision"] * float(collision),
            "reward/stuck": -w["stuck"] * max(stuck_score, ineffective_motion),
            "reward/oscillation": -w["oscillation"] * float(oscillating),
            "reward/action_smoothness": -w["action_smoothness"] * action_delta,
            "reward/unsafe_margin": -w["unsafe_margin"]
            * (safe_margin_deficit + safe_width_deficit),
            "reward/success": w["success"] * float(success),
            "reward/timeout": -w["timeout"] * float(timeout and not success),
            "reward/slack": -w["slack"],
        }

    def _distance_to_goal(self):
        goal = np.array([0.0, self.params.length + 0.5], dtype=np.float32)
        return float(np.linalg.norm(goal - self.pose[:2]))

    def _is_success(self):
        return (
            self.pose[1] >= self.params.length + 0.25
            and abs(self.pose[0]) < 0.25
            and abs(self._wrap_angle(self.pose[2])) < 0.35
        )

    def _is_collision(self):
        p = self.params
        in_passage_y = 0.0 <= self.pose[1] <= p.length
        if in_passage_y and abs(self.pose[0]) + self.robot_radius > p.width * 0.5:
            return True
        if p.obstacle_radius > 0.0:
            d = np.linalg.norm(
                self.pose[:2] - np.array([p.obstacle_x, p.obstacle_y])
            )
            if d < self.robot_radius + p.obstacle_radius:
                return True
        return False

    @staticmethod
    def _wrap_angle(angle):
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)
