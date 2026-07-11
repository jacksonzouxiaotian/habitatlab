#!/usr/bin/env python3

import math
from typing import Dict

import numpy as np
from procedural_env import ProceduralNarrowPassageEnv


class ProceduralRiskRecoveryEnv(ProceduralNarrowPassageEnv):
    """Recovery skill from pre-collision high-risk states.

    Unlike ProceduralRecoveryEnv, the robot starts in a risky but non-terminal
    passage state. The objective is to avoid collision, regain clearance, and
    return to a centered/aligned state that can hand control back to passage PPO.
    """

    def __init__(self, config: Dict = None):
        config = dict(config or {})
        config.setdefault("max_steps", 100)
        super().__init__(config)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        p = self.params
        half_free = p.width * 0.5 - self.robot_radius
        side = -1.0 if self.rng.random() < 0.5 else 1.0
        clearance = float(self.rng.uniform(0.03, 0.11))
        risky_x = side * max(0.0, half_free - clearance)
        yaw_bias = float(self.rng.uniform(0.45, 1.0))
        if self.rng.random() < 0.5:
            yaw_bias *= -1.0

        self.pose[:] = [
            risky_x,
            float(self.rng.uniform(0.0, max(0.1, p.length - 0.5))),
            yaw_bias,
        ]
        self.prev_action[:] = 0.0
        self.prev_dist = self._distance_to_goal()
        self.stuck_steps = int(self.rng.integers(0, 8))
        self.step_count = 0
        self.collision = False
        return self._obs(), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_cost = self._risk_cost()
        prev_pose = self.pose.copy()

        vx, wz = float(action[0]), float(action[1])
        self.pose[2] = self._wrap_angle(self.pose[2] + wz * self.dt)
        self.pose[0] += vx * math.sin(self.pose[2]) * self.dt
        self.pose[1] += vx * math.cos(self.pose[2]) * self.dt

        self.step_count += 1
        self.collision = self._is_collision()
        moved = float(np.linalg.norm(self.pose[:2] - prev_pose[:2]))
        if moved < 1e-3:
            self.stuck_steps += 1
        else:
            self.stuck_steps = max(0, self.stuck_steps - 2)

        obs = self._obs(action)
        recovered = self._is_recovered(obs)
        timeout = self.step_count >= self.max_steps
        done = recovered or self.collision or timeout
        reward = self._risk_recovery_reward(
            prev_cost, obs, action, recovered, timeout, moved=moved
        )
        self.prev_action = action.astype(np.float32)
        self.prev_dist = self._distance_to_goal()

        info = {
            "success": float(recovered),
            "risk_recovery_success": float(recovered),
            "collision": float(self.collision),
            "stuck": float(self.stuck_steps >= 20),
            "passage_width": self.params.width,
            "false_feasible": float(self.params.false_feasible),
            "min_clearance": min(obs[6], obs[7]),
        }
        info.update(self._last_reward_terms)
        return obs, reward, done, False, info

    def _risk_cost(self):
        obs = self._obs()
        return (
            2.0 * abs(float(obs[11]))
            + 1.2 * abs(float(obs[10]))
            + 1.5 * max(0.0, 0.14 - min(float(obs[6]), float(obs[7])))
            + 0.4 * float(obs[15])
        )

    def _is_recovered(self, obs):
        return (
            abs(float(obs[11])) < 0.12
            and abs(float(obs[10])) < 0.25
            and min(float(obs[6]), float(obs[7])) > 0.14
            and float(obs[15]) < 0.25
        )

    def _risk_recovery_reward(self, prev_cost, obs, action, recovered, timeout, moved=None):
        progress = prev_cost - self._risk_cost()
        terms = self._dense_reward_terms(
            progress=progress,
            obs=obs,
            action=action,
            success=recovered,
            timeout=timeout,
            collision=self.collision,
            moved=moved,
        )
        self._last_reward_terms = terms
        return float(sum(terms.values()))
