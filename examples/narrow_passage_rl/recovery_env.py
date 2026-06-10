#!/usr/bin/env python3

import math
from typing import Dict

import numpy as np
from procedural_env import ProceduralNarrowPassageEnv


class ProceduralRecoveryEnv(ProceduralNarrowPassageEnv):
    """Recovery skill task initialized near collision/stuck states."""

    def __init__(self, config: Dict = None):
        config = dict(config or {})
        config.setdefault("max_steps", 120)
        super().__init__(config)
        self.recovered_once = False

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        p = self.params
        side = -1.0 if self.rng.random() < 0.5 else 1.0
        wall_x = side * (p.width * 0.5 - self.robot_radius - 0.01)
        self.pose[:] = [
            wall_x,
            float(self.rng.uniform(0.15, max(0.2, p.length - 0.4))),
            float(self.rng.uniform(-1.0, 1.0)),
        ]
        self.prev_action[:] = 0.0
        self.prev_dist = self._distance_to_goal()
        self.stuck_steps = int(self.rng.integers(8, 18))
        self.collision = False
        self.recovered_once = False
        return self._obs(), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_recovery_cost = self._recovery_cost()
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
        reward = self._recovery_reward(prev_recovery_cost, obs, action, recovered, timeout)
        self.prev_action = action.astype(np.float32)
        self.prev_dist = self._distance_to_goal()

        info = {
            "success": float(recovered),
            "recovery_success": float(recovered),
            "collision": float(self.collision),
            "stuck": float(self.stuck_steps >= 20),
            "passage_width": self.params.width,
            "false_feasible": float(self.params.false_feasible),
            "min_clearance": min(obs[6], obs[7]),
        }
        return obs, reward, done, False, info

    def _recovery_cost(self):
        return (
            2.0 * abs(self.pose[0])
            + 1.0 * abs(self._wrap_angle(self.pose[2]))
            + 0.4 * min(1.0, self.stuck_steps / 20.0)
        )

    def _is_recovered(self, obs):
        min_clearance = min(obs[6], obs[7])
        return (
            abs(obs[11]) < 0.12
            and abs(obs[10]) < 0.25
            and min_clearance > 0.12
            and obs[15] < 0.25
        )

    def _recovery_reward(self, prev_cost, obs, action, recovered, timeout):
        reward = 2.0 * (prev_cost - self._recovery_cost())
        reward += 0.5 * min(obs[6], obs[7])
        reward -= 0.2 * abs(action[1])
        reward -= 20.0 * float(self.collision)
        reward -= 3.0 * obs[15]
        reward -= 0.01
        reward += 8.0 * float(recovered)
        reward -= 1.0 * float(timeout and not recovered)
        return float(reward)
