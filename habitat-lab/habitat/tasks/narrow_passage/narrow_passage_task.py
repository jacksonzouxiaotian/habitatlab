import math
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from habitat.tasks.narrow_passage.geometry import (
    NarrowPassageMemoryState,
    NarrowPassageState,
)
from habitat.tasks.nav.nav import NavigationTask
from habitat.core.registry import registry


@registry.register_task(name="NarrowPassageNav-v0")
class NarrowPassageNavTask(NavigationTask):
    """
    自定义狭窄通道决策任务
    """

    def __init__(self, config, sim, dataset):
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.entered_narrow = False
        self._np_state = NarrowPassageState()
        self._prev_position: Optional[np.ndarray] = None
        self._prev_distance: Optional[float] = None
        self._stuck_steps = 0
        self._last_action_vx = 0.0
        self._last_action_wz = 0.0
        self._failure_memory = defaultdict(int)
        self._memory_trigger_count = int(
            getattr(config, "memory_trigger_count", 1)
        )
        self._memory_reject_count = int(getattr(config, "memory_reject_count", 3))

    def reset(self, episode):
        self.entered_narrow = False
        agent_state = self._sim.get_agent_state()
        self._prev_position = np.array(agent_state.position, dtype=np.float32)
        self._prev_distance = self._distance_to_goal(episode)
        self._stuck_steps = 0
        self._last_action_vx = 0.0
        self._last_action_wz = 0.0
        self._failure_memory.clear()
        self._np_state = NarrowPassageState(
            distance_to_local_goal=float(self._prev_distance or 0.0),
            robot_radius=float(getattr(self._sim.agents[0], "radius", 0.18))
            if hasattr(self._sim, "agents")
            else 0.18,
        )
        return super().reset(episode)

    def step(self, action: Any, episode):
        self._cache_action(action)
        observations = super().step(action, episode)
        self._update_narrow_state(episode)
        return observations

    def get_narrow_passage_state(self) -> NarrowPassageState:
        return self._np_state

    def get_narrow_passage_memory_state(self) -> NarrowPassageMemoryState:
        key = self._memory_key(self._np_state)
        count = self._failure_memory.get(key, 0)
        return NarrowPassageMemoryState(
            failure_count=float(count),
            failed_state_count=float(len(self._failure_memory)),
            should_recover=float(count >= self._memory_trigger_count),
            should_reject=float(count >= self._memory_reject_count),
        )

    def _cache_action(self, action: Any) -> None:
        if isinstance(action, dict):
            action_args = action.get("action_args", action)
            linear = action_args.get("linear_velocity", 0.0)
            angular = action_args.get("angular_velocity", 0.0)
            self._last_action_vx = float(np.asarray(linear).reshape(-1)[0])
            self._last_action_wz = float(np.asarray(angular).reshape(-1)[0])

    def _goal_position(self, episode) -> Optional[np.ndarray]:
        if episode is None or not getattr(episode, "goals", None):
            return None
        return np.array(episode.goals[0].position, dtype=np.float32)

    def _distance_to_goal(self, episode) -> Optional[float]:
        goal = self._goal_position(episode)
        if goal is None:
            return None
        agent_pos = np.array(self._sim.get_agent_state().position, dtype=np.float32)
        delta = goal - agent_pos
        return float(np.linalg.norm(delta[[0, 2]]))

    def _heading_error(self, episode) -> float:
        goal = self._goal_position(episode)
        if goal is None:
            return 0.0
        agent_state = self._sim.get_agent_state()
        agent_pos = np.array(agent_state.position, dtype=np.float32)
        delta = goal - agent_pos
        goal_yaw = math.atan2(delta[0], -delta[2])
        rot = agent_state.rotation
        yaw = math.atan2(
            2.0 * (rot.real * rot.y + rot.x * rot.z),
            1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z),
        )
        err = goal_yaw - yaw
        return float((err + math.pi) % (2.0 * math.pi) - math.pi)

    def _lateral_offset(self, episode) -> float:
        goal = self._goal_position(episode)
        if goal is None or self._prev_position is None:
            return 0.0
        start = np.array(getattr(episode, "start_position", self._prev_position))
        pos = np.array(self._sim.get_agent_state().position, dtype=np.float32)
        line = goal[[0, 2]] - start[[0, 2]]
        norm = np.linalg.norm(line)
        if norm < 1e-6:
            return 0.0
        line = line / norm
        rel = pos[[0, 2]] - start[[0, 2]]
        return float(rel[0] * line[1] - rel[1] * line[0])

    def _memory_key(self, state: NarrowPassageState):
        return (
            int(abs(state.lateral_offset) / 0.10),
            int(abs(state.heading_error) / 0.25),
            int(state.stuck_score / 0.25),
            int(state.collision_flag > 0.5),
        )

    def _remember_failure(self) -> None:
        self._failure_memory[self._memory_key(self._np_state)] += 1

    def _update_narrow_state(self, episode):
        """
        根据局部运动、碰撞和目标相对状态更新 failure-aware state。
        """
        agent_state = self._sim.get_agent_state()
        pos = np.array(agent_state.position, dtype=np.float32)
        moved = 0.0
        if self._prev_position is not None:
            moved = float(np.linalg.norm(pos[[0, 2]] - self._prev_position[[0, 2]]))

        distance = self._distance_to_goal(episode)
        progress = 0.0
        if distance is not None and self._prev_distance is not None:
            progress = self._prev_distance - distance

        if moved < 1e-3 and progress < 1e-3:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        collided = 0.0
        prev_obs = getattr(self._sim, "_prev_sim_obs", {})
        if isinstance(prev_obs, dict):
            collided = float(bool(prev_obs.get("collided", False)))

        self._np_state = NarrowPassageState(
            distance_to_local_goal=float(distance or 0.0),
            heading_error=self._heading_error(episode),
            lateral_offset=self._lateral_offset(episode),
            current_vx=moved,
            current_wz=0.0,
            stuck_score=min(1.0, self._stuck_steps / 20.0),
            collision_flag=collided,
            previous_action_vx=self._last_action_vx,
            previous_action_wz=self._last_action_wz,
            robot_radius=self._np_state.robot_radius,
        )
        if collided > 0.5 or self._np_state.stuck_score >= 0.7:
            self._remember_failure()
        self._prev_position = pos
        self._prev_distance = distance
