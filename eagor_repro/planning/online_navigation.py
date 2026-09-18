"""Online map/frontier/A* navigation, with odometry-only execution feedback."""
from __future__ import annotations
import time
import numpy as np
from eagor_repro.planning.online_mapper import MapConfig, OnlineOccupancyMap
from eagor_repro.planning.frontier_planner import FrontierPlanner, DirectionCue
from eagor_repro.planning.grid_path_planner import astar, segment_safe


class OnlineNavigator:
    def __init__(self, config, position, forward_step=.25, turn_step_deg=15.):
        self.mapper = OnlineOccupancyMap(MapConfig(**dict(config.get('map', {}))), position)
        self.frontiers = FrontierPlanner(dict(config.get('frontier', {})))
        self.config = config
        self.forward_step = forward_step
        self.turn_step = np.deg2rad(turn_step_deg)
        self.goal = None
        self.path = []
        self.goal_since = 0
        self.last_progress = 0
        self.progress_position = np.asarray(position)[[0,2]].copy()
        self.last_plan = -100
        self.previous_action = None
        self.previous_position = None
        self.previous_rotation = None
        self.observe_steps = 0
        self.replans = self.switches = self.no_progress = self.recoveries = 0
        self.events = []
        self.timings = {}
        self.status = 'initial'
        self.invalid_paths = 0
        self.action_rejections = 0
        self.replan_reason = 'initial'
        self.pending_forward = False

    def invalidate(self, reason, step, cool=False):
        if cool and self.goal is not None:
            self.frontiers.cool(self.goal, step)
        self.goal = None
        self.path = []
        self.pending_forward = False
        self.replan_reason = reason

    def feedback(self, position, collision, step):
        if self.previous_position is None:
            return
        displacement = float(np.linalg.norm(np.asarray(position)-self.previous_position))
        if self.previous_action == 'move_forward' and (collision or displacement < .04):
            self.mapper.mark_contact(position, self.previous_rotation[:,0], step)
            self.invalidate('collision_contact', step, cool=True)
            self.recoveries += 1
        if np.linalg.norm(np.asarray(position)[[0,2]]-self.progress_position) > .2:
            self.progress_position = np.asarray(position)[[0,2]].copy()
            self.last_progress = step
        if step-self.last_progress >= int(self.config.get('stall_steps',24)):
            self.no_progress += 1
            self.invalidate('no_progress_timeout', step, cool=True)
            self.last_progress = step

    def _action(self, position, rotation, safe):
        m = self.mapper
        p = np.asarray(position)[[0,2]]
        if not self.path:
            return None
        end = p + np.asarray(rotation)[:,0][[0,2]]*self.forward_step
        if self.pending_forward:
            self.pending_forward = False
            if segment_safe(safe,m.cells(p),m.cells(end)):
                return 'move_forward'
        # Farthest line-of-sight point within a short lookahead; every shortcut
        # is rechecked against inflated obstacles and unknown cells.
        indices = np.argsort([np.linalg.norm(q-p) for q in self.path])
        nearest = int(indices[0])
        self.path = self.path[nearest:]
        waypoint = self.path[0]
        for q in self.path:
            if np.linalg.norm(q-p) > .65:
                break
            if segment_safe(safe, m.cells(p), m.cells(q)):
                waypoint = q
        delta = np.array([waypoint[0]-p[0],0.,waypoint[1]-p[1]])
        body = np.asarray(rotation).T @ delta
        angle = float(np.arctan2(body[1],body[0]))
        self.selected_heading = angle
        if abs(angle) > self.turn_step*.51:
            return 'turn_left' if angle > 0 else 'turn_right'
        if segment_safe(safe, m.cells(p),m.cells(end)):
            return 'move_forward'
        # Quantized yaw can put the nominal forward step outside a safe corridor.
        # Turn one bin only when the resulting swept segment is known-safe.
        forward = np.asarray(rotation)[:,0][[0,2]]
        left = np.asarray(rotation)[:,1][[0,2]]
        for sign in (1,-1):
            candidate = p + self.forward_step*(np.cos(self.turn_step)*forward + sign*np.sin(self.turn_step)*left)
            if segment_safe(safe,m.cells(p),m.cells(candidate)):
                self.pending_forward = True
                return 'turn_left' if sign == 1 else 'turn_right'
        return None

    def act(self, depth, position, rotation, cue: DirectionCue, step, collision=False):
        start_time = time.perf_counter()
        self.feedback(position, collision, step)
        self.mapper.update(depth, position, rotation, step)
        self.timings['mapping_ms'] = (time.perf_counter()-start_time)*1000
        p = np.asarray(position)[[0,2]]
        _,_,safe,_ = self.mapper.layers(step)
        if self.goal is not None and np.linalg.norm(p-self.goal) < .24:
            self.invalidate('subgoal_reached',step,cool=True)
            self.observe_steps = 0
        if self.goal is not None:
            if self.path:
                nearest=int(np.argmin([np.linalg.norm(q-p) for q in self.path]))
                self.path=self.path[nearest:]
            cells = self.mapper.cells(np.asarray(self.path))
            if not len(cells) or not np.all(safe[cells[:,0],cells[:,1]]):
                self.invalid_paths += 1
                self.invalidate('map_path_invalid',step)
        if self.goal is not None and step-self.goal_since > int(self.config.get('subgoal_budget',60)):
            self.invalidate('subgoal_timeout',step,cool=True)
        plan_start = time.perf_counter()
        if self.goal is None:
            self.goal, self.path, self.status = self.frontiers.choose(self.mapper,p,cue,step)
            self.events.append(dict(step=step, reason=self.replan_reason, selection=self.status,
                selected=None if self.goal is None else self.goal.tolist(), candidates=self.frontiers.last_scores))
            self.replans += 1
            self.last_plan = step
            if self.goal is not None:
                # Selected via Dijkstra's reachable tree, independently run A*.
                cells = astar(safe,self.mapper.cells(p),self.mapper.cells(self.goal))
                self.path = [self.mapper.world(c) for c in cells]
                self.switches += 1
                self.goal_since = step
        self.timings['planning_ms'] = (time.perf_counter()-plan_start)*1000
        follow_start = time.perf_counter()
        self.selected_heading = 0.
        action = self._action(position,rotation,safe)
        if action is None:
            if self.path:
                self.action_rejections += 1
                self.invalidate('unsafe_discrete_step',step,cool=True)
            self.observe_steps += 1
            self.status = 'bounded_observation'
            action = 'turn_left'
            if self.observe_steps > int(self.config.get('observation_budget',24)):
                self.status = 'exploration_exhausted'
                action = None  # Evaluator ends as failure, never task-success STOP.
        else:
            self.observe_steps = 0
        self.previous_action = action
        self.previous_position = np.array(position).copy()
        self.previous_rotation = np.array(rotation).copy()
        self.timings['following_ms'] = (time.perf_counter()-follow_start)*1000
        return action

    def telemetry(self, step):
        free, occupied, safe, _ = self.mapper.layers(step)
        return dict(planner_status=self.status, replans=self.replans, subgoal_switches=self.switches,
            no_progress_count=self.no_progress,recoveries=self.recoveries,
            free_cells=int(free.sum()),occupied_cells=int(occupied.sum()),visited_cells=int((self.mapper.visits>0).sum()),
            frontier_cells=self.frontiers.frontier_count, invalid_paths=self.invalid_paths,
            safety_escape_active=self.mapper.safety_escape_active,
            action_rejections=self.action_rejections, **self.mapper.frame_stats, **self.timings)
