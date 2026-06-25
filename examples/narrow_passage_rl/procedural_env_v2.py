#!/usr/bin/env python3
"""Harder narrow-passage benchmark environment (v2).

Supports multi-segment corridor geometries that expose failure modes of methods
that assume straight-line passage alignment.  The observation vector is
identical to ProceduralNarrowPassageEnv (19-dim) so all existing FSM / RL code
runs unchanged — with the key difference that heading_error and lateral_offset
are measured against the STRAIGHT start→goal line (same as NarrowPassageNav-v0
measures), not the local path tangent.  This exposes the FSM's structural
limitation on curved corridors.

Corridor types
--------------
  STRAIGHT       — baseline, same geometry as v1
  L_SHAPED       — 90-degree turn; heading-to-final-goal passes through inner wall
  S_SHAPED       — two opposite 90-degree turns
  NARROW_EXIT    — wide entry narrows to tight exit; requires late adaptation
  NARROW_ENTRY   — tight entry widens; conservative reject misses feasible passage
  ASYMMETRIC     — one-sided wall protrusion at midpoint; symmetric obs is wrong
  FALSE_FEASIBLE — passage looks navigable but is physically blocked inside

Usage
-----
    env = HarderNarrowPassageEnv({"corridor_types": ["L_SHAPED", "FALSE_FEASIBLE"]})
    obs, _ = env.reset()
    obs, reward, done, trunc, info = env.step(action)
    # info["corridor_type"], info["passage_width"], info["passable"]
"""

import enum
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        class _Env:
            pass

        class _Box:
            def __init__(self, low, high, shape=None, dtype=np.float32):
                self.shape = shape or np.asarray(low).shape
                self.low = np.full(self.shape, low if np.isscalar(low) else 0, dtype=dtype)
                self.high = np.full(self.shape, high if np.isscalar(high) else 1, dtype=dtype)
                self.dtype = dtype

            def sample(self):
                lo = np.where(np.isfinite(self.low), self.low, -1.0)
                hi = np.where(np.isfinite(self.high), self.high, 1.0)
                return np.random.uniform(lo, hi).astype(self.dtype)

        class _Spaces:
            Box = _Box

        class _Gym:
            Env = _Env

        gym, spaces = _Gym(), _Spaces()


# ── Corridor types ────────────────────────────────────────────────────────────

class CorridorType(enum.Enum):
    STRAIGHT = "straight"
    L_SHAPED = "l_shaped"
    S_SHAPED = "s_shaped"
    NARROW_EXIT = "narrow_exit"
    NARROW_ENTRY = "narrow_entry"
    ASYMMETRIC = "asymmetric"
    FALSE_FEASIBLE = "false_feasible"


_DEFAULT_PROBS = {
    CorridorType.STRAIGHT: 0.20,
    CorridorType.L_SHAPED: 0.20,
    CorridorType.S_SHAPED: 0.15,
    CorridorType.NARROW_EXIT: 0.15,
    CorridorType.NARROW_ENTRY: 0.10,
    CorridorType.ASYMMETRIC: 0.10,
    CorridorType.FALSE_FEASIBLE: 0.10,
}

# ── Corridor geometry helpers ─────────────────────────────────────────────────


@dataclass
class CorridorParams:
    ctype: CorridorType
    # Centerline path as list of (x, y) world-frame waypoints.
    path: List[Tuple[float, float]]
    # Width profile: list of (arc_s, width) pairs; interpolated.
    width_profile: List[Tuple[float, float]]
    # Blocking obstacle: (arc_s, lateral_n, radius).  None = no blocker.
    blocker: Optional[Tuple[float, float, float]] = None
    # Asymmetric protrusion: (arc_s, side [-1=left / +1=right], protrusion_amount).
    protrusion: Optional[Tuple[float, int, float]] = None
    total_arc: float = 0.0      # filled by build_corridor
    robot_radius: float = 0.18


@dataclass
class _PathFrame:
    s: float
    n: float               # signed lateral: + = left of direction of travel
    tangent: np.ndarray
    segment_idx: int


def _path_arcs(path):
    pts = [np.array(p, float) for p in path]
    arcs = [0.0]
    for i in range(1, len(pts)):
        arcs.append(arcs[-1] + float(np.linalg.norm(pts[i] - pts[i - 1])))
    return arcs, pts


def _project(world_pos, pts, arcs):
    """Project world_pos onto the polyline, returning _PathFrame."""
    best_d = float("inf")
    best = None
    for i in range(len(pts) - 1):
        ab = pts[i + 1] - pts[i]
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 1e-9:
            continue
        t = float(np.clip(np.dot(world_pos - pts[i], ab) / (ab_len ** 2), 0.0, 1.0))
        closest = pts[i] + t * ab
        d = float(np.linalg.norm(world_pos - closest))
        if d < best_d:
            best_d = d
            tang = ab / ab_len
            norm = np.array([-tang[1], tang[0]])   # 90° CCW = left of travel
            n = float(np.dot(world_pos - closest, norm))
            best = _PathFrame(
                s=arcs[i] + t * ab_len,
                n=n,
                tangent=tang.copy(),
                segment_idx=i,
            )
    return best


def _width_at(s, wp):
    if len(wp) == 1:
        return wp[0][1]
    for i in range(len(wp) - 1):
        s0, w0 = wp[i]
        s1, w1 = wp[i + 1]
        if s0 <= s <= s1:
            t = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
            return w0 + (w1 - w0) * t
    return wp[-1][1]


def _wall_clearances(frame, params):
    """Return (cl_body, cr_body): body-surface clearance from left/right wall."""
    W = _width_at(frame.s, params.width_profile)
    cl = W / 2 - frame.n
    cr = W / 2 + frame.n
    if params.protrusion is not None:
        s_p, side, amt = params.protrusion
        span = _width_at(s_p, params.width_profile) * 0.5
        if abs(frame.s - s_p) < span:
            if side < 0:
                cl -= amt
            else:
                cr -= amt
    # Convert wall-distance → body clearance (subtract robot radius)
    return cl - params.robot_radius, cr - params.robot_radius


def _check_blocker(frame, params):
    if params.blocker is None:
        return False
    s_b, n_b, r_b = params.blocker
    return math.hypot(frame.s - s_b, frame.n - n_b) < r_b + params.robot_radius


def _in_corridor(world_pos, pts, arcs, params, lat_slack=0.10):
    """True if world_pos is inside any corridor segment's passable rectangle.

    Uses unclamped t so that points before / after a segment's extent are
    correctly excluded (avoids false positives from path-end clamping).
    """
    for i in range(len(pts) - 1):
        ab = pts[i + 1] - pts[i]
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 1e-9:
            continue
        t = float(np.dot(world_pos - pts[i], ab)) / (ab_len ** 2)
        if t < -0.05 or t > 1.05:   # outside this segment's longitudinal span
            continue
        t_clamped = max(0.0, min(1.0, t))
        closest = pts[i] + t_clamped * ab
        d_lateral = float(np.linalg.norm(world_pos - closest))
        s = arcs[i] + t_clamped * ab_len
        W = _width_at(s, params.width_profile)
        if d_lateral <= W / 2 + lat_slack:
            return True
    return False


def _cast_ray(world_pos, yaw, pts, arcs, params, max_d=5.0, step=0.04):
    """Cast a ray from world_pos at heading yaw; return distance to first wall.

    For each ray point, segments are split into "strict" (unclamped t ∈ [0, 1])
    and "extension" (t ∈ [-0.05, 0) or (1, 1.05]).  We check strict segments
    first; extension segments are only used when no strict segment covers the
    point.  This prevents the inter-segment false-positive: a point well inside
    segment A's range being judged by segment B's (extension) geometry, which can
    report a negative clearance because the point is laterally displaced far from
    segment B's centreline.
    """
    dx, dy = math.sin(yaw), math.cos(yaw)
    ray_dir = np.array([dx, dy])
    for k in range(int(max_d / step) + 2):
        d = k * step
        if d > max_d:
            return max_d
        pt = world_pos + d * ray_dir
        if not _in_corridor(pt, pts, arcs, params):
            continue

        # Collect candidate segments partitioned by strict vs extension coverage.
        strict_segs = []
        ext_segs = []
        for i in range(len(pts) - 1):
            ab = pts[i + 1] - pts[i]
            ab_len = float(np.linalg.norm(ab))
            if ab_len < 1e-9:
                continue
            t = float(np.dot(pt - pts[i], ab)) / (ab_len ** 2)
            if t < -0.05 or t > 1.05:
                continue
            t_clamped = max(0.0, min(1.0, t))
            closest = pts[i] + t_clamped * ab
            tang = ab / ab_len
            norm = np.array([-tang[1], tang[0]])
            n = float(np.dot(pt - closest, norm))
            s = arcs[i] + t_clamped * ab_len
            W_i = _width_at(s, params.width_profile)
            if abs(n) > W_i / 2:
                continue
            entry = (i, t_clamped, n, tang.copy(), s)
            if 0.0 <= t <= 1.0:
                strict_segs.append(entry)
            else:
                ext_segs.append(entry)

        # Prefer strict coverage; fall back to extension only when the point is
        # in the "gap" between segment endpoints (not owned by any strict segment).
        for i, t_clamped, n, tang, s in (strict_segs if strict_segs else ext_segs):
            fr = _PathFrame(s=s, n=n, tangent=tang, segment_idx=i)
            cl, cr = _wall_clearances(fr, params)
            if cl < 0 or cr < 0:
                return d
            if _check_blocker(fr, params):
                return d
    return max_d


# ── Corridor generators ───────────────────────────────────────────────────────

def _build_corridor(ctype: CorridorType, rng: np.random.Generator,
                    width_range=(0.45, 0.90), robot_radius=0.18) -> CorridorParams:
    W = float(rng.uniform(*width_range))
    r = robot_radius

    if ctype == CorridorType.STRAIGHT:
        L = float(rng.uniform(2.5, 4.5))
        path = [(0.0, 0.0), (0.0, L)]
        wprofile = [(0.0, W), (L, W)]
        blocker = None
        prot = None

    elif ctype == CorridorType.L_SHAPED:
        L1 = float(rng.uniform(1.2, 2.5))  # length before turn
        L2 = float(rng.uniform(1.2, 2.5))  # length after turn
        sign = 1.0 if rng.random() > 0.5 else -1.0  # turn right or left
        turn_x = sign * L2
        path = [(0.0, 0.0), (0.0, L1), (turn_x, L1)]
        wprofile = [(0.0, W), (L1 + L2, W)]
        blocker = None
        prot = None

    elif ctype == CorridorType.S_SHAPED:
        L1 = float(rng.uniform(0.8, 1.5))
        L2 = float(rng.uniform(0.8, 1.5))
        L3 = float(rng.uniform(1.0, 2.0))
        sign = 1.0 if rng.random() > 0.5 else -1.0
        # right then left (or left then right)
        path = [
            (0.0, 0.0),
            (0.0, L1),
            (sign * L2, L1),
            (sign * L2, L1 + L3),
        ]
        wprofile = [(0.0, W), (L1 + L2 + L3, W)]
        blocker = None
        prot = None

    elif ctype == CorridorType.NARROW_EXIT:
        W_wide = float(rng.uniform(0.70, 1.0))
        W_narrow = float(rng.uniform(max(2.05 * r, 0.42), 0.60))
        L = float(rng.uniform(2.5, 4.5))
        trans = L * float(rng.uniform(0.4, 0.6))  # transition point
        path = [(0.0, 0.0), (0.0, L)]
        wprofile = [(0.0, W_wide), (trans, W_wide), (trans + 0.4, W_narrow), (L, W_narrow)]
        blocker = None
        prot = None
        W = W_narrow  # tightest width for paper reporting

    elif ctype == CorridorType.NARROW_ENTRY:
        W_narrow = float(rng.uniform(max(2.05 * r, 0.42), 0.60))
        W_wide = float(rng.uniform(0.70, 1.0))
        L = float(rng.uniform(2.5, 4.5))
        trans = L * float(rng.uniform(0.3, 0.5))
        path = [(0.0, 0.0), (0.0, L)]
        wprofile = [(0.0, W_narrow), (trans, W_narrow), (trans + 0.4, W_wide), (L, W_wide)]
        blocker = None
        prot = None
        W = W_narrow

    elif ctype == CorridorType.ASYMMETRIC:
        L = float(rng.uniform(2.5, 4.5))
        amt = float(rng.uniform(0.10, 0.25))   # protrusion inward amount
        s_p = L * float(rng.uniform(0.35, 0.65))
        side = 1 if rng.random() > 0.5 else -1
        path = [(0.0, 0.0), (0.0, L)]
        wprofile = [(0.0, W), (L, W)]
        blocker = None
        prot = (s_p, side, amt)

    elif ctype == CorridorType.FALSE_FEASIBLE:
        W_entry = float(rng.uniform(0.55, 0.80))   # wide-enough entry
        L = float(rng.uniform(2.5, 4.0))
        # Blocker is an obstacle placed midway, radius makes passage impossible
        s_b = L * float(rng.uniform(0.3, 0.5))
        n_b = float(rng.uniform(-0.05, 0.05))  # near centerline
        r_b = W_entry / 2 - r + float(rng.uniform(0.05, 0.12))  # blocks path
        path = [(0.0, 0.0), (0.0, L)]
        wprofile = [(0.0, W_entry), (L, W_entry)]
        blocker = (s_b, n_b, r_b)
        prot = None
        W = W_entry

    else:
        raise ValueError(f"Unknown ctype: {ctype}")

    arcs, _ = _path_arcs(path)
    params = CorridorParams(
        ctype=ctype,
        path=path,
        width_profile=wprofile,
        blocker=blocker,
        protrusion=prot,
        total_arc=arcs[-1],
        robot_radius=robot_radius,
    )
    return params, W


# ── Main environment class ────────────────────────────────────────────────────

class HarderNarrowPassageEnv(gym.Env):
    """Hard 2D narrow-passage benchmark with multi-segment corridors.

    Observation (19-dim) is layout-compatible with ProceduralNarrowPassageEnv.
    heading_error and lateral_offset use the STRAIGHT start→goal convention so
    existing FSM code runs unchanged and its structural failure on curved
    corridors is exposed.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}
        self.max_steps = int(cfg.get("max_steps", 400))
        self.dt = float(cfg.get("dt", 0.25))
        self.robot_radius = float(cfg.get("robot_radius", 0.18))
        self.max_vx = float(cfg.get("max_vx", 0.35))
        self.min_vx = float(cfg.get("min_vx", -0.15))
        self.max_wz = float(cfg.get("max_wz", 0.8))
        self.width_range = tuple(cfg.get("width_range", (0.45, 0.90)))
        self.yaw_noise = float(cfg.get("yaw_noise", 0.6))
        self.max_depth = float(cfg.get("max_depth", 5.0))
        self.rng = np.random.default_rng(cfg.get("seed", None))

        # Which corridor types to sample from (default: all)
        ctype_names = cfg.get("corridor_types", None)
        if ctype_names is not None:
            self._ctype_list = [CorridorType(n.lower()) for n in ctype_names]
        else:
            self._ctype_list = list(_DEFAULT_PROBS.keys())
        self._ctype_probs = np.array(
            [_DEFAULT_PROBS[ct] for ct in self._ctype_list], dtype=float
        )
        self._ctype_probs /= self._ctype_probs.sum()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([self.min_vx, -self.max_wz], dtype=np.float32),
            high=np.array([self.max_vx, self.max_wz], dtype=np.float32),
        )

        self._params = None
        self._pts = []
        self._arcs = []
        self._goal_world = np.zeros(2)
        self._start_world = np.zeros(2)
        self.pose = np.zeros(3, dtype=np.float32)  # (x, y, yaw)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.stuck_steps = 0
        self.collision = False
        self._min_body_margin = float("inf")

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        ctype = self.rng.choice(self._ctype_list, p=self._ctype_probs)
        self._params, self._episode_W = _build_corridor(
            ctype, self.rng, self.width_range, self.robot_radius
        )
        self._arcs, self._pts = _path_arcs(self._params.path)

        # Goal = last path point + small overshoot along final tangent
        last_tang = (self._pts[-1] - self._pts[-2])
        last_tang /= np.linalg.norm(last_tang) + 1e-9
        self._goal_world = self._pts[-1] + 0.4 * last_tang
        self._start_world = self._pts[0].copy()

        # Robot starts slightly behind the path start
        first_tang = (self._pts[1] - self._pts[0])
        first_tang /= np.linalg.norm(first_tang) + 1e-9
        start_pos = self._pts[0] - 0.9 * first_tang
        # Lateral jitter
        perp = np.array([-first_tang[1], first_tang[0]])
        start_pos += perp * float(self.rng.uniform(-0.25, 0.25))
        start_yaw = math.atan2(first_tang[0], first_tang[1]) + float(
            self.rng.uniform(-self.yaw_noise, self.yaw_noise)
        )

        self.pose = np.array([start_pos[0], start_pos[1], start_yaw], dtype=np.float32)
        self.prev_action[:] = 0.0
        self.step_count = 0
        self.stuck_steps = 0
        self.collision = False
        self._min_body_margin = float("inf")

        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_dist = self._dist_to_goal()
        prev_pos = self.pose[:2].copy()

        vx, wz = float(action[0]), float(action[1])
        new_yaw = float(self._wrap(self.pose[2] + wz * self.dt))
        new_x = self.pose[0] + vx * math.sin(new_yaw) * self.dt
        new_y = self.pose[1] + vx * math.cos(new_yaw) * self.dt

        self.pose[2] = new_yaw
        self.pose[0] = new_x
        self.pose[1] = new_y

        self.step_count += 1
        self.collision = self._is_collision()
        moved = float(np.linalg.norm(self.pose[:2] - prev_pos))
        progress = prev_dist - self._dist_to_goal()
        if moved < 1e-3 and progress < 1e-3:
            self.stuck_steps += 1
        else:
            self.stuck_steps = max(0, self.stuck_steps - 1)

        obs = self._obs(action)
        bm = float(obs[9])
        if bm < self._min_body_margin:
            self._min_body_margin = bm

        success = self._is_success()
        passable = self._params.blocker is None
        timeout = self.step_count >= self.max_steps
        # False-feasible episodes cannot succeed (path is physically blocked)
        if not passable:
            success = False
        done = success or self.collision or timeout

        reward = self._reward(prev_dist, obs, action, success, timeout)
        self.prev_action = action.astype(np.float32)

        info = {
            "success": float(success),
            "collision": float(self.collision),
            "stuck": float(self.stuck_steps >= 25),
            "passable": passable,
            "corridor_type": self._params.ctype.value,
            "passage_width": self._episode_W,
            "body_margin": self._episode_W / 2 - self.robot_radius,
            "min_body_margin": self._min_body_margin,
        }
        return obs, reward, done, False, info

    # ── Observation ───────────────────────────────────────────────────────────

    def _obs(self, action=None):
        """Analytical observation from path frame — layout matches ProceduralNarrowPassageEnv.

        Lateral clearances are computed exactly from the corridor geometry, not
        from ray-casting, so they are noise-free.  'Near' and 'far' sectors use
        lookahead along the path (not the robot heading), which is correct for a
        2D synthetic env without a simulated depth image.

        heading_error and lateral_offset use the STRAIGHT start→goal convention
        (same as NarrowPassageNav-v0) so the FSM's structural failure on L/S
        corridors is correctly exposed without giving extra information.
        """
        action = self.prev_action if action is None else action
        pos = self.pose[:2]

        # ── Path-frame clearances ─────────────────────────────────────────────
        fr = _project(pos, self._pts, self._arcs)
        inside = _in_corridor(pos, self._pts, self._arcs, self._params)

        if not inside or fr is None:
            # Approach / exit zone: open space, large clearances
            cl_body = 5.0
            cr_body = 5.0
            passage_width = 10.0
            body_margin = 5.0 - self.robot_radius
        else:
            W = _width_at(fr.s, self._params.width_profile)
            # Signed: fr.n > 0 = left of travel direction
            cl_body = W / 2 - fr.n - self.robot_radius
            cr_body = W / 2 + fr.n - self.robot_radius
            if self._params.protrusion is not None:
                s_p, side, amt = self._params.protrusion
                span = _width_at(s_p, self._params.width_profile) * 0.5
                if abs(fr.s - s_p) < span:
                    if side < 0:
                        cl_body -= amt
                    else:
                        cr_body -= amt
            passage_width = cl_body + cr_body + 2.0 * self.robot_radius
            body_margin = min(cl_body, cr_body)

        # ── Robot-heading raycasting for sector depths ────────────────────────
        # Six sectors: left-30°/center/right-30° × near/far.
        # Near: ray cast from the robot's current position.
        # Far:  ray cast from a point 1m ahead along the robot's heading.
        # This lets the robot see corners before hitting them — essential for
        # L/S-shaped corridors where the path-based analytical lookahead is blind.
        yaw = float(self.pose[2])
        MAX = self.max_depth
        ray_offset = math.pi / 6   # 30 degrees

        d_ln = _cast_ray(pos, yaw - ray_offset,
                         self._pts, self._arcs, self._params, MAX)
        d_cn = _cast_ray(pos, yaw,
                         self._pts, self._arcs, self._params, MAX)
        d_rn = _cast_ray(pos, yaw + ray_offset,
                         self._pts, self._arcs, self._params, MAX)

        # Far-sector origin: 1 m ahead along current heading
        far_origin = pos + np.array([math.sin(yaw), math.cos(yaw)])
        d_lf = _cast_ray(far_origin, yaw - ray_offset,
                         self._pts, self._arcs, self._params, MAX)
        d_cf = _cast_ray(far_origin, yaw,
                         self._pts, self._arcs, self._params, MAX)
        d_rf = _cast_ray(far_origin, yaw + ray_offset,
                         self._pts, self._arcs, self._params, MAX)

        # For FSM clearance estimates: keep path-frame left/right as the body
        # clearance signal (more accurate than raycasts at oblique angles).
        cl_near = cl_body
        cr_near = cr_body

        # Effective body clearances for FSM (min of current and near sector)
        cl = min(cl_body, cl_near)
        cr = min(cr_body, cr_near)

        # ── Heading error and lateral offset (path-tangent convention) ──────────
        # Use the current segment's tangent direction as the reference heading.
        # For straight corridors this equals the start→goal bearing (no change).
        # For L/S corridors it correctly uses the current segment direction:
        #   approach zone → "face corridor entry" (not the far goal in the next
        #   segment), segment 2 → "face west", etc.
        if fr is not None:
            tang = fr.tangent
            path_yaw = math.atan2(float(tang[0]), float(tang[1]))
            heading_error = self._wrap(path_yaw - float(self.pose[2]))
            # Lateral offset: signed distance from path centre, right-positive.
            # -fr.n because fr.n > 0 is LEFT of travel direction.
            lateral_offset = -float(fr.n)
        else:
            dx = float(self._goal_world[0] - pos[0])
            dy = float(self._goal_world[1] - pos[1])
            heading_error = self._wrap(math.atan2(dx, dy) - float(self.pose[2]))
            lateral_offset = 0.0

        dist_to_goal = self._dist_to_goal()
        stuck_score = min(1.0, self.stuck_steps / 25.0)

        return np.array([
            d_ln, d_cn, d_rn,
            d_lf, d_cf, d_rf,
            float(np.clip(cl, -1.0, 5.0)),
            float(np.clip(cr, -1.0, 5.0)),
            passage_width,
            body_margin,
            heading_error,
            lateral_offset,
            dist_to_goal,
            float(action[0]), float(action[1]),
            stuck_score,
            float(self.collision),
            float(self.prev_action[0]), float(self.prev_action[1]),
        ], dtype=np.float32)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _reward(self, prev_dist, obs, action, success, timeout):
        r = 2.0 * (prev_dist - self._dist_to_goal())
        r -= 0.4 * abs(obs[11])           # lateral penalty
        r -= 0.2 * abs(obs[10])           # heading penalty
        r += 0.15 * max(obs[9], -0.3)    # clearance reward (capped below)
        r -= 4.0 * obs[15]               # stuck penalty
        r -= 20.0 * float(self.collision)
        r -= 0.08 * float(np.abs(action - self.prev_action).sum())
        r -= 0.01
        r += 10.0 * float(success)
        r -= 1.0 * float(timeout and not success)
        return float(r)

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _dist_to_goal(self):
        return float(np.linalg.norm(self.pose[:2] - self._goal_world))

    def _is_success(self):
        dx = float(self.pose[0] - self._goal_world[0])
        dy = float(self.pose[1] - self._goal_world[1])
        return (
            math.hypot(dx, dy) < 0.30
            and abs(self._wrap(self.pose[2] - self._goal_yaw())) < 0.5
        )

    def _goal_yaw(self):
        # Use the last corridor segment's exit direction, not the start→goal
        # diagonal. For straight corridors these are equal; for L/S-shaped
        # corridors the exit heading is the last segment tangent direction.
        last = self._pts[-1] - self._pts[-2]
        last_len = float(np.linalg.norm(last))
        if last_len < 1e-9:
            dx = float(self._goal_world[0] - self._start_world[0])
            dy = float(self._goal_world[1] - self._start_world[1])
            return math.atan2(dx, dy)
        tang = last / last_len
        return float(math.atan2(tang[0], tang[1]))

    def _is_collision(self):
        pos = self.pose[:2]
        if not _in_corridor(pos, self._pts, self._arcs, self._params):
            return False
        fr = _project(pos, self._pts, self._arcs)
        if fr is None:
            return False
        cl, cr = _wall_clearances(fr, self._params)
        if cl < 0 or cr < 0:
            return True
        if _check_blocker(fr, self._params):
            return True
        return False

    @staticmethod
    def _wrap(a):
        return float((a + math.pi) % (2 * math.pi) - math.pi)

    # ── Episode-level summary ─────────────────────────────────────────────────

    @property
    def corridor_type(self):
        return self._params.ctype if self._params else None

    @property
    def is_passable(self):
        return self._params.blocker is None if self._params else True
