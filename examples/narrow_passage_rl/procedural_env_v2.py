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

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from narrow_passage.models.robot_morphology import (
    RobotMorphology,
    morphology_from_config,
)

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
                inferred = np.asarray(low).shape or np.asarray(high).shape
                self.shape = shape or inferred
                self.low = (
                    np.full(self.shape, low, dtype=dtype)
                    if np.isscalar(low)
                    else np.asarray(low, dtype=dtype).copy()
                )
                self.high = (
                    np.full(self.shape, high, dtype=dtype)
                    if np.isscalar(high)
                    else np.asarray(high, dtype=dtype).copy()
                )
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
    morphology: RobotMorphology = field(default_factory=RobotMorphology)


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


def _candidate_path_frames(world_pos, pts, arcs, params):
    """Return every local segment frame whose passage contains ``world_pos``.

    A corner belongs to the union of two orthogonal corridor arms.  Selecting
    only the numerically nearest polyline segment makes the OBB discontinuously
    jump between those arms and can report a collision even though the robot is
    inside the widened turning chamber.  The physical free space is the union,
    so collision queries must retain all locally valid frames at a junction.

    The small longitudinal tolerance matches :func:`_in_corridor`; it leaves
    the entry and exit end caps open while avoiding remote-segment matches.
    """

    frames = []
    for i in range(len(pts) - 1):
        ab = pts[i + 1] - pts[i]
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 1e-9:
            continue
        tangent = ab / ab_len
        raw_t = float(np.dot(world_pos - pts[i], ab)) / (ab_len ** 2)
        if raw_t < -0.05 or raw_t > 1.05:
            continue
        t = float(np.clip(raw_t, 0.0, 1.0))
        closest = pts[i] + t * ab
        normal = np.array([-tangent[1], tangent[0]])
        n = float(np.dot(world_pos - closest, normal))
        s = arcs[i] + t * ab_len
        if abs(n) <= _width_at(s, params.width_profile) / 2.0 + 0.10:
            frames.append(_PathFrame(
                s=s,
                n=n,
                tangent=tangent.copy(),
                segment_idx=i,
            ))
    return frames


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


def _turning_width_profile(
    total_arc: float,
    base_width: float,
    corner_arcs: List[float],
    morphology: RobotMorphology,
    *,
    ramp_length: float = 0.15,
) -> List[Tuple[float, float]]:
    """Build finite turning chambers around polyline vertices.

    A width spike at a single arc coordinate has zero physical extent and does
    not provide an OBB with room to rotate.  Each internal vertex therefore
    receives a constant-width plateau spanning one forward half-extent plus
    the configured safety margin on both sides, followed by a short linear
    transition back to the nominal corridor width.  Overlapping S-turn
    chambers use the maximum envelope, so the result remains a single ordered
    width profile.
    """

    total_arc = float(total_arc)
    base_width = float(base_width)
    half_plateau = morphology.half_length + morphology.safety_margin
    chamber_width = max(
        base_width, morphology.turning_required_width + 0.04
    )
    ramp_length = max(float(ramp_length), 1e-6)
    corners = [
        float(np.clip(corner, 0.0, total_arc)) for corner in corner_arcs
    ]
    breakpoints = {0.0, total_arc}
    for corner in corners:
        for offset in (
            -half_plateau - ramp_length,
            -half_plateau,
            half_plateau,
            half_plateau + ramp_length,
        ):
            breakpoints.add(float(np.clip(corner + offset, 0.0, total_arc)))

    def envelope_width(s: float) -> float:
        width = base_width
        for corner in corners:
            distance = abs(s - corner)
            if distance <= half_plateau:
                candidate = chamber_width
            elif distance <= half_plateau + ramp_length:
                fraction = 1.0 - (
                    (distance - half_plateau) / ramp_length
                )
                candidate = base_width + fraction * (
                    chamber_width - base_width
                )
            else:
                candidate = base_width
            width = max(width, candidate)
        return float(width)

    return [
        (float(s), envelope_width(float(s)))
        for s in sorted(breakpoints)
    ]


def _point_wall_clearances(frame, params):
    """Return point-to-wall clearances (used only by sensor ray casting)."""

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
    return cl, cr


def _relative_yaw(robot_yaw, tangent):
    path_yaw = math.atan2(float(tangent[0]), float(tangent[1]))
    return float((float(robot_yaw) - path_yaw + math.pi) % (2.0 * math.pi) - math.pi)


def _body_wall_clearances(frame, robot_yaw, params, *, safety=False):
    """Exact OBB clearances to the two parallel walls of the local segment."""

    cl, cr = _point_wall_clearances(frame, params)
    relative_yaw = _relative_yaw(robot_yaw, frame.tangent)
    half_extent = 0.5 * params.morphology.projected_width(relative_yaw)
    if safety:
        half_extent += params.morphology.safety_margin
    return cl - half_extent, cr - half_extent


def _check_blocker_point(frame, params):
    if params.blocker is None:
        return False
    s_b, n_b, r_b = params.blocker
    return math.hypot(frame.s - s_b, frame.n - n_b) < r_b


def _check_blocker_obb(frame, robot_yaw, params):
    """Exact circle-vs-OBB test expressed in the local path frame."""

    if params.blocker is None:
        return False
    s_b, n_b, r_b = params.blocker
    relative_yaw = _relative_yaw(robot_yaw, frame.tangent)
    ds = float(s_b - frame.s)
    dn = float(n_b - frame.n)
    # Path-frame (forward,left) -> robot-frame (forward,left).
    forward = ds * math.cos(relative_yaw) - dn * math.sin(relative_yaw)
    lateral = ds * math.sin(relative_yaw) + dn * math.cos(relative_yaw)
    q_forward = max(abs(forward) - params.morphology.half_length, 0.0)
    q_lateral = max(abs(lateral) - params.morphology.half_width, 0.0)
    return math.hypot(q_forward, q_lateral) < r_b


def _minimum_wall_width(params):
    """Minimum cross-sectional wall width, including a protrusion."""

    candidates = [float(width) for _, width in params.width_profile]
    if params.protrusion is not None:
        s_p, _, amount = params.protrusion
        candidates.append(float(_width_at(s_p, params.width_profile) - amount))
    return min(candidates)


def structural_margin_ground_truth(params):
    return float(
        _minimum_wall_width(params) - params.morphology.structural_required_width
    )


def ground_truth_passable(params):
    """Safety-aware OBB feasibility label from the same morphology as collision."""

    required = params.morphology.structural_required_width
    if _minimum_wall_width(params) + 1e-12 < required:
        return False
    if params.blocker is None:
        return True
    s_b, n_b, radius = params.blocker
    width = _width_at(s_b, params.width_profile)
    left_gap = width / 2.0 - (n_b + radius)
    right_gap = (n_b - radius) + width / 2.0
    return max(left_gap, right_gap) + 1e-12 >= required


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

    Free space is the union of strict polyline-segment rectangles.  Walls are
    sampled as a thin band around each segment's *lateral boundary*.  Merely
    sharing the longitudinal projection of a later, perpendicular segment is
    not a wall hit: that old test made an L/S arm several metres away appear as
    a 2-cm obstacle at the entrance (ray--segment cross-talk).
    """
    dx, dy = math.sin(yaw), math.cos(yaw)
    ray_dir = np.array([dx, dy])
    for k in range(int(max_d / step) + 2):
        d = k * step
        if d > max_d:
            return max_d
        pt = world_pos + d * ray_dir
        # Treat the polyline corridor as a union of free rectangles.  First
        # determine whether the sample is free; only then test proximity to a
        # *real lateral boundary*.  This preserves open entry/exit end caps.
        free_frames = []
        boundary_frames = []
        for i in range(len(pts) - 1):
            ab = pts[i + 1] - pts[i]
            ab_len = float(np.linalg.norm(ab))
            if ab_len < 1e-9:
                continue
            tang = ab / ab_len
            along = float(np.dot(pt - pts[i], tang))
            t = along / ab_len
            # Internal segment end caps are walls except where the adjacent
            # rectangle overlaps them.  The first entry and final exit remain
            # open.  Checking a metric band around the actual endpoint avoids
            # the remote-arm cross-talk caused by an unbounded projection test.
            endpoint = pts[i] if along < 0.0 else pts[i + 1]
            endpoint_n = float(np.dot(
                pt - endpoint, np.array([-tang[1], tang[0]])
            ))
            endpoint_s = arcs[i] if along < 0.0 else arcs[i + 1]
            endpoint_width = _width_at(endpoint_s, params.width_profile)
            near_internal_start = bool(
                i > 0 and -1.5 * step <= along < 0.0
            )
            near_internal_end = bool(
                i < len(pts) - 2
                and ab_len < along <= ab_len + 1.5 * step
            )
            if (
                (near_internal_start or near_internal_end)
                and abs(endpoint_n) <= endpoint_width / 2
            ):
                boundary_frames.append((
                    i,
                    0.0 if near_internal_start else 1.0,
                    endpoint_n,
                    tang.copy(),
                    endpoint_s,
                ))
            if t < 0.0 or t > 1.0:
                continue
            t_clamped = max(0.0, min(1.0, t))
            closest = pts[i] + t_clamped * ab
            norm = np.array([-tang[1], tang[0]])
            n = float(np.dot(pt - closest, norm))
            s = arcs[i] + t_clamped * ab_len
            W_i = _width_at(s, params.width_profile)
            if abs(n) <= W_i / 2:
                free_frames.append((i, t_clamped, n, tang.copy(), s))
            # A sample can cross a wall by at most roughly one ray step.  A
            # narrow band is sufficient and, unlike ``outside rectangle``,
            # cannot couple to a remote perpendicular arm.
            if abs(abs(n) - W_i / 2) <= 1.5 * step:
                boundary_frames.append((i, t_clamped, n, tang.copy(), s))

        point_is_free = False
        for i, t_clamped, n, tang, s in free_frames:
            fr = _PathFrame(s=s, n=n, tangent=tang, segment_idx=i)
            cl, cr = _point_wall_clearances(fr, params)
            if cl < 0 or cr < 0:
                continue
            if _check_blocker_point(fr, params):
                return max(d, 0.5 * step)
            point_is_free = True
            break
        if point_is_free:
            continue
        for i, t_clamped, n, tang, s in boundary_frames:
            fr = _PathFrame(s=s, n=n, tangent=tang, segment_idx=i)
            if _check_blocker_point(fr, params):
                return max(d, 0.5 * step)
            return max(d, 0.5 * step)
    return max_d


# ── Corridor generators ───────────────────────────────────────────────────────

def _build_corridor(
    ctype: CorridorType,
    rng: np.random.Generator,
    width_range=(0.45, 0.90),
    morphology: RobotMorphology | None = None,
    forced_width: float | None = None,
) -> CorridorParams:
    morphology = morphology or RobotMorphology()
    W = float(forced_width) if forced_width is not None else float(rng.uniform(*width_range))

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
        wprofile = _turning_width_profile(
            L1 + L2, W, [L1], morphology
        )
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
        corner_1 = L1
        corner_2 = L1 + L2
        wprofile = _turning_width_profile(
            L1 + L2 + L3, W, [corner_1, corner_2], morphology
        )
        blocker = None
        prot = None

    elif ctype == CorridorType.NARROW_EXIT:
        W_wide = float(rng.uniform(0.70, 1.0))
        W_narrow = (
            float(forced_width)
            if forced_width is not None
            else float(rng.uniform(max(morphology.width + 0.01, 0.42), 0.60))
        )
        L = float(rng.uniform(2.5, 4.5))
        trans = L * float(rng.uniform(0.4, 0.6))  # transition point
        path = [(0.0, 0.0), (0.0, L)]
        wprofile = [(0.0, W_wide), (trans, W_wide), (trans + 0.4, W_narrow), (L, W_narrow)]
        blocker = None
        prot = None
        W = W_narrow  # tightest width for paper reporting

    elif ctype == CorridorType.NARROW_ENTRY:
        W_narrow = (
            float(forced_width)
            if forced_width is not None
            else float(rng.uniform(max(morphology.width + 0.01, 0.42), 0.60))
        )
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
        if forced_width is not None:
            W = float(forced_width) + amt
        s_p = L * float(rng.uniform(0.35, 0.65))
        side = 1 if rng.random() > 0.5 else -1
        path = [(0.0, 0.0), (0.0, L)]
        wprofile = [(0.0, W), (L, W)]
        blocker = None
        prot = (s_p, side, amt)

    elif ctype == CorridorType.FALSE_FEASIBLE:
        W_entry = (
            float(forced_width)
            if forced_width is not None
            else float(rng.uniform(0.55, 0.80))
        )
        L = float(rng.uniform(2.5, 4.0))
        # Blocker is an obstacle placed midway, radius makes passage impossible
        s_b = L * float(rng.uniform(0.3, 0.5))
        n_b = float(rng.uniform(-0.05, 0.05))  # near centerline
        r_b = (
            W_entry / 2 - morphology.half_width
            + float(rng.uniform(0.05, 0.12))
        )  # blocks the OBB path
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
        morphology=morphology,
    )
    return params, _minimum_wall_width(params)


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
        self.morphology = morphology_from_config(cfg.get("morphology"))
        if "robot_radius" in cfg:
            legacy_width = 2.0 * float(cfg["robot_radius"])
            if abs(legacy_width - self.morphology.width) > 1e-12:
                raise ValueError(
                    "robot_radius conflicts with the strict OBB morphology; "
                    "set morphology.width instead"
                )
        # Read-only compatibility for plotting/legacy metadata. It is never used
        # for collision, clearance, success, or feasibility labels.
        self.robot_radius = self.morphology.half_width
        self.max_vx = float(cfg.get("max_vx", 0.35))
        self.min_vx = float(cfg.get("min_vx", -0.15))
        self.max_wz = float(cfg.get("max_wz", 0.8))
        self.width_range = tuple(cfg.get("width_range", (0.45, 0.90)))
        self.yaw_noise = float(cfg.get("yaw_noise", 0.6))
        self.start_pose_jitter = float(cfg.get("start_pose_jitter", 0.25))
        self.depth_noise = float(cfg.get("depth_noise", 0.0))
        self.obstacle_noise = float(cfg.get("obstacle_noise", 0.0))
        self.max_depth = float(cfg.get("max_depth", 5.0))
        self.rng = np.random.default_rng(cfg.get("seed", None))

        # Which corridor types to sample from (default: all)
        ctype_names = cfg.get("corridor_types", None)
        if ctype_names is not None:
            self._ctype_list = [CorridorType(n.lower()) for n in ctype_names]
        else:
            self._ctype_list = list(_DEFAULT_PROBS.keys())
        ctype_probs = cfg.get("corridor_type_probs", None)
        if ctype_probs is None:
            self._ctype_probs = np.array(
                [_DEFAULT_PROBS[ct] for ct in self._ctype_list], dtype=float
            )
        else:
            self._ctype_probs = np.array(
                [float(ctype_probs.get(ct.value, 0.0)) for ct in self._ctype_list],
                dtype=float,
            )
            if float(self._ctype_probs.sum()) <= 0.0:
                self._ctype_probs = np.ones(len(self._ctype_list), dtype=float)
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
        self.translation_stuck_steps = 0
        self.rotation_stuck_steps = 0
        self.collision = False
        self._min_body_margin = float("inf")

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        requested_type = options.get("corridor_type")
        ctype = (
            CorridorType(str(requested_type).lower())
            if requested_type is not None
            else self.rng.choice(self._ctype_list, p=self._ctype_probs)
        )
        forced_width = options.get("passage_width")
        self._params, self._episode_W = _build_corridor(
            ctype,
            self.rng,
            self.width_range,
            self.morphology,
            None if forced_width is None else float(forced_width),
        )
        self._apply_obstacle_noise()
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
        if "lateral_offset" in options:
            start_pos += perp * float(options["lateral_offset"])
        else:
            start_pos += perp * float(
                self.rng.uniform(-self.start_pose_jitter, self.start_pose_jitter)
            )
        path_yaw = math.atan2(first_tang[0], first_tang[1])
        if "yaw_deg" in options:
            start_yaw = path_yaw + math.radians(float(options["yaw_deg"]))
        else:
            start_yaw = path_yaw + float(
                self.rng.uniform(-self.yaw_noise, self.yaw_noise)
            )

        self.pose = np.array([start_pos[0], start_pos[1], start_yaw], dtype=np.float32)
        self.prev_action[:] = 0.0
        self.step_count = 0
        self.stuck_steps = 0
        self.translation_stuck_steps = 0
        self.rotation_stuck_steps = 0
        self.collision = False
        self._min_body_margin = float("inf")

        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_dist = self._dist_to_goal()
        prev_pos = self.pose[:2].copy()
        prev_yaw = float(self.pose[2])

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
        yaw_moved = abs(float(self._wrap(float(self.pose[2]) - prev_yaw)))
        progress = prev_dist - self._dist_to_goal()
        translation_expected = abs(vx) > 0.02
        rotation_expected = abs(wz) > 0.05
        if translation_expected and moved < 1e-3 and progress < 1e-3:
            self.translation_stuck_steps += 1
        else:
            self.translation_stuck_steps = max(0, self.translation_stuck_steps - 1)
        if rotation_expected and yaw_moved < 1e-3:
            self.rotation_stuck_steps += 1
        else:
            self.rotation_stuck_steps = max(0, self.rotation_stuck_steps - 1)
        # Backward-compatible observation slot: only failed commanded
        # translation can trigger recovery.  Deliberate probing/alignment does
        # not accumulate translation stuck.
        self.stuck_steps = self.translation_stuck_steps

        obs = self._obs(action)
        bm = float(obs[9])
        if bm < self._min_body_margin:
            self._min_body_margin = bm

        passable = ground_truth_passable(self._params)
        success = self._is_success() and passable
        timeout = self.step_count >= self.max_steps
        done = success or self.collision or timeout

        reward = self._reward(prev_dist, obs, action, success, timeout)
        self.prev_action = action.astype(np.float32)

        info = {
            "success": float(success),
            "collision": float(self.collision),
            "stuck": float(self.stuck_steps >= 25),
            "translation_stuck_counter": int(self.translation_stuck_steps),
            "rotation_stuck_counter": int(self.rotation_stuck_steps),
            "translation_expected": bool(translation_expected),
            "rotation_expected": bool(rotation_expected),
            "passable": passable,
            "corridor_type": self._params.ctype.value,
            "passage_width": self._episode_W,
            "structural_margin_gt": structural_margin_ground_truth(self._params),
            "body_margin": self._episode_W / 2 - self.morphology.half_width,
            "min_body_margin": self._min_body_margin,
            "morphology": self.morphology.record(),
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

        first_tangent = self._pts[1] - self._pts[0]
        first_tangent /= np.linalg.norm(first_tangent) + 1e-9
        approach_progress = float(np.dot(pos - self._pts[0], first_tangent))
        approaching_entry = bool(
            fr is not None and -self.max_depth <= approach_progress < 0.0
        )

        if approaching_entry:
            # Prospective entry geometry: strict feasibility must be observable
            # before the first physical contact.  This is a sensor/observation
            # quantity; collision remains disabled while the OBB is outside.
            entry_frame = _PathFrame(
                s=0.0,
                n=float(fr.n),
                tangent=first_tangent.copy(),
                segment_idx=0,
            )
            relative_yaw = _relative_yaw(float(self.pose[2]), first_tangent)
            projected_width = self.morphology.projected_width(relative_yaw)
            cl_body, cr_body = _body_wall_clearances(
                entry_frame, float(self.pose[2]), self._params
            )
            passage_width = cl_body + cr_body + projected_width
            # Physical current clearance remains open-space clearance. The
            # prospective cl/cr and width describe the sensed entry aperture.
            body_margin = 5.0 - self.morphology.half_width
        elif not inside or fr is None:
            # Approach / exit zone: open space, large clearances
            cl_body = 5.0
            cr_body = 5.0
            passage_width = 10.0
            body_margin = 5.0 - self.morphology.half_width
        else:
            relative_yaw = _relative_yaw(float(self.pose[2]), fr.tangent)
            projected_width = self.morphology.projected_width(relative_yaw)
            cl_body, cr_body = _body_wall_clearances(
                fr, float(self.pose[2]), self._params
            )
            passage_width = cl_body + cr_body + projected_width
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

        obs = np.array([
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
        if self.depth_noise > 0.0:
            noise = self.rng.normal(0.0, self.depth_noise, size=10).astype(np.float32)
            obs[:10] = np.clip(obs[:10] + noise, -1.0, self.max_depth)
        return obs

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

    def _apply_obstacle_noise(self):
        if self.obstacle_noise <= 0.0:
            return
        if self._params.blocker is not None:
            s, n, radius = self._params.blocker
            self._params.blocker = (
                s,
                n + float(self.rng.normal(0.0, self.obstacle_noise)),
                max(0.02, radius + float(self.rng.normal(0.0, self.obstacle_noise))),
            )
        if self._params.protrusion is not None:
            s, side, amount = self._params.protrusion
            self._params.protrusion = (
                s,
                side,
                max(0.01, amount + float(self.rng.normal(0.0, self.obstacle_noise))),
            )

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
        return self._collision_at_pose(self.pose)

    def _collision_at_pose(self, pose: np.ndarray) -> bool:
        """OBB collision query against the union of local corridor arms.

        At a polyline junction the free space is a union, not the cross-section
        of whichever centerline segment happens to be a few micrometres closer.
        A pose is wall-safe when at least one locally containing arm supports
        the OBB projection.  This removes a segment-selection discontinuity at
        L/S turning chambers without changing the footprint or wall widths.
        """

        pos = np.asarray(pose[:2], dtype=float)
        if not _in_corridor(pos, self._pts, self._arcs, self._params):
            return False
        frames = _candidate_path_frames(
            pos, self._pts, self._arcs, self._params
        )
        if not frames:
            return False
        wall_safe_frames = []
        for frame in frames:
            cl, cr = _body_wall_clearances(
                frame, float(pose[2]), self._params
            )
            if cl >= 0.0 and cr >= 0.0:
                wall_safe_frames.append(frame)
        if not wall_safe_frames:
            return True
        # Blockers currently occur only in the single-arm false-feasible scene.
        # Requiring every wall-safe representation to clear the same blocker
        # prevents a junction frame from masking physical contact if that scene
        # family is extended later.
        return all(
            _check_blocker_obb(frame, float(pose[2]), self._params)
            for frame in wall_safe_frames
        )

    def _wall_clearance_at_pose(self, pose: np.ndarray) -> float:
        """Return union-aware minimum lateral OBB clearance.

        At a junction each arm offers a valid local representation.  The union
        clearance is therefore the best supported arm clearance, not the value
        from a discontinuously selected nearest segment.
        """

        frames = _candidate_path_frames(
            np.asarray(pose[:2], dtype=float),
            self._pts,
            self._arcs,
            self._params,
        )
        clearances = []
        for frame in frames:
            cl, cr = _body_wall_clearances(
                frame, float(pose[2]), self._params
            )
            clearances.append(min(float(cl), float(cr)))
        return max(clearances) if clearances else float("inf")

    def action_is_safe(
        self,
        action: np.ndarray,
        *,
        horizon_steps: int = 3,
        minimum_clearance: float = 0.0,
    ) -> bool:
        """Short-horizon swept-OBB safety check for a velocity command."""

        action = np.clip(action, self.action_space.low, self.action_space.high)
        pose = np.asarray(self.pose, dtype=float).copy()
        for _ in range(max(1, int(horizon_steps))):
            vx, wz = float(action[0]), float(action[1])
            pose[2] = self._wrap(float(pose[2]) + wz * self.dt)
            pose[0] += vx * math.sin(float(pose[2])) * self.dt
            pose[1] += vx * math.cos(float(pose[2])) * self.dt
            if self._collision_at_pose(pose):
                return False
            if minimum_clearance > 0.0 and _in_corridor(
                pose[:2], self._pts, self._arcs, self._params
            ):
                if self._wall_clearance_at_pose(pose) < minimum_clearance:
                    return False
        return True

    def action_minimum_clearance(
        self,
        action: np.ndarray,
        *,
        horizon_steps: int = 3,
    ) -> float:
        """Return the minimum predicted swept-OBB wall clearance.

        The query is side-effect free and uses the same pose integration and
        union-aware OBB geometry as :meth:`action_is_safe`.  A negative value
        denotes predicted wall overlap; open space is reported as 0.5 m for a
        finite, CSV-friendly diagnostic value.
        """

        action = np.clip(action, self.action_space.low, self.action_space.high)
        pose = np.asarray(self.pose, dtype=float).copy()
        minimum_predicted = float("inf")
        for _ in range(max(1, int(horizon_steps))):
            vx, wz = float(action[0]), float(action[1])
            pose[2] = self._wrap(float(pose[2]) + wz * self.dt)
            pose[0] += vx * math.sin(float(pose[2])) * self.dt
            pose[1] += vx * math.cos(float(pose[2])) * self.dt
            if _in_corridor(pose[:2], self._pts, self._arcs, self._params):
                minimum_predicted = min(
                    minimum_predicted, self._wall_clearance_at_pose(pose)
                )
        return (
            float(minimum_predicted)
            if math.isfinite(minimum_predicted)
            else 0.50
        )

    def project_to_safe_action(
        self,
        action: np.ndarray,
        *,
        horizon_steps: int = 3,
        minimum_clearance: float = 0.0,
    ) -> tuple[np.ndarray, str]:
        """Project a desired command onto a small swept-OBB-safe action set.

        The projection is progress-first: preserve the desired command when it
        is safe, otherwise search a deterministic steering lattice at descending
        translational speeds.  Pure rotations are considered only when no
        non-zero forward candidate is swept-OBB safe.  It never alters the
        collision geometry or robot footprint.
        """

        desired = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        vx, wz = float(desired[0]), float(desired[1])
        turn_sign = 1.0 if wz >= 0.0 else -1.0
        if self.action_is_safe(
            desired,
            horizon_steps=horizon_steps,
            minimum_clearance=minimum_clearance,
        ):
            return desired, "desired_safe"

        # Complete the low-speed part of the safe action set.  A hand-picked
        # coupling between speed and turn rate omitted valid side-step commands
        # and produced +/-0.12-rad/s limit cycles at OBB boundaries.
        if vx > 0.0:
            speed_values = sorted({
                float(min(vx, cap))
                for cap in (
                    vx, 0.15, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02,
                    0.01, 0.008, 0.005, 0.003, 0.001,
                )
                if min(vx, cap) > 1e-9
            }, reverse=True)
            steering_values = list(dict.fromkeys([
                wz,
                wz - 0.05,
                wz + 0.05,
                *[0.05 * index for index in range(-5, 6)],
                -0.40,
                0.40,
                -0.60,
                0.60,
                -0.80,
                0.80,
            ]))
            # First preserve steering continuity within 0.15 rad/s, reducing
            # speed as necessary.  Only if no such action exists at any speed
            # may the full steering lattice make a larger evasive correction.
            for maximum_steering_deviation in (0.15, float("inf")):
                for candidate_vx in speed_values:
                    if (
                        candidate_vx < 0.01
                        and math.isfinite(maximum_steering_deviation)
                    ):
                        # Emergency creep exists to escape a local swept-OBB
                        # boundary.  Evaluate it in the clearance-prioritised
                        # pass below, rather than continuing a zero-progress
                        # near-desired rotation indefinitely.
                        continue
                    safe_progress: list[tuple[float, np.ndarray]] = []
                    for candidate_wz in steering_values:
                        if (
                            abs(float(candidate_wz) - wz)
                            > maximum_steering_deviation + 1e-12
                        ):
                            continue
                        candidate = np.clip(
                            np.asarray(
                                [candidate_vx, candidate_wz], dtype=np.float32
                            ),
                            self.action_space.low,
                            self.action_space.high,
                        ).astype(np.float32)
                        if not self.action_is_safe(
                            candidate,
                            horizon_steps=horizon_steps,
                            minimum_clearance=minimum_clearance,
                        ):
                            continue
                        clearance = self.action_minimum_clearance(
                            candidate, horizon_steps=horizon_steps
                        )
                        if candidate_vx < 0.01:
                            score = (
                                min(clearance, 0.50)
                                - 0.05 * abs(float(candidate[1]) - wz)
                                - 0.001 * abs(float(candidate[1]))
                            )
                        else:
                            score = (
                                -abs(float(candidate[1]) - wz)
                                + 0.01 * min(clearance, 0.50)
                                - 0.001 * abs(float(candidate[1]))
                            )
                        safe_progress.append((score, candidate))
                    if safe_progress:
                        _, candidate = max(
                            safe_progress, key=lambda item: item[0]
                        )
                        return (
                            candidate,
                            f"safe_progress_v{int(round(candidate_vx * 100)):02d}",
                        )

        rotation_values = list(dict.fromkeys([
            wz,
            *[0.01 * index for index in range(-12, 13) if index != 0],
            -0.20,
            0.20,
            -0.40,
            0.40,
        ]))
        rotation_candidates = [
            (
                np.asarray([0.0, candidate_wz], dtype=np.float32),
                (
                    "rotate_in_place_desired"
                    if abs(float(candidate_wz) - wz) < 1e-9
                    else "rotate_lattice"
                ),
            )
            for candidate_wz in rotation_values
        ]
        safe_rotations: list[tuple[float, np.ndarray, str]] = []
        for candidate, reason in rotation_candidates:
            candidate = np.clip(
                candidate, self.action_space.low, self.action_space.high
            ).astype(np.float32)
            if np.all(np.abs(candidate) < 1e-8):
                continue
            if self.action_is_safe(
                candidate,
                horizon_steps=horizon_steps,
                minimum_clearance=minimum_clearance,
            ):
                clearance = self.action_minimum_clearance(
                    candidate, horizon_steps=horizon_steps
                )
                score = min(clearance, 0.50) - 0.08 * abs(
                    float(candidate[1]) - wz
                )
                safe_rotations.append((score, candidate, reason))
        if safe_rotations:
            _, candidate, reason = max(
                safe_rotations, key=lambda item: item[0]
            )
            return candidate, reason
        # A forward commitment can end at a wall-adjacent swept-OBB boundary
        # where neither forward motion nor a finite observation rotation is
        # safe, while a short reverse command increases clearance.  Expose that
        # command as an explicit recovery action instead of an irreversible
        # STOP equilibrium.
        safe_backtracks: list[tuple[float, np.ndarray]] = []
        for reverse_vx in (-0.08, -0.03, -0.01):
            for reverse_wz in [0.01 * index for index in range(-20, 21)]:
                candidate = np.asarray(
                    [reverse_vx, reverse_wz], dtype=np.float32
                )
                if not self.action_is_safe(
                    candidate,
                    horizon_steps=horizon_steps,
                    minimum_clearance=minimum_clearance,
                ):
                    continue
                clearance = self.action_minimum_clearance(
                    candidate, horizon_steps=horizon_steps
                )
                safe_backtracks.append((
                    min(clearance, 0.50) - 0.005 * abs(reverse_wz),
                    candidate,
                ))
            if safe_backtracks:
                _, candidate = max(
                    safe_backtracks, key=lambda item: item[0]
                )
                return candidate, "safe_backtrack_recover"
        # STOP is a fallback, not a candidate that may outscore an available
        # observation rotation.  Scoring it together with non-zero actions
        # created permanent safe-stop equilibria at entries and junctions.
        return np.zeros(2, dtype=np.float32), "stop"

    @staticmethod
    def _wrap(a):
        return float((a + math.pi) % (2 * math.pi) - math.pi)

    # ── Episode-level summary ─────────────────────────────────────────────────

    @property
    def corridor_type(self):
        return self._params.ctype if self._params else None

    @property
    def is_passable(self):
        return ground_truth_passable(self._params) if self._params else True

    def navigation_diagnostics(self) -> dict:
        """Expose controller-facing path context without changing env state."""

        frame = _project(self.pose[:2], self._pts, self._arcs)
        if frame is None:
            return {
                "active_corridor_arm": -1,
                "local_goal": [float("nan"), float("nan")],
                "path_progress": float("nan"),
            }
        goal_index = min(frame.segment_idx + 1, len(self._pts) - 1)
        return {
            "active_corridor_arm": int(frame.segment_idx),
            "local_goal": [float(value) for value in self._pts[goal_index]],
            "path_progress": float(frame.s),
        }
