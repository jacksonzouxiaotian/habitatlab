"""Build the 19-dim observation vector from ROS2 sensor messages.

Used by fsm_nav_node to convert real-robot sensor data into the same
format that TurnCommitFSM / fsm_action expects.

Sector convention (degrees, 0 = forward, positive = left, negative = right):
  d_ln :  30 ..  75  near left
  d_cn : -20 ..  20  center (also used as d_cf)
  d_rn : -75 .. -30  near right
  d_lf :  75 .. 110  far left
  d_rf :-110 .. -75  far right
  cl   :  40 .. 100  left clearance wall
  cr   :-100 .. -40  right clearance wall
"""

import math
from typing import Optional

import numpy as np
from sensor_msgs.msg import LaserScan

_MAX_RANGE = 5.0


def _sector_min(scan: LaserScan, deg_lo: float, deg_hi: float) -> float:
    """Return the minimum valid range reading inside the angular sector [deg_lo, deg_hi].

    Angles are in degrees, 0=forward, positive=left (CCW), negative=right (CW).
    Returns _MAX_RANGE if no valid readings exist in the sector.
    """
    rad_lo = math.radians(min(deg_lo, deg_hi))
    rad_hi = math.radians(max(deg_lo, deg_hi))
    best = _MAX_RANGE
    for i, r in enumerate(scan.ranges):
        angle = scan.angle_min + i * scan.angle_increment
        if rad_lo <= angle <= rad_hi and math.isfinite(r):
            if scan.range_min < r < scan.range_max:
                best = min(best, r)
    return best


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def build_obs_19(
    scan: LaserScan,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    goal_x: float,
    goal_y: float,
    start_x: float,
    start_y: float,
    vx: float,
    wz: float,
    prev_vx: float,
    prev_wz: float,
    stuck_score: float,
    robot_half_width: float = 0.21,
    collision_override: Optional[float] = None,
) -> np.ndarray:
    """Assemble the 19-dim obs vector from ROS2 scan + pose + odometry.

    Parameters
    ----------
    scan            LaserScan message (2D LiDAR, horizontal)
    robot_x/y/yaw   Current robot pose in map frame (from TF or odom)
    goal_x/y        Target position in map frame
    start_x/y       Position at which the current goal was received
                    (used to compute lateral cross-track offset)
    vx / wz         Current body-frame velocities from odometry
    prev_vx/wz      Previous commanded velocities (FSM internal state)
    stuck_score     0-1, 1 = commanded forward but velocity dropped to ~0
    robot_half_width Robot body half-width in metres (default 0.21m for Go2)
    collision_override  If not None, overrides the LiDAR-proximity collision flag
    """
    # ── 6 depth sectors ──────────────────────────────────────────────────────
    d_ln = _sector_min(scan,  30,  75)
    d_cn = _sector_min(scan, -20,  20)
    d_rn = _sector_min(scan, -75, -30)
    d_lf = _sector_min(scan,  75, 110)
    d_cf = d_cn                          # reuse centre sector for "far centre"
    d_rf = _sector_min(scan, -110, -75)

    # ── Clearance and passage geometry ───────────────────────────────────────
    cl_raw = _sector_min(scan,  40, 100)
    cr_raw = _sector_min(scan, -100, -40)
    cl = max(0.0, cl_raw - robot_half_width)
    cr = max(0.0, cr_raw - robot_half_width)
    passage_width = cl + cr
    body_margin = min(cl, cr)

    # ── Goal features ────────────────────────────────────────────────────────
    dx, dy = goal_x - robot_x, goal_y - robot_y
    dist_to_goal = math.hypot(dx, dy)
    heading_error = _wrap_angle(math.atan2(dy, dx) - robot_yaw)

    # Lateral offset: signed cross-track error from start→goal line
    # Positive = robot is to the left of the line.
    sx, sy = goal_x - start_x, goal_y - start_y
    line_len = math.hypot(sx, sy)
    if line_len > 0.1:
        rx, ry = robot_x - start_x, robot_y - start_y
        lateral_offset = (sx * ry - sy * rx) / line_len
    else:
        lateral_offset = 0.0

    # ── Collision flag ───────────────────────────────────────────────────────
    if collision_override is not None:
        collision_flag = float(collision_override)
    else:
        # Proxy: any LiDAR reading in the forward ±25° within 0.25m
        front_min = _sector_min(scan, -25, 25)
        collision_flag = 1.0 if front_min < 0.25 else 0.0

    return np.array([
        d_ln, d_cn, d_rn,                          # 0-2  near sectors
        d_lf, d_cf, d_rf,                          # 3-5  far sectors
        cl, cr, passage_width, body_margin,        # 6-9  clearance
        heading_error, lateral_offset, dist_to_goal,  # 10-12 goal
        vx, wz,                                    # 13-14 velocity
        stuck_score, collision_flag,               # 15-16 safety
        prev_vx, prev_wz,                          # 17-18 previous command
    ], dtype=np.float32)
