"""FSM navigation node for narrow-passage traversal on a real quadruped.

Subscribes
----------
  /scan          sensor_msgs/LaserScan     2D LiDAR horizontal scan
  /odom          nav_msgs/Odometry         wheel odometry (velocity)
  /goal_pose     geometry_msgs/PoseStamped  navigation goal in map frame

Uses TF
-------
  map → base_link   SLAM-derived global pose (falls back to /odom frame if unavailable)

Publishes
---------
  /cmd_vel       geometry_msgs/Twist       velocity command to locomotion controller
  /narrow_mode   std_msgs/String           current FSM mode (COMMIT/ALIGN/RECOVER/…)

Parameters
----------
  control_rate       float  10.0   Control loop rate (Hz)
  robot_half_width   float  0.21   Robot body half-width (m) — calibrate per platform
  variant            str    "full" FSM variant: full / no_recovery / no_alignment
  enabled            bool   true   Set false to disable velocity output (safe stop)
  goal_topic         str    "/goal_pose"  Override goal topic if needed
  cmd_vel_topic      str    "/cmd_vel"    Override output topic if needed
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException

from narrow_memory_core.fsm import TurnCommitFSM
from narrow_memory_nav.obs_builder import build_obs_19


def _quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class FSMNavNode(Node):
    def __init__(self) -> None:
        super().__init__("fsm_nav_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("control_rate",     10.0)
        self.declare_parameter("robot_half_width",  0.21)
        self.declare_parameter("variant",           "full")
        self.declare_parameter("enabled",           True)
        self.declare_parameter("goal_topic",        "/goal_pose")
        self.declare_parameter("cmd_vel_topic",     "/cmd_vel")

        self._half_w  = float(self.get_parameter("robot_half_width").value)
        self._variant = str(self.get_parameter("variant").value)

        # ── FSM ───────────────────────────────────────────────────────────────
        self._fsm = TurnCommitFSM(variant=self._variant)
        self._prev_vx  = 0.0
        self._prev_wz  = 0.0

        # ── State ─────────────────────────────────────────────────────────────
        self._latest_scan: LaserScan | None = None
        self._latest_odom: Odometry | None  = None
        self._goal_x   = 0.0
        self._goal_y   = 0.0
        self._start_x: float | None = None
        self._start_y: float | None = None
        self._goal_received = False

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(LaserScan,    "/scan",  self._scan_cb, 10)
        self.create_subscription(Odometry,     "/odom",  self._odom_cb, 10)
        goal_topic = str(self.get_parameter("goal_topic").value)
        self.create_subscription(PoseStamped,  goal_topic, self._goal_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        cmd_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._cmd_pub  = self.create_publisher(Twist,  cmd_topic,      10)
        self._mode_pub = self.create_publisher(String, "/narrow_mode", 10)

        rate = float(self.get_parameter("control_rate").value)
        self.create_timer(1.0 / max(1.0, rate), self._timer_cb)

        self.get_logger().info(
            f"fsm_nav_node ready  variant={self._variant}  "
            f"half_w={self._half_w:.3f}m  rate={rate:.0f}Hz"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_scan = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _goal_cb(self, msg: PoseStamped) -> None:
        self._goal_x = msg.pose.position.x
        self._goal_y = msg.pose.position.y
        rx, ry, _ = self._map_pose()
        self._start_x, self._start_y = rx, ry
        self._goal_received = True
        self._fsm.reset()
        self.get_logger().info(
            f"Goal received: ({self._goal_x:.2f}, {self._goal_y:.2f})  "
            f"from ({rx:.2f}, {ry:.2f})"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _map_pose(self):
        """Return (x, y, yaw) in map frame; falls back to odom frame."""
        try:
            t = self._tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            x   = t.transform.translation.x
            y   = t.transform.translation.y
            yaw = _quat_to_yaw(t.transform.rotation)
            return x, y, yaw
        except (LookupException, ConnectivityException, Exception):
            pass

        if self._latest_odom is not None:
            p   = self._latest_odom.pose.pose
            yaw = _quat_to_yaw(p.orientation)
            return p.position.x, p.position.y, yaw

        return 0.0, 0.0, 0.0

    def _stuck_score(self) -> float:
        """1.0 if we commanded forward motion but the robot barely moved."""
        if self._latest_odom is None:
            return 0.0
        actual_vx = abs(float(self._latest_odom.twist.twist.linear.x))
        if self._prev_vx > 0.08 and actual_vx < 0.02:
            return 1.0
        return 0.0

    # ── Control loop ──────────────────────────────────────────────────────────

    def _timer_cb(self) -> None:
        if not bool(self.get_parameter("enabled").value):
            self._publish_zero()
            return

        if self._latest_scan is None or not self._goal_received:
            return

        rx, ry, ryaw = self._map_pose()

        vx = wz = 0.0
        if self._latest_odom is not None:
            vx = float(self._latest_odom.twist.twist.linear.x)
            wz = float(self._latest_odom.twist.twist.angular.z)

        sx = self._start_x if self._start_x is not None else rx
        sy = self._start_y if self._start_y is not None else ry

        obs = build_obs_19(
            self._latest_scan,
            rx, ry, ryaw,
            self._goal_x, self._goal_y,
            sx, sy,
            vx, wz,
            self._prev_vx, self._prev_wz,
            self._stuck_score(),
            self._half_w,
        )

        action, mode = self._fsm.step(obs)

        cmd = Twist()
        cmd.linear.x  = float(action[0])
        cmd.angular.z = float(action[1])
        self._cmd_pub.publish(cmd)

        mode_msg = String()
        mode_msg.data = mode
        self._mode_pub.publish(mode_msg)

        self._prev_vx = float(action[0])
        self._prev_wz = float(action[1])

    def _publish_zero(self) -> None:
        self._cmd_pub.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FSMNavNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
