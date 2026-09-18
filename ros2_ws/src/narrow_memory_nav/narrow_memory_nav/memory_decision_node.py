import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String

from narrow_memory_core import FailureAwareNavigator, FailureMemoryConfig
from narrow_memory_nav.scan_features import features_from_scan


class MemoryDecisionNode(Node):
    def __init__(self) -> None:
        super().__init__("memory_decision_node")
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("memory_trigger_count", 1)
        self.declare_parameter("memory_reject_count", 3)
        self.declare_parameter("clearance_low", 0.06)
        self.declare_parameter("clearance_high", 0.13)
        self.declare_parameter("lateral_low", 0.24)
        self.declare_parameter("lateral_high", 0.34)
        self.declare_parameter("heading_low", 0.45)
        self.declare_parameter("heading_high", 0.75)
        self.declare_parameter("high_risk_threshold", 0.75)
        self.declare_parameter("medium_risk_threshold", 0.50)
        self.declare_parameter("memory_recovery_risk", 0.45)
        self.declare_parameter("align_linear_limit", 0.10)
        self.declare_parameter("commit_linear_limit", 0.25)
        self.declare_parameter("recover_linear_x", -0.08)
        self.declare_parameter("recover_angular_z", 0.40)
        self.declare_parameter("reject_angular_z", 0.30)

        memory_cfg = FailureMemoryConfig(
            trigger_count=int(self.get_parameter("memory_trigger_count").value),
            reject_count=int(self.get_parameter("memory_reject_count").value),
        )
        self.navigator = FailureAwareNavigator(
            memory_config=memory_cfg,
            clearance_low=float(self.get_parameter("clearance_low").value),
            clearance_high=float(self.get_parameter("clearance_high").value),
            lateral_low=float(self.get_parameter("lateral_low").value),
            lateral_high=float(self.get_parameter("lateral_high").value),
            heading_low=float(self.get_parameter("heading_low").value),
            heading_high=float(self.get_parameter("heading_high").value),
            high_risk_threshold=float(
                self.get_parameter("high_risk_threshold").value
            ),
            medium_risk_threshold=float(
                self.get_parameter("medium_risk_threshold").value
            ),
            memory_recovery_risk=float(
                self.get_parameter("memory_recovery_risk").value
            ),
        )

        self.latest_scan: LaserScan | None = None
        self.latest_odom: Odometry | None = None
        self.latest_nav_cmd: Twist | None = None

        self.create_subscription(LaserScan, "/scan", self._scan_callback, 10)
        self.create_subscription(Odometry, "/odom", self._odom_callback, 10)
        self.create_subscription(Twist, "/nav2_cmd_vel", self._nav_cmd_callback, 10)

        self.decision_pub = self.create_publisher(String, "/narrow_decision", 10)
        self.can_pass_pub = self.create_publisher(Bool, "/narrow_can_pass", 10)
        self.risk_pub = self.create_publisher(Float32, "/narrow_risk", 10)
        self.safety_cmd_pub = self.create_publisher(Twist, "/safety_cmd_vel", 10)

        rate = float(self.get_parameter("control_rate").value)
        self.create_timer(1.0 / max(1.0, rate), self._timer_callback)
        self.get_logger().info("memory_decision_node started")

    def _scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def _odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _nav_cmd_callback(self, msg: Twist) -> None:
        self.latest_nav_cmd = msg

    def _stuck_score(self) -> float:
        if self.latest_odom is None or self.latest_nav_cmd is None:
            return 0.0
        vx = abs(float(self.latest_odom.twist.twist.linear.x))
        wz = abs(float(self.latest_odom.twist.twist.angular.z))
        cmd_vx = abs(float(self.latest_nav_cmd.linear.x))
        if cmd_vx > 0.08 and vx + 0.2 * wz < 0.02:
            return 1.0
        return 0.0

    def _make_safety_cmd(self, mode: str) -> Twist:
        cmd = Twist()
        if self.latest_nav_cmd is not None:
            cmd.linear.x = self.latest_nav_cmd.linear.x
            cmd.linear.y = self.latest_nav_cmd.linear.y
            cmd.linear.z = self.latest_nav_cmd.linear.z
            cmd.angular.x = self.latest_nav_cmd.angular.x
            cmd.angular.y = self.latest_nav_cmd.angular.y
            cmd.angular.z = self.latest_nav_cmd.angular.z

        if mode == "REJECT":
            cmd.linear.x = 0.0
            cmd.angular.z = float(self.get_parameter("reject_angular_z").value)
        elif mode == "RECOVER":
            cmd.linear.x = float(self.get_parameter("recover_linear_x").value)
            cmd.angular.z = float(self.get_parameter("recover_angular_z").value)
        elif mode == "ALIGN":
            limit = float(self.get_parameter("align_linear_limit").value)
            cmd.linear.x = min(cmd.linear.x, limit)
        elif mode == "COMMIT":
            limit = float(self.get_parameter("commit_linear_limit").value)
            cmd.linear.x = min(cmd.linear.x, limit)
        return cmd

    def _timer_callback(self) -> None:
        if self.latest_scan is None:
            return
        current_vx = 0.0
        current_wz = 0.0
        if self.latest_odom is not None:
            current_vx = float(self.latest_odom.twist.twist.linear.x)
            current_wz = float(self.latest_odom.twist.twist.angular.z)

        previous_action_vx = 0.0
        previous_action_wz = 0.0
        if self.latest_nav_cmd is not None:
            previous_action_vx = float(self.latest_nav_cmd.linear.x)
            previous_action_wz = float(self.latest_nav_cmd.angular.z)

        features = features_from_scan(
            self.latest_scan,
            current_vx=current_vx,
            current_wz=current_wz,
            previous_action_vx=previous_action_vx,
            previous_action_wz=previous_action_wz,
            stuck_score=self._stuck_score(),
        )
        decision = self.navigator.decide(features)

        decision_msg = String()
        decision_msg.data = decision.mode
        self.decision_pub.publish(decision_msg)

        can_pass_msg = Bool()
        can_pass_msg.data = decision.can_pass
        self.can_pass_pub.publish(can_pass_msg)

        risk_msg = Float32()
        risk_msg.data = decision.risk
        self.risk_pub.publish(risk_msg)

        self.safety_cmd_pub.publish(self._make_safety_cmd(decision.mode))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MemoryDecisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
