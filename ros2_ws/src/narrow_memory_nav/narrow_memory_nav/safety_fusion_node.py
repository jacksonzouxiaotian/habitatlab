import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class SafetyFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("safety_fusion_node")
        self.declare_parameter("enabled", False)
        self.latest_safety_cmd: Twist | None = None
        self.create_subscription(
            Twist, "/safety_cmd_vel", self._safety_cmd_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(0.05, self._timer_callback)
        self.get_logger().info("safety_fusion_node started")

    def _safety_cmd_callback(self, msg: Twist) -> None:
        self.latest_safety_cmd = msg

    def _timer_callback(self) -> None:
        if not bool(self.get_parameter("enabled").value):
            return
        if self.latest_safety_cmd is not None:
            self.cmd_pub.publish(self.latest_safety_cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
