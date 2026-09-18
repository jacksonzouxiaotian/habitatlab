import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("narrow_memory_nav")
    config_file = os.path.join(pkg_dir, "config", "narrow_memory_nav.yaml")

    # ── Launch arguments (override on the command line) ───────────────────────
    variant_arg = DeclareLaunchArgument(
        "variant", default_value="full",
        description="FSM variant: full | no_recovery | no_alignment"
    )
    enabled_arg = DeclareLaunchArgument(
        "enabled", default_value="true",
        description="Set false to disable velocity output (safe stop)"
    )

    return LaunchDescription([
        variant_arg,
        enabled_arg,

        # ── Main FSM navigation node ──────────────────────────────────────────
        Node(
            package="narrow_memory_nav",
            executable="fsm_nav_node",
            name="fsm_nav_node",
            output="screen",
            parameters=[
                config_file,
                {
                    "variant": LaunchConfiguration("variant"),
                    "enabled": LaunchConfiguration("enabled"),
                },
            ],
            # ── Topic remapping ───────────────────────────────────────────────
            # Adjust these to match your robot's actual topic names.
            # Examples:
            #   Go1/Go2: scan may be /scan or /livox/scan
            #   Goal: Nav2 publishes /goal_pose by default
            #   cmd_vel: may be /cmd_vel or /robot/cmd_vel
            remappings=[
                ("/scan",       "/scan"),
                ("/odom",       "/odom"),
                ("/goal_pose",  "/goal_pose"),
                ("/cmd_vel",    "/cmd_vel"),
            ],
        ),
    ])
