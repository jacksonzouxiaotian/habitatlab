# narrow_memory_nav

ROS 2 Humble wrapper for the narrow-passage failure memory logic in this repository.

## Build

```bash
cd /home/xiaotian/navigation/habitat-lab/ros2_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install
source install/setup.bash
```

## Run in observe mode

```bash
ros2 launch narrow_memory_nav narrow_memory_nav.launch.py
ros2 topic echo /narrow_decision
ros2 topic echo /narrow_can_pass
ros2 topic echo /narrow_risk
ros2 topic echo /safety_cmd_vel
```

By default `safety_fusion_node.enabled` is `false`, so the package does not publish
to the real `/cmd_vel` control path unless you explicitly enable it.

## Nav2 wiring

Remap Nav2's controller output from `/cmd_vel` to `/nav2_cmd_vel`, then inspect
`/safety_cmd_vel`. Only after simulation checks should you enable forwarding:

```bash
ros2 param set /safety_fusion_node enabled true
```

For Lite3 or another quadruped, remap the odometry and lidar topics in the launch
file, for example `/odom` to `/leg_odom2` and `/scan` to the lidar scan topic.
