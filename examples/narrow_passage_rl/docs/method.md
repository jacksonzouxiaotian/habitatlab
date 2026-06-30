# Method

The method is a geometry-guided, memory-aware navigation framework for narrow
passages.  It is decomposed into five explicit modules so that each component can
be tested independently.

## 1. Passage Geometry Encoder

At each step, the robot extracts an interpretable passage geometry vector:

```text
g_t = [
  passage_width,
  robot_body_width,
  clearance_left,
  clearance_right,
  min_clearance,
  width_body_ratio,
  entrance_angle,
  obstacle_asymmetry,
  local_goal_angle,
  velocity_history,
  stuck_score,
]
```

The encoder maps this vector to a compact embedding:

```text
z_g = GeometryEncoder(g_t)
```

This makes the method geometry-guided rather than a generic RGB-D policy.

## 2. Risk-Aware Traversability Estimator

The traversability estimator outputs:

```text
p_pass = P(passable | observation, geometry, memory)
r_risk = risk score
```

Risk is an interpretable fusion of learned and rule-based signals:

```text
risk =
  alpha * network_risk +
  beta  * memory_risk +
  gamma * clearance_risk +
  delta * stuck_risk
```

The current implementation uses explicit geometry and rule-based risk terms; the
`risk_head.py` module is the stable interface for learned risk heads.

## 3. Failure Memory Bank

After each episode or robot trial, the memory bank records:

```text
memory_item = {
  geometry_embedding,
  scene_id,
  passage_width,
  width_body_ratio,
  action_mode,
  outcome,
  min_clearance,
  oscillation_count,
  final_pose_error,
}
```

Retrieval uses geometry-aware similarity:

```text
sim =
  w1 * cosine(z_current, z_memory)
  - w2 * abs(width_ratio_current - width_ratio_memory)
  - w3 * abs(entrance_angle_current - entrance_angle_memory)
  - w4 * abs(clearance_asymmetry_current - clearance_asymmetry_memory)
```

The memory risk is the weighted failure rate of the retrieved top-k cases.

## 4. Decision Mode

The policy produces a high-level mode rather than directly outputting only
velocities.

| Mode | Trigger | Behavior |
|---|---|---|
| Commit | high `p_pass`, low risk | Stable traversal |
| Explore | uncertain, acceptable risk | Slow probing and alignment |
| Recover | stuck or oscillating | Back up and re-align |
| Reject | high historical failure or infeasible geometry | Do not enter |

## 5. Mode-Conditioned Controller

The low-level controller is conditioned on the selected mode:

```text
u_t = controller(mode, local_goal, depth, clearance)
```

| Mode | Controller |
|---|---|
| Commit | Normal navigation / learned policy |
| Explore | Low speed, heading alignment, safety filter |
| Recover | Reverse and rotate to re-localize |
| Reject | Stop or request global re-planning |

This decomposition is intended to transfer cleanly to ROS2/Nav2 and quadruped
robot execution.
