"""Baseline registry for paper experiments.

Implementation status values:
  implemented       — runnable/evaluated in this repository
  entrypoint        — training/evaluation entry point exists
  planned           — straightforward Habitat-Lab implementation planned
  ros2_adapter      — requires Nav2/ROS2 adapter or log replay
  style_baseline    — lightweight reproduction of a recent method family
  external_optional — full reproduction requires external checkpoints/packages
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    layer: str
    name: str
    purpose: str
    implementation_status: str
    habitat_lab_feasibility: str
    notes: str


BASELINE_SPECS = [
    BaselineSpec(
        "dwb",
        "traditional_planning",
        "DWB",
        "Nav2 velocity-space local planning baseline",
        "ros2_adapter",
        "ROS2/Nav2 bridge",
        "Represents common real-robot local planning.",
    ),
    BaselineSpec(
        "teb",
        "traditional_planning",
        "TEB",
        "Time-elastic-band optimization baseline",
        "external_optional",
        "external adapter/log replay",
        "Useful classical optimizer, but not native Habitat-Lab.",
    ),
    BaselineSpec(
        "rpp",
        "traditional_planning",
        "RPP",
        "Regulated Pure Pursuit path tracking baseline",
        "ros2_adapter",
        "ROS2/Nav2 bridge",
        "Strong stable path follower in Nav2.",
    ),
    BaselineSpec(
        "mppi",
        "traditional_planning",
        "MPPI",
        "Sampling/MPC local controller baseline",
        "ros2_adapter",
        "ROS2/Nav2 bridge",
        "Tests whether sampling optimization handles boundary passages.",
    ),
    BaselineSpec(
        "smac_hybrid_astar",
        "traditional_planning",
        "Smac Hybrid-A* / State Lattice",
        "Kinodynamic global planning baseline",
        "ros2_adapter",
        "ROS2/Nav2 bridge",
        "Tests whether better global planning solves local passage failures.",
    ),
    BaselineSpec(
        "ppo_depth",
        "learning_navigation",
        "PPO-depth",
        "End-to-end RL with depth/local goal",
        "planned",
        "direct Habitat-Lab policy",
        "Shows ordinary RL without explicit geometry.",
    ),
    BaselineSpec(
        "ppo_geometry",
        "learning_navigation",
        "PPO-geometry",
        "RL with explicit passage geometry",
        "implemented",
        "current SB3/Habitat evaluation",
        "Already evaluated as PPO v2 geometry baseline.",
    ),
    BaselineSpec(
        "recurrent_ppo",
        "learning_navigation",
        "Recurrent PPO / GRU-PPO",
        "Generic temporal memory RL baseline",
        "implemented",
        "lightweight PyTorch fallback; SB3-contrib pending",
        "5k-step CPU fallback evaluated; final SB3-Contrib run still pending.",
    ),
    BaselineSpec(
        "sac_geometry",
        "learning_navigation",
        "SAC-geometry",
        "Off-policy continuous-control baseline",
        "implemented",
        "current SB3 evaluation",
        "SAC v2 Habitat mined-val result is 0/151.",
    ),
    BaselineSpec(
        "td3_geometry",
        "learning_navigation",
        "TD3-geometry",
        "Off-policy continuous-control baseline",
        "entrypoint",
        "current training/eval entry point",
        "Training was too slow on CPU in the current pass.",
    ),
    BaselineSpec(
        "bc_dagger",
        "learning_navigation",
        "BC / DAgger",
        "Imitation learning from planner/FSM expert",
        "planned",
        "direct supervised policy",
        "Tests whether imitation of an expert handles boundary cases.",
    ),
    BaselineSpec(
        "knn_failure_memory",
        "memory_history",
        "kNN Failure Memory",
        "Failure retrieval from hand geometry features",
        "implemented",
        "direct current memory module",
        "Evaluated in eval_memory_baselines.py.",
    ),
    BaselineSpec(
        "vanilla_episodic_memory",
        "memory_history",
        "Vanilla Episodic Memory",
        "Embedding-only failure retrieval",
        "implemented",
        "direct current memory module",
        "Evaluated in eval_memory_baselines.py.",
    ),
    BaselineSpec(
        "transformer_history",
        "memory_history",
        "Transformer History",
        "History-conditioned policy without explicit failure bank",
        "planned",
        "direct sequence model",
        "Controls for generic long-horizon history.",
    ),
    BaselineSpec(
        "replay_memory_policy",
        "memory_history",
        "Replay Memory Policy",
        "Episode embedding concatenated to policy input",
        "planned",
        "direct policy wrapper",
        "Controls for simple replay-style historical context.",
    ),
    BaselineSpec(
        "viplanner_style",
        "recent_strong_navigation",
        "ViPlanner-style",
        "Learned local planner with traversability cost",
        "style_baseline",
        "lightweight Habitat-Lab implementation",
        "Depth/local map + goal -> local waypoint sequence.",
    ),
    BaselineSpec(
        "nomad_style",
        "recent_strong_navigation",
        "NoMaD-style diffusion waypoint policy",
        "Diffusion waypoint/action sequence baseline",
        "style_baseline",
        "lightweight synthetic benchmark implementation",
        "History + goal -> K-step action or waypoint sequence.",
    ),
    BaselineSpec(
        "vint_gnm_style",
        "recent_strong_navigation",
        "ViNT/GNM-style visual navigation",
        "Foundation-style visual navigation comparison",
        "external_optional",
        "optional visual baseline",
        "Not the main geometry-boundary comparison.",
    ),
    BaselineSpec(
        "hierarchical_legged_rl",
        "recent_strong_navigation",
        "Hierarchical legged RL waypoint follower",
        "Quadruped confined-space baseline",
        "style_baseline",
        "lightweight waypoint follower",
        "Matches the four-legged narrow-space motivation.",
    ),
    BaselineSpec(
        "privileged_geometry_rl",
        "recent_strong_navigation",
        "Privileged-geometry RL",
        "Quadruped narrow-pipe style baseline",
        "style_baseline",
        "synthetic/Habitat privileged feature policy",
        "Tests whether privileged geometry alone is enough.",
    ),
]


def by_layer(layer: str) -> list[BaselineSpec]:
    return [spec for spec in BASELINE_SPECS if spec.layer == layer]
