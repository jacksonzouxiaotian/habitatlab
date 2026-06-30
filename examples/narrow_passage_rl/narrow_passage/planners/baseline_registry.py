"""Classical and sampling baseline planner specifications."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselinePlannerSpec:
    name: str
    family: str
    implementation_status: str
    notes: str


BASELINE_PLANNERS = {
    "dwb": BaselinePlannerSpec("DWB", "classical_local_planner", "ros2_adapter", "Nav2 DWB local planner"),
    "teb": BaselinePlannerSpec("TEB", "classical_local_planner", "planned", "Timed Elastic Band baseline"),
    "rpp": BaselinePlannerSpec("RPP", "classical_local_planner", "ros2_adapter", "Regulated Pure Pursuit"),
    "mppi": BaselinePlannerSpec("MPPI", "sampling_mpc_planner", "ros2_adapter", "Nav2 MPPI controller"),
    "ppo": BaselinePlannerSpec("PPO", "learning_baseline", "implemented", "SB3/Habitat PPO baselines"),
    "sac": BaselinePlannerSpec("SAC", "learning_baseline", "implemented", "SB3 SAC v2 baseline"),
    "td3": BaselinePlannerSpec("TD3", "learning_baseline", "train_entrypoint", "Training/eval entry points implemented"),
}

