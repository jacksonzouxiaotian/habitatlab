# Baseline Taxonomy

The paper baseline suite is organized into four layers.  The goal is not merely
to beat classical navigation, but to show that near the geometric feasibility
boundary, standard planners, standard RL, generic history models, and recent
visual navigation models all lack explicit failure memory and geometry-aware
risk reasoning.

## Layer 1: Traditional Planning

| Baseline | Why it matters | Habitat-Lab implementation plan |
|---|---|---|
| DWB | Nav2 velocity-space local planner | ROS2/Nav2 adapter; replay same passage starts/goals |
| TEB | Time-elastic-band optimization | ROS2 adapter or external log replay |
| RPP | Robust path-following controller in Nav2 | ROS2/Nav2 adapter |
| MPPI | Sampling/MPC local controller | ROS2/Nav2 MPPI adapter |
| Smac Hybrid-A* / State Lattice | Kinodynamic global planning | Nav2 Smac global path + local controller evaluation |

These baselines answer whether a stronger classical or kinodynamic planner is
enough.  The expected failure mode is still local hesitation, unsafe clearance,
or repeated entry into a historically failed passage.

## Layer 2: Learning-Based Navigation

| Baseline | Inputs | Purpose | Status |
|---|---|---|---|
| PPO-depth | depth + local goal | End-to-end RL without explicit geometry | planned |
| PPO-geometry | 19-D geometry vector | Geometry input without failure memory | implemented/evaluated |
| Recurrent PPO / GRU-PPO | geometry + hidden state | Tests whether generic temporal memory is enough | SB3-Contrib smoke evaluated; full training pending |
| Geometry + GRU PPO | geometry + recurrent state | Stronger recurrent RL baseline | SB3-Contrib / lightweight fallback evaluated |
| SAC / TD3 | geometry + continuous action | Off-policy continuous-control baselines | SAC evaluated; TD3 entry point implemented |
| BC / DAgger | expert trajectories | Imitation from planner/FSM expert | smoke evaluated; full training pending |

The key comparison is:

```text
PPO
PPO + GRU
PPO + Geometry
PPO + Geometry + GRU
Ours w/o Failure Memory
Ours Full
```

## Layer 3: Memory / History Methods

| Baseline | Has history | Has failure labels | Geometry similarity | Decision mode |
|---|---:|---:|---:|---:|
| GRU-PPO | yes | no | no | no |
| kNN Failure Memory | yes | yes | partial | no |
| Replay Memory Policy | yes | weak | no | no |
| Transformer History | yes | no | no | no |
| Vanilla Episodic Memory | yes | yes | no | no |
| Geometry-Guided Failure Memory | yes | yes | yes | yes |

This is the most important comparison for the paper.  It separates generic
history conditioning from explicit failure memory with geometric similarity and
mode-level decisions.

Current synthetic v2 memory baseline results are available in
`results/narrow_passage_rl/paper_table_memory_baselines.md`.  In the 5-round
repeated-passage benchmark, kNN failure memory and vanilla episodic memory both
learn to reject false-feasible passages, but they reject more passable corridors
and save fewer wasted steps than Geometry-Guided Failure Memory.

Current smoke results for RecurrentPPO, BC, DAgger, and Replay Memory Policy are
available in `results/narrow_passage_rl/paper_table_new_baselines_smoke.md`.
These are deployment checks, not final long-training scores.

## Layer 4: Recent Strong Navigation Methods

These methods should be treated as lightweight style baselines unless a full
model/checkpoint is available.

| Baseline | Why include it | Practical version in this benchmark |
|---|---|---|
| ViPlanner-style learned local planner | Learned local planning with geometry/semantic traversability costs | depth/local map + goal -> waypoint sequence, trained on narrow-passage v2 |
| NoMaD-style diffusion waypoint policy | Modern diffusion navigation policy with history and goal conditioning | observation history + goal -> K-step action/waypoint sequence |
| ViNT/GNM-style visual navigation | Foundation-style visual navigation | optional visual baseline, not the main narrow-geometry comparison |
| Quadruped confined-space RL | Closest to legged narrow-space navigation | hierarchical RL waypoint follower / privileged-geometry RL |

References:

- ViPlanner: https://arxiv.org/abs/2310.00982
- NoMaD: https://arxiv.org/abs/2310.07896
- GNM: https://arxiv.org/abs/2210.03370
- ViNT: https://arxiv.org/abs/2306.14846
- Dexterous legged locomotion in confined 3D spaces: https://arxiv.org/abs/2403.03848
- Quadrupedal narrow pipe inspection: https://arxiv.org/abs/2412.13621

## What Can Be Implemented Directly in Habitat-Lab?

| Feasibility | Baselines |
|---|---|
| Directly runnable in current codebase | PPO-geometry, SAC-geometry, FSM ablations, kNN/geometry failure memory, vanilla episodic memory, SB3-Contrib RecurrentPPO, BC/DAgger, replay-memory policy, lightweight GRU-PPO, synthetic ViPlanner-style waypoint policy |
| Needs modest new code | PPO-depth, Transformer history, NoMaD-style diffusion waypoint policy |
| Needs ROS2/Nav2 bridge | DWB, RPP, MPPI, Smac Hybrid-A* |
| Needs external package/log replay | TEB, full ViNT/GNM/NoMaD/ViPlanner checkpoints |
| Better treated as paper context or style baseline | full foundation navigation models, full quadruped locomotion policies |
