# Baseline Taxonomy

Baselines are grouped by evidential role.  The main paper table should not mix
formal baselines with diagnostic or smoke runs.

## Formal Main Baselines

These baselines are suitable for the main comparison table.

| Family | Method | Train domain | Eval domain | Purpose |
|---|---|---|---|---|
| Classical | APF+Gap | none | Habitat HM3D | Classical depth/GPS reactive navigation |
| Learning | PPO v2 geometry sensor | Synthetic v2 | Habitat HM3D | On-policy learning-only transfer baseline |
| Learning | SAC v2 geometry sensor | Synthetic v2 | Habitat HM3D | Off-policy continuous-control baseline |
| Learning | TD3 synthetic-to-Habitat | Synthetic v2 | Habitat HM3D | Off-policy deterministic continuous-control baseline |
| Ours | Geometry-FSM | none | Synthetic v2 / Habitat HM3D | Geometry/risk/mode controller |

Legacy PPO with the v1 `obs[10]` yaw/heading mismatch is excluded from formal
baseline tables and is provenance only.

Formal table:

```text
results/narrow_passage_rl/paper_table_formal_baselines.md
```

## Diagnostic Baselines

Diagnostic baselines explain failure modes or metric weaknesses.

| Method | Why diagnostic |
|---|---|
| TD3 Habitat-native | Shows that nominal success can be exploited; strict clearance-aware success remains low. |
| Habitat-Baselines PPO config | Task/policy wiring smoke test, not a final PPO curve. |

Diagnostic table:

```text
results/narrow_passage_rl/paper_table_diagnostic_baselines.md
```

## Smoke / Appendix-Only Learning Baselines

These verify code paths and provide preliminary negative controls.  They should
not be used as main baselines unless rerun with the full multi-seed protocol.

| Method | Status | Purpose |
|---|---|---|
| GRU-PPO lightweight | smoke | Generic recurrent hidden-state negative control |
| RecurrentPPO | smoke / single run | SB3-Contrib recurrent policy path |
| BC-FSM | smoke | Imitation from Geometry-FSM expert |
| DAgger-FSM | smoke | Interactive imitation baseline |
| Replay Memory Policy | smoke | Generic history/replay observation baseline |

Smoke table:

```text
results/narrow_passage_rl/paper_table_smoke_baselines.md
```

## Memory Baselines

The memory claim is not one-shot success improvement.  The correct comparison is
repeated false-feasible exposure and wasted-step reduction.

| Method | History | Failure labels | Geometry similarity | Decision role |
|---|---:|---:|---:|---:|
| no memory | no | no | no | always attempt |
| local intra-episode memory | within episode | yes | coarse | recovery only, no cross-round reject |
| kNN failure memory | cross episode | yes | partial hand geometry | reject |
| vanilla episodic memory | cross episode | yes | embedding similarity | reject |
| geometry-guided cross-episode failure memory | cross episode | yes | explicit geometry/type similarity | reject / cautious mode |

Memory table:

```text
results/narrow_passage_rl/paper_table_repeated_failure_memory.md
```

## FSM Ablations

FSM ablations test which controller modules matter under perturbation.

| Ablation | Removed component | Interpretation |
|---|---|---|
| w/o recovery | Backward recovery after collision/stuck | Tests recovery contribution |
| w/o heading alignment | Goal/corridor heading correction | Critical under yaw perturbation |
| w/o lateral alignment | Centerline lateral correction | Tests side-offset centering |

In nominal HM3D anchors, ablations may also reach 100% because starts are mostly
well aligned.  Use Habitat stress validation to expose differences.

## Out-of-Scope Or External Baselines

ROS/Nav2 planners such as DWB, TEB, RPP, MPPI, and Smac Hybrid-A* are relevant
real-robot baselines, but they require ROS2/Nav2 integration or log replay and
are not part of the current Habitat main table.

Foundation or diffusion navigation models such as ViNT/GNM/NoMaD/ViPlanner can
be discussed as related work or future stronger baselines unless full task-
matched checkpoints and evaluation scripts are available.
