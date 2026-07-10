# Experiment Protocol

This document defines the splits, metrics, and reporting rules used by the
narrow-passage experiments.  It follows the same interpretation as the root
README: Habitat nominal 100% is anchor validation, not a complete robustness
proof, and memory is evaluated through repeated infeasible commitment reduction.

## Dataset Splits

### Procedural v2

`procedural_env_v2.py` generates the synthetic benchmark.  The main benchmark
contains seven corridor types:

| Type | Purpose |
|---|---|
| straight | Basic passage traversal |
| l_shaped | Junction following and turn commitment |
| s_shaped | Multi-turn local geometry |
| narrow_entry | Entrance alignment |
| narrow_exit | Exit clearance |
| asymmetric | Uneven left/right clearance |
| false_feasible | Looks feasible at entry but is physically infeasible |

Main table protocol:

- 500 episodes per seed.
- 3 seeds for the main synthetic table.
- Report mean and standard deviation across seeds.

Repeated-failure memory protocol:

- 5 repeated rounds.
- 20 passable corridors and 15 false-feasible corridors per round.
- Same corridor seeds are replayed across rounds.
- Report wasted false-feasible steps and steps saved vs no memory.

### Habitat HM3D

Habitat uses the custom `NarrowPassageNav-v0` task and mined HM3D anchors.

| Split | Episodes | Scenes | Use |
|---|---:|---:|---|
| Val set A | 157 | 20 held-out HM3D scenes | Original Habitat anchor table |
| Val set B | 151 | 20 held-out HM3D scenes | Current mined validation split |
| Extreme-narrow subset | 24 | subset of Val anchors | `body_margin < 0.05 m` stress slice |

The mined anchors are mostly well aligned.  Therefore the nominal Habitat table
is called **Habitat HM3D Anchor Validation**.

## Seeds

| Experiment | Seeds |
|---|---|
| Synthetic v2 main benchmark | `0, 1, 2` |
| Repeated failure memory | deterministic fixed corridor seeds, replayed by round |
| Habitat deterministic FSM/APF | deterministic pass over episode split |
| PPO v2 geometry transfer | 3 seeds, reported as 5.7%, 0.0%, 0.6% |
| SAC/TD3 transfer rows | single checkpoint unless otherwise stated |
| Smoke baselines | single seed unless explicitly rerun |

Single-seed or smoke rows must be labeled as such and should not be mixed into
the formal main baseline table.

## Difficulty Bins

Habitat difficulty is derived from `body_margin`.

| Bin | Definition |
|---|---|
| narrow | `body_margin <= 0.15 m` |
| normal | `0.15 m < body_margin <= 0.40 m` |
| wide | `body_margin > 0.40 m` |
| extreme-narrow | `body_margin < 0.05 m` |

## Metric Definitions

| Metric | Definition |
|---|---|
| success | Reaches the local goal under the task's distance/alignment condition. |
| strict success | Success plus no collision/stuck and clearance above the strict threshold. |
| collision | Simulator collision flag or task collision measure. With `allow_sliding=False`, boundary stopping may suppress physical penetration. |
| near collision | Minimum body margin below the safety threshold, default `0.05 m`. |
| minimum clearance | Minimum observed `body_margin` over the episode. Negative values in Habitat can arise from depth artifacts near walls and should be reported transparently. |
| timeout | Reaches `max_steps` without success, collision, or stuck termination. |
| stuck | Low progress or high stuck score under the task/window definition. |
| passable false reject | A method rejects a passable corridor before attempting it. |
| false-feasible reject | A method rejects an infeasible false-feasible corridor before committing to it. |
| wasted FF steps | Steps spent attempting false-feasible passages that are not rejected. |
| retrieval precision | Among rejected passages, the fraction that are truly false-feasible. |

The strict-success diagnostic is especially important for learning baselines.
TD3 Habitat-native reaches high nominal success but low strict success, showing
that nominal goal-reaching alone is insufficient for this task.

## False-Feasible Definition

In procedural v2, a false-feasible passage is an entrance that appears locally
passable but becomes blocked or too narrow inside.  The correct behavior is not
to count success; repeated-failure memory should reduce future commitment to
the same infeasible passage.

In Habitat, false-feasible anchors are not currently part of the main mined
HM3D split.  Do not claim Habitat false-feasible results unless a separate
mined/labeled dataset and CSV exist.

## Reporting Rules

- Report Habitat nominal 100% as **nominal anchor validation**.
- Pair nominal Habitat results with stress validation.
- Report strict success for RL diagnostics.
- Report repeated-failure metrics for memory claims.
- Keep smoke baselines in appendix/status tables unless rerun with the full
  multi-seed, same-eval-domain protocol.
