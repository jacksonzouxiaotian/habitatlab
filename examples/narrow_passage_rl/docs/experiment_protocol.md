# Experiment Protocol

## Benchmarks

### Synthetic Narrow-Passage Benchmark

Four scenario families are used:

| Scenario | Purpose |
|---|---|
| Straight Passage | Basic narrow-passage traversal |
| Angled Entrance | Entrance-angle and alignment robustness |
| Asymmetric Obstacles | Uneven left/right clearance |
| Multi-Passage Choice | Avoid repeated failed passages with memory |

Each family is evaluated at width/body ratios:

```text
0.70, 0.75, 0.85, 0.95, 1.10
```

### Habitat HM3D NarrowPassageNav-v0

Habitat episodes are mined from HM3D scenes using geodesic and clearance
criteria.  The current mined validation set contains 151 episodes from 20 held
out scenes, plus 24 extreme-narrow episodes.

## Four-Layer Baseline Suite

The baseline suite is organized into four layers.  The paper claim is not only
"ours beats DWB"; the claim is that near the geometric feasibility boundary,
methods without explicit failure memory and geometry-aware risk tend to hesitate,
retry failed passages, or make unsafe local decisions.

| Layer | Purpose | Methods |
|---|---|---|
| Traditional planning | Common real-robot navigation stacks | DWB, TEB, RPP, MPPI, Smac Hybrid-A* / State Lattice |
| Learning navigation | Ordinary RL/IL without explicit failure memory | PPO-depth, PPO-geometry, Recurrent PPO, SAC/TD3, BC/DAgger |
| Memory/history methods | Distinguish generic history from failure memory | GRU-PPO, kNN failure memory, replay-memory policy, Transformer history, vanilla episodic memory |
| Recent strong navigation | Modern visual/semantic/diffusion-style comparison | ViPlanner-style, NoMaD-style, ViNT/GNM-style, quadruped confined-space style baselines |

Implementation status and exact method names are listed in
`docs/baseline_taxonomy.md` and `configs/eval_baselines.yaml`.

## PPO / RL Baseline Reporting

PPO-family results must be reported under one of three categories:

| Category | Meaning |
|---|---|
| Synthetic PPO | SB3/SB3-Contrib training and evaluation on synthetic v2 |
| Synthetic-to-Habitat PPO | SB3 checkpoint trained on synthetic v2 and evaluated on HM3D mined-val |
| Habitat-Baselines PPO smoke test | Habitat task/policy wiring check using `ppo_narrow_passage.yaml` |

Do not describe the Habitat-Baselines smoke config as a complete paper PPO
training curve.  The paper-facing learning baselines are the SB3/SB3-Contrib
scripts in `examples/narrow_passage_rl/`.

## Curriculum Status

`configs/train_ppo.yaml` documents a planned width/body-ratio curriculum, but
`train_sb3_v2.py` does not parse that YAML and does not currently implement
staged curriculum scheduling.  Until a staged runner is added, papers and README
text should not claim curriculum learning for PPO/SAC/TD3/RecurrentPPO runs.

## Core Ablations

| Ablation | Question answered |
|---|---|
| Ours w/o Memory | Is failure memory useful beyond geometry/risk? |
| Ours w/o Geometry | Are explicit passage features necessary? |
| Ours w/o Risk Head | Does risk estimation improve boundary decisions? |
| Ours w/o Recovery | Does recovery behavior matter after stuck/oscillation? |
| Ours Full | Full Geometry-Guided Failure Memory method |

## Memory Comparison Matrix

| Method | Has history | Has failure labels | Geometry similarity | Decision mode |
|---|---:|---:|---:|---:|
| GRU-PPO | yes | no | no | no |
| kNN Failure Memory | yes | yes | partial | no |
| Vanilla Episodic Memory | yes | yes | no | no |
| Geometry-Guided Failure Memory | yes | yes | yes | yes |

## Metrics

| Metric | Definition |
|---|---|
| Success Rate | Reaches the goal and calls stop within the success radius |
| Collision Rate | Any collision or body-intersection event |
| Near Collision Rate | Clearance below the safety threshold |
| Oscillation Count | Repeated left/right command sign changes near the entrance |
| Stuck Rate | Timeout or low progress for the stuck window |
| Recovery Success | Recovers after entering Recover mode |
| Min Clearance | Minimum body clearance over the episode |
| Time to Goal | Number of steps or seconds until success |
| SPL | Success weighted by path efficiency |
| Repeated Failure Rate | Re-enters a passage with prior failed memory |

## Seeds and Repetitions

- Synthetic benchmark: 3 seeds by default (`0, 1, 2`), 500 episodes per seed.
- Habitat deterministic methods: single deterministic pass over each split.
- RL Habitat transfer: checkpoint-specific deterministic evaluation.
- Learning baselines should report multiple seeds when used as strong baselines.
  Single-seed or smoke-test rows must be labeled as such.

## Robot Body Width

The robot body width is treated as a physical parameter and is used to compute:

```text
body_margin = passage_width / 2 - robot_body_width / 2
width_body_ratio = passage_width / robot_body_width
```

The default body width is `0.36 m` unless an experiment config overrides it.

## Outcome Definitions

- `success`: goal reached and stop action called.
- `collision`: simulator collision, body intersection, or negative body margin
  depending on the benchmark.
- `near_collision`: body margin below `0.05 m`.
- `oscillation`: repeated alternation of steering direction near the entrance.
- `stuck`: low progress or high stuck score for the configured window.
- `recovery`: the controller enters Recover mode and later returns to Commit or
  Explore.
- `reject`: the memory/risk gate refuses to enter a passage.

## Reporting

Paper tables should report both aggregate rates and per-difficulty breakdowns.
For Habitat FSM rows with 100% SR, also report the mined-anchor protocol,
success/collision definitions, and stress-test variants.  Current stress tests
include +60 degree initial yaw on extreme-narrow anchors; planned harder tests
include randomized yaw, lateral offset, dynamic obstacles, additional anchors,
and false-feasible anchors.
For memory experiments, always include repeated-failure rate and wasted steps on
false-feasible passages.
