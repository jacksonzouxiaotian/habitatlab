# Geometry-Guided Failure-Aware Narrow-Passage Navigation

This repository is a fork of
[Habitat-Lab](https://github.com/facebookresearch/habitat-lab).  The fork keeps
the Habitat-Lab v0.3.3 structure, but the narrow-passage research code is
isolated in a small set of added directories.

The paper studies local navigation for quadruped robots near geometric
feasibility limits: tight entrances, low clearance, misleading passable-looking
openings, and repeated failed attempts.  The main system is not a general
PointNav/ObjectNav policy.  It is a geometry-guided, risk-aware, failure-aware
local navigation framework.

## What This Repository Adds

The primary contribution lives here:

```text
examples/narrow_passage_rl/
  procedural_env_v2.py                  # Synthetic v2 narrow-passage benchmark
  eval_harder_benchmark.py              # Main procedural benchmark
  eval_habitat_geometry_fsm.py          # Geometry-FSM on Habitat HM3D anchors
  eval_habitat_stress_validation.py     # Habitat stress validation
  eval_repeated_failure_memory.py       # Repeated infeasible commitment memory test
  eval_habitat_sb3.py                   # SB3 PPO/SAC/TD3 Habitat evaluation
  train_sb3_v2.py                       # PPO/SAC/TD3 synthetic v2 training
  train_recurrent_ppo_v2.py             # RecurrentPPO smoke baseline
  train_bc_dagger_v2.py                 # BC/DAgger smoke baselines
  train_replay_memory_policy_v2.py      # Generic replay-memory smoke baseline
  failure_memory.py                     # Episode-local passage memory
  cross_episode_memory.py               # Cross-episode failure memory
  fair_reward_wrapper.py                # Dense/fair reward wrapper for RL baselines
  results/narrow_passage_rl/            # CSV, Markdown, LaTeX tables

habitat-lab/habitat/tasks/narrow_passage/
  narrow_passage_task.py                # NarrowPassageNav-v0 task
  sensors.py                            # 19-D geometry feature sensor
  rewards.py                            # Narrow-passage reward/measures
  geometry.py                           # Depth-to-geometry feature extraction

habitat-baselines/habitat_baselines/
  config/narrow_passage/                # Narrow-passage PPO configs
  rl/ppo/narrow_passage_policy.py       # Low-dimensional geometry policy
```

Everything else is inherited Habitat-Lab infrastructure unless explicitly noted.

## Main Claim

Near the boundary of geometric feasibility, learning-only policies and generic
history mechanisms are brittle.  The central claim is that explicitly modeling
passage geometry, clearance risk, decision mode, and failure history gives a
more interpretable and stable local navigation system.

The method uses:

- A 19-D depth-derived geometry feature vector.
- A mode-switching controller: align, commit, explore, recover, reject.
- Clearance/risk-aware decisions rather than only goal distance.
- Cross-episode failure memory to avoid repeated commitment to infeasible
  passages.

RL is used as a fair baseline and diagnostic local skill, not as the primary
paper contribution.

## Main Experiments

### 1. Procedural v2 Benchmark

Synthetic v2 includes straight, L-shaped, S-shaped, narrow-entry, narrow-exit,
asymmetric, and false-feasible corridors.

Main result:

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rule baseline | 25.4 | 60.5 | 8.2 | 6.5 | 27.2 | 24.8 | 33.5 | 0.0 |
| Geometry-FSM | 70.3 | 92.8 | 79.6 | 70.0 | 96.7 | 61.4 | 50.6 | 0.0 |

Removing alignment under entry jitter drops the FSM from 64.1% to 19.1% overall,
which identifies heading/lateral alignment as a critical local-control module.

Primary files:

- `examples/narrow_passage_rl/eval_harder_benchmark.py`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_main.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_ablation.md`

### 2. Habitat HM3D Nominal Anchor Validation

Habitat experiments use mined HM3D narrow-passage anchors and the
`NarrowPassageNav-v0` task.  The nominal anchor table checks whether the
geometry controller works on scanned scenes when the mined starts are mostly
well aligned.

| Method | Setting | Success |
|---|---|---:|
| PPO v2 geometry, 3 seeds | Synthetic-to-Habitat transfer | 2.1% +/- 2.6% |
| SAC v2 geometry | Synthetic-to-Habitat transfer | 0.0% |
| TD3 synthetic-to-Habitat | Synthetic-to-Habitat transfer | 2.0% |
| APF+Gap | Classical depth/GPS baseline | 93.6% |
| Geometry-FSM | Nominal HM3D anchors | 100.0% |

Important interpretation: the 100% Geometry-FSM result is nominal anchor
validation, not a complete robustness proof.

Primary files:

- `examples/narrow_passage_rl/eval_habitat_geometry_fsm.py`
- `examples/narrow_passage_rl/eval_habitat_apf_gap.py`
- `examples/narrow_passage_rl/eval_habitat_sb3.py`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_habitat.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_formal_baselines.md`

### 3. Habitat Stress Validation

Nominal HM3D anchors are mostly well aligned, so stress validation is used to
expose module sensitivity under perturbation.

The stress protocol evaluates:

- Start yaw perturbation: 0, 30, 60 degrees.
- Lateral start offset: 0.0 m, 0.10 m, 0.20 m.
- Depth-sector dropout: 0%, 10%, 30%.
- Feature Gaussian noise: sigma = 0.0, 0.02, 0.05.
- Extreme-narrow subset: `body_margin < 0.05 m`.

The +60 degree extreme-narrow slice exposes heading alignment as the key module:
full FSM succeeds, while removing heading alignment fails.

Primary files:

- `examples/narrow_passage_rl/eval_habitat_stress_validation.py`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_habitat_stress.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_stress_validation.csv`

### 4. Strict-Safety RL Diagnostic

TD3 trained directly in Habitat can reach 100% nominal success, but only 2.0%
strict clearance-aware success.  98.0% of episodes are success-but-unsafe and
near-collision rate is 100%.

This diagnostic is included to show why nominal goal-reaching success is not a
sufficient metric for narrow-passage traversal.

Primary files:

- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_diagnostic_baselines.md`
- `examples/narrow_passage_rl/eval_habitat_sb3.py`

### 5. Repeated Failure Memory

Failure memory does not improve one-shot nominal Habitat success.  Its
contribution is to suppress repeated commitments to previously failed infeasible
passages.

Repeated false-feasible evaluation uses 5 rounds of the same 20 passable and 15
false-feasible corridors.

| Method | Passable SR | Passable false reject | Final FF reject | Wasted FF steps | Steps saved |
|---|---:|---:|---:|---:|---:|
| no_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 |
| local_intra_episode_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 |
| kNN Failure Memory | 0.860 | 0.060 | 1.000 | 5500 | 11000 |
| Vanilla Episodic Memory | 0.820 | 0.130 | 1.000 | 8140 | 8360 |
| Geometry-Guided Cross-Episode Failure Memory | 0.900 | 0.030 | 1.000 | 4400 | 12100 |

Primary files:

- `examples/narrow_passage_rl/eval_repeated_failure_memory.py`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_repeated_failure_memory.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/repeated_failure_memory.csv`

### 6. D_min Calibration

The robot body-width threshold can be calibrated from outcomes.  Starting from a
wrong conservative estimate, the Bayesian calibrator converges from 0.56 m to
0.39 m within roughly 75 episodes.

| Agent | Success | Reject rate |
|---|---:|---:|
| Oracle D_true = 0.36 m | 94.0% | 0% |
| Fixed wrong D_hat = 0.56 m | 67.7% | 32% |
| Calibrated | 89.3% | 8% |

Primary files:

- `examples/narrow_passage_rl/eval_dmin_calibration.py`
- `examples/narrow_passage_rl/dmin_calibrator.py`
- `examples/narrow_passage_rl/results/narrow_passage_rl/dmin_calibration.png`

## Formal, Diagnostic, And Smoke Baselines

The learning baselines are intentionally split by evidential role.

Formal main baselines:

- PPO v2 geometry sensor, 3 seeds, synthetic-to-Habitat transfer.
- SAC v2 geometry sensor.
- TD3 synthetic-to-Habitat transfer.
- APF+Gap classical baseline.
- Geometry-FSM.

Diagnostic baselines:

- TD3 Habitat-native nominal success vs strict clearance-aware success.

Smoke / appendix-only baselines:

- GRU-PPO lightweight.
- RecurrentPPO smoke.
- BC-FSM.
- DAgger-FSM.
- Replay Memory Policy.

Legacy PPO runs with the v1 `obs[10]` yaw/heading mismatch are excluded from
formal baseline tables.

Tables:

- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_formal_baselines.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_diagnostic_baselines.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_smoke_baselines.md`

## Current RL Interpretation

The current RL results should be read as baseline and diagnostic evidence, not
as the main narrow-passage solution.

The fair comparison uses the same 19-D geometry sensor for PPO/SAC/TD3.  PPO v2
geometry transfer reaches 2.1% +/- 2.6% on Habitat, SAC reaches 0.0%, and TD3
synthetic-to-Habitat reaches 2.0% despite 77.8% success in the synthetic v2
environment.  This supports the paper's claim that learning-only policies are
fragile near geometric feasibility boundaries.

The Habitat-native TD3 result is intentionally reported as a diagnostic:

| Method | Nominal success | Strict success | Unsafe success | Interpretation |
|---|---:|---:|---:|---|
| TD3 Habitat-native | 100.0% | 2.0% | 98.0% | RL can exploit nominal goal-reaching without safe clearance |

The split between nominal and strict success occurs because nominal success only
checks the local goal/alignment condition, while strict success also requires
clearance-safe traversal.  Therefore RL's 100% nominal result should be used to
motivate strict clearance-aware evaluation, not to claim that RL safely solved
the task.

The next RL improvement should be framed as **risk-constrained local RL skill**:

- train with strict-success reward, where success-but-unsafe does not receive
  the terminal success bonus;
- add an action shield that limits forward motion or triggers recovery when
  `body_margin`, left/right clearance, or heading alignment are unsafe;
- condition the policy on the FSM mode so RL learns local control inside
  `ALIGN`, `COMMIT`, `EXPLORE`, and `RECOVER` rather than replacing the
  geometry/risk decision layer.

This keeps the RL contribution aligned with the paper: RL is useful as a local
skill and safety diagnostic, while explicit geometry, risk, and failure memory
remain the core method.

## What Not To Overclaim

- HM3D nominal 100% is not a complete robustness proof.  It is nominal anchor validation on
  mined, mostly well-aligned starts.
- Habitat stress validation is the evidence for module sensitivity under yaw,
  lateral, dropout, noise, and extreme-clearance perturbations.
- Failure memory is not for improving one-shot passable-anchor success.  It
  reduces repeated infeasible commitment and wasted attempts.
- Smoke RL baselines are not main baselines.  They verify code paths and belong
  in appendix/status tables unless rerun under the full protocol.
- Real-robot experiments should be described as pilot hardware validation unless
  accompanied by full videos, logs, bags, and statistics.

## Reproduction Commands

Run from the repository root.

```bash
# 1. Procedural v2 benchmark
python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500

# 2. Habitat stress validation
python examples/narrow_passage_rl/eval_habitat_stress_validation.py \
    --preset paper \
    --split val \
    --num-episodes -1

# 3. Repeated failure memory
python examples/narrow_passage_rl/eval_repeated_failure_memory.py \
    --n-rounds 5 \
    --n-passable 20 \
    --n-ff 15 \
    --max-steps 220

# 4. D_min calibration
python examples/narrow_passage_rl/eval_dmin_calibration.py

# 5. Regenerate paper tables from existing CSV artifacts
python examples/narrow_passage_rl/make_paper_tables.py
```

Habitat commands require a working Habitat/HM3D installation and the generated
narrow-passage dataset under:

```text
data/datasets/narrow_passage/{split}/{split}.json.gz
```

## More Documentation

- Detailed experiment scripts and tables:
  `examples/narrow_passage_rl/README.md`
- Method notes:
  `examples/narrow_passage_rl/docs/method.md`
- Reproducibility notes:
  `examples/narrow_passage_rl/docs/reproducibility.md`
- Experiment protocol:
  `examples/narrow_passage_rl/docs/experiment_protocol.md`

## Habitat-Lab Base

This repository remains a Habitat-Lab fork.  Please cite Habitat-Lab and
Habitat-Sim when using the underlying simulator infrastructure.
