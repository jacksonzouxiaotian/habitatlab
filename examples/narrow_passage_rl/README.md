# Geometry-Guided Failure-Aware Narrow-Passage Navigation

This directory contains the narrow-passage research code added on top of the
Habitat-Lab fork.  It is the main entry point for reproducing the paper's
experiments and tables.

## What This Directory Adds

```text
examples/narrow_passage_rl/
  procedural_env_v2.py                  # Synthetic v2 benchmark
  eval_harder_benchmark.py              # Main procedural benchmark
  eval_habitat_geometry_fsm.py          # Geometry-FSM on HM3D anchors
  eval_habitat_apf_gap.py               # APF+Gap Habitat baseline
  eval_habitat_sb3.py                   # PPO/SAC/TD3 Habitat evaluation
  eval_habitat_stress_validation.py     # Formal Habitat stress validation
  eval_repeated_failure_memory.py       # Repeated false-feasible memory test
  eval_dmin_calibration.py              # D_min self-calibration
  train_sb3_v2.py                       # PPO/SAC/TD3 on synthetic v2
  train_recurrent_ppo_v2.py             # RecurrentPPO smoke baseline
  train_bc_dagger_v2.py                 # BC/DAgger smoke baselines
  train_replay_memory_policy_v2.py      # Generic replay-memory smoke baseline
  record_habitat_video.py               # Habitat video/keyframe generation
  results/narrow_passage_rl/            # CSV/Markdown/LaTeX/figures
```

The Habitat task and sensors used by these scripts live in:

```text
habitat-lab/habitat/tasks/narrow_passage/
```

## Main Claim

The paper focuses on local narrow-passage navigation near geometric feasibility
limits.  The main contribution is not a generic RL navigation policy.  The claim
is that explicit passage geometry, clearance risk, decision mode, and failure
history are more stable and interpretable than learning-only or generic-history
baselines in this regime.

## Main Experiments

### 1. Procedural v2 Benchmark

Script:

```bash
python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500
```

What it tests:

- Straight corridors.
- L-shaped and S-shaped corridors.
- Narrow entry and narrow exit.
- Asymmetric obstacles.
- False-feasible, physically infeasible passages.

Key result:

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rule baseline | 25.4 | 60.5 | 8.2 | 6.5 | 27.2 | 24.8 | 33.5 | 0.0 |
| Geometry-FSM | 70.3 | 92.8 | 79.6 | 70.0 | 96.7 | 61.4 | 50.6 | 0.0 |

Output tables:

- `results/narrow_passage_rl/paper_table_main.md`
- `results/narrow_passage_rl/paper_table_ablation.md`

### 2. Habitat HM3D Nominal Anchor Validation

Scripts:

```bash
python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py
python examples/narrow_passage_rl/eval_habitat_apf_gap.py
python examples/narrow_passage_rl/eval_habitat_sb3.py --algo td3
```

What it tests:

- Scene-based validation in HM3D using `NarrowPassageNav-v0`.
- Mined anchors from held-out HM3D scenes.
- Formal learning baselines and classical APF+Gap baseline.

Important interpretation:

The 100% Geometry-FSM rows are nominal anchor validation.  The mined anchors are
mostly well aligned, so this result should not be described as complete robustness.
Stress validation is required to expose module sensitivity.

Main table:

- `results/narrow_passage_rl/paper_table_habitat.md`
- `results/narrow_passage_rl/paper_table_formal_baselines.md`

### 3. Habitat Stress Validation

Script:

```bash
python examples/narrow_passage_rl/eval_habitat_stress_validation.py \
    --preset paper \
    --split val \
    --num-episodes -1
```

Stress settings:

- Start yaw perturbation: 0, 30, 60 degrees.
- Lateral start offset: 0.0 m, 0.10 m, 0.20 m.
- Depth-sector dropout: 0%, 10%, 30%.
- Feature Gaussian noise: sigma = 0.0, 0.02, 0.05.
- Extreme-narrow subset: `body_margin < 0.05 m`.

Compared methods:

- Full Geometry-FSM.
- FSM without heading alignment.
- FSM without recovery.
- FSM without lateral alignment.
- APF+Gap, when the interface is available.

Output:

- `results/narrow_passage_rl/habitat_stress_validation.csv`
- `results/narrow_passage_rl/paper_table_habitat_stress.md`
- `results/narrow_passage_rl/paper_table_habitat_stress.tex`

### 4. Strict-Safety RL Diagnostic

This is a diagnostic table, not the main method ranking.

TD3 trained directly in Habitat reaches 100% nominal success but only 2.0%
strict clearance-aware success.  This demonstrates that nominal goal-reaching
success can be exploited by RL and must be reported together with clearance
safety.

Output:

- `results/narrow_passage_rl/paper_table_diagnostic_baselines.md`

### 5. Repeated Failure Memory

Script:

```bash
python examples/narrow_passage_rl/eval_repeated_failure_memory.py \
    --n-rounds 5 \
    --n-passable 20 \
    --n-ff 15 \
    --max-steps 220
```

Claim tested:

Memory does not improve one-shot nominal Habitat success.  Its contribution is
to suppress repeated commitments to previously failed infeasible passages.

Key result:

| Method | Passable SR | Passable false reject | Final FF reject | Wasted FF steps | Steps saved | Precision |
|---|---:|---:|---:|---:|---:|---:|
| no_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 | - |
| local_intra_episode_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 | - |
| kNN Failure Memory | 0.860 | 0.060 | 1.000 | 5500 | 11000 | 0.916 |
| Vanilla Episodic Memory | 0.820 | 0.130 | 1.000 | 8140 | 8360 | 0.736 |
| Geometry-Guided Cross-Episode Failure Memory | 0.900 | 0.030 | 1.000 | 4400 | 12100 | 0.963 |

Output:

- `results/narrow_passage_rl/repeated_failure_memory.csv`
- `results/narrow_passage_rl/paper_table_repeated_failure_memory.md`
- `results/narrow_passage_rl/paper_table_repeated_failure_memory.tex`

### 6. D_min Calibration

Script:

```bash
python examples/narrow_passage_rl/eval_dmin_calibration.py
```

What it tests:

- Online calibration of the effective robot width / minimum traversable
  clearance threshold.
- Recovery from an initially conservative wrong estimate.

Output:

- `results/narrow_passage_rl/dmin_calibration.png`
- `results/narrow_passage_rl/dmin_calib_episodes.csv`
- `results/narrow_passage_rl/dmin_calib_convergence.csv`

## Formal, Diagnostic, And Smoke Baselines

Use these tables for paper organization:

```text
results/narrow_passage_rl/paper_table_formal_baselines.md
results/narrow_passage_rl/paper_table_diagnostic_baselines.md
results/narrow_passage_rl/paper_table_smoke_baselines.md
```

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

Legacy PPO runs with the v1 `obs[10]` yaw/heading mismatch are provenance only
and are excluded from formal baseline tables.

## Current RL Interpretation

RL is currently used in three roles:

1. Formal learning baselines: PPO/SAC/TD3 with the same 19-D geometry input.
2. Safety diagnostic: Habitat-native TD3 nominal success vs strict success.
3. Smoke/appendix paths: recurrent, imitation, and replay-memory policies.

The important diagnostic result is:

| Method | Nominal SR | Strict SR | Success-but-unsafe | Near collision |
|---|---:|---:|---:|---:|
| TD3 Habitat-native | 1.000 | 0.020 | 0.980 | 1.000 |

This means the policy learned to satisfy the nominal Habitat success condition,
but not to maintain safe clearance through the passage.  It should be reported
as nominal-metric exploitation, not as a safe RL traversal policy.

The AAAI-facing framing is:

- learning-only transfer is weak near geometric feasibility boundaries;
- nominal success can hide unsafe clearance behavior;
- strict clearance-aware success, near-collision, and minimum clearance must be
  reported for RL baselines;
- future RL should be a risk-constrained local skill under a geometry/risk
  decision layer.

Planned safe-RL direction:

```text
a_rl = policy(obs, mode)
a_safe = geometry_action_shield(a_rl, body_margin, clearance_left, clearance_right, heading_error)
reward = progress + alignment + strict_success - unsafe_margin - near_collision - stuck
```

Under this framing, RL contributes a fair baseline, a safety diagnostic, and a
path toward local skill learning.  The main method remains geometry-guided
decision making with failure memory.

## What Not To Overclaim

- HM3D nominal 100% is not a complete robustness proof.  It is nominal anchor validation on
  mined, mostly well-aligned starts.
- Habitat stress validation is the evidence for module sensitivity under yaw,
  lateral, dropout, noise, and extreme-clearance perturbations.
- Failure memory is not for improving one-shot passable-anchor success.  It
  reduces repeated infeasible commitment and wasted attempts.
- Smoke RL baselines are not main baselines.  They verify code paths and belong
  in appendix/status tables unless rerun under the full protocol.

## Reproduction Commands

Run from the repository root.

```bash
# Procedural v2 benchmark
python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500

# Habitat stress validation
python examples/narrow_passage_rl/eval_habitat_stress_validation.py \
    --preset paper \
    --split val \
    --num-episodes -1

# Repeated failure memory
python examples/narrow_passage_rl/eval_repeated_failure_memory.py \
    --n-rounds 5 \
    --n-passable 20 \
    --n-ff 15 \
    --max-steps 220

# D_min calibration
python examples/narrow_passage_rl/eval_dmin_calibration.py

# Regenerate paper tables from available CSV artifacts
python examples/narrow_passage_rl/make_paper_tables.py
```

Habitat runs require:

```text
data/scene_datasets/hm3d/
data/datasets/narrow_passage/{split}/{split}.json.gz
```

## Dataset Generation

Mine anchors from HM3D and generate `NarrowPassageNav-v0` episode files:

```bash
python examples/narrow_passage_rl/mine_habitat_passages.py \
    --scenes-dir data/scene_datasets/hm3d/val \
    --target-episodes 800 \
    --out-train data/datasets/narrow_passage/anchors_train.csv \
    --out-val data/datasets/narrow_passage/anchors_val.csv

python examples/narrow_passage_rl/generate_habitat_episodes.py \
    --anchors data/datasets/narrow_passage/anchors_val.csv \
    --split val \
    --output data/datasets/narrow_passage/val/val.json.gz
```

## Video And Figures

Record Habitat videos with overlays:

```bash
python examples/narrow_passage_rl/record_habitat_video.py \
    --method geometry_fsm \
    --split val \
    --episode-index 0 \
    --output-dir video_dir/narrow_passage_habitat \
    --save-keyframes
```

Render top-down synthetic trajectories:

```bash
python examples/narrow_passage_rl/render_trajectories.py \
    --panels l_shaped s_shaped false_feasible narrow_entry \
    --dpi 150
```

## Result Files

Important table outputs:

```text
results/narrow_passage_rl/paper_table_main.md
results/narrow_passage_rl/paper_table_ablation.md
results/narrow_passage_rl/paper_table_habitat.md
results/narrow_passage_rl/paper_table_habitat_stress.md
results/narrow_passage_rl/paper_table_formal_baselines.md
results/narrow_passage_rl/paper_table_diagnostic_baselines.md
results/narrow_passage_rl/paper_table_smoke_baselines.md
results/narrow_passage_rl/paper_table_repeated_failure_memory.md
```

Important raw CSV outputs:

```text
results/narrow_passage_rl/harder_benchmark_episodes.csv
results/narrow_passage_rl/habitat_stress_validation.csv
results/narrow_passage_rl/repeated_failure_memory.csv
results/narrow_passage_rl/habitat_td3_habitat_native_strict_eval.csv
```

## Implementation Notes

The 19-D geometry feature vector contains depth sectors, left/right clearance,
passage width, body margin, heading error, lateral offset, distance to goal,
previous action, stuck score, and collision flag.  Habitat exposes this through
`NarrowPassageGeometrySensor`.

The Geometry-FSM outputs local velocity commands `(v_x, omega_z)` and uses
interpretable modes: align, commit, explore, recover, reject, and follow-space
for L/S-shaped turns.

The real-robot interface should reproduce the same 19-D feature vector from
depth, localization, and proprioception.  Hardware experiments should be
reported as pilot validation unless accompanied by full videos, logs, bags, and
statistics.
