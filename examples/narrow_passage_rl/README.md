# Geometry-Guided Failure-Aware Narrow-Passage Navigation

This directory contains the narrow-passage research code added on top of the
Habitat-Lab fork.  It is the main entry point for reproducing the paper's
experiments and tables.

Naming convention:

- `Geometry-FSM` is the implementation name used in scripts and CSV files.
- `DEGNAV-Rule` is the paper-facing name for `Geometry-FSM`.
- `DEGNAV-RL` is a diagnostic policy that tests whether PPO can learn only the
  high-level mode selector `pi(m_t | b_t)` over `Commit`, `Explore`, `Recover`,
  and `Reject`.
- PPO/SAC/TD3 are learning-only direct-control baselines that map geometry
  observations directly to velocity actions.

Geometry-FSM is referred to as DEGNAV-Rule in the paper.

## What This Directory Adds

```text
examples/narrow_passage_rl/
  procedural_env_v2.py                  # Synthetic v2 benchmark
  eval_harder_benchmark.py              # Main procedural benchmark
  eval_habitat_geometry_fsm.py          # Geometry-FSM / DEGNAV-Rule on HM3D anchors
  eval_habitat_apf_gap.py               # APF+Gap Habitat baseline
  eval_habitat_sb3.py                   # PPO/SAC/TD3 direct-control Habitat evaluation
  eval_habitat_stress_validation.py     # Formal Habitat stress validation
  eval_repeated_failure_memory.py       # Repeated false-feasible memory test
  eval_memory_transfer_interference.py  # Memory transfer/interference test
  eval_dmin_calibration.py              # D_min self-calibration
  train_sb3_v2.py                       # PPO/SAC/TD3 direct-control training
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

The main method is DEGNAV-Rule / Geometry-FSM.  DEGNAV-RL is retained as a
diagnostic appendix experiment.

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
| Reactive rule baseline | 25.4 | 60.5 | 8.2 | 6.5 | 27.2 | 24.8 | 33.5 | 0.0 |
| DEGNAV-Rule / Geometry-FSM | 70.3 | 92.8 | 79.6 | 70.0 | 96.7 | 61.4 | 50.6 | 0.0 |

Output tables:

- `results/narrow_passage_rl/paper_table_procedural_v2_main.md`
- `results/narrow_passage_rl/paper_table_harder_ablation.md`

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
- Direct-control learning baselines and classical APF+Gap baseline.

Important interpretation:

The 100.0% DEGNAV-Rule / Geometry-FSM rows are nominal anchor validation under
the current mining protocol.  The mined anchors are mostly well aligned, so this
result should not be described as complete robustness.  Stress validation is
required to expose module sensitivity.

Diagnostic and provenance tables:

- `results/narrow_passage_rl/paper_table_habitat.md`
- `results/narrow_passage_rl/paper_table_formal_baselines.md` (mixed-split Habitat diagnostic comparison)

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

- Full DEGNAV-Rule / Geometry-FSM.
- FSM without heading alignment.
- FSM without recovery.
- FSM without lateral alignment.
- APF+Gap, when the interface is available.

Output:

- `results/narrow_passage_rl/habitat_stress_validation.csv`
- `results/narrow_passage_rl/paper_table_habitat_stress.md`
- `results/narrow_passage_rl/paper_table_habitat_stress.tex`

### 4. Clearance-Aware RL Diagnostic

This is a diagnostic table, not the main method ranking.

TD3 trained directly in Habitat reaches 100% nominal success but only 2.0%
clearance-aware strict success.  This demonstrates that nominal goal-reaching
success can be exploited by RL and must be reported together with clearance
diagnostics.

The Habitat clearance-related metrics are derived from depth observations and an approximate robot body-margin model. They are used as clearance-aware diagnostic indicators rather than calibrated physical safety measurements.

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

### 6. Memory Transfer And Interference

Script:

```bash
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset smoke
```

Claim tested:

This experiment tests whether geometry-guided failure memory transfers from
repeated false-feasible passages to similar new false-feasible passages, while
avoiding false rejection on similar feasible passages.  This is distinct from
the repeated identical false-feasible table above.

Output:

- `results/narrow_passage_rl/memory_transfer_interference.csv`
- `results/narrow_passage_rl/paper_table_memory_transfer_interference.md`
- `results/narrow_passage_rl/paper_table_memory_transfer_interference.tex`

Paper-preset result:

| Method | Passable SR | Final FF reject | Transfer reject new FF | Interference false reject |
|---|---:|---:|---:|---:|
| no_memory | 90.0 +/- 2.2 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| vanilla episodic memory | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| kNN failure memory | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| geometry-guided cross-episode failure memory | 90.0 +/- 2.2 | 100.0 +/- 0.0 | 93.0 +/- 8.6 | 0.0 +/- 0.0 |

Interpretation: generic episodic/kNN memory can suppress repeated infeasible
commitments, but in this protocol it also over-generalizes and rejects all
similar feasible passages.  Geometry-guided failure memory preserves passable
success while transferring rejection to similar new false-feasible passages.

### 7. D_min Calibration

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

## Diagnostic And Smoke Baselines

Use these tables for paper organization:

```text
results/narrow_passage_rl/paper_table_formal_baselines.md  # mixed-split Habitat diagnostic comparison
results/narrow_passage_rl/paper_table_diagnostic_baselines.md
results/narrow_passage_rl/paper_table_smoke_baselines.md
```

Mixed-split Habitat diagnostic comparison:

- PPO v2 geometry sensor, 3 seeds, synthetic-to-Habitat direct-control transfer.
- SAC v2 geometry sensor direct-control baseline.
- TD3 synthetic-to-Habitat direct-control baseline.
- APF+Gap classical baseline.
- DEGNAV-Rule / Geometry-FSM.

Clearance-aware diagnostic evaluation:

- TD3 Habitat-native nominal success vs clearance-aware strict success.

Smoke / appendix-only baselines:

- GRU-PPO lightweight.
- RecurrentPPO smoke.
- BC-FSM.
- DAgger-FSM.
- Replay Memory Policy.

Legacy PPO runs with the v1 `obs[10]` yaw/heading mismatch are provenance only
and are excluded from formal baseline tables.

## Current RL Interpretation

RL is currently used in four clearly separated roles:

1. Direct-control learning baselines: PPO/SAC/TD3 policies with the same
   19-D geometry input.
2. Clearance-aware diagnostic evaluation: Habitat-native TD3 nominal success vs clearance-aware strict success.
3. Smoke/appendix paths: recurrent, imitation, and replay-memory policies.
4. DEGNAV-RL: a diagnostic policy for high-level mode selection over the
   explicit belief state, not a direct velocity policy.

The important diagnostic result is:

| Method | Nominal SR | Strict SR | Success-but-unsafe | Near collision |
|---|---:|---:|---:|---:|
| TD3 Habitat-native | 1.000 | 0.020 | 0.980 | 1.000 |

This means the policy learned to satisfy the nominal Habitat success condition,
but not the depth-derived body-margin diagnostic.  It should be reported as
nominal-metric exploitation, not as a calibrated clearance result.

The AAAI-facing framing is:

- learning-only transfer is weak near geometric feasibility boundaries;
- nominal success can hide low body-margin behavior;
- clearance-aware strict success, near-collision, and minimum clearance must be
  reported for RL baselines;
- DEGNAV-RL is intended to test only the high-level mode selector
  `pi(m_t | b_t)`, while the geometry/risk decision variables remain explicit.

DEGNAV-RL formulation:

```text
b_t = (p_feas, E[Delta], Var[Delta], heading_error, lateral_error,
       stuckness, contact, memory_risk)
m_t ~ pi(m_t | b_t),  m_t in {Commit, Explore, Recover, Reject}
u_t = mode_conditioned_controller(m_t, geometry, local_goal)
```

Under this framing, PPO/SAC/TD3 remain direct-control baselines, while
DEGNAV-RL is a diagnostic policy that tests whether the mode selector can be
learned without replacing DEGNAV-Rule's explicit geometry/risk logic.

## DEGNAV-RL: Diagnostic Belief-Guided Mode-Selection PPO

DEGNAV-RL trains PPO on the compact feasibility-belief state `b_t` instead of
the raw 19-D geometry observation.  The policy is intended to output a
high-level mode:

```text
0 = COMMIT
1 = EXPLORE
2 = RECOVER
3 = REJECT
```

The wrapper then realizes that mode with the shared mode-conditioned controller.
This is not the same as the PPO/SAC/TD3 direct-control baselines in
`train_sb3_v2.py`, which output velocity actions directly.
DEGNAV-RL is included as a diagnostic policy rather than a competitive final
method. Under the current reward and action interface, the learned policy
collapses to Commit and Explore and does not demonstrate meaningful Recover or
Reject behavior.

Current 3-seed result on procedural v2, using 1M PPO steps per seed:

| Variant | Overall SR | Strict SR | Collision | Near collision | Reject | Mode usage |
|---|---:|---:|---:|---:|---:|---|
| DEGNAV-RL full belief | 26.7% | 26.7% | 62.3% | 70.9% | 0.0% | Commit 31.0%, Explore 69.0%, Recover 0.0%, Reject 0.0% |
| no p_feas | 25.9% | 25.9% | 61.9% | 72.5% | 0.0% | Commit 62%, Explore 38% |
| no delta_var | 28.4% | 28.4% | 58.9% | 70.3% | 0.0% | Commit 29%, Explore 71% |
| no memory | 28.4% | 28.4% | 58.9% | 70.3% | 0.0% | Commit 29%, Explore 71% |
| no alignment | 25.8% | 25.8% | 61.5% | 72.2% | 0.0% | Commit 37%, Explore 63% |
| geometry only | 30.9% | 30.9% | 56.0% | 68.1% | 0.0% | Explore 100% |

Interpretation:

- DEGNAV-RL is no longer `not run`, but it should remain a diagnostic learning
  variant rather than the main method.
- It is far below DEGNAV-Rule / Geometry-FSM and still has high collision and
  near-collision rates.
- The learned mode selector did not learn `Reject` or `Recover` under the
  current reward/controller interface.  This is useful evidence that simply
  learning `pi(m_t | b_t)` is not enough; the explicit rule-based feasibility,
  risk, and failure-memory checks remain important.
- The negative result supports five design conclusions: sparse reward alone is
  insufficient, mode semantics are not learned automatically, explicit failure
  memory may still be necessary, Recover/Reject require dedicated reward or
  supervision, and longer training alone may not solve mode collapse.
- The belief-state ablation does not show a clean advantage for the full belief
  vector.  `geometry_only` performs best in this run, so the paper should not
  claim that the current DEGNAV-RL policy successfully exploits uncertainty or
  memory risk.

Smoke training:

```bash
python examples/narrow_passage_rl/train_belief_mode_ppo.py \
    --total-steps 2048 \
    --num-envs 1 \
    --ctypes straight_only \
    --ablation full \
    --eval-episodes 20 \
    --save-dir data/degnav_rl_belief_mode_smoke
```

Paper-scale training:

```bash
for seed in 0 1 2; do
  python examples/narrow_passage_rl/train_belief_mode_ppo.py \
      --total-steps 1000000 \
      --num-envs 8 \
      --seed ${seed} \
      --ctypes full \
      --ablation full \
      --eval-episodes 500 \
      --device cuda \
      --save-dir data/degnav_rl_belief_mode_full_seed${seed}
done
```

Evaluation:

```bash
for seed in 0 1 2; do
  python examples/narrow_passage_rl/eval_belief_mode_ppo.py \
      --model data/degnav_rl_belief_mode_full_seed${seed}/belief_mode_ppo.zip \
      --episodes 500 \
      --seed $((1000 + seed)) \
      --ctypes full \
      --ablation full \
      --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed${seed}_eval.csv
done
```

Belief-state ablations use the same scripts:

```bash
python examples/narrow_passage_rl/train_belief_mode_ppo.py \
    --total-steps 1000000 \
    --num-envs 8 \
    --ctypes full \
    --ablation no_p_feas \
    --eval-episodes 500 \
    --save-dir data/degnav_rl_belief_mode_no_p_feas

python examples/narrow_passage_rl/eval_belief_mode_ppo.py \
    --model data/degnav_rl_belief_mode_no_p_feas/belief_mode_ppo.zip \
    --episodes 500 \
    --ctypes full \
    --ablation no_p_feas \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_p_feas_eval.csv
```

Supported ablations are `full`, `no_p_feas`, `no_delta_var`, `no_memory`,
`no_alignment`, and `geometry_only`.  `geometry_only` changes the policy
observation to six features: `d_hat`, `body_margin`, `clearance_left`,
`clearance_right`, `heading_error`, and `lateral_error`.

To regenerate the DEGNAV-RL ablation table from one or more eval CSVs:

```bash
python examples/narrow_passage_rl/make_paper_tables.py \
    --belief-mode-inputs \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed0_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed1_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed2_eval.csv \
    --belief-mode-ablation-inputs \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed*_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_p_feas_seed*_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_delta_var_seed*_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_memory_seed*_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_alignment_seed*_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_geometry_only_seed*_eval.csv
```

Training outputs include `belief_mode_ppo.zip`,
`belief_mode_ppo_config.json`, `belief_mode_ppo_quick_eval.csv`, and
`belief_mode_ppo_quick_summary.md`.

## What Not To Overclaim

- HM3D nominal 100.0% is not a complete robustness proof.  It is nominal anchor
  validation under the current mining protocol on mined, mostly well-aligned
  starts.
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

# Memory transfer/interference smoke test
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset smoke

# Memory transfer/interference paper run
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset paper

# D_min calibration
python examples/narrow_passage_rl/eval_dmin_calibration.py

# Regenerate paper tables from available CSV artifacts
python examples/narrow_passage_rl/make_paper_tables.py

# Width-margin phase diagram for paper figures
python examples/narrow_passage_rl/plot_margin_phase.py \
    --inputs \
      examples/narrow_passage_rl/results/narrow_passage_rl/harder_benchmark_episodes.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed0_eval.csv \
    --labels DEGNAV-Rule DEGNAV-RL \
    --output-dir examples/narrow_passage_rl/results/narrow_passage_rl \
    --bin-width 0.02 \
    --margin-min -0.10 \
    --margin-max 0.10
```

The margin-phase script writes
`results/narrow_passage_rl/figures/margin_phase_all_methods.{png,pdf}` and
`results/narrow_passage_rl/tables/margin_phase_summary.csv`.  It uses
`delta_mean` directly when available, or computes `d_hat - w_req_cons` from the
CSV columns.

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
results/narrow_passage_rl/paper_table_procedural_v2_main.md
results/narrow_passage_rl/paper_table_harder_ablation.md
results/narrow_passage_rl/paper_table_habitat.md
results/narrow_passage_rl/paper_table_habitat_stress.md
results/narrow_passage_rl/paper_table_formal_baselines.md
results/narrow_passage_rl/paper_table_diagnostic_baselines.md
results/narrow_passage_rl/paper_table_smoke_baselines.md
results/narrow_passage_rl/paper_table_repeated_failure_memory.md
results/narrow_passage_rl/paper_table_memory_transfer_interference.md
results/narrow_passage_rl/paper_table_degnav_rl_diagnostic.md
results/narrow_passage_rl/paper_table_belief_mode_ablation.md
```

Important raw CSV outputs:

```text
results/narrow_passage_rl/harder_benchmark_episodes.csv
results/narrow_passage_rl/habitat_stress_validation.csv
results/narrow_passage_rl/repeated_failure_memory.csv
results/narrow_passage_rl/memory_transfer_interference.csv
results/narrow_passage_rl/belief_mode_full_seed*_eval.csv
results/narrow_passage_rl/belief_mode_<ablation>_seed*_eval.csv
results/narrow_passage_rl/habitat_td3_habitat_native_strict_eval.csv
```

## Implementation Notes

The 19-D geometry feature vector contains depth sectors, left/right clearance,
passage width, body margin, heading error, lateral offset, distance to goal,
previous action, stuck score, and collision flag.  Habitat exposes this through
`NarrowPassageGeometrySensor`.

The Geometry-FSM / DEGNAV-Rule outputs local velocity commands `(v_x, omega_z)`
through interpretable modes: align, commit, explore, recover, reject, and
follow-space for L/S-shaped turns.  DEGNAV-RL is only a diagnostic test of
mode-selection learning, and should not be described as replacing this interface
with direct velocity regression or as the current final method.

The real-robot interface should reproduce the same 19-D feature vector from
depth, localization, and proprioception.  Hardware experiments should be
reported as pilot validation unless accompanied by full videos, logs, bags, and
statistics.
