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

Naming convention for the paper:

- `Geometry-FSM` is the implementation name used in scripts and CSV files.
- `DEGNAV-Rule` is the paper-facing name for `Geometry-FSM`.
- `DEGNAV-RL` is a diagnostic policy that tests whether PPO can learn only the
  high-level mode selector `pi(m_t | b_t)` over an explicit feasibility-belief
  state.
- PPO/SAC/TD3 are learning-only direct-control baselines that map geometry
  observations directly to velocity actions.

Geometry-FSM is referred to as DEGNAV-Rule in the paper.

## What This Repository Adds

The primary contribution lives here:

```text
examples/narrow_passage_rl/
  procedural_env_v2.py                  # Synthetic v2 narrow-passage benchmark
  eval_harder_benchmark.py              # Main procedural benchmark
  eval_habitat_geometry_fsm.py          # Geometry-FSM / DEGNAV-Rule on Habitat HM3D anchors
  eval_habitat_stress_validation.py     # Habitat stress validation
  eval_repeated_failure_memory.py       # Repeated infeasible commitment memory test
  eval_memory_transfer_interference.py  # Memory transfer/interference test
  eval_habitat_sb3.py                   # SB3 PPO/SAC/TD3 direct-control Habitat evaluation
  train_belief_mode_ppo.py              # DEGNAV-RL high-level mode-selection PPO
  eval_belief_mode_ppo.py               # DEGNAV-RL procedural v2 evaluation
  train_sb3_v2.py                       # PPO/SAC/TD3 direct-control synthetic v2 training
  train_recurrent_ppo_v2.py             # RecurrentPPO smoke baseline
  train_bc_dagger_v2.py                 # BC/DAgger smoke baselines
  train_replay_memory_policy_v2.py      # Generic replay-memory smoke baseline
  plot_margin_phase.py                  # Rule-vs-DEGNAV width-margin phase diagram
  failure_memory.py                     # Episode-local passage memory
  cross_episode_memory.py               # Cross-episode failure memory
  fair_reward_wrapper.py                # Dense/fair reward wrapper for RL baselines
  narrow_passage/models/belief_state.py # Compact feasibility-belief state
  narrow_passage/envs/belief_mode_env.py# Discrete mode wrapper for DEGNAV-RL
  narrow_passage/metrics/strict_metrics.py # Shared clearance diagnostic metrics
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

The main paper method is DEGNAV-Rule / Geometry-FSM.

DEGNAV-RL is included as a diagnostic policy rather than a competitive final
method.  It tests the high-level selector `pi(m_t | b_t)`, where `m_t` is one of
`Commit`, `Explore`, `Recover`, and `Reject`; the low-level velocity realization
remains the same mode-conditioned controller used by DEGNAV-Rule.  PPO/SAC/TD3
are separate direct-control baselines: they map the same geometry observations
directly to velocity actions.

## Main Experiments

### 1. Procedural v2 Benchmark

Synthetic v2 includes straight, L-shaped, S-shaped, narrow-entry, narrow-exit,
asymmetric, and false-feasible corridors.

Main result:

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reactive rule baseline | 25.4 | 60.5 | 8.2 | 6.5 | 27.2 | 24.8 | 33.5 | 0.0 |
| DEGNAV-Rule / Geometry-FSM | 70.3 | 92.8 | 79.6 | 70.0 | 96.7 | 61.4 | 50.6 | 0.0 |

Removing alignment under entry jitter drops the FSM from 64.1% to 19.1% overall,
which identifies heading/lateral alignment as a critical local-control module.

False-feasible outcome decomposition:

| Method | Episodes | Traversal success | Correct reject | Collision | Timeout/stuck | Wasted attempts |
|---|---:|---:|---:|---:|---:|---:|
| Reactive rule baseline | 1500 | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% |
| DEGNAV-Rule / Geometry-FSM | 1500 | 0.0% | 0.0% | 2.8% | 97.2% | 100.0% |

This is an important negative/provenance result: 0% traversal success is not
equivalent to correct rejection.  The current benchmark-labeled false-feasible
rows show execution failure or wasted attempts, not learned or rule-based
abstention.

Reject is tracked only when a controller explicitly emits the Reject mode.  The
default DEGNAV-Rule / Geometry-FSM row should not be described as solving
false-feasible rejection.  A separate diagnostic candidate can be run with
`--methods rule_baseline geometry_fsm feasibility_reject` to test conservative
observable-geometry rejection and report false-reject rate separately.

Primary files:

- `examples/narrow_passage_rl/eval_harder_benchmark.py`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_procedural_v2_main.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_harder_ablation.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/raw/false_feasible_outcomes.csv`
- `examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_false_feasible_outcomes.md`

### 2. Habitat HM3D Nominal Anchor Validation

Habitat experiments use mined HM3D narrow-passage anchors and the
`NarrowPassageNav-v0` task.  The nominal anchor table checks whether the
geometry controller works on scanned scenes when the mined starts are mostly
well aligned.

| Method | Setting | Success |
|---|---|---:|
| PPO v2 geometry, 3 seeds | Synthetic-to-Habitat direct-control transfer | 2.1% +/- 2.6% |
| SAC v2 geometry | Synthetic-to-Habitat direct-control transfer | 0.0% |
| TD3 synthetic-to-Habitat | Synthetic-to-Habitat direct-control transfer | 2.0% |
| APF+Gap | Classical depth/GPS baseline | 93.6% |
| DEGNAV-Rule / Geometry-FSM | Nominal HM3D anchors | 100.0% |

Important interpretation: the 100.0% DEGNAV-Rule / Geometry-FSM result is
nominal anchor validation under the current mining protocol, not a complete
robustness proof.

Habitat experiments should be interpreted in three separate categories:

- Nominal same-split evaluation: unperturbed HM3D anchors evaluated on a single
  split/protocol.
- Perturbation or stress evaluation: yaw, lateral offset, dropout, noise, and
  extreme-narrow stressors used to expose module sensitivity.
- Clearance-aware diagnostic evaluation: depth-derived body-margin diagnostics
  such as clearance-aware strict success, near-collision, and
  success-but-unsafe.

The Habitat clearance-related metrics are derived from depth observations and an approximate robot body-margin model. They are used as clearance-aware diagnostic indicators rather than calibrated physical safety measurements.

`paper_table_formal_baselines.md` is a mixed-split Habitat diagnostic
comparison because it contains both HM3D Val set A and mined Val set B rows. It
must not be used for fair method ranking in the main paper.

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

The +60 degree extreme-narrow slice exposes heading-alignment sensitivity:
full FSM remains at 100.0% nominal success, while removing heading alignment
drops to 79.2% with higher timeout. Clearance-aware strict success is reported
separately as a diagnostic body-margin proxy.

Primary files:

- `examples/narrow_passage_rl/eval_habitat_stress_validation.py`
- `examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_habitat_stress_nominal.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_habitat_clearance_diagnostic.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_habitat_stress_key_slices.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_stress_validation.csv`

### 4. Clearance-Aware RL Diagnostic

TD3 trained directly in Habitat can reach 100% nominal success, but only 2.0%
clearance-aware strict success.  98.0% of episodes are success-but-unsafe and
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

### 6. Memory Transfer And Interference

The repeated-failure table uses repeated identical false-feasible passages.  A
separate transfer/interference script tests whether geometry-anchored memory
also rejects similar-but-new false-feasible passages without over-rejecting
similar feasible passages.

Current paper-preset result: geometry-guided cross-episode failure memory keeps
passable success at 90.0% +/- 2.2%, reaches 100.0% final rejection on repeated
false-feasible passages, rejects 93.0% +/- 8.6% of similar new false-feasible
passages, and has 0.0% interference false rejection on similar feasible
passages.  Vanilla episodic memory and kNN failure memory reject false-feasible
passages aggressively, but also falsely reject 100.0% of similar feasible
passages in this protocol.

Primary files:

- `examples/narrow_passage_rl/eval_memory_transfer_interference.py`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_transfer_interference.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/memory_transfer_interference.csv`

### 7. D_min Calibration

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

## Diagnostic And Smoke Baselines

The learning baselines are intentionally split by evidential role.

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

Legacy PPO runs with the v1 `obs[10]` yaw/heading mismatch are excluded from
formal baseline tables.

Tables:

- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_formal_baselines.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_diagnostic_baselines.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_smoke_baselines.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_degnav_rl_diagnostic.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_belief_mode_ablation.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_transfer_interference.md`

## Current RL Interpretation

The current RL results should be read as baseline and diagnostic evidence, not
as the main narrow-passage solution.

The fair comparison uses the same 19-D geometry sensor for PPO/SAC/TD3.  PPO v2
geometry transfer reaches 2.1% +/- 2.6% on Habitat, SAC reaches 0.0%, and TD3
synthetic-to-Habitat reaches 2.0% despite 77.8% success in the synthetic v2
environment.  This supports the paper's claim that learning-only policies are
fragile near geometric feasibility boundaries.

These PPO/SAC/TD3 policies are direct-control baselines.  They do not implement
DEGNAV-RL because they output velocity actions directly instead of selecting a
high-level mode from an explicit belief state.

The Habitat-native TD3 result is intentionally reported as a diagnostic:

| Method | Nominal success | Strict success | Unsafe success | Interpretation |
|---|---:|---:|---:|---|
| TD3 Habitat-native | 100.0% | 2.0% | 98.0% | RL can exploit nominal goal-reaching while violating the body-margin diagnostic |

The split between nominal and strict success occurs because nominal success only
checks the local goal/alignment condition, while strict success also requires
clearance-aware strict success.  Therefore RL's 100% nominal result should be used to
motivate clearance-aware diagnostic evaluation, not to claim that RL solved the
task.

DEGNAV-RL is framed as a diagnostic mode-selection policy:

- build an explicit belief state
  `b_t = (p_feas, E[Delta], Var[Delta], heading_error, lateral_error,
  stuckness, contact, memory_risk)`;
- learn only `pi(m_t | b_t)` over `Commit`, `Explore`, `Recover`, and `Reject`;
- keep the mode-conditioned low-level controller and strict clearance-aware
  safety checks outside the learned policy.

This keeps the RL contribution aligned with the paper: the diagnostic policy can
test mode selection, while explicit geometry, risk, and failure memory remain
the core method.  DEGNAV-RL is included as a diagnostic policy rather than a
competitive final method. Under the current reward and action interface, the
learned policy collapses to Commit and Explore and does not demonstrate
meaningful Recover or Reject behavior.

DEGNAV-RL entry points:

- `examples/narrow_passage_rl/train_belief_mode_ppo.py`
- `examples/narrow_passage_rl/eval_belief_mode_ppo.py`
- `examples/narrow_passage_rl/narrow_passage/envs/belief_mode_env.py`
- `examples/narrow_passage_rl/narrow_passage/models/belief_state.py`

Current diagnostic comparison, with eval domains shown explicitly.  This is not
a single leaderboard: the DEGNAV-RL row is procedural v2, while the
direct-control rows below are Habitat diagnostics/provenance.

| Method | Decision type | Eval domain | Overall SR | Strict SR | Collision | Near collision | Reject |
|---|---|---|---:|---:|---:|---:|---:|
| PPO direct velocity, mined-val provenance | Direct velocity | Habitat HM3D mined-val | 6.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| TD3 synthetic-to-Habitat | Direct velocity | Habitat HM3D mined-val | 2.0% | 0.0% | 0.0% | 100.0% | - |
| DEGNAV-RL | Learned high-level mode selector | Procedural v2 | 26.7% | 26.7% | 62.3% | 70.9% | 0.0% |
| DEGNAV-Rule / Geometry-FSM | Rule mode selector | Habitat nominal anchors | 100.0% nominal anchor validation | 2.0% on HM3D nominal anchors | 0.0% | 100.0% | 0.0% |

DEGNAV-RL mode distribution over 1500 procedural v2 evaluation episodes:
Commit: 31.0%, Explore: 69.0%, Recover: 0.0%, Reject: 0.0%.

Interpretation: DEGNAV-RL is a diagnostic learning variant, not the final
method.  It still has high collision and near-collision rates and did not learn
to use `Reject` or `Recover` in the current setup.  Its role in the paper should
therefore be: learning the mode selector alone is not enough unless safety
constraints and failure-memory/reject behavior are made explicit.

This negative result supports five design conclusions: sparse reward alone is
insufficient, mode semantics are not learned automatically, explicit failure
memory may still be necessary, Recover/Reject require dedicated reward or
supervision, and longer training alone may not solve mode collapse.

Belief-state ablations are also diagnostic rather than a positive contribution
claim: `geometry_only` reaches 30.9% strict SR, while the full belief state
reaches 26.7%.  This suggests the current DEGNAV-RL reward/controller interface
is not yet exploiting `p_feas`, uncertainty, or memory risk; those signals remain
most reliable in the interpretable DEGNAV-Rule path.

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

# 4. Memory transfer/interference
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset paper

# 5. D_min calibration
python examples/narrow_passage_rl/eval_dmin_calibration.py

# 6. DEGNAV-RL diagnostic mode-selection run
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

  python examples/narrow_passage_rl/eval_belief_mode_ppo.py \
      --model data/degnav_rl_belief_mode_full_seed${seed}/belief_mode_ppo.zip \
      --episodes 500 \
      --seed $((1000 + seed)) \
      --ctypes full \
      --ablation full \
      --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed${seed}_eval.csv
done

# 7. Rule-vs-DEGNAV width-margin phase analysis
python examples/narrow_passage_rl/eval_harder_benchmark.py \
    --methods rule_baseline geometry_fsm \
    --episodes 500 \
    --seeds 0 1 2 \
    --log-belief-diagnostics \
    --log-outcome-decomposition \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv

python examples/narrow_passage_rl/plot_margin_phase.py \
    --inputs examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv \
    --output-dir examples/narrow_passage_rl/results/narrow_passage_rl/figures \
    --table-dir examples/narrow_passage_rl/results/narrow_passage_rl/tables \
    --methods rule_baseline geometry_fsm \
    --labels "Reactive rule baseline,DEGNAV-Rule" \
    --bin-width 0.05 \
    --margin-min -0.30 \
    --margin-max 0.30 \
    --min-bin-count 5

# Optional infeasible-side margin probe; keep separate from the main figure
python examples/narrow_passage_rl/eval_harder_benchmark.py \
    --methods rule_baseline geometry_fsm feasibility_reject \
    --episodes 500 \
    --seeds 0 1 2 \
    --width-range 0.30 0.55 \
    --log-belief-diagnostics \
    --log-outcome-decomposition \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_infeasible_probe.csv

# 8. Regenerate paper tables from existing CSV artifacts
python examples/narrow_passage_rl/make_paper_tables.py
```

The infeasible-side probe is a sensitivity analysis, not the main paired figure.
It increases negative-margin support and tests the diagnostic
`feasibility_reject` variant.  Current results show that this conservative gate
does not solve false-feasible rejection: it introduces false rejects and leaves
most false-feasible blockers as timeout/wasted-attempt cases.  Use repeated
failure-memory experiments for the memory/rejection claim.

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
