# Narrow-Passage Navigation with Geometric FSM and Failure Memory

This directory contains all experiments for the narrow-passage navigation paper.
We evaluate on two independent setups:

1. **Synthetic procedural environment** — primary benchmark, fully self-contained.
2. **Habitat real HM3D scenes** — generalization experiment using `NarrowPassageNav-v0`.

---

## Key Results

### Habitat HM3D Generalization

**Val set A** (original 157 episodes, 20 held-out scenes):

| Method | SR | Narrow | Normal | Wide | Notes |
|---|---|---|---|---|---|
| PPO-SB3 baseline | 0.0% | 0.0% | 0.0% | 0.0% | Legacy obs[10] format mismatch (v1 yaw vs. Habitat heading) |
| SB3 PPO synthetic-to-Habitat geometry | 5.7% | 5.1% | 9.1% | 0.0% | Single seed, synthetic v2 env, 5M steps |
| PPO w/ geometry sensor (3 seeds) | 2.1% (±2.6%) | 1.7% | 3.0% | 1.4% | Seeds: 5.7%, 0.0%, 0.6% |
| APF+Gap (Khatib 1986, Meng 2002) | 93.6% | 92.4% | 100% | 82.6% | Classical; depth + GPS only |
| **Geometry-FSM (ours)** | **100%** | **100%** | **100%** | **100%** | |
| **FSM + Failure Memory (ours)** | **100%** | **100%** | **100%** | **100%** | |

**Val set B** (mined 151 episodes, 20 held-out HM3D scenes — independent split):

| Method | SR | Narrow (73) | Normal (55) | Wide (23) |
|---|---|---|---|---|
| SB3 PPO synthetic-to-Habitat geometry | 6.0% | 5.5% | 9.1% | 0.0% |
| SAC v2 (geometry sensor) | 0.0% | 0.0% | 0.0% | 0.0% |
| **Geometry-FSM (ours)** | **100%** | **100%** | **100%** | **100%** |

PPO results are grouped into three distinct categories: synthetic PPO, SB3
synthetic-to-Habitat transfer, and Habitat-Baselines smoke tests.  The rows
above are synthetic-to-Habitat transfer baselines: policies trained on synthetic
v2 geometry observations and evaluated on mined HM3D episodes.  They should not
be described as complete Habitat-Baselines PPO training curves.  Their low HM3D
SR is used as evidence that pure learned policies are unstable under narrow
passage geometry transfer, motivating explicit geometry/risk/failure memory.

**FSM ablation** — all components remain at 100% on Habitat:

| Variant | SR | Narrow | Normal | Wide |
|---|---|---|---|---|
| Full FSM | 100% | 100% | 100% | 100% |
| w/o recovery | 100% | 100% | 100% | 100% |
| w/o alignment | 100% | 100% | 100% | 100% |

On unperturbed Habitat episodes, recovery/alignment ablations remain at 100%
because the mined anchors are already well aligned.  The 100% Habitat rows are
therefore not the only evidence for robustness; they must be reported with the
stress ablations and the mining/success/collision protocol.  A controlled
Habitat stress test rotates the initial heading by 60° on the 24 extreme-narrow
episodes (body_margin < 0.05 m), revealing the alignment module:

| Variant (+60° heading perturb) | SR | Successes |
|---|---:|---:|
| Full FSM | 100% | 24/24 |
| w/o recovery | 100% | 24/24 |
| w/o all alignment | 0% | 0/24 |
| w/o heading alignment | 0% | 0/24 |
| w/o lateral alignment | 100% | 24/24 |

Thus, heading alignment is the critical Habitat module under start-pose perturbation; lateral
centering and recovery are not the bottleneck for these mined anchors.  The
stress evaluator now also exposes lateral start offset, start-distance shift,
feature noise, and depth-sector dropout knobs.  Goal perturbation,
false-feasible Habitat anchors, dynamic obstacles, and unseen-room
generalization still require regenerated datasets or simulator extensions before
they should be claimed as completed.

**Inference speed** (CPU, n=10,000 calls):

| Method | Mean latency | Speedup vs PPO |
|---|---|---|
| Geometry-FSM | 20.9 µs | **7.1×** |
| FSM + TurnCommit | 21.5 µs | 6.9× |
| FSM + Cross-Memory | 26.0 µs | 5.7× |
| PPO-SB3 | 148.0 µs | 1.0× (baseline) |

**Cross-Episode Memory** (8 agents × 100 corridors: 60 passable + 40 false-feasible):

| Condition | FF wasted steps R1 | R2 | R3-8 | FF reject R3+ | Steps saved (8 rounds) |
|---|---|---|---|---|---|
| **Cross-episode memory** | 7,500 | 900 | **0** | **100%** | **87,600** |
| Local (intra-episode) memory | 12,000 | 12,000 | 12,000 | 0% | 0 |
| No memory (baseline) | 12,000 | 12,000 | 12,000 | 0% | 0 |

Cross-memory passable SR stays at **90%** throughout all rounds (no false rejections of passable corridors).
R1 already achieves 37.5% FF rejection because FF seeds processed later in R1 find matching failures
from earlier FF seeds within the same round (sequential within-round learning).

**Memory baseline comparison** (5 agents × 35 corridors: 20 passable + 15 false-feasible):

| Method | Passable SR ↑ | Passable Reject ↓ | Final FF Reject ↑ | Wasted FF Steps ↓ | Steps Saved vs No Memory ↑ |
|---|---:|---:|---:|---:|---:|
| no_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 |
| kNN Failure Memory | 0.860 | 0.060 | 1.000 | 5500 | 11000 |
| Vanilla Episodic Memory | 0.820 | 0.130 | 1.000 | 8140 | 8360 |
| **Geometry-Guided Failure Memory** | **0.900** | **0.030** | **1.000** | **4400** | **12100** |

This separates "has memory" from the proposed geometry-guided failure memory.
kNN and embedding-only episodic memory both learn to reject false-feasible
passages, but they reject more passable corridors and waste more repeated-failure
steps. Raw results: `results/narrow_passage_rl/memory_baselines.csv`.

**Learning baseline status**:

| Method | Domain | Train budget | Eval episodes | Success ↑ | Collision ↓ | Notes |
|---|---|---:|---:|---:|---:|---|
| SB3 PPO synthetic-to-Habitat geometry | Habitat HM3D mined-val | 5M synthetic steps | 151 | 0.060 | - | Synthetic v2 checkpoint transferred to HM3D |
| SAC v2 geometry | Habitat HM3D mined-val | 2M steps | 151 | 0.000 | - | Existing SB3 checkpoint |
| GRU-PPO lightweight | Synthetic v2 | 5k steps | 50 | 0.000 | 0.980 | PyTorch fallback; SB3-Contrib unavailable in current env |
| RecurrentPPO | Synthetic v2 | 3M steps | 500 | 0.130 | 0.456 | SB3-Contrib MlpLstmPolicy + fair reward |
| BC-FSM | Synthetic v2 | 5179 expert transitions | 40 | 0.500 | 0.475 | Supervised imitation smoke |
| DAgger-FSM | Synthetic v2 | 6346 transitions | 40 | 0.300 | 0.650 | One DAgger iteration smoke |
| Replay Memory Policy | Synthetic v2 | 1024 steps | 40 | 0.000 | 0.025 | PPO + generic replay embedding smoke |

The GRU-PPO run is intentionally labeled as a lightweight fallback rather than
a final SB3-Contrib RecurrentPPO result. It is useful as an early negative
control: generic recurrent hidden state did not solve the boundary-passage
problem under a small CPU training budget.
After installing `sb3-contrib==2.7.1` in the `habitat` environment, the formal
SB3-Contrib `RecurrentPPO` path runs end-to-end.  With fair reward and 3M
training steps, it reaches 13.0% SR but 45.6% collision on synthetic v2: it
learns some straight/asymmetric passages, but still fails on L/S turns,
narrow-exit cases, and false-feasible safety.

---

### Harder Synthetic Benchmark — v2 (500 episodes, 7 corridor types)

The v2 benchmark extends the procedural environment to include L-shaped, S-shaped,
and false-feasible corridors. A geometry-guided `TurnCommitFSM` detects junction
asymmetry and commits to the correct turn direction.
RL-based policies (SB3 PPO) achieve near-zero SR on L/S-shaped types — this is the
primary negative result motivating the geometry-first approach.

Results below are mean across 3 independent seeds (std in parentheses), 500 episodes per seed.

**Main comparison:**

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feas. |
|---|---|---|---|---|---|---|---|---|
| Rule baseline | 25.4 (1.0) | 60.5 | 8.2 | 6.5 | 27.2 | 24.8 | 33.5 | 0.0 |
| **Geometry-FSM (ours)** | **70.3 (1.1)** | **92.8** | **79.6** | **70.0** | **96.7** | **61.4** | **50.6** | **0.0** |
| FSM + local memory | 70.3 (1.1) | 92.8 | 79.6 | 70.0 | 96.7 | 61.4 | 50.6 | 0.0 |
| FSM + cross memory | 70.3 (1.1) | 92.8 | 79.6 | 70.0 | 96.7 | 61.4 | 50.6 | 0.0 |

**FSM ablation** (entry_jitter σ=0.25 m to stress-test recovery):

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feas. |
|---|---|---|---|---|---|---|---|---|
| **Geometry-FSM (full)** | **64.1 (0.9)** | **82.1** | **74.6** | **65.9** | **94.8** | 45.8 | 45.7 | 0.0 |
| FSM w/o recovery | 64.6 (0.7) | 80.3 | 77.1 | 62.2 | 95.3 | 52.3 | 48.2 | 0.0 |
| FSM w/o alignment | 19.1 (0.4) | 44.5 | **6.1** | **6.5** | 23.0 | 24.8 | 16.5 | 0.0 |

`false_feasible` corridors (physically impassable) yield 0% SR with 0% collision — correctly
rejected by body-margin gating. Memory variants are identical to base FSM in single-run eval;
differentiation requires multi-round repeated-passage experiments (see `eval_memory_differentiation.py`).

**Cross-episode Memory Differentiation** (`eval_memory_differentiation.py`):

| Round | no-memory FF attempt% | with-memory FF attempt% | wasted steps |
|---|---|---|---|
| 1 | 100% | 40% | 400 |
| 2+ | 100% | **0%** | **0** |

From round 2 onward, with-memory correctly rejects all false-feasible passages (0 wasted steps)
while the memoryless agent continues to waste 400 steps per episode. Narrow passages (true passable)
are always attempted correctly by both methods (SR 100%).

### Fair RL Reward Wrapper

Learning baselines now use `FairNarrowPassageRewardWrapper` by default in
`train_sb3_v2.py`, `train_recurrent_ppo_v2.py`, and
`train_replay_memory_policy_v2.py`.  The wrapper removes an open-space reward
loophole in the native v2 reward: raw `body_margin` is about 4.82 m outside the
passage, so an RL policy could collect a large clearance bonus by staying near
the entrance until timeout.  The fair reward clips clearance to the narrow-passage
scale and adds timeout / no-progress / outside-idle penalties.

Use `--reward-mode native` only to reproduce legacy runs.  New PPO, SAC, TD3,
RecurrentPPO, and Replay Memory Policy baselines should use the default:
`--reward-mode fair`.

```bash
python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500
```

### Trajectory Visualization

Top-down figures comparing method behavior across corridor types:

```bash
python examples/narrow_passage_rl/render_trajectories.py \
    --panels l_shaped s_shaped false_feasible narrow_entry \
    --dpi 150
# Output → results/narrow_passage_rl/trajectory_comparison.png
```

Mode color scheme: COMMIT=green · EXPLORE=orange · ALIGN=red · RECOVER=dark-red ·
FOLLOW_SPACE=teal · rule_baseline=blue.

---

### D_min Self-Calibration (300 synthetic episodes)

| Agent | SR | Reject rate | Notes |
|---|---|---|---|
| Oracle (D_true = 0.36 m) | 94.0% | 0% | Perfect knowledge |
| Fixed wrong (D_hat = 0.56 m) | 67.7% | 32% | Rejects feasible passages |
| **Calibrated (ours)** | **89.3%** | 8% | Bayesian update from outcomes |

D_hat converges from 0.56 m → 0.39 m (9% error) within ~75 episodes.

---

## Repository Layout

```
examples/narrow_passage_rl/
├── configs/
│   ├── train_ppo.yaml              # PPO protocol notes; curriculum planned, not auto-run
│   ├── eval_baselines.yaml         # Baseline evaluation protocol
│   ├── eval_ablation.yaml          # Module ablation protocol
│   └── sim2real.yaml               # Synthetic-to-Habitat transfer checks
│
├── narrow_passage/                 # Paper-facing research framework
│   ├── envs/
│   │   ├── narrow_passage_task.py  # Stable adapter for NarrowPassageNav-v0
│   │   ├── passage_generator.py    # Mining / dataset generation entry points
│   │   └── metrics.py              # Success, safety, memory-specific metrics
│   ├── models/
│   │   ├── geometry_encoder.py     # Explicit passage geometry vector → z_g
│   │   ├── risk_head.py            # Risk fusion / traversability estimator
│   │   ├── failure_memory.py       # Geometry-aware failure memory bank
│   │   └── policy.py               # Commit / Explore / Recover / Reject modes
│   ├── baselines/
│   │   └── registry.py             # Four-layer baseline taxonomy
│   ├── planners/
│   │   ├── astar_baseline.py       # Global graph-search baseline spec
│   │   ├── rrt_baseline.py         # Sampling baseline spec
│   │   ├── dwa_baseline.py         # DWB/DWA local planner adapter spec
│   │   ├── smac_baseline.py        # Smac Hybrid-A* / State Lattice spec
│   │   └── mppi_baseline.py        # MPPI local planner adapter spec
│   └── scripts/
│       ├── train.py                # Stable training entry point
│       ├── evaluate.py             # Stable evaluation entry point
│       ├── run_ablation.py         # Stable ablation entry point
│       ├── make_paper_tables.py    # Table rendering entry point
│       └── visualize_memory.py     # Memory visualization entry point
│
├── docs/
│   ├── method.md                   # Five-module algorithm description
│   ├── baseline_taxonomy.md        # Four-layer baseline suite and feasibility
│   ├── experiment_protocol.md      # Seeds, scenes, metrics, definitions
│   ├── habitat_stress_validation.md # Habitat 100% FSM stress-test protocol
│   └── reproducibility.md          # Environment, checkpoints, commands
│
├── results/
│   ├── raw/                        # Raw future experiment dumps
│   ├── tables/                     # Paper table outputs
│   ├── figures/                    # Paper figures
│   └── narrow_passage_rl/          # Current result CSV/MD/PNG artifacts
│
Legacy implementation scripts retained for reproducibility:
├── procedural_env.py               # Synthetic 2-D corridor simulator (primary)
├── procedural_env_v2.py            # v2: L/S-shaped, asymmetric, false-feasible corridors
├── risk_estimator.py               # Geometric risk estimator (width from depth)
├── failure_memory.py               # Episode-local failure memory
├── cross_episode_memory.py         # Cross-episode failure memory (persistent)
├── dmin_calibrator.py              # Bayesian D_min self-calibration module
├── run_rl_experiments.py           # Synthetic ablation suite
│
├── eval_harder_benchmark.py        # v2 harder benchmark (TurnCommitFSM + 7 corridor types)
├── eval_memory_differentiation.py  # Cross-episode memory differentiation experiment
├── eval_habitat_geometry_fsm.py    # Geometry-FSM evaluator on Habitat
├── eval_habitat_apf_gap.py         # APF+Gap classical baseline
├── eval_habitat_ppo_policy.py      # Trained NarrowPassagePolicy evaluator
├── eval_habitat_fsm_ablations.py   # FSM variant ablation (no_recovery / no_alignment)
├── eval_dmin_calibration.py        # D_min self-calibration experiment
├── render_trajectories.py          # Top-down trajectory visualization (paper figures)
│
├── mine_habitat_passages.py        # Auto-mine narrow passages from HM3D scenes
├── generate_habitat_episodes.py    # Anchor CSV → Habitat JSON dataset
├── eval_multi_agent_memory.py      # Cross-episode memory (Experiment ①)
├── eval_memory_baselines.py        # kNN / vanilla episodic / geometry-guided memory comparison
├── eval_inference_speed.py         # FSM vs RL inference latency comparison (Experiment ⑥)
├── train_sb3_v2.py                 # SB3 PPO/SAC/TD3 training on v2 env (fair RL baseline)
├── train_gru_ppo_v2.py             # Lightweight PyTorch GRU-PPO baseline fallback
├── train_recurrent_ppo_v2.py       # Formal SB3-Contrib RecurrentPPO baseline
├── collect_expert_trajectories.py  # FSM expert data collector for BC/DAgger
├── train_bc_dagger_v2.py           # BC and DAgger imitation baselines
├── replay_memory_wrapper.py        # Generic replay-memory observation wrapper
├── train_replay_memory_policy_v2.py # PPO/SAC/TD3 with generic replay memory input
├── make_paper_tables.py            # Render Markdown / LaTeX result tables
├── plot_delta_d_phase.py           # ΔD phase-transition curve (FSM vs baselines)
├── plot_dmin_calibration.py        # D_min calibration convergence figure
│
└── results/narrow_passage_rl/
    ├── results_rl_summary.csv              # All results (synthetic + Habitat)
    ├── harder_benchmark_summary.csv        # v2 harder benchmark results
    ├── harder_benchmark_episodes.csv       # v2 per-episode details
    ├── paper_table_main.{md,tex}           # Synthetic main table
    ├── paper_table_ablation.{md,tex}       # Synthetic ablation
    ├── paper_table_habitat.{md,tex}        # HM3D main comparison table
    ├── paper_table_habitat_ablation.{md,tex} # HM3D FSM ablation
    ├── memory_differentiation.csv          # Multi-round memory reject experiment
    ├── memory_baselines.csv                # kNN / vanilla episodic / geometry-guided memory results
    ├── paper_table_memory_baselines.md     # Memory baseline comparison table
    ├── gru_ppo_v2_eval.csv                 # Lightweight GRU-PPO synthetic v2 evaluation
    ├── gru_ppo_v2_summary.csv              # Lightweight GRU-PPO summary row
    ├── paper_table_learning_baselines.md   # PPO / SAC / GRU-PPO status table
    ├── paper_table_new_baselines_smoke.md  # RecurrentPPO / BC / DAgger / replay smoke table
    ├── ppo_seed{0,1,2}_episodes.csv        # PPO 3-seed Habitat eval (5.7%, 0.0%, 0.6%)
    ├── delta_d_phase.png                   # ΔD phase-transition figure
    ├── dmin_calibration.png                # D_min calibration figure
    ├── traj_seed*.png                      # Top-down trajectory comparison figures
    ├── habitat_fsm_v2_episodes.csv         # Per-episode FSM results (157 ep)
    ├── habitat_memory_v2_episodes.csv      # Per-episode FSM+memory results
    ├── habitat_apf_gap_episodes.csv        # Per-episode APF+Gap results
    ├── habitat_sb3_hard_episodes.csv       # PPO-SB3 on Habitat (fixed formula → 0.0%)
    ├── ppo_v2_episodes.csv                 # PPO v2 checkpoint on Habitat (5.7%)
    ├── sb3_on_v2_episodes.csv              # PPO-SB3 on v2 synthetic env (0.0%)
    ├── habitat_ppo_policy_episodes.csv     # Per-episode trained policy results (v1)
    ├── habitat_sac_v2_mined_val.csv        # SAC v2 checkpoint on mined Habitat val (0.0%)
    ├── habitat_fsm_extreme_ablation_perturb60.csv # Habitat stress ablation (+60° heading)
    └── habitat_fsm_ablation_episodes.csv   # FSM ablation (3 variants × 157 ep)

habitat-lab/habitat/tasks/narrow_passage/
├── narrow_passage_task.py     # NarrowPassageNav-v0 task, sensors, measures
├── rewards.py                 # NarrowPassageReward
├── sensors.py                 # NarrowPassageGeometrySensor (depth → 19-dim features)
└── geometry.py                # Shared geometry helpers (FEATURE_NAMES, MEMORY_FEATURE_NAMES)

habitat-baselines/habitat_baselines/
├── config/narrow_passage/ppo_narrow_passage.yaml   # Habitat-Baselines smoke-test config
└── rl/ppo/narrow_passage_policy.py                 # NarrowPassagePolicy (GRU + Gaussian)
```

---

## Dataset

HM3D val split mined with `mine_habitat_passages.py`. Scene-level split.

**Val set A** (original, episode-level mining from first pass):

| Split | Episodes | Scenes | Narrow | Normal | Wide |
|---|---|---|---|---|---|
| train | 638 | 82 | 320 | 224 | 94 |
| val A | 157 | 20 | 79 | 55 | 23 |

**Val set B** (independent second mining pass, --target-episodes 800):

| Split | Episodes | Scenes | Narrow | Normal | Wide |
|---|---|---|---|---|---|
| train | 608 | 80 | — | — | — |
| val B | 151 | 20 | 73 | 55 | 23 |

Difficulty: `narrow` = body_margin ≤ 0.15 m · `normal` = 0.15–0.40 m · `wide` > 0.40 m.
Tightest episode: body_margin ≈ 0.011 m (≈ half a finger of clearance).

---

## Installation

```bash
pip install -e habitat-lab/
pip install -e habitat-baselines/
pip install stable-baselines3 gymnasium numpy matplotlib
```

For Habitat experiments: install `habitat-sim` and place HM3D data under
`data/scene_datasets/hm3d/`.

---

## Synthetic Environment Experiments

```bash
# Full ablation suite (primary synthetic env)
python examples/narrow_passage_rl/run_rl_experiments.py

# FSM + failure memory (quick eval)
python examples/narrow_passage_rl/eval_mode_fsm.py --use-memory 1

# v2 harder benchmark (L/S-shaped, asymmetric, false-feasible)
python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500

# Memory/history baselines for repeated false-feasible passages
python examples/narrow_passage_rl/eval_memory_baselines.py \
    --n-rounds 5 --n-passable 20 --n-ff 15 --max-steps 220

# Lightweight GRU-PPO baseline fallback when sb3-contrib is unavailable
python examples/narrow_passage_rl/train_gru_ppo_v2.py \
    --total-steps 5000 --rollout-steps 512 --epochs 3 \
    --batch-size 128 --eval-episodes 50 \
    --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/gru_ppo_v2_smoke

# Formal SB3-Contrib RecurrentPPO smoke baseline
python examples/narrow_passage_rl/train_recurrent_ppo_v2.py \
    --reward-mode fair \
    --total-steps 1024 --n-steps 128 --batch-size 64 --eval-episodes 40 \
    --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_ppo_v2_smoke

# Expert data → BC / DAgger smoke baselines
python examples/narrow_passage_rl/collect_expert_trajectories.py \
    --episodes 40 --max-steps 300 \
    --output-npz examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke.npz \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke_episodes.csv
python examples/narrow_passage_rl/train_bc_dagger_v2.py \
    --algo bc \
    --dataset examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke.npz \
    --epochs 5 --eval-episodes 40
python examples/narrow_passage_rl/train_bc_dagger_v2.py \
    --algo dagger \
    --dataset examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke.npz \
    --epochs 3 --dagger-iters 1 --dagger-episodes 10 --eval-episodes 40

# Replay Memory Policy smoke baseline
python examples/narrow_passage_rl/train_replay_memory_policy_v2.py \
    --reward-mode fair \
    --algo ppo --total-steps 1024 --eval-episodes 40 \
    --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/replay_memory_policy_v2_smoke

# D_min self-calibration experiment (300 episodes)
python examples/narrow_passage_rl/eval_dmin_calibration.py --n-episodes 300
python examples/narrow_passage_rl/plot_dmin_calibration.py
```

---

## Habitat Experiments

### 1. Mine the Dataset

```bash
conda activate habitat
python examples/narrow_passage_rl/mine_habitat_passages.py \
    --scenes-dir data/scene_datasets/hm3d/val \
    --target-episodes 800 \
    --out-train data/datasets/narrow_passage/anchors_train.csv \
    --out-val   data/datasets/narrow_passage/anchors_val.csv
```

### 2. Generate JSON Datasets

```bash
python examples/narrow_passage_rl/generate_habitat_episodes.py \
    --anchors data/datasets/narrow_passage/anchors_train.csv \
    --split train --output data/datasets/narrow_passage/train/train.json.gz

python examples/narrow_passage_rl/generate_habitat_episodes.py \
    --anchors data/datasets/narrow_passage/anchors_val.csv \
    --split val --output data/datasets/narrow_passage/val/val.json.gz
```

### 3. Habitat-Baselines PPO Smoke Test

`habitat-baselines/habitat_baselines/config/narrow_passage/ppo_narrow_passage.yaml`
is a smoke-test config for the NarrowPassageNav-v0 task components and the
custom policy wiring.  It is not the main paper PPO training pipeline.  The
paper learning baselines use SB3/SB3-Contrib scripts under
`examples/narrow_passage_rl/`.

```bash
conda activate habitat
python -m habitat_baselines.run \
    --config-name narrow_passage/ppo_narrow_passage \
    habitat_baselines.total_num_steps=5e6 \
    habitat_baselines.num_environments=4
# Checkpoints → data/narrow_passage_checkpoints/
```

### 4. Evaluate All Methods

```bash
conda activate habitat

# Geometry-FSM (no memory)
python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py --use-memory 0

# Geometry-FSM + failure memory
python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py --use-memory 1

# APF+Gap classical baseline
python examples/narrow_passage_rl/eval_habitat_apf_gap.py

# Trained NarrowPassagePolicy (RL)
python examples/narrow_passage_rl/eval_habitat_ppo_policy.py \
    --ckpt data/narrow_passage_checkpoints/latest.pth

# FSM ablations (no_recovery / no_alignment)
python examples/narrow_passage_rl/eval_habitat_fsm_ablations.py \
    --variants full no_recovery no_alignment

# Stress validation: yaw perturbation on extreme-narrow anchors
python examples/narrow_passage_rl/eval_habitat_fsm_ablations.py \
    --split extreme_narrow \
    --variants full no_heading_alignment no_lateral_alignment \
    --heading-perturb-deg 60 \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/habitat_fsm_stress_yaw60.csv

# Stress validation: lateral offset and noisy geometry features
python examples/narrow_passage_rl/eval_habitat_fsm_ablations.py \
    --split extreme_narrow \
    --variants full no_recovery no_alignment \
    --lateral-perturb-m 0.2 \
    --feature-noise-std 0.03 \
    --depth-dropout-prob 0.10 \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/habitat_fsm_stress_lat_noise.csv
```

### 5. Generate Paper Tables and Figures

```bash
python examples/narrow_passage_rl/make_paper_tables.py \
    --input      examples/narrow_passage_rl/results/narrow_passage_rl/results_rl_summary.csv \
    --output-dir examples/narrow_passage_rl/results/narrow_passage_rl/

python examples/narrow_passage_rl/plot_delta_d_phase.py
python examples/narrow_passage_rl/plot_dmin_calibration.py
```

---

## Method Summary

### Geometry-FSM

Mode-switching controller driven by live depth-derived passage features:

| Mode | Trigger | Action |
|---|---|---|
| ALIGN | \|heading_error\| > 40° | Rotate in place toward goal |
| COMMIT | Normal geometry | 0.20 m/s forward + alignment correction |
| EXPLORE | Medium misalignment | 0.08 m/s forward + stronger correction |
| RECOVER | Collision or stuck_score > 0.80 | Back up + reorient |

Features (19-dim from `NarrowPassageGeometrySensor`): depth at 6 sectors, clearance left/right, passage width, body margin, heading error, lateral offset, distance to goal, velocities, stuck score, collision flag.

### TurnCommitFSM (v2 extension)

Wraps the base Geometry-FSM with a junction-detection layer for L/S-shaped corridors.
Detects corner entry via near-ray asymmetry (`max_side − min_side > 2.0`, `min_side < 0.30 m`,
`d_center ≤ 1.5 × min_side`), then commits to the open-arm direction for up to 60 steps.
Natural exit fires when `peak_heading_error > 30°` and `abs(heading_error) < 0.35 rad`,
followed by an 8-step cooldown to prevent immediate false re-trigger.

### APF+Gap Baseline

Artificial Potential Field (Khatib 1986) + Gap Navigation (Meng & Burdick 2002).
Uses only raw depth + GPS/compass — no trained model. Relative gap threshold
`max(0.35 m, depth_max × 0.60)` adapts to passage width.

### D_min Self-Calibration

Bayesian posterior `p(D_min | traversal outcomes)` over a 200-bin grid.
Update rule: success at width W → `likelihood ∝ sigmoid((W - D_min) / σ)`.
With ε=0.25 epsilon-exploration, converges from D_init=0.56 m to within 9%
of D_true=0.36 m in ~75 episodes.

---

## Sim-to-Real Transfer

The FSM is a feature-driven controller: the same code transfers to a real quadruped
without retraining, as long as the 19-dim observation vector can be reproduced from
real sensors.

### Sensor Mapping

| Feature | Real-robot source |
|---|---|
| `d_ln, d_cn, d_rn, d_lf, d_cf, d_rf` | Depth camera (e.g. RealSense D435): min-pool at 6 fixed azimuth angles on the horizontal projection |
| `cl, cr, passage_width, body_margin` | Min-distance left/right from depth scan minus robot effective radius |
| `heading_error, lateral_offset` | SLAM / UWB localization + goal position |
| `dist_to_goal` | Same localization |
| `stuck_score, collision` | Velocity estimate + contact force / IMU jerk |

### Control Interface

FSM outputs `(v_x, ω_z)`. Map to the quadruped's locomotion controller velocity
interface. Verify `ω_z` sign convention (CW/CCW). Scale `v_x_max` to the gait range.

### Geometry Calibration

1. Run `eval_dmin_calibration.py` (or `dmin_calibrator.py` on-robot) to calibrate
   the effective body radius from traversal outcomes (~75 episodes to converge).
2. Log depth features from a known L-shaped corner; verify `asymmetry = max_side − min_side`
   exceeds 2.0 before the junction. If not, lower the threshold in `_follow_space_mode`.

### Deployment Checklist

1. Reproduce the 19-dim vector in a known corridor; compare against simulator output.
2. Test straight corridors (FSM without TurnCommitFSM): verify `CORRIDOR_FOLLOW`
   and `RECOVER` modes.
3. Test L-shaped corridor: verify `FOLLOW_SPACE` trigger fires at the junction.
4. Run D_min calibration on-robot.
5. Enable `cross_episode_memory.py` (transfers unchanged).

### Main Sim-to-Real Risks

| Risk | Mitigation |
|---|---|
| Depth noise / missing values (glass, dark) | Median-filter sectors; require minimum valid-point count |
| Quadruped effective radius varies with gait | D_min calibrator; add 5 cm margin to `min_side` threshold |
| Localization drift | Local odometry for `lateral_offset`; global SLAM for `heading_error` |
| `FOLLOW_SPACE` fails at real L-junction | Log asymmetry signal; lower threshold or add dead-reckoning fallback |

---

## Key Implementation Note: Heading Error Sign

The original `atan2(delta_x, -delta_z)` formula caused heading_error = 0 when
the robot faced **away** from the goal. Habitat's forward direction is
`[-sin(yaw), 0, -cos(yaw)]`; the correct formula is `atan2(-delta_x, -delta_z)`.
The same fix applies to `generate_habitat_episodes.py::yaw_from_start_goal`.
