# Narrow-Passage Navigation with Geometric FSM and Failure Memory

This directory contains all experiments for the narrow-passage navigation paper.
We evaluate on two independent setups:

1. **Synthetic procedural environment** — primary benchmark, fully self-contained.
2. **Habitat real HM3D scenes** — generalization experiment using `NarrowPassageNav-v0`.

---

## Key Results

### Habitat HM3D Generalization (157 val episodes, 20 held-out scenes)

| Method | SR | Narrow | Normal | Wide | Notes |
|---|---|---|---|---|---|
| PPO-SB3 baseline | 10.2% | 3.8% | 18.2% | 13.0% | Trained on synthetic env |
| PPO w/ geometry sensor | 6.4% | 2.5% | 10.9% | 8.7% | 5M steps, 99%+ train SR → fails to generalize |
| APF+Gap (Khatib 1986, Meng 2002) | 93.6% | 92.4% | 100% | 82.6% | Depth + GPS only |
| **Geometry-FSM (ours)** | **100%** | **100%** | **100%** | **100%** | |
| **FSM + Failure Memory (ours)** | **100%** | **100%** | **100%** | **100%** | |

**FSM ablation** — all components remain at 100%, showing structural robustness:

| Variant | SR | Narrow | Normal | Wide |
|---|---|---|---|---|
| Full FSM | 100% | 100% | 100% | 100% |
| w/o recovery | 100% | 100% | 100% | 100% |
| w/o alignment | 100% | 100% | 100% | 100% |

---

### Harder Synthetic Benchmark — v2 (500 episodes, 7 corridor types)

The v2 benchmark extends the procedural environment to include L-shaped, S-shaped,
and false-feasible corridors. A geometry-guided `TurnCommitFSM` detects junction
asymmetry and commits to the correct turn direction.
RL-based policies (SB3 PPO) achieve near-zero SR on L/S-shaped types — this is the
primary negative result motivating the geometry-first approach.

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feas. |
|---|---|---|---|---|---|---|---|---|
| Rule baseline | 24.0% | 57.1% | 5.0% | 5.1% | 34.5% | 23.1% | 31.3% | 0% |
| **Geometry-FSM (ours)** | **71.8%** | **94.6%** | **86.1%** | **65.4%** | **94.8%** | 71.2% | 47.9% | 0% |
| FSM w/o alignment | 25.6% | 57.1% | 6.9% | 7.7% | 37.9% | 28.9% | 29.2% | 0% |

`false_feasible` corridors (physically impassable) always yield 0% — correctly
rejected by body-margin gating; collision rate = 0% throughout.

```bash
python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500
```

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
├── procedural_env.py               # Synthetic 2-D corridor simulator (primary)
├── procedural_env_v2.py            # v2: L/S-shaped, asymmetric, false-feasible corridors
├── risk_estimator.py               # Geometric risk estimator (width from depth)
├── failure_memory.py               # Episode-local failure memory
├── cross_episode_memory.py         # Cross-episode failure memory (persistent)
├── dmin_calibrator.py              # Bayesian D_min self-calibration module
├── run_rl_experiments.py           # Synthetic ablation suite
│
├── eval_harder_benchmark.py        # v2 harder benchmark (TurnCommitFSM + 7 corridor types)
├── eval_habitat_geometry_fsm.py    # Geometry-FSM evaluator on Habitat
├── eval_habitat_apf_gap.py         # APF+Gap classical baseline
├── eval_habitat_ppo_policy.py      # Trained NarrowPassagePolicy evaluator
├── eval_habitat_fsm_ablations.py   # FSM variant ablation (no_recovery / no_alignment)
├── eval_dmin_calibration.py        # D_min self-calibration experiment
│
├── mine_habitat_passages.py        # Auto-mine narrow passages from HM3D scenes
├── generate_habitat_episodes.py    # Anchor CSV → Habitat JSON dataset
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
    ├── delta_d_phase.png                   # ΔD phase-transition figure
    ├── dmin_calibration.png                # D_min calibration figure
    ├── habitat_fsm_v2_episodes.csv         # Per-episode FSM results (157 ep)
    ├── habitat_memory_v2_episodes.csv      # Per-episode FSM+memory results
    ├── habitat_apf_gap_episodes.csv        # Per-episode APF+Gap results
    ├── habitat_ppo_v2_episodes.csv         # Per-episode PPO-SB3 results
    ├── habitat_ppo_policy_episodes.csv     # Per-episode trained policy results
    └── habitat_fsm_ablation_episodes.csv   # FSM ablation (3 variants × 157 ep)

habitat-lab/habitat/tasks/narrow_passage/
├── narrow_passage_task.py     # NarrowPassageNav-v0 task, sensors, measures
├── rewards.py                 # NarrowPassageReward
├── sensors.py                 # NarrowPassageGeometrySensor (depth → 19-dim features)
└── geometry.py                # Shared geometry helpers (FEATURE_NAMES, MEMORY_FEATURE_NAMES)

habitat-baselines/habitat_baselines/
├── config/narrow_passage/ppo_narrow_passage.yaml   # PPO training config
└── rl/ppo/narrow_passage_policy.py                 # NarrowPassagePolicy (GRU + Gaussian)
```

---

## Dataset

HM3D val split mined with `mine_habitat_passages.py`. Scene-level 80/20 split.

| Split | Episodes | Scenes | Narrow | Normal | Wide |
|---|---|---|---|---|---|
| train | 638 | 82 | 320 | 224 | 94 |
| val | 157 | 20 | 79 | 55 | 23 |

Difficulty: `narrow` = body_margin ≤ 0.15 m · `normal` = 0.15–0.40 m · `wide` > 0.40 m.
Tightest val episode: body_margin ≈ 0.011 m (≈ half a finger of clearance).

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

### 3. Train PPO Baseline

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
