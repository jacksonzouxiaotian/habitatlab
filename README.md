# Geometry-Guided Narrow-Passage Navigation for Quadruped Robots

> **This repository is a fork of [Habitat-Lab](https://github.com/facebookresearch/habitat-lab).**
> Our contributions are confined to `examples/narrow_passage_rl/` and
> `habitat-lab/habitat/tasks/narrow_passage/`. All other files are the
> unmodified Habitat-Lab v0.3.3 codebase.

This repository contains simulation experiments for a paper on geometry-guided,
failure-aware narrow-passage navigation for quadruped robots. The approach uses a
hand-crafted Finite State Machine (FSM) driven by live depth-derived passage geometry.
Reinforcement learning is evaluated as a baseline and shown to fail at generalizing
to unseen corridor geometries — motivating the geometry-first design.

Real-robot experiments (quadruped hardware) are conducted separately and are
referenced in the paper.

---

## Our Contributions

```
examples/narrow_passage_rl/          ← all experiment code (new directory)
│
├── procedural_env.py                 # Synthetic 2-D corridor simulator
├── procedural_env_v2.py              # v2: L/S-shaped, asymmetric, false-feasible
├── risk_estimator.py                 # Geometric risk estimator
├── failure_memory.py                 # Episode-local failure memory
├── cross_episode_memory.py           # Cross-episode persistent failure memory
├── dmin_calibrator.py                # Bayesian D_min self-calibration
│
├── eval_harder_benchmark.py          # v2 benchmark + TurnCommitFSM
├── eval_habitat_geometry_fsm.py      # Geometry-FSM on Habitat HM3D
├── eval_habitat_apf_gap.py           # APF+Gap classical baseline
├── eval_habitat_ppo_policy.py        # Trained NarrowPassagePolicy (RL)
├── eval_habitat_fsm_ablations.py     # FSM ablation variants
├── eval_dmin_calibration.py          # D_min calibration experiment
│
├── mine_habitat_passages.py          # Auto-mine narrow passages from HM3D
├── generate_habitat_episodes.py      # Anchor CSV → Habitat JSON dataset
├── make_paper_tables.py              # Render Markdown / LaTeX tables
├── plot_delta_d_phase.py             # ΔD phase-transition figure
├── plot_dmin_calibration.py          # D_min calibration convergence figure
│
└── results/narrow_passage_rl/        # All result CSVs and paper tables

habitat-lab/habitat/tasks/narrow_passage/   ← new Habitat task (new directory)
├── narrow_passage_task.py            # NarrowPassageNav-v0 task + sensors
├── rewards.py
├── sensors.py                        # NarrowPassageGeometrySensor (depth → 19-dim)
└── geometry.py

habitat-baselines/habitat_baselines/
├── config/narrow_passage/ppo_narrow_passage.yaml   ← new training config
└── rl/ppo/narrow_passage_policy.py                 ← new policy network
```

---

## Key Results

### 1. Harder Synthetic Benchmark — v2 (500 episodes, 7 corridor types)

Tests generalization to L-shaped, S-shaped, and false-feasible corridors.
RL (PPO/SB3) achieves near-zero SR on L/S-shaped types.

Mean across 3 seeds (std in parentheses), 500 episodes per seed.

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feas. |
|---|---|---|---|---|---|---|---|---|
| Rule baseline | 25.4 (1.0) | 60.5 | 8.2 | 6.5 | 27.2 | 24.8 | 33.5 | 0.0 |
| **Geometry-FSM (ours)** | **70.3 (1.1)** | **92.8** | **79.6** | **70.0** | **96.7** | **61.4** | **50.6** | **0.0** |

FSM ablation (entry jitter σ=0.25 m): removing alignment drops Overall from 64.1% → 19.1% (−45 pp),
L-shaped from 74.6% → 6.1%. `false_feasible` corridors yield 0% SR — correctly rejected by
body-margin gating.

### 2. Habitat HM3D Generalization (157 val episodes, 20 held-out scenes)

| Method | SR | Narrow | Normal | Wide | Notes |
|---|---|---|---|---|---|
| PPO-SB3 baseline | 10.2% | 3.8% | 18.2% | 13.0% | Trained on synthetic env, fails to generalize |
| PPO w/ geometry sensor | 6.4% | 2.5% | 10.9% | 8.7% | 5M steps, 99%+ train SR — sim-to-real gap |
| APF+Gap (Khatib 1986) | 93.6% | 92.4% | 100% | 82.6% | Depth + GPS only, no learning |
| **Geometry-FSM (ours)** | **100%** | **100%** | **100%** | **100%** | |
| **FSM + Failure Memory (ours)** | **100%** | **100%** | **100%** | **100%** | |

FSM ablation — all variants stay at 100%, showing structural robustness.

### 3. D_min Self-Calibration (300 synthetic episodes)

| Agent | SR | Reject rate | Notes |
|---|---|---|---|
| Oracle (D_true = 0.36 m) | 94.0% | 0% | Perfect body-width knowledge |
| Fixed wrong (D_hat = 0.56 m) | 67.7% | 32% | Over-rejects feasible passages |
| **Calibrated (ours)** | **89.3%** | 8% | Bayesian update from outcomes |

D_hat converges from 0.56 m → 0.39 m (9% error) within ~75 episodes.

---

## Method: Geometry-FSM

A mode-switching controller driven by 19-dimensional depth-derived features:

```
obs = [d_ln, d_cn, d_rn,          # near depth: left / center / right
       d_lf, d_cf, d_rf,          # far depth
       cl, cr,                    # clearance left / right
       passage_width, body_margin, # passage geometry
       heading_error, lateral_offset, dist_to_goal,
       action[0], action[1],      # previous velocities
       stuck_score, collision,
       prev_action[0], prev_action[1]]
```

| Mode | Trigger | Action |
|---|---|---|
| ALIGN | \|heading_error\| > 40° | Rotate in place toward goal |
| COMMIT | Normal geometry | 0.20 m/s forward + alignment correction |
| EXPLORE | Medium misalignment | 0.08 m/s forward + stronger correction |
| RECOVER | Collision or stuck\_score > 0.80 | Back up + reorient |
| FOLLOW\_SPACE | L/S-junction: depth asymmetry > 2.0 m | Commit to open-arm direction |

**TurnCommitFSM** (v2): wraps the base FSM with junction detection.
Detects corner entry via near-ray asymmetry, commits to the open-arm direction
for up to 60 steps. Natural exit fires when `peak_he > 30°` AND `|he| < 0.35 rad`.

---

## Sim-to-Real Transfer

The FSM is a feature-driven controller: as long as the 19-dim feature vector can
be reproduced on the real robot, the controller transfers without retraining.

### Sensor Mapping

| Feature | Real-robot source |
|---|---|
| `d_ln, d_cn, d_rn, d_lf, d_cf, d_rf` | Depth camera (e.g. RealSense D435): min-pool at 6 fixed azimuth angles on the horizontal projection |
| `cl, cr, passage_width, body_margin` | Min-distance left/right obstacle from depth scan, minus robot effective radius |
| `heading_error, lateral_offset` | SLAM / UWB localization + goal position |
| `dist_to_goal` | Same localization |
| `stuck_score, collision` | Velocity estimate + contact force sensors / IMU jerk |

### Control Interface

FSM outputs `(v_x, ω_z)` velocity commands. Map linearly to the quadruped's
locomotion controller velocity interface. Confirm sign convention (CW/CCW for `ω_z`).
The FSM's `v_x_max ≈ 0.06–0.12 m/s` is a soft limit; scale to the quadruped's
gait range as needed.

### Geometry Calibration

Thresholds in `_follow_space_mode` (`min_side < 0.30 m`, `asymmetry > 2.0`) were
tuned for the synthetic 2-D environment. For the real robot:

1. Run `dmin_calibrator.py` in an open corridor — Bayesian-updates D_min from
   traversal outcomes, converges within ~75 episodes.
2. Log depth features from a known L-shaped corner and verify `asymmetry` reaches
   the 2.0 threshold before the junction. If not, adjust the threshold down.

### Deployment Steps

1. **Feature reproduction** — log the 19-dim vector on the robot in a known corridor
   and compare numerically against the simulator for the same geometry.
2. **Straight-corridor test** — deploy FSM without TurnCommitFSM; verify
   `CORRIDOR_FOLLOW` and `RECOVER` modes behave as expected.
3. **L-shaped test** — enable TurnCommitFSM; verify `FOLLOW_SPACE` triggers at the
   junction (check depth asymmetry signal in real time).
4. **D_min calibration** — run `dmin_calibrator.py` on-robot to adapt to real
   body dimensions.
5. **Failure memory** — `cross_episode_memory.py` transfers unchanged.

### Main Sim-to-Real Risks

| Risk | Mitigation |
|---|---|
| Depth noise / missing values (glass, dark surfaces) | Median-filter depth sectors; require a minimum valid-point count |
| Quadruped effective radius varies with gait | Use D_min calibrator; add 5 cm safety margin to `min_side` threshold |
| Localization drift in long corridors | Use local odometry for short-horizon `lateral_offset`; global for `heading_error` |
| `FOLLOW_SPACE` fails at real L-junction | Log asymmetry signal; lower threshold or add dead-reckoning fallback |

---

## Installation

```bash
# Base Habitat stack
pip install -e habitat-lab/
pip install -e habitat-baselines/

# Experiment dependencies
pip install stable-baselines3 gymnasium numpy matplotlib
```

For Habitat experiments: install `habitat-sim` and place HM3D data under
`data/scene_datasets/hm3d/`. Use the `habitat` conda environment.

---

## Quick Start

```bash
# Synthetic v2 harder benchmark (no Habitat required)
python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500

# Habitat HM3D — Geometry-FSM
conda run -n habitat python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py

# Habitat HM3D — APF+Gap baseline
conda run -n habitat python examples/narrow_passage_rl/eval_habitat_apf_gap.py

# D_min self-calibration
python examples/narrow_passage_rl/eval_dmin_calibration.py --n-episodes 300

# Generate paper tables
python examples/narrow_passage_rl/make_paper_tables.py \
    --input      examples/narrow_passage_rl/results/narrow_passage_rl/results_rl_summary.csv \
    --output-dir examples/narrow_passage_rl/results/narrow_passage_rl/
```

Full experiment details: [examples/narrow_passage_rl/README.md](examples/narrow_passage_rl/README.md)

---

## Dataset

HM3D val split, auto-mined with `mine_habitat_passages.py`. Scene-level 80/20 split.

| Split | Episodes | Narrow | Normal | Wide |
|---|---|---|---|---|
| train | 638 | 320 | 224 | 94 |
| val | 157 | 79 | 55 | 23 |

`narrow` = body_margin ≤ 0.15 m · `normal` = 0.15–0.40 m · `wide` > 0.40 m.

```bash
conda run -n habitat python examples/narrow_passage_rl/mine_habitat_passages.py \
    --scenes-dir data/scene_datasets/hm3d/val --target-episodes 800 \
    --out-train data/datasets/narrow_passage/anchors_train.csv \
    --out-val   data/datasets/narrow_passage/anchors_val.csv

python examples/narrow_passage_rl/generate_habitat_episodes.py \
    --anchors data/datasets/narrow_passage/anchors_train.csv \
    --split train --output data/datasets/narrow_passage/train/train.json.gz
```

---

## Implementation Notes

**Heading error sign fix**: `atan2(delta_x, −delta_z)` caused `heading_error = 0`
when the robot faced *away* from the goal. Habitat's forward direction is
`[−sin(yaw), 0, −cos(yaw)]`; correct formula is `atan2(−delta_x, −delta_z)`.
Without the fix, FSM achieved 2.7% on HM3D; after: 100%.

**body_margin formula**: Uses `min(clearance_left, clearance_right) − robot_radius`
(tight side, not average), reflecting actual worst-case clearance at the current
lateral position.

---

## Habitat-Lab Base

The rest of this repository is [Habitat-Lab v0.3.3](https://github.com/facebookresearch/habitat-lab)
by Meta AI Research, released under the MIT License.
