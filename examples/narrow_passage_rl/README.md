# Narrow-Passage Navigation with Geometric FSM and Failure Memory

This directory contains all experiments for the narrow-passage navigation paper.
We evaluate on two independent setups:

1. **Synthetic procedural environment** — the primary benchmark, fully self-contained.
2. **Habitat real HM3D scenes** — generalization experiment using `NarrowPassageNav-v0`.

---

## Overview

Narrow-passage navigation is a hard subproblem in mobile robotics: the robot must
thread through corridors whose width is close to the robot's body diameter.
End-to-end RL policies struggle to generalize across scenes; classical geometric
controllers do not exploit cross-episode experience.

We propose a **Geometry-FSM** with an optional **Failure Memory** module:

- **Geometry-FSM**: a mode-switching controller (ALIGN → COMMIT → EXPLORE → RECOVER)
  driven by live depth-derived passage geometry (clearance, heading error, lateral offset).
- **Failure Memory**: stores geometric fingerprints of previously failed passages so the
  robot can *reject* visually similar dead-ends in future episodes without re-attempting them.

---

## Results

### Synthetic Procedural Environment (Primary)

| Method | Easy SR | Normal SR | Hard SR |
|--------|---------|-----------|---------|
| PPO baseline | — | — | ~5% |
| Geometry-FSM (no memory) | — | — | ~70% |
| **Ours (FSM + Memory)** | — | — | **83.2%** |

Full ablation table: `results/narrow_passage_rl/paper_table_ablation.{md,tex}`

### Habitat HM3D Generalization (157 val episodes, 20 scenes)

| Method | Success Rate | Collision Rate | Avg Min Clearance |
|--------|-------------|----------------|--------------------|
| PPO baseline (v2, clean) | 10.2% (16/157) | 0.0% | — |
| Geometry-FSM (no memory) | **100.0%** (157/157) | 0.0% | −0.088 m |
| **FSM + Failure Memory** | **100.0%** (157/157) | 0.0% | −0.088 m |

The PPO policy achieves ~99% success on its *training* scenes but only 10.2% on
held-out HM3D val scenes, confirming that geometric methods generalize where RL
overfits to scene appearance.

Full Habitat table: `results/narrow_passage_rl/paper_table_habitat.{md,tex}`

---

## Repository Layout

```
examples/narrow_passage_rl/
├── procedural_env.py           # Synthetic 2-D corridor simulator (primary benchmark)
├── risk_estimator.py           # Geometric risk estimator (passage width from depth)
├── failure_memory.py           # Cross-episode failure memory module
├── run_rl_experiments.py       # Run full synthetic ablation suite
├── eval_habitat_geometry_fsm.py  # Geometry-FSM evaluator on Habitat
├── mine_habitat_passages.py    # Auto-mine narrow passages from HM3D scenes
├── generate_habitat_episodes.py  # Convert anchor CSV → Habitat JSON dataset
├── make_paper_tables.py        # Render Markdown / LaTeX result tables
└── results/narrow_passage_rl/
    ├── results_rl_summary.csv      # All results (synthetic + Habitat)
    ├── paper_table_main.{md,tex}   # Main result table
    ├── paper_table_ablation.{md,tex}
    ├── paper_table_habitat.{md,tex}
    ├── habitat_fsm_v2_episodes.csv     # Per-episode FSM results
    └── habitat_memory_v2_episodes.csv  # Per-episode FSM+memory results

habitat-lab/habitat/tasks/narrow_passage/
├── narrow_passage_task.py   # NarrowPassageNav-v0 task, sensors, measures
├── rewards.py               # NarrowPassageReward
├── sensors.py               # NarrowPassageGeometrySensor (live depth features)
└── geometry.py              # Shared geometry helpers

habitat-baselines/habitat_baselines/config/narrow_passage/
└── ppo_narrow_passage.yaml  # PPO training config for NarrowPassageNav-v0
```

---

## Installation

```bash
# Clone and install habitat-lab (this repo)
pip install -e habitat-lab/
pip install -e habitat-baselines/

# Synthetic experiments only need:
pip install stable-baselines3 gymnasium numpy
```

For Habitat experiments, install `habitat-sim` and place HM3D scene data under
`data/scene_datasets/hm3d/`.

---

## Synthetic Environment Experiments

```bash
# Run full ablation suite (PPO / FSM / FSM+memory / ablations)
python examples/narrow_passage_rl/run_rl_experiments.py

# Quick single run: FSM + failure memory
python examples/narrow_passage_rl/eval_mode_fsm.py --use-memory 1

# Render a rollout
python examples/narrow_passage_rl/render_rollout.py --use-memory 1
```

---

## Habitat Experiments

### 1. Mine the Dataset

Automatically extract narrow-passage episodes from HM3D val scenes:

```bash
conda activate habitat
python examples/narrow_passage_rl/mine_habitat_passages.py \
    --scenes-dir data/scene_datasets/hm3d/val \
    --target-episodes 800 \
    --min-straightness 0.65 \
    --max-geodesic 10.0 \
    --out-train data/datasets/narrow_passage/anchors_train.csv \
    --out-val   data/datasets/narrow_passage/anchors_val.csv
```

Episodes are bucketed by body-margin difficulty:
`narrow` (0–0.15 m) · `normal` (0.15–0.40 m) · `wide` (> 0.40 m).

### 2. Generate Habitat Datasets

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
    habitat_baselines.checkpoint_folder=data/narrow_passage_checkpoints \
    habitat_baselines.total_num_steps=5e6 \
    habitat_baselines.num_environments=4
```

### 4. Evaluate Geometry-FSM

```bash
# Without failure memory (no-memory ablation)
python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py \
    --num-episodes -1 --use-memory 0

# With failure memory (full method)
python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py \
    --num-episodes -1 --use-memory 1 \
    --output-csv results/narrow_passage_rl/habitat_memory_episodes.csv
```

### 5. Generate Paper Tables

```bash
python examples/narrow_passage_rl/make_paper_tables.py \
    --habitat-input examples/narrow_passage_rl/results/narrow_passage_rl/results_rl_summary.csv \
    --output-dir   examples/narrow_passage_rl/results/narrow_passage_rl/
```

---

## Key Bug Fix: `_heading_error()` Sign

The original `atan2(delta_x, -delta_z)` formula caused heading_error = 0 when the
robot faced **away** from the goal. Habitat's forward direction is
`[-sin(yaw), 0, -cos(yaw)]`; the correct formula is `atan2(-delta_x, -delta_z)`.

This affected both the training reward and the success measure.
The same sign fix was applied to `generate_habitat_episodes.py::yaw_from_start_goal`.

---

## Dataset

The Habitat dataset (`data/datasets/narrow_passage/`) is generated from the
**HM3D val split** (100 scenes) using `mine_habitat_passages.py`.
Scene-level 80/20 split: 82 train scenes / 20 val scenes.

| Split | Episodes | Scenes | Narrow | Normal | Wide |
|-------|----------|--------|--------|--------|------|
| train | 638 | 82 | 320 | 224 | 94 |
| val   | 157 | 20 |  79 |  55 | 23 |

Tightest episode: body_margin ≈ 0.011 m (≈ half a finger of clearance).
