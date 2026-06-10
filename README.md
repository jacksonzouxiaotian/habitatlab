# Narrow-Passage RL and Failure Memory Experiments

This repository is a code-only export of the project-specific work for
narrow-passage navigation. It does not mirror the full Habitat-Lab repository.

The core idea is to use reinforcement learning as a local passage traversal and
recovery skill, not as a generic PointNav or ObjectNav policy:

```text
Geometry / Risk Estimator
        ↓
Failure-aware Mode Decision
        ↓
Mode-conditioned Local RL Policy
        ↓
Recovery / Retry / Reject
```

The experiments support four paper contributions:

1. Geometry-aware narrow-passage local policy.
2. Failure-aware recovery policy.
3. Passage-centric failure memory.
4. Mode decision over commit, align, recover, and reject.

## Repository Layout

```text
examples/narrow_passage_rl/
  procedural_env.py              # 2D procedural narrow-passage environment
  recovery_env.py                # collision/stuck recovery task
  risk_recovery_env.py           # pre-collision high-risk recovery task
  train_sb3.py                   # PPO training for passage traversal
  train_recovery_sb3.py          # PPO training for recovery
  train_risk_recovery_sb3.py     # PPO training for risk recovery
  eval_sb3.py                    # passage PPO evaluation
  eval_rule_baseline.py          # rule baseline evaluation
  eval_mode_fsm.py               # geometry/risk FSM evaluation
  eval_memory_fsm.py             # failure-memory-gated FSM evaluation
  run_rl_experiments.py          # one-command paper experiment runner
  make_paper_tables.py           # CSV to Markdown/LaTeX tables
  render_rollout.py              # rollout video renderer
  render_memory_demo.py          # explicit failure-memory demo renderer
  generate_habitat_episodes.py   # Habitat episode generator from anchors
  HABITAT_INTEGRATION.md         # scene-based Habitat validation notes

habitat-lab/habitat/tasks/narrow_passage/
  geometry.py                    # geometry and memory feature definitions
  sensors.py                     # NarrowPassageGeometrySensor, MemorySensor
  measures.py                    # success, collision, stuck measures
  rewards.py                     # narrow-passage reward
  narrow_passage_task.py         # Habitat task wrapper with local memory

habitat-baselines/habitat_baselines/
  config/narrow_passage/ppo_narrow_passage.yaml
  rl/ppo/narrow_passage_policy.py

results/narrow_passage_rl/
  results_rl_summary.csv
  paper_table_main.md/.tex
  paper_table_ablation.md/.tex
```

## Current Experimental Status

The procedural experiments already provide the main controlled evidence:

- Rule baseline versus PPO local passage policy.
- Ablations for failure state, clearance features, and curriculum.
- Standalone recovery policy.
- Risk recovery policy.
- Geometry FSM and memory-gated FSM.
- False-feasible and narrow-passage subgroup metrics.

Representative hard-setting results from the current run:

```text
Rule baseline:
  success_rate = 0.620
  collision_rate = 0.334

Passage PPO:
  success_rate = 0.832
  collision_rate = 0.168

Memory-gated FSM:
  success_rate = 0.834
  collision_rate = 0.150
  reject_rate = 0.016
```

## Environment

Use the same conda environment used for Habitat-Lab and Stable-Baselines3:

```bash
conda activate habitat
```

The procedural scripts require:

- Python 3.9+
- numpy
- gym or gymnasium
- stable-baselines3
- matplotlib for video rendering
- ffmpeg for mp4 output, or Pillow for gif output

## Main Procedural Experiments

Run all paper-style evaluations:

```bash
python examples/narrow_passage_rl/run_rl_experiments.py \
  --difficulty hard \
  --episodes 500
```

Generate paper tables:

```bash
python examples/narrow_passage_rl/make_paper_tables.py
```

Outputs:

```text
results/narrow_passage_rl/results_rl_summary.csv
results/narrow_passage_rl/paper_table_main.md
results/narrow_passage_rl/paper_table_ablation.md
results/narrow_passage_rl/paper_table_main.tex
results/narrow_passage_rl/paper_table_ablation.tex
```

## Training Commands

Passage policy with curriculum:

```bash
python examples/narrow_passage_rl/train_sb3.py \
  --difficulty easy \
  --total-steps 1000000 \
  --save-dir data/narrow_passage_sb3_easy

python examples/narrow_passage_rl/train_sb3.py \
  --difficulty hard \
  --total-steps 1000000 \
  --load-model data/narrow_passage_sb3_easy/ppo_narrow_passage.zip \
  --save-dir data/narrow_passage_sb3_hard
```

Recovery policy:

```bash
python examples/narrow_passage_rl/train_recovery_sb3.py \
  --difficulty hard \
  --total-steps 1000000 \
  --save-dir data/narrow_passage_recovery
```

Risk recovery policy:

```bash
python examples/narrow_passage_rl/train_risk_recovery_sb3.py \
  --difficulty hard \
  --total-steps 1000000 \
  --save-dir data/narrow_passage_risk_recovery
```

## Key Evaluation Commands

Rule baseline:

```bash
python examples/narrow_passage_rl/eval_rule_baseline.py \
  --difficulty hard \
  --episodes 500
```

Passage PPO:

```bash
python examples/narrow_passage_rl/eval_sb3.py \
  --model data/narrow_passage_sb3_hard/ppo_narrow_passage.zip \
  --difficulty hard \
  --episodes 500
```

Memory-gated FSM:

```bash
python examples/narrow_passage_rl/eval_memory_fsm.py \
  --passage-model data/narrow_passage_sb3_hard/ppo_narrow_passage.zip \
  --risk-recovery-model data/narrow_passage_risk_recovery/ppo_risk_recovery.zip \
  --difficulty hard \
  --episodes 500
```

## Video Generation

Memory demo without failure memory:

```bash
python examples/narrow_passage_rl/render_memory_demo.py \
  --max-attempts 2 \
  --output video_dir/memory_demo_no_memory.mp4
```

Memory demo with failure memory:

```bash
python examples/narrow_passage_rl/render_memory_demo.py \
  --use-memory \
  --max-attempts 2 \
  --output video_dir/memory_demo_with_memory.mp4
```

If ffmpeg is unavailable, use `.gif` output:

```bash
python examples/narrow_passage_rl/render_memory_demo.py \
  --use-memory \
  --max-attempts 2 \
  --output video_dir/memory_demo_with_memory.gif
```

## Habitat-Lab Scene Validation

This export includes the Habitat-Lab integration files, but not the full
Habitat-Lab source tree. Copy the files into a Habitat-Lab checkout using the
same relative paths.

Generate manual anchor template:

```bash
python examples/narrow_passage_rl/generate_habitat_episodes.py \
  --write-template data/datasets/narrow_passage/anchors_template.csv
```

Create train/val datasets from manually selected passage anchors:

```bash
python examples/narrow_passage_rl/generate_habitat_episodes.py \
  --anchors data/datasets/narrow_passage/anchors_train.csv \
  --split train \
  --output data/datasets/narrow_passage/train/train.json.gz

python examples/narrow_passage_rl/generate_habitat_episodes.py \
  --anchors data/datasets/narrow_passage/anchors_val.csv \
  --split val \
  --output data/datasets/narrow_passage/val/val.json.gz
```

Train in Habitat-Baselines:

```bash
python -m habitat_baselines.run \
  --config-name narrow_passage/ppo_narrow_passage \
  --run-type train
```

Evaluate:

```bash
python -m habitat_baselines.run \
  --config-name narrow_passage/ppo_narrow_passage \
  --run-type eval \
  habitat.dataset.split=val
```

## Paper Positioning

The RL policy should be described as a local skill conditioned on passage
geometry, risk, and failure memory. It should not be framed as a standalone
general PointNav or ObjectNav policy.

Recommended method table:

```text
Rule baseline
Passage PPO
PPO w/o failure state
PPO w/o clearance
PPO w/o curriculum
PPO + collision recovery
PPO + risk recovery
Geometry FSM
Memory-gated FSM
```
