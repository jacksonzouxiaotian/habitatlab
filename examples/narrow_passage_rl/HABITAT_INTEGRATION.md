# Habitat Narrow-Passage RL Integration

This is the minimal Habitat-Lab path for turning the procedural narrow-passage
RL idea into scene-based experiments.

## 1. Create Manual Passage Anchors

Start with 20-50 hand-picked HM3D/MP3D/Replica passages. Write a CSV template:

```bash
python examples/narrow_passage_rl/generate_habitat_episodes.py \
  --write-template data/datasets/narrow_passage/anchors_template.csv
```

Edit the CSV so each row describes one passage:

- `scene_id`: Habitat scene path.
- `start_x,start_y,start_z`: pose before the passage.
- `goal_x,goal_y,goal_z`: local goal after the passage.
- `passage_x,passage_y,passage_z`: passage center.
- `passage_width`, `passage_length`, `difficulty`, `false_feasible`.

## 2. Generate Habitat PointNav-Compatible Episodes

```bash
python examples/narrow_passage_rl/generate_habitat_episodes.py \
  --anchors data/datasets/narrow_passage/anchors_train.csv \
  --split train \
  --output data/datasets/narrow_passage/train/train.json.gz
```

Use the same format for validation:

```bash
python examples/narrow_passage_rl/generate_habitat_episodes.py \
  --anchors data/datasets/narrow_passage/anchors_val.csv \
  --split val \
  --output data/datasets/narrow_passage/val/val.json.gz
```

## 3. Train

The first Habitat version uses a PointNav-like navigation agent, but replaces
the task observations and reward with narrow-passage components:

- `narrow_passage_features`: depth ROI + clearance + alignment + stuck state.
- `narrow_passage_memory`: episode-local failure memory features.
- `narrow_passage_reward`: progress + centering + alignment + clearance + failure penalties.

```bash
python -m habitat_baselines.run \
  --config-name narrow_passage/ppo_narrow_passage \
  --run-type train
```

If your checkout uses the older entry point:

```bash
python habitat-baselines/habitat_baselines/run.py \
  --config-name narrow_passage/ppo_narrow_passage \
  --run-type train
```

## 4. Evaluate

```bash
python -m habitat_baselines.run \
  --config-name narrow_passage/ppo_narrow_passage \
  --run-type eval \
  habitat.dataset.split=val
```

For the paper, report narrow-passage metrics instead of only SPL:

- Success rate
- Collision rate
- Stuck rate
- Minimum clearance
- False-feasible collision rate
- Recovery success rate
- Memory-triggered reject/retry rate

## 5. Recommended Ablations

Use the same episode set for every method:

- PointNav PPO: depth + pointgoal only.
- Geometry rule: hand-coded centerline/clearance controller.
- RL-only: `narrow_passage_features`, no memory sensor.
- Ours without memory: geometry + recovery, no `narrow_passage_memory`.
- Ours full: geometry + memory + failure-aware recovery/mode decision.
