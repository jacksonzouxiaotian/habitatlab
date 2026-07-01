# Reproducibility

## Environment

Record the exact versions used for final experiments:

```bash
python --version
python -c "import habitat; print(habitat.__version__)"
python -c "import habitat_sim; print(habitat_sim.__version__)"
nvidia-smi
```

Expected setup:

- Python: 3.9+
- Habitat-Lab: this fork
- Habitat-Sim: installed in the `habitat` conda environment
- CUDA/GPU: NVIDIA GPU with EGL support for Habitat-Sim rendering

## Data

HM3D scenes should be placed under:

```text
data/scene_datasets/hm3d/
```

Narrow-passage datasets:

```text
data/datasets/narrow_passage/train/train.json.gz
data/datasets/narrow_passage/val/val.json.gz
data/datasets/narrow_passage/extreme_narrow/extreme_narrow.json.gz
```

## Checkpoints

Default checkpoint paths:

```text
data/narrow_passage_sb3_v2_ppo/ppo_narrow_passage_v2.zip
data/narrow_passage_sb3_v2_sac/sac_narrow_passage_v2.zip
data/narrow_passage_sb3_v2_td3/td3_narrow_passage_v2.zip
data/narrow_passage_checkpoints/latest.pth
```

Large checkpoints and HM3D data are not committed to the repository.

## Reproduce Tables

Run the main experiments:

```bash
python examples/narrow_passage_rl/narrow_passage/scripts/train.py --algo ppo
python examples/narrow_passage_rl/narrow_passage/scripts/evaluate.py --method fsm
python examples/narrow_passage_rl/narrow_passage/scripts/run_ablation.py
python examples/narrow_passage_rl/narrow_passage/scripts/make_paper_tables.py
```

Run the newly added memory/history baselines:

```bash
python examples/narrow_passage_rl/eval_memory_baselines.py \
  --n-rounds 5 --n-passable 20 --n-ff 15 --max-steps 220
```

This writes:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/memory_baselines.csv
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_baselines.md
```

Run the lightweight GRU-PPO fallback baseline:

```bash
python examples/narrow_passage_rl/train_gru_ppo_v2.py \
  --total-steps 5000 --rollout-steps 512 --epochs 3 \
  --batch-size 128 --eval-episodes 50 \
  --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/gru_ppo_v2_smoke
```

This writes:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/gru_ppo_v2_eval.csv
examples/narrow_passage_rl/results/narrow_passage_rl/gru_ppo_v2_summary.csv
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_learning_baselines.md
```

Run the formal SB3-Contrib RecurrentPPO smoke baseline:

```bash
python examples/narrow_passage_rl/train_recurrent_ppo_v2.py \
  --total-steps 1024 --n-steps 128 --batch-size 64 --eval-episodes 40 \
  --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_ppo_v2_smoke \
  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/recurrent_ppo_v2_smoke_eval.csv \
  --output-summary examples/narrow_passage_rl/results/narrow_passage_rl/recurrent_ppo_v2_smoke_summary.csv
```

Collect FSM expert trajectories and run BC/DAgger smoke baselines:

```bash
python examples/narrow_passage_rl/collect_expert_trajectories.py \
  --episodes 40 --max-steps 300 \
  --output-npz examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke.npz \
  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke_episodes.csv

python examples/narrow_passage_rl/train_bc_dagger_v2.py \
  --algo bc \
  --dataset examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke.npz \
  --epochs 5 --batch-size 256 --eval-episodes 40 \
  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/bc_v2_smoke_eval.csv \
  --output-summary examples/narrow_passage_rl/results/narrow_passage_rl/bc_v2_smoke_summary.csv

python examples/narrow_passage_rl/train_bc_dagger_v2.py \
  --algo dagger \
  --dataset examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke.npz \
  --epochs 3 --batch-size 256 --dagger-iters 1 --dagger-episodes 10 \
  --eval-episodes 40 \
  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/dagger_v2_smoke_eval.csv \
  --output-summary examples/narrow_passage_rl/results/narrow_passage_rl/dagger_v2_smoke_summary.csv
```

Run the generic Replay Memory Policy smoke baseline:

```bash
python examples/narrow_passage_rl/train_replay_memory_policy_v2.py \
  --algo ppo --total-steps 1024 --eval-episodes 40 \
  --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/replay_memory_policy_v2_smoke \
  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/replay_memory_policy_v2_smoke_eval.csv \
  --output-summary examples/narrow_passage_rl/results/narrow_passage_rl/replay_memory_policy_v2_smoke_summary.csv
```

For Habitat experiments, use the `habitat` environment:

```bash
conda run -n habitat python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py
conda run -n habitat python examples/narrow_passage_rl/eval_habitat_sb3.py --algo ppo
conda run -n habitat python examples/narrow_passage_rl/eval_habitat_fsm_ablations.py
```

## Runtime Notes

- PPO v2 training: multi-hour CPU run depending on `total_steps`.
- SAC/TD3: slower CPU training; use GPU-capable PyTorch if available.
- Habitat evaluation requires EGL/GPU access outside restricted sandboxes.
- The `habitat` environment used for the formal RecurrentPPO smoke run has
  `stable-baselines3==2.7.1` and `sb3-contrib==2.7.1`.
- The older lightweight GRU-PPO row remains as a fallback result from the base
  Python environment; prefer `train_recurrent_ppo_v2.py` for final experiments.
