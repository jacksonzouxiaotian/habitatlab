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
