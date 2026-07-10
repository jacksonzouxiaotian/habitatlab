# Reproducibility

This document lists the environment, data paths, commands, expected outputs, and
known limitations for reproducing the narrow-passage results.

## Environment

The repository is a Habitat-Lab fork.  Habitat experiments require a working
Habitat/HM3D setup with EGL-capable rendering.

Recommended environment:

| Component | Version / note |
|---|---|
| Python | 3.9+ |
| Habitat-Lab | this fork, based on Habitat-Lab v0.3.3 |
| Habitat-Sim | installed in the `habitat` conda environment |
| NumPy | 1.26.x in the tested environment |
| PyTorch | CUDA-capable install recommended for training |
| Stable-Baselines3 | 2.7.1 for SB3 baselines |
| sb3-contrib | 2.7.1 for RecurrentPPO |
| GPU | NVIDIA GPU with EGL support for Habitat rendering |

Useful checks:

```bash
python --version
python -c "import habitat; print(habitat.__version__)"
python -c "import stable_baselines3; print(stable_baselines3.__version__)"
python -c "import sb3_contrib; print(sb3_contrib.__version__)"
nvidia-smi
```

## Required Data Paths

HM3D scenes:

```text
data/scene_datasets/hm3d/
```

Narrow-passage datasets:

```text
data/datasets/narrow_passage/train/train.json.gz
data/datasets/narrow_passage/val/val.json.gz
```

Optional anchor CSVs:

```text
data/datasets/narrow_passage/anchors_train.csv
data/datasets/narrow_passage/anchors_val.csv
```

Large HM3D assets and trained checkpoints are not committed to the repository.

## Main Commands

Run from repository root.

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

# D_min calibration
python examples/narrow_passage_rl/eval_dmin_calibration.py

# Regenerate paper tables from available CSV artifacts
python examples/narrow_passage_rl/make_paper_tables.py
```

Use `conda run -n habitat` or activate the `habitat` environment for Habitat
commands if your default shell is not already in that environment.

## Dataset Generation

If the narrow-passage dataset is missing, mine anchors and generate episodes:

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

## Expected Output Files

Main Markdown/LaTeX tables:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_main.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_ablation.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_habitat.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_habitat_stress.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_formal_baselines.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_diagnostic_baselines.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_smoke_baselines.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_repeated_failure_memory.md
```

Important raw CSVs:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/harder_benchmark_episodes.csv
examples/narrow_passage_rl/results/narrow_passage_rl/habitat_stress_validation.csv
examples/narrow_passage_rl/results/narrow_passage_rl/repeated_failure_memory.csv
examples/narrow_passage_rl/results/narrow_passage_rl/habitat_td3_habitat_native_strict_eval.csv
```

Figures/videos:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/dmin_calibration.png
video_dir/narrow_passage_habitat/
results/narrow_passage_rl/keyframes/
```

## Known Limitations

- Habitat nominal 100% is nominal anchor validation, not a complete robustness
  proof.
- Habitat stress validation currently perturbs existing mined episodes; dynamic
  obstacles, false-feasible Habitat anchors, goal perturbation, and unseen-room
  splits require additional data/simulator work.
- Failure memory is evaluated through repeated false-feasible exposure and
  wasted-step reduction, not one-shot passable-anchor success.
- Habitat-Baselines PPO config is a smoke-test path; the paper-facing learning
  baselines are the SB3/SB3-Contrib scripts.
- Smoke baselines such as BC/DAgger/RecurrentPPO/Replay Memory Policy should
  remain appendix/status rows unless rerun with the full protocol.
- `make_paper_tables.py` regenerates tables from available CSV artifacts; it
  does not train models or mine HM3D scenes by itself.
