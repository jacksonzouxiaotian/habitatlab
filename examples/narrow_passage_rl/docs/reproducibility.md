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

Optional NaVILA/VLN assets used by the current diagnostics:

```text
/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/models/navila-llama3-8b-8f/
/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data/datasets/R2R_VLNCE_v1-3_preprocessed/
/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data/ddppo-models/gibson-2plus-resnet50.pth
/home/xiaotian/vla/NaVILA/
/home/xiaotian/vla/Open-Nav/
```

The R2R JSON files are present locally, but the matching MP3D scene assets are
not available in the current reproducibility snapshot.  VLN diagnostics
therefore report runtime, action parsing, instruction sensitivity, and adapter
behavior only; they do not report SR, SPL, NE, nDTW, or standard R2R/RxR
generalization.

Formal R2R/VLN-CE paired evaluation additionally requires licensed MP3D scene
assets in the NaVILA/VLN-CE Habitat layout:

```text
/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data/scene_datasets/mp3d/{scene}/{scene}.glb
/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data/scene_datasets/mp3d/{scene}/{scene}.navmesh
```

The current preflight confirms the NaVILA checkpoint, R2R `val_unseen`
annotations, DDPPO depth checkpoint, data symlinks, and Habitat 0.1.7 imports,
but reports 0/11 required `val_unseen` `.glb` files and 0/11 `.navmesh` files.
This is a data-availability blocker, not a code-path blocker.

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

NaVILA/VLN diagnostics:

```bash
# Model artifact/runtime smoke test
CUDA_VISIBLE_DEVICES=0 \
python examples/narrow_passage_rl/vln_dataset_free_smoke.py \
    --output-dir examples/narrow_passage_rl/results/vln_dataset_free

# 640-inference controlled batch diagnostic
CUDA_VISIBLE_DEVICES=0 \
python examples/narrow_passage_rl/vln_batch_diagnostic.py \
    --r2r-samples 256 \
    --output-dir examples/narrow_passage_rl/results/vln_batch_diagnostic

# R2R text-only corpus and language/action prior diagnostic
python examples/narrow_passage_rl/vln_r2r_text_analysis.py

# Paired deterministic NaVILA-to-DEGNAV safety-adapter replay
python examples/narrow_passage_rl/eval_vln_safety_adapter.py \
    --predictions examples/narrow_passage_rl/results/vln_batch_diagnostic/predictions.csv \
    --output-dir examples/narrow_passage_rl/results/vln_safety_adapter_smoke

# R2R/VLN-CE paired evaluation preflight
conda run -n navila-eval python examples/narrow_passage_rl/vln_r2r_preflight.py \
    --ensure-symlinks

# Open-Nav installation and asset preflight
PYTHONPATH=/home/xiaotian/vla/Open-Nav:/home/xiaotian/vla/habitat-lab-v0.1.7 \
conda run -n navila-eval \
python examples/narrow_passage_rl/opennav_preflight.py
```

After the MP3D assets are installed and the preflight reports `PASS`, run the
standard NaVILA paired evaluation:

```bash
cd /home/xiaotian/vla/NaVILA/evaluation
bash scripts/eval/r2r.sh \
    /media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/models/navila-llama3-8b-8f \
    1 0 "0"

python scripts/eval_jsons.py \
    ./eval_out/navila-llama3-8b-8f/VLN-CE-v1/val_unseen \
    1
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

Start from the paper-ready index:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_ready_results.md
```

Main Markdown/LaTeX tables:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_procedural_v2_main.md
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_procedural_v2_main.md
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_procedural_v2_ablation_core.md
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_false_feasible_outcomes.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_habitat_same_split.md
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_habitat_stress_nominal.md
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_habitat_stress_key_slices.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_repeated_failure_memory.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_transfer_interference.md
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_calibration_extended.md
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_margin_phase_summary.md
```

Diagnostic or smoke outputs:

```text
examples/narrow_passage_rl/results/vln_dataset_free/report.md
examples/narrow_passage_rl/results/vln_batch_diagnostic/report.md
examples/narrow_passage_rl/results/vln_batch_diagnostic/issues_and_improvements.csv
examples/narrow_passage_rl/results/vln_r2r_text_analysis/report.md
examples/narrow_passage_rl/results/vln_r2r_text_analysis/figures/r2r_instruction_length.png
examples/narrow_passage_rl/results/vln_r2r_text_analysis/figures/r2r_val_unseen_language_cues.png
examples/narrow_passage_rl/results/vln_r2r_text_analysis/figures/navila_action_prior_by_language_cue.png
examples/narrow_passage_rl/results/vln_safety_adapter_smoke/report.md
examples/narrow_passage_rl/results/vln_safety_adapter_smoke/comparison_summary.csv
examples/narrow_passage_rl/results/vln_r2r_eval_preflight/r2r_vlnce_preflight.md
examples/narrow_passage_rl/results/vln_r2r_eval_preflight/r2r_vlnce_preflight.json
examples/narrow_passage_rl/results/opennav_preflight/opennav_preflight.md
examples/narrow_passage_rl/results/opennav_preflight/opennav_preflight.json
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_diagnostic_baselines.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_smoke_baselines.md
```

Important raw CSVs:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/harder_benchmark_episodes.csv
examples/narrow_passage_rl/results/narrow_passage_rl/habitat_stress_validation.csv
examples/narrow_passage_rl/results/narrow_passage_rl/repeated_failure_memory.csv
examples/narrow_passage_rl/results/narrow_passage_rl/habitat_td3_habitat_native_strict_eval.csv
examples/narrow_passage_rl/results/vln_batch_diagnostic/predictions.csv
examples/narrow_passage_rl/results/vln_safety_adapter_smoke/adapted_predictions.csv
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
- NaVILA/VLN results in this repository are currently diagnostic.  They show
  local model loading, action-format behavior, stop failure, action-prior bias,
  and deterministic adapter behavior, but they do not provide standard VLN
  benchmark metrics.
- `make_paper_tables.py` regenerates tables from available CSV artifacts; it
  does not train models or mine HM3D scenes by itself.
