# Narrow-Passage Navigation Research Framework

This fork contains a focused narrow-passage navigation framework built on top of
Habitat-Lab.  The research code lives under:

```text
examples/narrow_passage_rl/
```

Geometry-FSM is the implementation name used by scripts and result files.
Geometry-FSM is referred to as DEGNAV-Rule in the paper.

Paper-facing naming:

- `DEGNAV-Rule / Geometry-FSM`: interpretable feasibility-belief controller.
- `DEGNAV-RL`: learning variant that learns only the high-level mode selector
  `pi(m_t | b_t)` over `Commit`, `Explore`, `Recover`, and `Reject`.
- `PPO/SAC/TD3 direct-control baselines`: learning-only policies that map the
  19-D geometry observation directly to velocity actions.
- `Diagnostic/smoke baselines`: code-path checks or metric diagnostics, not the
  main method ranking.

## Where The Code Lives

```text
examples/narrow_passage_rl/
  procedural_env_v2.py                  # Synthetic v2 benchmark
  eval_harder_benchmark.py              # Main procedural benchmark
  eval_habitat_geometry_fsm.py          # Geometry-FSM / DEGNAV-Rule on HM3D anchors
  eval_habitat_apf_gap.py               # APF+Gap Habitat baseline
  eval_habitat_sb3.py                   # PPO/SAC/TD3 direct-control Habitat evaluation
  eval_habitat_stress_validation.py     # Formal Habitat stress validation
  eval_repeated_failure_memory.py       # Repeated false-feasible memory test
  eval_memory_transfer_interference.py  # Memory transfer/interference test
  eval_dmin_calibration.py              # D_min self-calibration
  train_belief_mode_ppo.py              # DEGNAV-RL high-level mode-selection PPO
  eval_belief_mode_ppo.py               # DEGNAV-RL procedural v2 evaluation
  train_sb3_v2.py                       # PPO/SAC/TD3 direct-control training
  train_recurrent_ppo_v2.py             # RecurrentPPO smoke baseline
  train_bc_dagger_v2.py                 # BC/DAgger smoke baselines
  train_replay_memory_policy_v2.py      # Generic replay-memory smoke baseline
  plot_margin_phase.py                  # Width-margin phase diagram
  record_habitat_video.py               # Habitat video/keyframe generation
  narrow_passage/models/belief_state.py # Compact feasibility-belief state
  narrow_passage/envs/belief_mode_env.py# Discrete mode wrapper for DEGNAV-RL
  narrow_passage/metrics/strict_metrics.py # Shared strict safety metrics
  results/narrow_passage_rl/            # CSV/Markdown/LaTeX/figures
```

The Habitat task and sensors used by these scripts live in:

```text
habitat-lab/habitat/tasks/narrow_passage/
```

## Formal Baseline Summary

Use this file as the single main formal baseline table for the paper:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_formal_baselines.md
```

Headline values from that table:

| Method | Role | Train domain | Eval domain | Seeds | Success |
|---|---|---|---|---:|---:|
| PPO v2 geometry sensor | Direct-control RL baseline | Synthetic v2 -> Habitat | HM3D Val set A | 3 | 2.1% +/- 2.6% |
| SAC v2 geometry sensor | Direct-control RL baseline | Synthetic v2 -> Habitat | HM3D mined Val set B | 1 | 0.0% |
| TD3 synthetic-to-Habitat | Direct-control RL baseline | Synthetic v2 -> Habitat | HM3D mined Val set B | 1 | 2.0% |
| APF+Gap | Classical local baseline | None | HM3D Val set A | deterministic | 93.6% |
| DEGNAV-Rule / Geometry-FSM | Proposed rule controller | None | HM3D mined Val set B | deterministic | 100.0% nominal anchor validation |

The 100.0% DEGNAV-Rule / Geometry-FSM row is **nominal anchor validation under
the current mining protocol**, not a complete robustness claim.  The mined HM3D
starts are mostly well aligned, so this table must be read together with the
stress-validation table.

Older single-run PPO mined-val logs reported 6.0% on 151 episodes.  That number
is kept only as legacy/single-run provenance and is not the formal 3-seed
baseline.

## Habitat Stress Validation

Stress validation is the module-sensitivity evidence for DEGNAV-Rule /
Geometry-FSM.  It perturbs yaw, lateral offset, depth-sector dropout, feature
noise, and the extreme-narrow subset.

```bash
python examples/narrow_passage_rl/eval_habitat_stress_validation.py \
    --preset paper \
    --split val \
    --num-episodes -1
```

Outputs:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/habitat_stress_validation.csv
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_habitat_stress.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_habitat_stress.tex
```

## Diagnostic Baselines

Diagnostic baselines are not the main method ranking.  They explain metric
failure modes.  The current key diagnostic is Habitat-native TD3:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_diagnostic_baselines.md
```

| Method | Purpose | Nominal success | Strict success | Success-but-unsafe | Near collision |
|---|---|---:|---:|---:|---:|
| TD3 Habitat-native | Nominal metric exploitation diagnostic | 100.0% | 2.0% | 98.0% | 100.0% |

This row shows that nominal Habitat success can be optimized without safe
clearance.  It should not be presented as a safe TD3 traversal result.

## Smoke / Appendix-Only Baselines

Smoke baselines verify that training/evaluation paths run end to end.  They are
not main paper baselines unless rerun under the full protocol.

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_smoke_baselines.md
```

Examples:

- GRU-PPO lightweight.
- RecurrentPPO smoke.
- BC-FSM.
- DAgger-FSM.
- Replay Memory Policy.

The Habitat-Baselines `ppo_narrow_passage.yaml` config is a task/policy
smoke-test config, not the main paper PPO training pipeline.

## Memory Contribution

Memory does not improve one-shot nominal Habitat success.  Its contribution is
to suppress repeated commitments to previously failed infeasible passages.

```bash
python examples/narrow_passage_rl/eval_repeated_failure_memory.py \
    --n-rounds 5 \
    --n-passable 20 \
    --n-ff 15 \
    --max-steps 220
```

Main output:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_repeated_failure_memory.md
```

For transfer/interference beyond repeated identical passages:

```bash
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset paper
```

Outputs:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/memory_transfer_interference.csv
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_transfer_interference.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_transfer_interference.tex
```

## DEGNAV-RL Mode-Selection Baseline

DEGNAV-RL learns only `pi(m_t | b_t)` over `Commit`, `Explore`, `Recover`, and
`Reject`.  It does not output direct velocities.  The mode-conditioned velocity
controller is shared with DEGNAV-Rule / Geometry-FSM.

Smoke run:

```bash
python examples/narrow_passage_rl/train_belief_mode_ppo.py \
    --total-steps 2048 \
    --num-envs 1 \
    --ctypes straight_only \
    --ablation full \
    --eval-episodes 20 \
    --save-dir data/degnav_rl_belief_mode_smoke
```

Evaluation:

```bash
python examples/narrow_passage_rl/eval_belief_mode_ppo.py \
    --model data/degnav_rl_belief_mode_smoke/belief_mode_ppo.zip \
    --episodes 50 \
    --ctypes full \
    --ablation full \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_ppo_eval.csv
```

Supported belief-state ablations are `full`, `no_p_feas`, `no_delta_var`,
`no_memory`, `no_alignment`, and `geometry_only`.

## Reproduction Commands

Run from the repository root:

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

# Memory transfer/interference
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset paper

# DEGNAV-RL smoke training
python examples/narrow_passage_rl/train_belief_mode_ppo.py \
    --total-steps 2048 \
    --num-envs 1 \
    --ctypes straight_only \
    --ablation full \
    --eval-episodes 20 \
    --save-dir data/degnav_rl_belief_mode_smoke

# DEGNAV-RL evaluation
python examples/narrow_passage_rl/eval_belief_mode_ppo.py \
    --model data/degnav_rl_belief_mode_smoke/belief_mode_ppo.zip \
    --episodes 50 \
    --ctypes full \
    --ablation full \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_ppo_eval.csv

# D_min calibration
python examples/narrow_passage_rl/eval_dmin_calibration.py

# Width-margin phase diagram
python examples/narrow_passage_rl/plot_margin_phase.py \
    --inputs \
      examples/narrow_passage_rl/results/narrow_passage_rl/harder_benchmark_episodes.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_ppo_eval.csv \
    --labels DEGNAV-Rule DEGNAV-RL \
    --output-dir examples/narrow_passage_rl/results/narrow_passage_rl

# Regenerate paper tables from available CSV artifacts
python examples/narrow_passage_rl/make_paper_tables.py
```

Habitat runs require:

```text
data/scene_datasets/hm3d/
data/datasets/narrow_passage/{split}/{split}.json.gz
```

## Video And Figures

Record Habitat videos with overlays:

```bash
python examples/narrow_passage_rl/record_habitat_video.py \
    --method geometry_fsm \
    --split val \
    --episode-index 0 \
    --output-dir video_dir/narrow_passage_habitat \
    --save-keyframes
```

Core documents:

- `examples/narrow_passage_rl/README.md`
- `examples/narrow_passage_rl/docs/method.md`
- `examples/narrow_passage_rl/docs/baseline_taxonomy.md`
- `examples/narrow_passage_rl/docs/experiment_protocol.md`
- `examples/narrow_passage_rl/docs/reproducibility.md`
