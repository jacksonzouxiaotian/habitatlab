# Narrow-Passage Navigation Research Framework

This fork contains a focused narrow-passage navigation framework built on top of
Habitat-Lab.  The research code lives under:

```text
examples/narrow_passage_rl/
```

The directory is organized as a paper-facing framework rather than a loose set of
Habitat modifications:

```text
examples/narrow_passage_rl/
  configs/                 # Reproducible train/eval protocol configs
  narrow_passage/
    baselines/             # Four-layer baseline registry
    envs/                  # Habitat task adapters, passage generation, metrics
    models/                # Geometry encoder, risk head, memory, policy modes
    planners/              # Classical / sampling baseline registry
    scripts/               # Stable command entry points
  results/
    raw/                   # Raw per-episode outputs
    tables/                # Paper-ready tables
    figures/               # Paper figures
  docs/                    # Method, protocol, reproducibility notes
```

Quick entry points:

```bash
# Train RL baseline on the synthetic v2 benchmark
python examples/narrow_passage_rl/narrow_passage/scripts/train.py --algo ppo

# Run memory/history baselines
python examples/narrow_passage_rl/eval_memory_baselines.py \
  --n-rounds 5 --n-passable 20 --n-ff 15 --max-steps 220

# Run lightweight GRU-PPO fallback baseline
python examples/narrow_passage_rl/train_gru_ppo_v2.py \
  --total-steps 5000 --eval-episodes 50 \
  --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/gru_ppo_v2_smoke

# Run formal SB3-Contrib RecurrentPPO smoke baseline
python examples/narrow_passage_rl/train_recurrent_ppo_v2.py \
  --reward-mode fair \
  --total-steps 1024 --n-steps 128 --batch-size 64 --eval-episodes 40 \
  --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/recurrent_ppo_v2_smoke

# Collect expert data and run BC/DAgger smoke baselines
python examples/narrow_passage_rl/collect_expert_trajectories.py \
  --episodes 40 --output-npz examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke.npz
python examples/narrow_passage_rl/train_bc_dagger_v2.py \
  --algo bc --dataset examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke.npz \
  --epochs 5 --eval-episodes 40
python examples/narrow_passage_rl/train_bc_dagger_v2.py \
  --algo dagger --dataset examples/narrow_passage_rl/results/narrow_passage_rl/expert_fsm_v2_smoke.npz \
  --epochs 3 --dagger-iters 1 --dagger-episodes 10 --eval-episodes 40

# Run generic replay-memory policy smoke baseline
python examples/narrow_passage_rl/train_replay_memory_policy_v2.py \
  --reward-mode fair \
  --algo ppo --total-steps 1024 --eval-episodes 40 \
  --save-dir examples/narrow_passage_rl/results/narrow_passage_rl/checkpoints/replay_memory_policy_v2_smoke

# Evaluate Habitat baselines and FSM variants
python examples/narrow_passage_rl/narrow_passage/scripts/evaluate.py --method fsm

# Run FSM ablations
python examples/narrow_passage_rl/narrow_passage/scripts/run_ablation.py
```

New baseline result tables:

- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_baselines.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_learning_baselines.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_new_baselines_smoke.md`

Learning baselines use `FairNarrowPassageRewardWrapper` by default.  It clips
open-space clearance reward and penalizes timeout / no-progress behavior, which
prevents policies from receiving high return by staying outside the passage.
Use `--reward-mode native` only to reproduce legacy reward runs.

## Current Results Snapshot

Full per-episode CSVs, Markdown tables, and LaTeX tables are under
`examples/narrow_passage_rl/results/narrow_passage_rl/`.  The most important
current results are:

### Habitat HM3D Mined-Val

| Method | Episodes | Success | Notes |
|---|---:|---:|---|
| SB3 PPO synthetic-to-Habitat geometry | 151 | 6.0% | Trained on synthetic v2, evaluated on mined HM3D |
| SAC v2 geometry | 151 | 0.0% | SB3 SAC policy |
| Geometry-FSM | 151 | 100.0% | Feature-driven controller |
| Habitat FSM stress ablation, +60 deg heading | 24 | 100.0% full / 0.0% no heading alignment | Extreme-narrow subset |

The Habitat-Baselines `ppo_narrow_passage.yaml` config is a task/policy
smoke-test config, not the main paper PPO training pipeline.  Learning-baseline
claims should distinguish: synthetic PPO, SB3 synthetic-to-Habitat transfer, and
Habitat-Baselines smoke tests.

### Memory / History Baselines

| Method | Passable SR | Passable Reject | Final False-Feasible Reject | Wasted FF Steps |
|---|---:|---:|---:|---:|
| no_memory | 0.900 | 0.000 | 0.000 | 16500 |
| kNN Failure Memory | 0.860 | 0.060 | 1.000 | 5500 |
| Vanilla Episodic Memory | 0.820 | 0.130 | 1.000 | 8140 |
| Geometry-Guided Failure Memory | 0.900 | 0.030 | 1.000 | 4400 |

### Learning Baselines

| Method | Domain | Train budget | Eval episodes | Success | Collision | Notes |
|---|---|---:|---:|---:|---:|---|
| SB3 PPO synthetic-to-Habitat geometry | Habitat HM3D mined-val | 5M synthetic steps | 151 | 0.060 | - | Synthetic v2 checkpoint transferred to HM3D |
| SAC v2 geometry | Habitat HM3D mined-val | 2M steps | 151 | 0.000 | - | Existing SB3 checkpoint |
| RecurrentPPO | Synthetic v2 | 3M steps | 500 | 0.130 | 0.456 | SB3-Contrib + fair reward |

RecurrentPPO learns some straight/asymmetric passages, but it remains weak on
L/S turns, narrow exits, and false-feasible safety.  This is the fair-reward
replacement for the older native-reward timeout result.

The 100% Habitat FSM rows are reported with the current mined-anchor protocol.
They should be read together with the +60 deg yaw stress ablation.  The Habitat
stress evaluator now supports yaw perturbation, lateral offset, start-distance
shift, feature noise, and depth-sector dropout.  Goal perturbation,
false-feasible Habitat anchors, dynamic obstacles, and unseen-room
generalization remain planned dataset/simulator extensions until new stress CSVs
are generated.

### Paper Videos

`examples/narrow_passage_rl/record_habitat_video.py` records RGB/depth MP4s for
`geometry_fsm`, `ppo_sb3`, `apf_gap`, and `ppo_policy`, with overlayed episode
state and optional keyframes.

| Method | Episode | Artifact | Result |
|---|---|---|---|
| Geometry-FSM | `hm3d_narrow_000008` | `video_dir/narrow_passage_habitat/geometry_fsm_ep000_hm3d_narrow_000008.mp4` | 28 steps, success=1 |
| SB3 PPO synthetic-to-Habitat | `hm3d_narrow_000008` | `video_dir/narrow_passage_habitat/ppo_sb3_ep000_hm3d_narrow_000008.mp4` | 500 steps, timeout |

Keyframes are under `results/narrow_passage_rl/keyframes/`.

### Newly Deployed Baselines

These are smoke runs that verify training/evaluation paths; they are not final
long-training scores.

| Baseline | Train data / budget | Eval episodes | Success | Collision |
|---|---:|---:|---:|---:|
| FSM expert trajectories | 40 episodes / 5179 transitions | 40 | 0.750 | 0.100 |
| BC-FSM | 5179 expert transitions / 5 epochs | 40 | 0.500 | 0.475 |
| DAgger-FSM | 6346 transitions / 1 DAgger iter | 40 | 0.300 | 0.650 |
| Replay Memory Policy | 1024 env steps | 40 | 0.000 | 0.025 |

Core documents:

- `examples/narrow_passage_rl/docs/method.md`
- `examples/narrow_passage_rl/docs/baseline_taxonomy.md`
- `examples/narrow_passage_rl/docs/experiment_protocol.md`
- `examples/narrow_passage_rl/docs/reproducibility.md`
- `examples/narrow_passage_rl/README.md`

The key idea is geometry-guided navigation with explicit passage features,
risk-aware traversability estimation, a failure memory bank, high-level decision
modes, and a mode-conditioned controller.
