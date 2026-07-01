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

## Current Results Snapshot

Full per-episode CSVs, Markdown tables, and LaTeX tables are under
`examples/narrow_passage_rl/results/narrow_passage_rl/`.  The most important
current results are:

### Habitat HM3D Mined-Val

| Method | Episodes | Success | Notes |
|---|---:|---:|---|
| PPO v2 geometry | 151 | 6.0% | SB3 geometry policy |
| SAC v2 geometry | 151 | 0.0% | SB3 SAC policy |
| Geometry-FSM | 151 | 100.0% | Feature-driven controller |
| Habitat FSM stress ablation, +60 deg heading | 24 | 100.0% full / 0.0% no heading alignment | Extreme-narrow subset |

### Memory / History Baselines

| Method | Passable SR | Passable Reject | Final False-Feasible Reject | Wasted FF Steps |
|---|---:|---:|---:|---:|
| no_memory | 0.900 | 0.000 | 0.000 | 16500 |
| kNN Failure Memory | 0.860 | 0.060 | 1.000 | 5500 |
| Vanilla Episodic Memory | 0.820 | 0.130 | 1.000 | 8140 |
| Geometry-Guided Failure Memory | 0.900 | 0.030 | 1.000 | 4400 |

### Newly Deployed Baselines

These are smoke runs that verify training/evaluation paths; they are not final
long-training scores.

| Baseline | Train data / budget | Eval episodes | Success | Collision |
|---|---:|---:|---:|---:|
| FSM expert trajectories | 40 episodes / 5179 transitions | 40 | 0.750 | 0.100 |
| BC-FSM | 5179 expert transitions / 5 epochs | 40 | 0.500 | 0.475 |
| DAgger-FSM | 6346 transitions / 1 DAgger iter | 40 | 0.300 | 0.650 |
| RecurrentPPO | 1024 env steps | 40 | 0.000 | 0.000 |
| Replay Memory Policy | 1024 env steps | 40 | 0.000 | 0.025 |

The 3M-step RecurrentPPO run should be logged as a final result only after its
500-episode evaluation CSV is written.

Core documents:

- `examples/narrow_passage_rl/docs/method.md`
- `examples/narrow_passage_rl/docs/baseline_taxonomy.md`
- `examples/narrow_passage_rl/docs/experiment_protocol.md`
- `examples/narrow_passage_rl/docs/reproducibility.md`
- `examples/narrow_passage_rl/README.md`

The key idea is geometry-guided navigation with explicit passage features,
risk-aware traversability estimation, a failure memory bank, high-level decision
modes, and a mode-conditioned controller.
