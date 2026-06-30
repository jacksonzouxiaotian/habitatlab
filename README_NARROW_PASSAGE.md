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

# Evaluate Habitat baselines and FSM variants
python examples/narrow_passage_rl/narrow_passage/scripts/evaluate.py --method fsm

# Run FSM ablations
python examples/narrow_passage_rl/narrow_passage/scripts/run_ablation.py
```

New baseline result tables:

- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_baselines.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_learning_baselines.md`

Core documents:

- `examples/narrow_passage_rl/docs/method.md`
- `examples/narrow_passage_rl/docs/baseline_taxonomy.md`
- `examples/narrow_passage_rl/docs/experiment_protocol.md`
- `examples/narrow_passage_rl/docs/reproducibility.md`
- `examples/narrow_passage_rl/README.md`

The key idea is geometry-guided navigation with explicit passage features,
risk-aware traversability estimation, a failure memory bank, high-level decision
modes, and a mode-conditioned controller.
