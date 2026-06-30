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

# Evaluate Habitat baselines and FSM variants
python examples/narrow_passage_rl/narrow_passage/scripts/evaluate.py --method fsm

# Run FSM ablations
python examples/narrow_passage_rl/narrow_passage/scripts/run_ablation.py
```

Core documents:

- `examples/narrow_passage_rl/docs/method.md`
- `examples/narrow_passage_rl/docs/baseline_taxonomy.md`
- `examples/narrow_passage_rl/docs/experiment_protocol.md`
- `examples/narrow_passage_rl/docs/reproducibility.md`
- `examples/narrow_passage_rl/README.md`

The key idea is geometry-guided navigation with explicit passage features,
risk-aware traversability estimation, a failure memory bank, high-level decision
modes, and a mode-conditioned controller.
