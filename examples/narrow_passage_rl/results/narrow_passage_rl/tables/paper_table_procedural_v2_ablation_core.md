# Table: Procedural v2 Core Ablations

| Variant | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feasible traversal success | Correct reject | Collision | Near collision | Timeout/stuck |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DEGNAV full | 69.9±1.6% | 88.8±5.4% | 79.8±2.3% | 75.3±3.6% | 94.5±1.5% | 59.3±5.0% | 46.1±4.5% | 0.0±0.0% | 0.0±0.0% | 17.0±1.4% | 41.9±0.3% | 13.1±1.9% |
| w/o alignment | 2.4±0.5% | 7.8±0.7% | 0.0±0.0% | 0.0±0.0% | 1.9±2.0% | 2.1±0.3% | 2.8±1.1% | 0.0±0.0% | 0.0±0.0% | 48.1±1.8% | 75.1±0.9% | 49.5±1.5% |
| w/o recovery | 69.9±1.6% | 88.8±5.4% | 79.8±2.3% | 75.3±3.6% | 94.5±1.5% | 59.3±5.0% | 46.1±4.5% | 0.0±0.0% | 0.0±0.0% | 17.0±1.4% | 41.9±0.3% | 13.1±1.9% |
| deterministic margin only | 69.3±1.5% | 88.8±5.4% | 79.8±3.2% | 73.7±3.2% | 93.5±2.4% | 59.3±5.0% | 44.0±2.5% | 0.0±0.0% | 0.0±0.0% | 17.8±1.6% | 41.9±0.3% | 12.9±2.1% |
| w/o yaw prior | 70.1±1.7% | 89.1±4.9% | 79.8±2.3% | 74.9±4.3% | 94.5±1.5% | 60.2±3.6% | 46.8±3.7% | 0.0±0.0% | 0.0±0.0% | 16.9±1.0% | 41.9±0.3% | 13.1±1.9% |

Notes:
- Values are mean±std across seeds. Rates are computed per seed first, then averaged.
- False-feasible traversal success is reported separately from correct rejection; 0% traversal success is not treated as correct rejection.
- `deterministic margin only` replaces probabilistic feasibility gating with `d_hat - w_req_cons > tau_margin` while preserving the yaw-aware width prior and alignment controller.
- `w/o yaw prior` keeps probabilistic uncertainty and alignment but uses a fixed frontal required-width prior.
- Recovery should be interpreted conservatively here: this unperturbed procedural benchmark does not strongly activate recovery, so recovery benefit should be assessed in stress or stuck-specific settings.
- This table is computed from episode-level procedural v2 rows and is intended to isolate belief, yaw-prior, alignment, and recovery components.
- Loaded raw episode rows from `examples/narrow_passage_rl/results/narrow_passage_rl/raw/procedural_ablation_core.csv`.
- Exact evaluation command: `python examples/narrow_passage_rl/eval_harder_benchmark.py --variants full no_alignment no_recovery deterministic_margin no_yaw_prior --episodes 500 --seeds 0 1 2 --log-belief-diagnostics --log-outcome-decomposition --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/procedural_ablation_core.csv`.
