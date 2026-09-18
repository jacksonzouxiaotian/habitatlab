# Table: Procedural v2 Core Ablations

| Variant | Overall | Δ Overall vs full | Straight | L-shaped | S-shaped | Narrow entry | Asymmetric | Collision | Timeout/stuck |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DEGNAV full | 69.9±1.6% | +0.0±0.0 pp | 88.8±5.4% | 79.8±2.3% | 75.3±3.6% | 59.3±5.0% | 46.1±4.5% | 17.0±1.4% | 13.1±1.9% |
| w/o alignment | 2.4±0.5% | -67.5±2.1 pp | 7.8±0.7% | 0.0±0.0% | 0.0±0.0% | 2.1±0.3% | 2.8±1.1% | 48.1±1.8% | 49.5±1.5% |
| w/o recovery | 69.9±1.6% | +0.0±0.0 pp | 88.8±5.4% | 79.8±2.3% | 75.3±3.6% | 59.3±5.0% | 46.1±4.5% | 17.0±1.4% | 13.1±1.9% |
| deterministic margin | 69.3±1.5% | -0.6±0.2 pp | 88.8±5.4% | 79.8±3.2% | 73.7±3.2% | 59.3±5.0% | 44.0±2.5% | 17.8±1.6% | 12.9±2.1% |
| w/o yaw prior | 70.1±1.7% | +0.1±0.4 pp | 89.1±4.9% | 79.8±2.3% | 74.9±4.3% | 60.2±3.6% | 46.8±3.7% | 16.9±1.0% | 13.1±1.9% |

Notes:
- Values are mean±std across seeds. Rates are computed per seed first, then averaged.
- Δ Overall is computed as a paired per-seed difference from DEGNAV full, then averaged.
- The current benchmark identifies alignment as the dominant measured component. Deterministic-margin and no-yaw-prior variants remain close to full DEGNAV, so the benchmark does not fully isolate the probabilistic belief and yaw-prior contributions.
- `deterministic margin` replaces probabilistic feasibility gating with `d_hat - w_req_cons > tau_margin` while preserving the yaw-aware width prior and alignment controller.
- `w/o yaw prior` keeps probabilistic uncertainty and alignment but uses a fixed frontal required-width prior.
- Recovery should be interpreted conservatively here: w/o recovery matches full DEGNAV on this unperturbed benchmark, so recovery benefit should be assessed in stress or stuck-specific settings.
- No statistical significance is claimed without a separate statistical test.
- Loaded raw episode rows from `examples/narrow_passage_rl/results/narrow_passage_rl/raw/procedural_ablation_core.csv`.
- Benchmark version: procedural v2 / HarderNarrowPassageEnv.
- Common configuration: 500 episodes per seed, seeds 0, 1, 2 for every listed variant.
- Rows per variant: DEGNAV full=1500, w/o alignment=1500, w/o recovery=1500, deterministic margin=1500, w/o yaw prior=1500.
- Exact evaluation command: `python examples/narrow_passage_rl/eval_harder_benchmark.py --variants full no_alignment no_recovery deterministic_margin no_yaw_prior --episodes 500 --seeds 0 1 2 --log-belief-diagnostics --log-outcome-decomposition --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/procedural_ablation_core.csv`.
