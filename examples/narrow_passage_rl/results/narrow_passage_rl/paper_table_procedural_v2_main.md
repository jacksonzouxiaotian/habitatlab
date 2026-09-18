# Table: Procedural v2 Main Benchmark

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feasible traversal success |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reactive rule baseline | 25.4±1.2% | 60.7±3.8% | 8.4±3.1% | 6.4±2.7% | 27.8±5.9% | 24.8±1.6% | 33.6±4.6% | 0.0±0.0% |
| DEGNAV-Rule | **70.3±1.4%** | **92.8±2.6%** | **79.4±7.0%** | **70.3±4.4%** | **96.5±2.0%** | **61.7±9.3%** | **50.3±3.9%** | 0.0±0.0% |

Notes:
- Values are mean±std across seeds; rates are computed per seed first, then averaged.
- The main table contains only the primary reactive baseline and the main paper method.
- Local-memory and cross-episode memory rows are intentionally omitted here because their single-encounter procedural v2 results match DEGNAV-Rule; memory is evaluated separately under recurrence and transfer.
- False-feasible traversal success alone does not distinguish correct rejection, collision, or timeout/stuck, so it is not bolded as a success criterion.
- Use the false-feasible outcome decomposition table for explicit rejection, wasted attempts, collision, and timeout/stuck analysis.
- Loaded raw episode rows from `examples/narrow_passage_rl/results/narrow_passage_rl/harder_benchmark_episodes.csv`.
- Benchmark version: procedural v2 / HarderNarrowPassageEnv.
- Seeds: 42, 43, 44.
- Episode budget: 500 episodes per seed per method for the overall metric.
- Regeneration command: `python examples/narrow_passage_rl/eval_harder_benchmark.py --methods rule_baseline geometry_fsm --episodes 500 --seeds 42 43 44`.
