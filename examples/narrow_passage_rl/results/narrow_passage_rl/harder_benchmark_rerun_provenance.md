# Harder Benchmark Schema Rerun Provenance

- Generated: 2026-07-13
- Git commit: aa6f8c60b257e6552e4b0af74035d65d71e5aed8
- Evaluator: examples/narrow_passage_rl/eval_harder_benchmark.py
- Config reference: examples/narrow_passage_rl/configs/eval_baselines.yaml
- Table generator: examples/narrow_passage_rl/scripts/make_harder_benchmark_tables.py
- Validator: examples/narrow_passage_rl/scripts/validate_harder_benchmark_schema.py
- Old-result backup: examples/narrow_passage_rl/results/narrow_passage_rl/legacy/harder_benchmark_stale_20260713_162608
- Seeds: 42, 43, 44
- Episodes: 500 per method per seed
- Methods: rule_baseline, geometry_fsm, fsm_no_recovery, fsm_no_alignment, fsm_local_memory, fsm_cross_memory
- Rerun command:   python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500 --seeds 3 --seed 42
- Validation command:   python examples/narrow_passage_rl/scripts/validate_harder_benchmark_schema.py --csv examples/narrow_passage_rl/results/narrow_passage_rl/harder_benchmark_episodes.csv
- Table command:   python examples/narrow_passage_rl/scripts/make_harder_benchmark_tables.py --csv examples/narrow_passage_rl/results/narrow_passage_rl/harder_benchmark_episodes.csv --output-dir examples/narrow_passage_rl/results/narrow_passage_rl

Notes:
- The previous CSV was stale and lacked canonical belief/mode margin-phase fields.
- The new CSV contains 9000 rows and passes schema validation.
- Belief/mode field coverage is 0.833 because rule_baseline has no belief or four-mode interface and correctly logs NaN for those fields.

Scope correction:
- This rerun regenerated the main harder benchmark outputs only: `harder_benchmark_episodes.csv`, `harder_benchmark_summary.csv`, `paper_table_procedural_v2_main.md`, and `paper_table_harder_main.tex`.
- `paper_table_harder_ablation.md` and `.tex` were restored to the previous entry-jitter ablation protocol backup because the main no-jitter CSV must not be used to rewrite ablation claims.
- `benchmark_ablation_episodes.csv` remains an older-schema ablation artifact and should be rerun separately before using it for margin-phase logging.
