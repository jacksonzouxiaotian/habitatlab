status: stale backup before schema rerun
reason: harder_benchmark_episodes.csv lacked canonical belief/mode margin-phase fields
config: examples/narrow_passage_rl/configs/eval_baselines.yaml synthetic_v2 episodes=500 seeds=[0,1,2]; paper table protocol uses seeds 42-44
rerun_command: python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500 --seeds 3 --seed 42
table_command: python examples/narrow_passage_rl/scripts/make_harder_benchmark_tables.py
validation_command: python examples/narrow_passage_rl/scripts/validate_harder_benchmark_schema.py
