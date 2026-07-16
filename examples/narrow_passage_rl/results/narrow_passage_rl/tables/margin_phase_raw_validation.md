# Margin-Phase Raw CSV Validation

**Status:** PASS

## Provenance

- Exact command:

```bash
python examples/narrow_passage_rl/eval_harder_benchmark.py \
  --methods rule_baseline geometry_fsm \
  --episodes 500 \
  --seeds 0 1 2 \
  --log-belief-diagnostics \
  --log-outcome-decomposition \
  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv
```
- Git commit: `5f70a286a54a08c162727c54728f610bde660c58`
- Working tree dirty at validation time: `yes`
- CSV path: `examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv`
- Metadata path: `examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.meta.json`
- Episode semantics: `--episodes is per method per seed.`
- Episodes per seed per method: `500`

## Compact Summary

- Total rows: `3000`
- Unique paired scenarios: `1500`
- Rows per method: `{'rule_baseline': 1500, 'geometry_fsm': 1500}`
- Rows per seed: `{'0': 1000, '1': 1000, '2': 1000}`
- Corridor-type counts per method-paired scenario set: `{'narrow_exit': 205, 's_shaped': 235, 'asymmetric': 141, 'straight': 320, 'l_shaped': 302, 'false_feasible': 150, 'narrow_entry': 147}`

| Method | Success | Collision | Reject | Finite margin coverage | Delta min | Delta median | Delta max |
|:---|---:|---:|---:|---:|---:|---:|---:|
| rule_baseline | 26.1% | 63.2% | 0.0% | 100.0% | -0.0622 | 0.2383 | 0.5759 |
| geometry_fsm | 69.9% | 17.0% | 0.0% | 100.0% | -0.0622 | 0.2383 | 0.5759 |

## Sanity Checks

| Check | Result | Detail |
|:---|:---:|:---|
| target CSV exists | PASS | examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv |
| metadata JSON exists | PASS | examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.meta.json |
| required columns exist | PASS | all present |
| required methods present | PASS | ['geometry_fsm', 'rule_baseline'] |
| required seeds present | PASS | ['0', '1', '2'] |
| paired scenario_id sets within each seed | PASS | seed 0: 500 shared scenarios; seed 1: 500 shared scenarios; seed 2: 500 shared scenarios |
| no duplicate method + seed + scenario_id | PASS | [] |
| delta_mean equals d_hat - w_req_cons for finite rows | PASS | [] |
| p_feas is in [0,1] | PASS | [] |
| boolean outcome columns contain only 0/1 | PASS | [] |
| correct_reject implies reject | PASS | [] |
| false_reject implies reject | PASS | [] |
| correct_reject and false_reject never both true | PASS | [] |
| rule_baseline has belief_used_by_policy == 0 | PASS | [] |
| geometry_fsm has belief_used_by_policy == 1 | PASS | [] |
| margin_snapshot_source is populated | PASS | [] |
| paired scenarios share corridor geometry and start diagnostics | PASS | [] |

## Missing-Value Counts For Required Columns

| Column | Missing count |
|:---|---:|
| method | 0 |
| scenario_id | 0 |
| episode_id | 0 |
| seed | 0 |
| corridor_type | 0 |
| passage_width | 0 |
| entry_yaw | 0 |
| lateral_offset | 0 |
| success | 0 |
| collision | 0 |
| near_collision | 0 |
| reject | 0 |
| correct_reject | 0 |
| false_reject | 0 |
| timeout | 0 |
| stuck | 0 |
| d_hat | 0 |
| w_req_prior | 0 |
| w_req_cons | 0 |
| delta_mean | 0 |
| delta_var | 0 |
| p_feas | 0 |
| risk | 0 |
| body_margin | 0 |
| min_clearance | 0 |
| margin_snapshot_step | 0 |
| margin_snapshot_source | 0 |
| belief_used_by_policy | 0 |
| mode_commit_count | 1500 |
| mode_explore_count | 1500 |
| mode_recover_count | 1500 |
| mode_reject_count | 1500 |
