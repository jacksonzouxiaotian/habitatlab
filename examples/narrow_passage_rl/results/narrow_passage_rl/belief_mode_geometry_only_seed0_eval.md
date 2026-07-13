# DEGNAV-RL Procedural v2 Evaluation

| Metric | Value |
|:---|---:|
| episodes | 500 |
| ablation | geometry_only |
| success | 0.3080 |
| strict_success | 0.3080 |
| collision | 0.5620 |
| near_collision | 0.6820 |
| reject | 0.0000 |
| correct_reject | 0.0000 |
| false_reject | 0.0000 |
| timeout | 0.1300 |
| avg_min_clearance | 0.0002 |

## Per-Corridor Success

| Corridor | Episodes | Success | Strict success |
|:---|---:|---:|---:|
| asymmetric | 62 | 0.3710 | 0.3710 |
| false_feasible | 45 | 0.0000 | 0.0000 |
| l_shaped | 104 | 0.0000 | 0.0000 |
| narrow_entry | 50 | 0.3000 | 0.3000 |
| narrow_exit | 76 | 0.7895 | 0.7895 |
| s_shaped | 78 | 0.0000 | 0.0000 |
| straight | 85 | 0.6588 | 0.6588 |

## Mode Distribution

| Mode | Count | Fraction |
|:---|---:|---:|
| explore | 43046 | 1.0000 |
