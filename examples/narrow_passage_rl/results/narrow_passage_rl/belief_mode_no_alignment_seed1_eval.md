# DEGNAV-RL Procedural v2 Evaluation

| Metric | Value |
|:---|---:|
| episodes | 500 |
| ablation | no_alignment |
| success | 0.2320 |
| strict_success | 0.2320 |
| collision | 0.6380 |
| near_collision | 0.7380 |
| reject | 0.0000 |
| correct_reject | 0.0000 |
| false_reject | 0.0000 |
| timeout | 0.1300 |
| avg_min_clearance | -0.0083 |

## Per-Corridor Success

| Corridor | Episodes | Success | Strict success |
|:---|---:|---:|---:|
| asymmetric | 62 | 0.4032 | 0.4032 |
| false_feasible | 45 | 0.0000 | 0.0000 |
| l_shaped | 104 | 0.0000 | 0.0000 |
| narrow_entry | 51 | 0.3137 | 0.3137 |
| narrow_exit | 75 | 0.2533 | 0.2533 |
| s_shaped | 78 | 0.0000 | 0.0000 |
| straight | 85 | 0.6588 | 0.6588 |

## Mode Distribution

| Mode | Count | Fraction |
|:---|---:|---:|
| commit | 8973 | 0.2298 |
| explore | 30082 | 0.7702 |
