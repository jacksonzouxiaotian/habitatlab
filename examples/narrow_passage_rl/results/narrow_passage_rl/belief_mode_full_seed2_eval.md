# DEGNAV-RL Procedural v2 Evaluation

| Metric | Value |
|:---|---:|
| episodes | 500 |
| ablation | full |
| success | 0.3000 |
| strict_success | 0.3000 |
| collision | 0.5940 |
| near_collision | 0.6960 |
| reject | 0.0000 |
| correct_reject | 0.0000 |
| false_reject | 0.0000 |
| timeout | 0.1060 |
| avg_min_clearance | 0.0142 |

## Per-Corridor Success

| Corridor | Episodes | Success | Strict success |
|:---|---:|---:|---:|
| asymmetric | 49 | 0.3061 | 0.3061 |
| false_feasible | 57 | 0.0000 | 0.0000 |
| l_shaped | 101 | 0.0000 | 0.0000 |
| narrow_entry | 41 | 0.3171 | 0.3171 |
| narrow_exit | 68 | 0.8676 | 0.8676 |
| s_shaped | 69 | 0.0000 | 0.0000 |
| straight | 115 | 0.5478 | 0.5478 |

## Mode Distribution

| Mode | Count | Fraction |
|:---|---:|---:|
| explore | 38645 | 1.0000 |
