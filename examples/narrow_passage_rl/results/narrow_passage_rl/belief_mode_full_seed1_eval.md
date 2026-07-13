# DEGNAV-RL Procedural v2 Evaluation

| Metric | Value |
|:---|---:|
| episodes | 500 |
| ablation | full |
| success | 0.3020 |
| strict_success | 0.3020 |
| collision | 0.5920 |
| near_collision | 0.6960 |
| reject | 0.0000 |
| correct_reject | 0.0000 |
| false_reject | 0.0000 |
| timeout | 0.1060 |
| avg_min_clearance | 0.0144 |

## Per-Corridor Success

| Corridor | Episodes | Success | Strict success |
|:---|---:|---:|---:|
| asymmetric | 49 | 0.3061 | 0.3061 |
| false_feasible | 57 | 0.0000 | 0.0000 |
| l_shaped | 101 | 0.0000 | 0.0000 |
| narrow_entry | 41 | 0.3171 | 0.3171 |
| narrow_exit | 69 | 0.8696 | 0.8696 |
| s_shaped | 69 | 0.0000 | 0.0000 |
| straight | 114 | 0.5526 | 0.5526 |

## Mode Distribution

| Mode | Count | Fraction |
|:---|---:|---:|
| explore | 38692 | 1.0000 |
