# DEGNAV-RL Procedural v2 Evaluation

| Metric | Value |
|:---|---:|
| episodes | 500 |
| ablation | full |
| success | 0.1980 |
| strict_success | 0.1980 |
| collision | 0.6840 |
| near_collision | 0.7340 |
| reject | 0.0000 |
| correct_reject | 0.0000 |
| false_reject | 0.0000 |
| timeout | 0.1180 |
| avg_min_clearance | 0.0109 |

## Per-Corridor Success

| Corridor | Episodes | Success | Strict success |
|:---|---:|---:|---:|
| asymmetric | 49 | 0.2245 | 0.2245 |
| false_feasible | 57 | 0.0000 | 0.0000 |
| l_shaped | 101 | 0.0000 | 0.0000 |
| narrow_entry | 41 | 0.3415 | 0.3415 |
| narrow_exit | 69 | 0.1739 | 0.1739 |
| s_shaped | 70 | 0.0000 | 0.0000 |
| straight | 113 | 0.5487 | 0.5487 |

## Mode Distribution

| Mode | Count | Fraction |
|:---|---:|---:|
| commit | 34763 | 1.0000 |
