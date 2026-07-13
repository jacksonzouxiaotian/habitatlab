# DEGNAV-RL Procedural v2 Evaluation

| Metric | Value |
|:---|---:|
| episodes | 500 |
| ablation | no_p_feas |
| success | 0.2320 |
| strict_success | 0.2320 |
| collision | 0.6500 |
| near_collision | 0.7480 |
| reject | 0.0000 |
| correct_reject | 0.0000 |
| false_reject | 0.0000 |
| timeout | 0.1180 |
| avg_min_clearance | -0.0143 |

## Per-Corridor Success

| Corridor | Episodes | Success | Strict success |
|:---|---:|---:|---:|
| asymmetric | 62 | 0.3871 | 0.3871 |
| false_feasible | 45 | 0.0000 | 0.0000 |
| l_shaped | 104 | 0.0000 | 0.0000 |
| narrow_entry | 50 | 0.3000 | 0.3000 |
| narrow_exit | 76 | 0.2763 | 0.2763 |
| s_shaped | 78 | 0.0128 | 0.0128 |
| straight | 85 | 0.6471 | 0.6471 |

## Mode Distribution

| Mode | Count | Fraction |
|:---|---:|---:|
| commit | 35010 | 1.0000 |
