# DEGNAV-RL Procedural v2 Evaluation

| Metric | Value |
|:---|---:|
| episodes | 500 |
| ablation | no_p_feas |
| success | 0.2340 |
| strict_success | 0.2340 |
| collision | 0.6460 |
| near_collision | 0.7460 |
| reject | 0.0000 |
| correct_reject | 0.0000 |
| false_reject | 0.0000 |
| timeout | 0.1200 |
| avg_min_clearance | -0.0138 |

## Per-Corridor Success

| Corridor | Episodes | Success | Strict success |
|:---|---:|---:|---:|
| asymmetric | 62 | 0.3871 | 0.3871 |
| false_feasible | 45 | 0.0000 | 0.0000 |
| l_shaped | 104 | 0.0000 | 0.0000 |
| narrow_entry | 51 | 0.3137 | 0.3137 |
| narrow_exit | 75 | 0.2800 | 0.2800 |
| s_shaped | 78 | 0.0128 | 0.0128 |
| straight | 85 | 0.6471 | 0.6471 |

## Mode Distribution

| Mode | Count | Fraction |
|:---|---:|---:|
| commit | 35447 | 1.0000 |
