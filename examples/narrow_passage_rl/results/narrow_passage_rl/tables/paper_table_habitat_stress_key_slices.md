# Table: Habitat Stress Key Slices

Key slices for manuscript discussion. Clearance columns are diagnostic body-margin proxies, not calibrated physical contact measurements.

| Stress | Method | Episodes | Success rate | Collision | Timeout | Avg steps | Strict success (diagnostic) | Near collision (diagnostic) | Avg min clearance (diagnostic) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | apf_gap | 151 | 96.7% | 0.0% | 0.0% | 94.8 | 1.3% | 100.0% | -0.112 |
| nominal | fsm_full | 151 | 100.0% | 0.0% | 0.0% | 48.1 | 2.0% | 100.0% | -0.107 |
| yaw60 | apf_gap | 151 | 95.4% | 0.0% | 0.0% | 95.4 | 0.7% | 100.0% | -0.122 |
| yaw60 | fsm_full | 151 | 100.0% | 0.0% | 0.0% | 51.5 | 0.7% | 100.0% | -0.117 |
| yaw60 | fsm_no_heading_alignment | 151 | 76.8% | 0.0% | 16.6% | 160.1 | 0.0% | 100.0% | -0.132 |
| extreme_yaw60 | apf_gap | 24 | 79.2% | 0.0% | 0.0% | 91.9 | 0.0% | 100.0% | -0.139 |
| extreme_yaw60 | fsm_full | 24 | 100.0% | 0.0% | 0.0% | 50.3 | 0.0% | 100.0% | -0.134 |
| extreme_yaw60 | fsm_no_heading_alignment | 24 | 79.2% | 0.0% | 12.5% | 138.3 | 0.0% | 100.0% | -0.144 |
| lat020 | apf_gap | 151 | 92.7% | 0.0% | 0.0% | 93.2 | 1.3% | 100.0% | -0.113 |
| lat020 | fsm_full | 151 | 98.0% | 0.0% | 0.0% | 47.9 | 1.3% | 100.0% | -0.108 |
| lat020 | fsm_no_heading_alignment | 151 | 95.4% | 0.0% | 0.7% | 75.2 | 0.7% | 100.0% | -0.118 |

Notes:
- Loaded raw Habitat stress rows from `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_stress_validation.csv`.
- Regenerate with: `python examples/narrow_passage_rl/eval_habitat_stress_validation.py --preset paper --split val --num-episodes -1`.
