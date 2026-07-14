# Table: Extended Required-Width Calibration

| Prior W_hat | Type | Final posterior mean | Final q95 | Mean p_feas | Empirical feasible | Reject | Unsafe attempt | Brier | ECE |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.26 | under-conservative | 0.3453±0.0017 | 0.3580±0.0014 | 0.6582 | 59.7% | 28.1% | 12.2% | 0.04357 | 0.06152 |
| 0.31 | under-conservative | 0.3429±0.0070 | 0.3551±0.0071 | 0.6482 | 58.7% | 30.6% | 10.8% | 0.04491 | 0.06152 |
| 0.36 | oracle | 0.3399±0.0026 | 0.3526±0.0025 | 0.6431 | 59.0% | 30.0% | 11.0% | 0.03794 | 0.05313 |
| 0.46 | over-conservative | 0.3379±0.0023 | 0.3519±0.0022 | 0.6413 | 60.2% | 31.8% | 8.9% | 0.03858 | 0.03916 |
| 0.56 | over-conservative | 0.3434±0.0029 | 0.3562±0.0013 | 0.6046 | 57.8% | 32.8% | 10.1% | 0.03760 | 0.03775 |

Notes:
- `outcome_success` in the raw episode CSV is the oracle physical feasibility label under `true_width`, used for p_feas reliability analysis.
- `unsafe_attempt` is an attempted passage with `passage_width < true_width`, exposing under-conservative priors.
- Reliability claims should be made from `calibration_episode_predictions.csv` and `calibration_reliability_bins.csv`, not from the summary row alone.
- Exact command: `python examples/narrow_passage_rl/eval_dmin_calibration.py --priors 0.26 0.31 0.36 0.46 0.56 --true-width 0.36 --episodes 300 --seeds 0 1 2 --output-dir examples/narrow_passage_rl/results/narrow_passage_rl`.
- Loaded calibration prior sweep from `examples/narrow_passage_rl/results/narrow_passage_rl/raw/calibration_prior_sweep.csv`.
