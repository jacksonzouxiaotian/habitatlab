# Table: DEGNAV-RL Diagnostic Belief-State Ablation

DEGNAV-RL is included as a diagnostic policy rather than a competitive final method. Under the current reward and action interface, the learned policy collapses to Commit and Explore and does not demonstrate meaningful Recover or Reject behavior.

| Ablation | Policy input | Overall SR | Strict SR | Collision | Near collision | Reject | Correct reject | False reject |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | All belief features | 26.7% | 26.7% | 62.3% | 70.9% | 0.0% | 0.0% | 0.0% |
| no_p_feas | p_feas replaced by 0.5 | 25.9% | 25.9% | 61.9% | 72.5% | 0.0% | 0.0% | 0.0% |
| no_delta_var | delta_var replaced by constant | 28.4% | 28.4% | 58.9% | 70.3% | 0.0% | 0.0% | 0.0% |
| no_memory | memory_risk replaced by 0 | 28.4% | 28.4% | 58.9% | 70.3% | 0.0% | 0.0% | 0.0% |
| no_alignment | heading/lateral errors replaced by 0 | 25.8% | 25.8% | 61.5% | 72.2% | 0.0% | 0.0% | 0.0% |
| geometry_only | d_hat, body margin, clearances, heading/lateral | 30.9% | 30.9% | 56.0% | 68.1% | 0.0% | 0.0% | 0.0% |

Notes:
- These ablations test the learned high-level mode selector input, not the low-level mode-conditioned controller.
- The full belief state is not better than `geometry_only` in this run, so this table should not be used to claim that belief-guided PPO solves the task.
- Full-belief mode usage over 1500 evaluation episodes is Commit: 31.0%, Explore: 69.0%, Recover: 0.0%, Reject: 0.0%.
- Loaded DEGNAV-RL ablation CSVs from `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed0_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed1_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed2_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_geometry_only_seed0_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_geometry_only_seed1_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_geometry_only_seed2_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_alignment_seed0_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_alignment_seed1_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_alignment_seed2_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_delta_var_seed0_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_delta_var_seed1_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_delta_var_seed2_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_memory_seed0_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_memory_seed1_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_memory_seed2_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_p_feas_seed0_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_p_feas_seed1_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_p_feas_seed2_eval.csv`.
