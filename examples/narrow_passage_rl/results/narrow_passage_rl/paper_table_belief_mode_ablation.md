# Table: DEGNAV-RL Belief-State Ablation

| Ablation | Policy input | Overall SR | Strict SR | Collision | Near collision | Reject | Correct reject | False reject |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | All belief features | not run | not run | not run | not run | not run | not run | not run |
| no_p_feas | p_feas replaced by 0.5 | not run | not run | not run | not run | not run | not run | not run |
| no_delta_var | delta_var replaced by constant | not run | not run | not run | not run | not run | not run | not run |
| no_memory | memory_risk replaced by 0 | not run | not run | not run | not run | not run | not run | not run |
| no_alignment | heading/lateral errors replaced by 0 | not run | not run | not run | not run | not run | not run | not run |
| geometry_only | d_hat, body margin, clearances, heading/lateral | not run | not run | not run | not run | not run | not run | not run |

Notes:
- These ablations test the learned high-level mode selector input, not the low-level mode-conditioned controller.
- No DEGNAV-RL ablation CSVs found; rows are marked as not run.
