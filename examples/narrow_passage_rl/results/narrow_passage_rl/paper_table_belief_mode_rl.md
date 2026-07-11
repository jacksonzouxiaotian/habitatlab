# Table: Belief-Mode RL Comparison

| Method | Input | Decision type | Overall SR | Strict SR | Collision | Near collision | Reject | Correct reject | False reject |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO direct velocity (single-run mined-val) | 19-D geometry observation; single-run provenance | Direct velocity | 6.0% | 0.0% | 0.0% | 100.0% | 0.0% | not run | not run |
| SAC direct velocity | 19-D geometry observation | Direct velocity | 0.0% | 0.0% | 0.0% | 100.0% | not run | not run | not run |
| TD3 direct velocity | 19-D geometry observation | Direct velocity | 2.0% | 0.0% | 0.0% | 100.0% | not run | not run | not run |
| APF+Gap | Depth + local goal | Classical reactive planner | 93.6% | not run | not run | not run | not run | not run | not run |
| DEGNAV-Rule / Geometry-FSM | 19-D geometry observation + rule belief | Rule mode selector | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% | not run | not run |
| DEGNAV-RL | Explicit feasibility belief state | Learned high-level mode selector | not run | not run | not run | not run | not run | not run | not run |

Notes:
- DEGNAV-RL learns only the high-level mode selector over the explicit belief state; the velocity realization remains the same mode-conditioned controller as DEGNAV-Rule.
- Learning-only PPO/SAC/TD3 baselines output direct velocity actions from geometry observations.
- The PPO direct-velocity row here is a single-run mined-val provenance row when loaded from `habitat_ppo_v2_mined_val.csv`; the formal main PPO result remains 2.1% +/- 2.6% in `paper_table_formal_baselines.md`.
- DEGNAV-RL: no evaluation CSV found. Expected one of `eval_belief_mode_ppo.csv`, `degnav_rl_belief_mode_ppo_eval.csv`, `belief_mode_ppo_eval.csv`, or `belief_mode_ppo_quick_eval.csv`.
