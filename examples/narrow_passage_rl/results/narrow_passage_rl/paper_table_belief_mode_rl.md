# Table: DEGNAV-RL Diagnostic Belief-Mode Comparison

DEGNAV-RL is included as a diagnostic policy rather than a competitive final method. Under the current reward and action interface, the learned policy collapses to Commit and Explore and does not demonstrate meaningful Recover or Reject behavior.

This table is a diagnostic provenance comparison, not a main-paper leaderboard:
the DEGNAV-RL row is procedural v2, while the direct-control rows are Habitat
diagnostics or mixed-provenance baselines.

| Method | Input | Decision type | Overall SR | Strict SR | Collision | Near collision | Reject | Correct reject | False reject |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO direct velocity (single-run mined-val) | 19-D geometry observation; single-run provenance | Direct velocity | 6.0% | 0.0% | 0.0% | 100.0% | 0.0% | not run | not run |
| SAC direct velocity | 19-D geometry observation | Direct velocity | 0.0% | 0.0% | 0.0% | 100.0% | not run | not run | not run |
| TD3 direct velocity | 19-D geometry observation | Direct velocity | 2.0% | 0.0% | 0.0% | 100.0% | not run | not run | not run |
| APF+Gap | Depth + local goal | Classical reactive planner | 93.6% | not run | not run | not run | not run | not run | not run |
| DEGNAV-Rule / Geometry-FSM | 19-D geometry observation + rule belief | Rule mode selector | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% | not run | not run |
| DEGNAV-RL | Explicit feasibility belief state | Learned high-level mode selector | 26.7% | 26.7% | 62.3% | 70.9% | 0.0% | 0.0% | 0.0% |

Notes:
- DEGNAV-RL is intended to test only the high-level mode selector over the explicit belief state; the velocity realization remains the same mode-conditioned controller as DEGNAV-Rule.
- Learning-only PPO/SAC/TD3 baselines output direct velocity actions from geometry observations.
- The DEGNAV-RL row is a procedural v2 mode-selection result; Habitat DEGNAV-RL evaluation is not included yet.
- DEGNAV-RL full-belief mode usage is Commit: 31.0%, Explore: 69.0%, Recover: 0.0%, Reject: 0.0%; it should not be presented as learned recovery or learned rejection.
- Do not claim DEGNAV-RL improves over DEGNAV-Rule or solves the task.
- The PPO direct-velocity row here is a single-run mined-val provenance row when loaded from `habitat_ppo_v2_mined_val.csv`; the 3-seed PPO Val set A result is listed in `paper_table_formal_baselines.md`, which is a mixed-split Habitat diagnostic comparison rather than a fair main ranking.
- DEGNAV-RL row loaded from `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed0_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed1_eval.csv`, `examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed2_eval.csv`.
