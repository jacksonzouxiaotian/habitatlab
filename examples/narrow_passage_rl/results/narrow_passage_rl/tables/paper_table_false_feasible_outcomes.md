# Table: False-Feasible Outcome Decomposition

| Method | Episodes | Traversal success | Explicit reject | Correct reject | Collision | Near collision | Timeout/stuck | Wasted attempts |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reactive rule baseline | 1500 (3 seeds) | 0.0±0.0% (0/1500) | 0.0±0.0% (0/1500) | 0.0±0.0% (0/1500) | 100.0±0.0% (1500/1500) | 57.6±2.6% (864/1500) | 0.0±0.0% (0/1500) | 100.0±0.0% (1500/1500) |
| DEGNAV-Rule | 1500 (3 seeds) | 0.0±0.0% (0/1500) | 0.0±0.0% (0/1500) | 0.0±0.0% (0/1500) | 2.8±0.7% (42/1500) | 14.5±1.0% (218/1500) | 97.2±0.7% (1458/1500) | 100.0±0.0% (1500/1500) |

Notes:
- One-shot false-feasible traversal success does not establish correct rejection. This table separates explicit abstention from collision and non-collision execution failure.
- DEGNAV-Rule reduces hard collision relative to the reactive baseline, but one-shot correct rejection remains weak; recurrence handling is evaluated separately through memory.
- `explicit reject` means the controller selected Reject.
- `correct_reject = explicit reject and passable_label == false`.
- Timeout/stuck remains an execution failure category and is not labeled as safe rejection.
- `wasted_attempt = attempted traversal on a false-feasible passage without correct rejection`.
- `success == false` is never converted into correct rejection.
- Collision and near-collision are logged independently and may overlap.
- Loaded raw outcome rows from `examples/narrow_passage_rl/results/narrow_passage_rl/raw/false_feasible_outcomes.csv`.
- Protocol: one-shot benchmark-labeled false-feasible passages.
- Methods included: Reactive rule baseline and DEGNAV-Rule.
- Each listed method uses 3 seeds and 500 false-feasible episodes per seed.
