# Table: False-Feasible Outcome Decomposition

0% traversal success is not equivalent to correct rejection; this table decomposes abstention and execution failure.

| Method | Episodes | Traversal success | Correct reject | False reject | Collision | Near collision | Timeout/stuck | Wasted attempts |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reactive rule baseline | 1500 | 0.0% | 0.0% | 0.0% | 100.0% | 57.6% | 0.0% | 100.0% |
| DEGNAV-Rule / Geometry-FSM | 1500 | 0.0% | 0.0% | 0.0% | 2.8% | 14.5% | 97.2% | 100.0% |

Definitions:
- `correct_reject = reject and passable_label == false`.
- `false_reject = reject and passable_label == true`.
- `wasted_attempt = attempted execution on a false-feasible passage without correct rejection`.
- `success == false` is never converted into correct rejection.
