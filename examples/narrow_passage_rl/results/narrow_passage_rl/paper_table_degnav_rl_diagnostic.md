# Table: DEGNAV-RL Diagnostic Learning Variant

DEGNAV-RL is included as a diagnostic policy rather than a competitive final method. Under the current reward and action interface, the learned policy collapses to Commit and Explore and does not demonstrate meaningful Recover or Reject behavior.

The main method remains DEGNAV-Rule / Geometry-FSM. This table should be used in
the appendix or diagnostic discussion, not as a canonical main result table.

| Method | Decision type | Eval domain | Overall SR | Strict SR | Collision | Near collision | Reject | Correct reject | False reject | Mode usage |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| DEGNAV-RL full belief | Diagnostic high-level mode selector | Procedural v2 | 26.7% | 26.7% | 62.3% | 70.9% | 0.0% | 0.0% | 0.0% | Commit 31.0%, Explore 69.0%, Recover 0.0%, Reject 0.0% |
| DEGNAV-RL geometry-only | Learned high-level mode selector ablation | Procedural v2 | 30.9% | 30.9% | 56.0% | 68.1% | 0.0% | 0.0% | 0.0% | Explore 100% |

Notes:
- DEGNAV-RL does not output direct velocity commands; the low-level velocity realization is the shared mode-conditioned controller.
- The full belief state is not better than `geometry_only` in this run.
- Reject, correct reject, and false reject are all 0.0%, so the learned selector is not yet using the failure-aware decision modes.
- This negative result supports the interpretation that sparse reward alone is insufficient, mode semantics are not learned automatically, explicit failure memory may still be necessary, Recover/Reject require dedicated reward or supervision, and longer training alone may not solve mode collapse.
- Do not claim DEGNAV-RL improves over DEGNAV-Rule or solves narrow-passage navigation.
