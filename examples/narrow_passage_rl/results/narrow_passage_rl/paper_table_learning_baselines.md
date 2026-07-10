# Table Index: Learning Baselines

Learning baselines are split into three groups to avoid mixing formal Habitat
comparisons with diagnostic and smoke-test runs.

## A. Formal Main Baselines

Use `paper_table_formal_baselines.md` for the main paper table.

Included:
- PPO v2 geometry sensor, 3 seeds, synthetic-to-Habitat transfer.
- SAC v2 geometry sensor.
- TD3 synthetic-to-Habitat transfer.
- APF+Gap classical baseline.
- Geometry-FSM.

Legacy PPO runs with the v1 `obs[10]` yaw/heading mismatch are excluded and
should not be treated as main baselines.

## B. Diagnostic Baselines

Use `paper_table_diagnostic_baselines.md` for rows that explain metric failure
modes rather than method ranking.

Included:
- TD3 Habitat-native nominal success vs. strict clearance-aware success.

This row shows that nominal Habitat success can be exploited by a learned policy:
TD3 reaches 100.0% nominal success but only 2.0% strict success, with 98.0%
success-but-unsafe and 100.0% near-collision.

## C. Smoke / Appendix-Only Baselines

Use `paper_table_smoke_baselines.md` for code-path checks and preliminary
negative controls.

Included:
- GRU-PPO lightweight.
- RecurrentPPO smoke.
- BC-FSM.
- DAgger-FSM.
- Replay Memory Policy.

These rows should stay in an appendix/status table unless rerun under the full
multi-seed, same-eval-domain protocol.
