# Habitat Same-Split Validation Report

## Source Audit

| File | Method | Split | Episodes | Unique episode IDs | Seeds | Available requested fields | Notes |
|---|---|---|---:|---:|---|---|---|
| `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_ppo_v2_episodes.csv` | PPO direct-control baseline | not logged; manifest: HM3D Val set A | 157 | 157 | 3 per manifest; no seed column | episode_id, success, steps | method column absent; method inferred from manifest/file name |
| `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_apf_gap_episodes.csv` | APF+Gap classical baseline | not logged; manifest: HM3D Val set A | 157 | 157 | deterministic | episode_id, success, steps | method column absent; method inferred from manifest/file name |
| `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_fsm_v2_episodes.csv` | DEGNAV-Rule / Geometry-FSM | not logged; manifest: HM3D Val set A | 157 | 157 | deterministic | scene_id, episode_id, success, collision, near_collision, steps | method column absent; method inferred from manifest/file name |
| `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_memory_v2_episodes.csv` | DEGNAV-Rule + failure memory | not logged; manifest: HM3D Val set A | 157 | 157 | deterministic | scene_id, episode_id, success, collision, near_collision, steps | method column absent; method inferred from manifest/file name |

## Fairness Check

- Episode ID intersection: 157.
- Episode ID union: 157.
- The four source files contain identical episode ID sets, so no episode intersection filtering was needed beyond verifying equality.
- The table does not mix procedural data, stress slices, or mined Val set B source CSVs.
- Success is treated as Habitat `narrow_passage_success` provenance, following the corresponding evaluators.
- SPL/path_length are unavailable for all methods and are omitted.
- Collision and near-collision are unavailable for PPO and APF+Gap, so they are reported as `not logged` and are not fair all-method ranking metrics.
- Strict success is unavailable in these source CSVs and is not reported. If added later, it must be named `Clearance-aware Strict SR (diagnostic)`.

## Generated Outputs

- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_habitat_same_split.md`
- `examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_habitat_same_split.tex`

## Recommended Rerun If Full Diagnostic Columns Are Needed

Rerun PPO/APF+Gap on the same HM3D Val set A episode list with evaluators that log `collision`, `near_collision`, `min_clearance`, and `strict_success`. Do not replace this table with `paper_table_formal_baselines.md`, because that table intentionally mixes HM3D Val set A and mined Val set B.
