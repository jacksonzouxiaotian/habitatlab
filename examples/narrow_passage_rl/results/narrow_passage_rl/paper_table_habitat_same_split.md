# Table: Habitat HM3D Same-Split Nominal Anchor Comparison

This table uses only the `habitat_formal_same_split` provenance from `results_manifest.yaml`. All included methods are evaluated on the same HM3D Val set A episode IDs. It is a nominal anchor comparison, not a calibrated clearance-performance table.

| Method | Success ↑ | Collision ↓ | Near-collision ↓ (diagnostic) | Steps ↓ | Episodes | Seeds |
|---|---:|---:|---:|---:|---:|---|
| PPO direct-control baseline | 10.2% | not logged | not logged | 498.0 | 157 | 3 per manifest; no seed column |
| APF+Gap classical baseline | 93.6% | not logged | not logged | 111.5 | 157 | deterministic |
| DEGNAV-Rule / Geometry-FSM | 100.0% | 0.0% | 100.0% | 54.0 | 157 | deterministic |
| DEGNAV-Rule + failure memory | 100.0% | 0.0% | 100.0% | 54.0 | 157 | deterministic |

Notes:
- Success and steps are the fair same-split comparison metrics available for all four methods.
- `SPL`, `path_length`, and `strict_success` are not present in these source CSVs, so they are not reported here.
- Collision and near-collision are not logged in the PPO/APF+Gap source CSVs; those columns are diagnostic only and must not be used for all-method ranking.
- The Habitat clearance-related metrics are derived from depth observations and an approximate robot body-margin model. They are used as clearance-aware diagnostic indicators rather than calibrated physical safety measurements.

Provenance:
- Split: HM3D Val set A, same 157 episode IDs across all included methods.
- Anchor source: `habitat_formal_same_split` in `results_manifest.yaml`; unperturbed mined HM3D narrow-passage anchors.
- Evaluation table generated: 2026-07-13
- Episode intersection used: 157 / union 157 episode IDs.
- Source CSV paths:
  - `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_ppo_v2_episodes.csv`
  - `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_apf_gap_episodes.csv`
  - `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_fsm_v2_episodes.csv`
  - `examples/narrow_passage_rl/results/narrow_passage_rl/habitat_memory_v2_episodes.csv`

Residual risks:
- `scene_id` is only logged by the DEGNAV-Rule / Geometry-FSM and failure-memory CSVs.
- PPO seed provenance is stored in the manifest, but the CSV does not contain a per-row `seed` column, so per-seed mean/std cannot be reconstructed from this file alone.
- For a full all-method clearance-aware diagnostic table, rerun PPO and APF+Gap with evaluators that log `collision`, `near_collision`, `min_clearance`, and `strict_success` on the same episode list.
