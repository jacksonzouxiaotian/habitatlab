# Table: Mixed-Split Habitat Diagnostic Comparison

This table is a mixed-split Habitat diagnostic comparison, not a canonical main
paper ranking.  It contains rows evaluated on both HM3D Val set A and mined Val
set B, so it cannot be used for a fair method ordering in the main paper.
Geometry-FSM is referred to as DEGNAV-Rule in the paper.  Legacy PPO runs with
the v1 `obs[10]` yaw/heading mismatch are excluded from the current diagnostic
comparison.

Success metric: nominal `NarrowPassageNav-v0` success unless otherwise stated.
Clearance-aware strict success is reported separately in
`paper_table_diagnostic_baselines.md`.

| Method | Category | Train domain | Eval domain | Train budget | Seeds | Eval episodes | Success metric | SR | Collision | Notes |
|:---|:---|:---|:---|:---|:---:|---:|:---|---:|---:|:---|
| PPO v2 geometry sensor | Direct-control RL baseline | Synthetic v2 `HarderNarrowPassageEnv` | Habitat HM3D Val set A | 5M synthetic steps | 3 | 157 | Nominal Habitat success | 2.1% ± 2.6% | - | Maps 19-D geometry directly to velocity actions; seeds: 5.7%, 0.0%, 0.6% |
| SAC v2 geometry sensor | Direct-control RL baseline | Synthetic v2 `HarderNarrowPassageEnv` | Habitat HM3D mined Val set B | 2M synthetic steps | 1 | 151 | Nominal Habitat success | 0.0% | - | Same 19-D geometry interface as PPO v2 |
| TD3 synthetic-to-Habitat | Direct-control RL baseline | Synthetic v2 `HarderNarrowPassageEnv` | Habitat HM3D mined Val set B | 3M synthetic steps | 1 | 151 | Nominal Habitat success | 2.0% | 0.0% | Synthetic v2 SR = 77.8%; transfer collapses on HM3D |
| APF+Gap | Classical local baseline | None | Habitat HM3D Val set A | No learning | deterministic | 157 | Nominal Habitat success | 93.6% | 0.0% | Depth + GPS reactive planner |
| DEGNAV-Rule / Geometry-FSM | Proposed rule controller | None | Habitat HM3D mined Val set B | No learning | deterministic | 151 | Nominal anchor validation under the current mining protocol | 100.0% | 0.0% | Mined anchors are mostly well aligned; see stress validation for perturbations |

Interpretation:
- The learning rows use the corrected v2 geometry sensor format and are
  direct-control diagnostic baselines, not DEGNAV-RL.
- This table mixes HM3D Val set A and mined Val set B.  Do not cite it as a
  same-split main paper ranking.
- The low PPO/SAC/TD3 transfer scores support the claim that learning-only
  policies are fragile near geometric feasibility boundaries.
- DEGNAV-Rule / Geometry-FSM's 100.0% row is nominal anchor validation under
  the current mining protocol, not a standalone robustness claim; use
  `paper_table_habitat_stress.md` for module sensitivity under Habitat
  perturbations.
