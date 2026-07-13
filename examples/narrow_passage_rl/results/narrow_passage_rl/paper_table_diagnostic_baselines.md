# Table: Diagnostic Learning Baselines

These rows are diagnostic evidence, not the main baseline table.  They show why
the paper reports clearance-aware diagnostic metrics in addition to nominal
goal-reaching success.

The Habitat clearance-related metrics are derived from depth observations and an
approximate robot body-margin model. They are used as clearance-aware diagnostic
indicators rather than calibrated physical safety measurements.

Geometry-FSM is referred to as DEGNAV-Rule in the paper.  The TD3 row below is
not DEGNAV-RL: it is a direct-control policy that maps geometry observations
directly to velocity actions.

| Method | Purpose | Train domain | Eval domain | Train budget | Seeds | Eval episodes | Success metric | SR | Strict SR | Collision | Near collision | Notes |
|:---|:---|:---|:---|:---|:---:|---:|:---|---:|---:|---:|---:|:---|
| TD3 Habitat-native | Nominal metric exploitation diagnostic | Habitat HM3D mined Val set B | Habitat HM3D mined Val set B | 1M Habitat steps | 1 | 151 | Nominal Habitat success | 100.0% | 2.0% | 0.0% | 100.0% | Direct-control TD3; 98.0% success-but-unsafe; avg min clearance = -0.104 m |

Interpretation:
- TD3 Habitat-native can optimize the nominal distance/alignment success signal,
  but clearance-aware strict success remains very low.
- This diagnostic row motivates reporting `strict_success`,
  `clearance_safe`, `success_but_unsafe`, near-collision rate, and minimum
  clearance for learning baselines.
- Do not present this row as evidence that TD3 achieves calibrated physical
  safety; it is evidence that nominal success alone is insufficient.
