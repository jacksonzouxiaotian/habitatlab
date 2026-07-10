# Table: Diagnostic Learning Baselines

These rows are diagnostic evidence, not the main baseline table.  They show why
the paper reports clearance-aware safety metrics in addition to nominal
goal-reaching success.

| Method | Purpose | Train domain | Eval domain | Train budget | Seeds | Eval episodes | Success metric | SR | Strict SR | Collision | Near collision | Notes |
|:---|:---|:---|:---|:---|:---:|---:|:---|---:|---:|---:|---:|:---|
| TD3 Habitat-native | Nominal metric exploitation diagnostic | Habitat HM3D mined Val set B | Habitat HM3D mined Val set B | 1M Habitat steps | 1 | 151 | Nominal Habitat success | 100.0% | 2.0% | 0.0% | 100.0% | 98.0% success-but-unsafe; avg min clearance = -0.104 m |

Interpretation:
- TD3 Habitat-native can optimize the nominal distance/alignment success signal,
  but strict clearance-aware success remains very low.
- This diagnostic row motivates reporting `strict_success`,
  `clearance_safe`, `success_but_unsafe`, near-collision rate, and minimum
  clearance for learning baselines.
- Do not present this row as evidence that TD3 safely solves narrow-passage
  traversal; it is evidence that nominal success alone is insufficient.
