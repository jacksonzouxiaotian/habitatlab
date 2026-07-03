# Habitat Stress Validation

The unperturbed HM3D mined-val FSM rows can reach 100% success because the
mined anchors are reasonably aligned with the passage entrance.  Stress
validation is therefore reported next to the main Habitat table to show which
controller modules survive realistic start-pose and sensing perturbations.

## Implemented Stress Tests

`eval_habitat_fsm_ablations.py` supports the following perturbations without
regenerating the dataset:

| Stress type | CLI flag | Purpose |
|---|---|---|
| Initial yaw perturbation | `--heading-perturb-deg` | Tests heading alignment under start orientation error |
| Lateral start offset | `--lateral-perturb-m` | Tests entrance centering under side offset |
| Start distance shift | `--start-distance-shift-m` | Tests whether success depends on a fixed entrance distance |
| Depth/geometry noise | `--feature-noise-std` | Tests noisy clearance and passage-width estimates |
| Depth-sector dropout | `--depth-dropout-prob` | Tests robustness to missing local depth sectors |
| Extreme-narrow subset | `--split extreme_narrow` | Tests anchors with body margin below 0.05 m |

Example commands:

```bash
conda activate habitat

# Yaw stress on extreme-narrow anchors.
python examples/narrow_passage_rl/eval_habitat_fsm_ablations.py \
    --split extreme_narrow \
    --variants full no_heading_alignment no_lateral_alignment \
    --heading-perturb-deg 60 \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/habitat_fsm_stress_yaw60.csv

# Lateral offset stress.
python examples/narrow_passage_rl/eval_habitat_fsm_ablations.py \
    --split extreme_narrow \
    --variants full no_lateral_alignment no_alignment \
    --lateral-perturb-m 0.2 \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/habitat_fsm_stress_lat02.csv

# Feature-level sensor stress.
python examples/narrow_passage_rl/eval_habitat_fsm_ablations.py \
    --split extreme_narrow \
    --variants full no_recovery no_alignment \
    --feature-noise-std 0.03 \
    --depth-dropout-prob 0.10 \
    --seed 0 \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/habitat_fsm_stress_depth_noise.csv
```

For the paper table, report the perturbation parameters in the row label, for
example `Extreme-narrow +60 deg yaw` or `Extreme-narrow +0.2 m lateral offset`.

## Planned Stress Tests

The following tests require new mined anchors, dataset regeneration, or simulator
extensions.  Do not report them as completed until the corresponding CSVs exist.

| Stress type | Required work | Purpose |
|---|---|---|
| Goal perturbation | Regenerate episodes with randomized exit-side goals | Checks overfitting to a fixed local goal point |
| False-feasible Habitat anchors | Mine entrances that look passable but become blocked inside | Tests failure-memory reject decisions |
| Dynamic obstacle | Add temporary blockers or pedestrian proxies near entrances | Tests recovery and re-planning under transient blockage |
| Scene generalization | Build unseen HM3D val/test room split | Checks that success is not scene memorization |

## Reporting Rule

When the main Habitat table contains 100% FSM success, include:

- mined split size and body-margin distribution,
- `allow_sliding=False`,
- success/collision definitions,
- at least one start-pose stress row,
- whether each stress row is implemented from the existing dataset or from a
  regenerated stress split.
