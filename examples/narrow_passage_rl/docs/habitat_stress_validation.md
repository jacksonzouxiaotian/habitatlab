# Habitat Stress Validation

Habitat HM3D nominal anchors are useful scene-based validation, but they are not
enough to claim broad robustness.  The mined starts are mostly aligned with the
passage entrance, so full Geometry-FSM and some ablations can all reach 100% in
the nominal table.  Stress validation perturbs the same task to reveal module
sensitivity.

## Why Nominal Anchors Are Not Enough

Nominal anchors mainly answer:

- Can the controller use HM3D depth-derived geometry features?
- Does the controller reach the local passage goal on scanned scenes?
- Does the mined dataset wiring work in `NarrowPassageNav-v0`?

They do not fully answer:

- What happens if the robot starts with wrong yaw?
- What happens if the robot is laterally offset from the entrance?
- What happens when depth sectors drop out or geometry estimates are noisy?
- Which FSM module actually matters near the clearance boundary?

Therefore the 100% nominal row should be reported as **Habitat HM3D Anchor
Validation**, and the stress table should be cited for robustness claims.

## Stress Factors

The formal script is:

```bash
python examples/narrow_passage_rl/eval_habitat_stress_validation.py \
    --preset paper \
    --split val \
    --num-episodes -1
```

It evaluates:

| Factor | Values | Purpose |
|---|---|---|
| start yaw perturbation | 0, 30, 60 degrees | Tests heading alignment |
| lateral start offset | 0.0 m, 0.10 m, 0.20 m | Tests lateral centering |
| depth-sector dropout | 0%, 10%, 30% | Tests missing local depth sectors |
| feature Gaussian noise | sigma = 0.0, 0.02, 0.05 | Tests noisy geometry features |
| extreme narrow subset | `body_margin < 0.05 m` | Tests near-limit clearance |

Compared methods:

- Full Geometry-FSM.
- FSM w/o heading alignment.
- FSM w/o recovery.
- FSM w/o lateral alignment.
- APF+Gap when the interface is available.

## Metrics

Stress validation reports:

- Success rate.
- Strict success rate.
- Collision rate.
- Near-collision rate.
- Average minimum clearance.
- Average steps.
- Timeout rate.

Strict success should be used when comparing against learning baselines or
when nominal success appears saturated.

## How To Interpret Results

Interpretation should focus on module sensitivity under perturbation.

Examples:

- If full FSM succeeds under +60 degree yaw but w/o heading alignment fails,
  heading alignment is the exposed critical module.
- If lateral-offset stress hurts w/o lateral alignment more than full FSM,
  lateral centering is useful.
- If dropout/noise reduces strict success while nominal success stays high,
  report the safety degradation rather than only goal-reaching success.
- If all methods remain saturated under a stress factor, the stress is not
  strong enough to separate modules and should not be overinterpreted.

## Outputs

```text
results/narrow_passage_rl/habitat_stress_validation.csv
results/narrow_passage_rl/paper_table_habitat_stress.md
results/narrow_passage_rl/paper_table_habitat_stress.tex
```

If these files are missing, rerun the script above before citing the stress
table.

## Limitations

The current stress validation perturbs existing mined episodes.  The following
require new data generation or simulator extensions and should not be claimed
as completed unless corresponding CSVs exist:

- Goal perturbation around the exit.
- Habitat false-feasible anchors blocked inside the passage.
- Dynamic obstacles near or inside the passage.
- A separate unseen-room/test-room split.
