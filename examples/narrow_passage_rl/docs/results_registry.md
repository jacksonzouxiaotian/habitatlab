# Narrow-Passage Results Registry

This document explains how to read the canonical result manifest:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/results_manifest.yaml
```

The manifest is the paper-facing provenance index for the AAAI narrow-passage
experiments.  Every entry records the raw CSV files, generated paper tables,
domain, split, methods, seeds, episode counts, metrics, and whether the result
is safe to cite in the main paper.

## Status Labels

| Status | Meaning | Main-paper use |
|:---|:---|:---|
| `canonical` | Controlled result with clear provenance and a stable task definition. | Safe to cite when `allowed_in_main_paper: true`. |
| `diagnostic` | Result used to explain a failure mode, metric issue, or design choice. | Safe only as diagnostic evidence when explicitly labeled. |
| `smoke` | Small run used to verify that code paths work. | Do not cite as experimental evidence. |
| `legacy` | Older or mixed-provenance result kept for reproducibility history. | Do not cite as current formal evidence. |

Two rules matter most:

1. Mixed-split tables are never canonical.
2. Smoke baselines are never allowed in the main paper.

## Safe-To-Cite Entries

The following entries are intended to be safe for the AAAI paper, subject to the
notes in the manifest:

| Manifest key | Status | What it supports |
|:---|:---|:---|
| `procedural_v2_main` | canonical | Main procedural v2 comparison for DEGNAV-Rule. |
| `procedural_v2_ablation` | canonical | Rule-controller ablations in procedural v2. |
| `habitat_formal_same_split` | canonical | Same-split HM3D anchor-validation provenance. |
| `habitat_stress_nominal` | canonical | Stress-tested Habitat robustness and module sensitivity. |
| `repeated_failure_memory` | canonical | Memory reduces repeated false-feasible commitment and wasted attempts. |
| `memory_transfer_interference` | canonical | Geometry-anchored memory transfers to similar infeasible passages without false rejection on similar feasible ones. |
| `dmin_calibration` | canonical | Clearance / width-margin calibration evidence. |
| `margin_phase` | canonical | Width-margin phase diagram around the feasibility boundary. |
| `habitat_clearance_diagnostic` | diagnostic | Nominal success can hide unsafe RL behavior; strict metrics are required. |
| `rl_direct_control_baselines` | diagnostic | PPO/SAC/TD3 direct-control transfer baselines are fragile near geometric feasibility limits. |
| `degnav_rl_diagnostic` | diagnostic | DEGNAV-RL is a diagnostic policy; current runs collapse to Commit/Explore and do not demonstrate meaningful Recover or Reject behavior. |

For diagnostic entries, the main text should say what the diagnostic proves.  For
example, `habitat_clearance_diagnostic` supports the need for strict
clearance-aware metrics; it does not support a claim that TD3 safely solves
Habitat narrow-passage traversal.

## Entries Not Safe For Main Claims

| Manifest key | Status | Reason |
|:---|:---|:---|
| `legacy_old_procedural_table` | legacy | Older procedural result format and legacy PPO/SB3 provenance. |
| `legacy_old_ablation_table` | legacy | Older procedural ablation/case table with superseded method names and metrics. |
| `legacy_mixed_split_formal_baselines` | legacy | Mixes Habitat Val set A and mined Val set B rows; not a canonical ranking. |
| `smoke_baselines` | smoke | Verifies RecurrentPPO, BC, DAgger, and replay-memory code paths only. |

These files should stay in the repository for reproducibility history, but they
should not be used as current paper evidence.

The old root paths `paper_table_main.md` and `paper_table_ablation.md` are
deliberately small deprecation stubs.  Their preserved contents live under:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/legacy/
```

## Method Names

Use these paper-facing names consistently:

| Implementation name | Paper-facing name |
|:---|:---|
| `Geometry-FSM` | `DEGNAV-Rule` |
| `FSM + Failure Memory` | `DEGNAV-Rule + failure memory` |
| `BeliefModeEnv` / belief-mode PPO | `DEGNAV-RL` |
| PPO/SAC/TD3 velocity policies | Learning-only direct-control baselines |

DEGNAV-RL is included as a diagnostic policy rather than a competitive final
method. It tests the high-level mode selector `pi(m_t | b_t)` over the explicit
belief state and does not output direct velocity commands. Under the current
reward and action interface, the learned policy collapses to Commit and Explore
and does not demonstrate meaningful Recover or Reject behavior. The
direct-control PPO/SAC/TD3 baselines map geometry observations directly to
velocity actions.

## Habitat 100 Percent Anchor Validation

The DEGNAV-Rule / Geometry-FSM Habitat row with 100.0 percent success must be
described as:

```text
100.0% nominal anchor validation under the current mining protocol.
```

Do not describe it as universal robustness, perfect generalization, or a solved
HM3D task.  The nominal mined anchors are mostly well aligned, so the stress
validation table is the correct evidence for perturbation robustness and module
sensitivity.

Use:

- `habitat_formal_same_split` for nominal same-split anchor-validation
  provenance.
- `habitat_stress_nominal` for yaw, lateral offset, dropout, noise, and
  extreme-narrow stress validation.
- `habitat_clearance_diagnostic` when explaining why nominal success can be
  unsafe for direct-control RL.

## Memory Claim

The memory contribution is not one-shot Habitat passable-anchor success.  In the
nominal Habitat table, Geometry-FSM and FSM + Failure Memory both reach 100.0
percent nominal anchor-validation success.

The correct memory claim is:

```text
Failure memory suppresses repeated commitments to previously failed infeasible
passages and reduces wasted false-feasible attempts.
```

Use `repeated_failure_memory` and `memory_transfer_interference` for this claim.

## Regenerating Paper Tables

The manifest is an index, not a table generator.  To regenerate the paper tables
from current CSVs, run:

```bash
python examples/narrow_passage_rl/make_paper_tables.py
```

To regenerate the margin-phase figure:

```bash
python examples/narrow_passage_rl/plot_margin_phase.py \
  --inputs \
    examples/narrow_passage_rl/results/narrow_passage_rl/harder_benchmark_episodes.csv \
    examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed0_eval.csv \
  --labels DEGNAV-Rule DEGNAV-RL \
  --output-dir examples/narrow_passage_rl/results/narrow_passage_rl \
  --bin-width 0.02 \
  --margin-min -0.10 \
  --margin-max 0.10
```

After regenerating tables, update `results_manifest.yaml` if a raw CSV, split,
seed set, episode count, or paper-facing table path changes.
