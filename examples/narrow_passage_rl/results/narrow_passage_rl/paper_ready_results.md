# Paper-Ready Narrow-Passage Result Index

This is the citation-facing index for the AAAI narrow-passage paper.  Use this
file before browsing the many historical `paper_table_*.md` artifacts in this
directory.

The rule of thumb is simple:

- **Main paper tables/figures** below are safe to cite with their stated scope.
- **Diagnostic/appendix tables** can be cited only for the specific diagnostic
  claim described here.
- **Do-not-cite tables** are preserved for provenance but should not be used in
  the manuscript.

## Main Paper Tables And Figures

| Claim | Table / Figure | Raw provenance | Use |
| :--- | :--- | :--- | :--- |
| DEGNAV-Rule improves procedural narrow-passage traversal over the reactive rule baseline. | `paper_table_procedural_v2_main.md` | `harder_benchmark_episodes.csv`, `harder_benchmark_summary.csv` | Main synthetic benchmark. |
| Alignment, probabilistic margin, yaw prior, and recovery ablations. | `tables/paper_table_procedural_v2_ablation_core.md` | `raw/procedural_ablation_core.csv` | Main component ablation. |
| 0% false-feasible traversal success is not correct rejection. | `tables/paper_table_false_feasible_outcomes.md` | `raw/false_feasible_outcomes.csv` | Required provenance table for false-feasible outcomes. |
| DEGNAV-Rule works on nominal HM3D anchors under the current mining protocol. | `paper_table_habitat_same_split.md` | `habitat_ppo_v2_episodes.csv`, `habitat_apf_gap_episodes.csv`, `habitat_fsm_v2_episodes.csv`, `habitat_memory_v2_episodes.csv` | Same-split nominal anchor validation. |
| Habitat perturbations expose module sensitivity. | `tables/paper_table_habitat_stress_nominal.md` and `tables/paper_table_habitat_stress_key_slices.md` | `raw/habitat_stress_all.csv` | Stress validation; nominal success is the primary metric. |
| Failure memory reduces repeated infeasible commitment. | `paper_table_repeated_failure_memory.md` | `repeated_failure_memory.csv` | Main memory contribution under repeated false-feasible passages. |
| Geometry-guided memory transfers rejection while avoiding interference. | `paper_table_memory_transfer_interference.md` | `memory_transfer_interference.csv` | Main memory generalization/interference table. |
| Required-width calibration handles under- and over-conservative priors. | `tables/paper_table_calibration_extended.md` | `raw/calibration_prior_sweep.csv`, `raw/calibration_episode_predictions.csv`, `tables/calibration_reliability_bins.csv` | Calibration and reliability evidence. |
| Performance changes around the estimated feasibility boundary. | `figures/margin_phase_rule_vs_degnav_rule.{pdf,png}` and `tables/paper_table_margin_phase_summary.md` | `raw/margin_phase_rule_vs_degnav_episodes.csv` | Main Rule-vs-DEGNAV margin-phase figure. |

## Diagnostic Or Appendix-Only Results

| Diagnostic question | Table / Figure | Safe interpretation |
| :--- | :--- | :--- |
| Do Habitat clearance proxies agree with nominal success? | `tables/paper_table_habitat_clearance_diagnostic.md` | Clearance-aware strict success and near-collision are depth-derived body-margin diagnostics, not calibrated contact measurements. |
| Can Habitat-native TD3 exploit nominal success? | `paper_table_diagnostic_baselines.md` | Shows nominal success can be high while clearance diagnostics remain poor. |
| Did DEGNAV-RL learn reliable Recover/Reject behavior? | `paper_table_degnav_rl_diagnostic.md`, `paper_table_belief_mode_ablation.md` | No. DEGNAV-RL remains a diagnostic policy that collapses to Commit/Explore and is not the main method. |
| What happens with a deliberately narrow width-range probe? | `figures/infeasible_probe/` and `tables/infeasible_probe/` | Diagnostic sensitivity only. The conservative reject gate increases explicit Reject but introduces false rejects and does not solve false-feasible blockers. |
| Are smoke baselines runnable? | `paper_table_smoke_baselines.md`, `paper_table_new_baselines_smoke.md` | Code-path checks only; do not use as final baselines. |
| What are mixed-split direct-control RL numbers? | `paper_table_formal_baselines.md`, `paper_table_learning_baselines.md` | Mixed-split Habitat diagnostics only; not a fair main-paper ranking. |

## Do Not Cite As Main Results

- `paper_table_main.md` and `paper_table_ablation.md`: deprecated stubs; old
  contents are preserved under `legacy/`.
- `paper_table_formal_baselines.md`: mixed HM3D Val set A / mined Val set B
  diagnostic comparison, not a canonical ranking.
- Legacy PPO/SB3 tables with older observation mismatch or buggy metric
  definitions.
- Smoke baseline tables unless the manuscript explicitly labels them as smoke
  or appendix status checks.

## Remaining Caveats To State In The Paper

- The current one-shot false-feasible benchmark does **not** demonstrate
  correct Reject behavior for default DEGNAV-Rule; use the outcome decomposition
  and memory tables for that discussion.
- The conservative `feasibility_reject` gate is diagnostic only.  It increases
  negative-margin support and explicit Reject, but its false-reject rate makes
  it unsuitable as the main method without additional design.
- Habitat clearance metrics are derived from depth observations and an
  approximate body-margin model.  Treat them as diagnostics, not calibrated
  contact measurements.
- DEGNAV-RL is a diagnostic learned mode selector, not a competitive final
  method.
- The same-split Habitat nominal table is anchor validation under the current
  mining protocol, not a universal robustness proof.
