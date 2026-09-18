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

## Latest Formal Habitat / MP3D Evidence (2026-09-05)

These artifacts supersede older Habitat smoke or mixed-split rows when making
claims about official MP3D PointNav or the new MP3D-derived narrow-passage
validation. The two protocols must remain in separate tables.

| Scope | Current artifact | Status and permitted claim |
| :--- | :--- | :--- |
| Official Habitat PointNav v1, MP3D val, 495 fixed episodes | `figures/habitat_official_pointnav_comparison.{pdf,png,svg}` and `../official_pointnav_mp3d_v1_20260904/paper_table_official_pointnav.md` | Complete official-reference comparison for ForwardOnly, Random, RandomForward, GoalFollower, PointNav PPO, DD-PPO transfer, and the NavMesh oracle. |
| MP3D-derived narrow passages, 19D methods, 80 fixed episodes | `figures/habitat_mp3d_narrow_method_comparison.{pdf,png,svg}`, `figures/habitat_mp3d_candidate_outcomes.{pdf,png,svg}`, and `../degnav_e2e_mp3d_narrow_v1_20260904/19d_comparison/formal_mp3d_comparison.md` | Complete scoped comparison; Geometry, kNN memory, and DEGNAV memory are identical, so it is negative evidence for a memory gain. |
| MP3D-derived narrow passages, raw-Depth DEGNAV-E2E | `figures/habitat_mp3d_narrow_method_comparison.{pdf,png,svg}`, `figures/habitat_mp3d_candidate_outcomes.{pdf,png,svg}`, and `../degnav_e2e_mp3d_narrow_v1_20260904/paper_table_degnav_e2e.md` | Complete three-seed validation: 83.33±0.59% all Success, 95.42±1.18% decision accuracy, and 79.17±2.95% Correct Reject. Offline BC, not PPO; visual-to-mode with a fixed controller. |

The full protocol separation, environment, invalid-run quarantine, and
limitations are recorded in
`../formal_mp3d_summary_20260905/formal_experiment_work_report.md`.

## Main Paper Tables And Figures

| Claim | Table / Figure | Raw provenance | Use |
| :--- | :--- | :--- | :--- |
| DEGNAV-Rule improves procedural narrow-passage traversal over the reactive rule baseline. | `paper_table_procedural_v2_main.md` | `harder_benchmark_episodes.csv`, `harder_benchmark_summary.csv` | Main synthetic benchmark. |
| Alignment, probabilistic margin, yaw prior, and recovery ablations. | `tables/paper_table_procedural_v2_ablation_core.md` | `raw/procedural_ablation_core.csv` | Main component ablation. |
| 0% false-feasible traversal success is not correct rejection. | `tables/paper_table_false_feasible_outcomes.md` | `raw/false_feasible_outcomes.csv` | Required provenance table for false-feasible outcomes. |
| DEGNAV-Rule works on nominal HM3D anchors under the current mining protocol. | `paper_table_habitat_same_split.md` | `habitat_ppo_v2_episodes.csv`, `habitat_apf_gap_episodes.csv`, `habitat_fsm_v2_episodes.csv`, `habitat_memory_v2_episodes.csv` | Same-split nominal anchor validation. |
| Habitat perturbations expose module sensitivity; the appended nominal block audits belief feasibility variants. | `tables/paper_table_habitat_stress_nominal.md` and `tables/paper_table_habitat_stress_key_slices.md` | `raw/habitat_stress_all.csv`, `audits/habitat_rq1_20260821_110829/{summary,episodes,steps}.csv` | Historical stress validation plus a separately labeled new belief-gated nominal RQ1 block; point-estimate/no-uncertainty are equivalent aliases. |
| Failure memory reduces repeated infeasible commitment. | `paper_table_repeated_failure_memory.md` | `repeated_failure_memory.csv` | Main memory contribution under repeated false-feasible passages. |
| Geometry-guided memory transfers rejection while avoiding interference. | `paper_table_memory_transfer_interference.md` | `memory_transfer_interference.csv` | Main memory generalization/interference table. |
| Required-width calibration handles under- and over-conservative priors. | `figures/calibration_posterior_convergence_paper.{pdf,png,svg}` and `tables/paper_table_calibration_extended.md` | `raw/calibration_prior_sweep.csv`, `raw/calibration_episode_predictions.csv`, `tables/calibration_reliability_bins.csv` | Calibration convergence and residual-bias evidence; synthetic threshold experiment, not Habitat closed-loop navigation. |
| Performance changes around the estimated feasibility boundary. | `figures/margin_phase_rule_vs_degnav_rule.{pdf,png}` and `tables/paper_table_margin_phase_summary.md` | `raw/margin_phase_rule_vs_degnav_episodes.csv` | Main Rule-vs-DEGNAV margin-phase figure. |

## Diagnostic Or Appendix-Only Results

| Diagnostic question | Table / Figure | Safe interpretation |
| :--- | :--- | :--- |
| Do Habitat clearance proxies agree with nominal success? | `tables/paper_table_habitat_clearance_diagnostic.md` | Clearance-aware strict success and near-collision are depth-derived body-margin diagnostics, not calibrated contact measurements. |
| Can Habitat-native TD3 exploit nominal success? | `paper_table_diagnostic_baselines.md` | Shows nominal success can be high while clearance diagnostics remain poor. |
| Did DEGNAV-RL learn reliable Recover/Reject behavior? | `paper_table_degnav_rl_diagnostic.md`, `paper_table_belief_mode_ablation.md` | No. DEGNAV-RL remains a diagnostic policy that collapses to Commit/Explore and is not the main method. |
| Is the downloaded NaVILA checkpoint runnable locally? | `../vln_dataset_free/report.md` | Yes. This is a model artifact/runtime smoke test only; it reports parseable actions, latency, and GPU memory, not navigation success. |
| What failures appear before standard VLN data is available? | `../vln_batch_diagnostic/report.md`, `../vln_batch_diagnostic/issues_and_improvements.csv` | The 640-inference diagnostic exposes stop failure, action-prior bias, weak left/right grounding, and visual perturbation sensitivity. R2R text is paired with unmatched demo frames, so this is not R2R SR/SPL. |
| Does a deterministic NaVILA-to-DEGNAV interface remove obvious control-channel hazards? | `../vln_safety_adapter_smoke/report.md`, `../vln_safety_adapter_smoke/comparison_summary.csv` | Yes for trusted stop/directive and synthetic low-clearance gates in paired replay. This is interface logic, not learned VLN improvement or physical safety. |
| What happens with a deliberately narrow width-range probe? | `figures/infeasible_probe/` and `tables/infeasible_probe/` | Diagnostic sensitivity only. The conservative reject gate increases explicit Reject but introduces false rejects and does not solve false-feasible blockers. |
| Are smoke baselines runnable? | `paper_table_smoke_baselines.md`, `paper_table_new_baselines_smoke.md` | Code-path checks only; do not use as final baselines. |
| What are mixed-split direct-control RL numbers? | `paper_table_formal_baselines.md`, `paper_table_learning_baselines.md` | Mixed-split Habitat diagnostics only; not a fair main-paper ranking. |
| Does replacing the 19D actor input with raw Depth already prove a fully end-to-end controller? | `../degnav_e2e_mp3d_narrow_v1_20260904/paper_table_degnav_e2e.md` | No. It proves a raw-Depth recurrent visual-to-mode actor under scene-disjoint validation; the low-level controller is fixed and checkpoint selection used the same validation corpus. |

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
- The NaVILA/VLN diagnostics are runtime and interface evidence only.  They do
  not replace standard paired R2R/RxR evaluation with MP3D scenes.
- The same-split Habitat nominal table is anchor validation under the current
  mining protocol, not a universal robustness proof.
- DEGNAV-E2E is offline behavior cloning, not PPO. Its current checkpoint was
  selected with MP3D-val teacher labels and evaluated closed-loop on the same
  validation corpus; report it as validation, not an independent test result.
- Across 240 DEGNAV-E2E seed-episodes, Recover was selected in only one episode
  and that episode timed out. Current evidence does not establish reliable
  recovery behavior.
