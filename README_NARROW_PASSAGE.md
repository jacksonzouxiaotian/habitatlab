# Narrow-Passage Navigation Research Framework

This fork contains a focused narrow-passage navigation framework built on top of
Habitat-Lab.  The research code lives under:

```text
examples/narrow_passage_rl/
```

Geometry-FSM is the implementation name used by scripts and result files.
Geometry-FSM is referred to as DEGNAV-Rule in the paper.

Paper-facing naming:

- `DEGNAV-Rule / Geometry-FSM`: interpretable feasibility-belief controller.
- `DEGNAV-RL`: diagnostic policy that tests whether PPO can learn only the
  high-level mode selector `pi(m_t | b_t)` over `Commit`, `Explore`, `Recover`,
  and `Reject`.
- `PPO/SAC/TD3 direct-control baselines`: learning-only policies that map the
  19-D geometry observation directly to velocity actions.
- `Diagnostic/smoke baselines`: code-path checks or metric diagnostics, not the
  main method ranking.

## Latest formal MP3D result status (2026-09-05)

The official Habitat PointNav v1 MP3D-val queue is complete on the same 495
episode IDs for every available seed. PointNav PPO reaches `78.52±0.34%`
Success and `0.64±0.01` SPL; DD-PPO reaches `93.87±0.42%` Success and
`0.85±0.00` SPL, with the required Gibson-2+→MP3D transfer label. The random,
reactive, and NavMesh-oracle rows are complete as well.

The separate MP3D-derived narrow-passage validation uses 80 scene-disjoint
episodes, a 0.36 m body, continuous velocity control, and no sliding.
Geometry-19D/no-memory, kNN-memory-19D, and DEGNAV-memory-19D are identical:
`82.50%` all-episode Success, `98.44%` feasible Success, and `0%` Correct
Reject. This is a negative memory result. The corrected-history raw-Depth
DEGNAV-E2E three-seed run is complete: `83.33±0.59%` all-episode Success,
`99.48±0.74%` feasible Success, `95.42±1.18%` decision accuracy, and
`79.17±2.95%` Correct Reject, with `2.08±0.74%` False Reject and zero feasible
collisions. It uses offline BC and a fixed low-level controller, not PPO. No
earlier invalid E2E number is admissible.

Current evidence and reproduction details:

- `examples/narrow_passage_rl/results/formal_mp3d_summary_20260905/final_method_status.md`
- `examples/narrow_passage_rl/results/formal_mp3d_summary_20260905/formal_experiment_work_report.md`
- `examples/narrow_passage_rl/results/formal_mp3d_summary_20260905/environment_snapshot.md`
- `examples/narrow_passage_rl/results/official_pointnav_mp3d_v1_20260904/`
- `examples/narrow_passage_rl/results/degnav_e2e_mp3d_narrow_v1_20260904/`

The official PointNav and derived narrow-passage numbers are intentionally kept
in separate tables because their agent radius, actions, sliding, Success
definition, and episode set differ. Neither new paper-facing table includes a
strict-success column.

## Where The Code Lives

```text
examples/narrow_passage_rl/
  procedural_env_v2.py                  # Synthetic v2 benchmark
  eval_harder_benchmark.py              # Main procedural benchmark
  eval_habitat_geometry_fsm.py          # Geometry-FSM / DEGNAV-Rule on HM3D anchors
  eval_habitat_apf_gap.py               # APF+Gap Habitat baseline
  eval_habitat_sb3.py                   # PPO/SAC/TD3 direct-control Habitat evaluation
  eval_habitat_stress_validation.py     # Formal Habitat stress validation
  eval_repeated_failure_memory.py       # Repeated false-feasible memory test
  eval_memory_transfer_interference.py  # Memory transfer/interference test
  eval_dmin_calibration.py              # D_min self-calibration
  train_belief_mode_ppo.py              # DEGNAV-RL high-level mode-selection PPO
  eval_belief_mode_ppo.py               # DEGNAV-RL procedural v2 evaluation
  train_sb3_v2.py                       # PPO/SAC/TD3 direct-control training
  train_recurrent_ppo_v2.py             # RecurrentPPO smoke baseline
  train_bc_dagger_v2.py                 # BC/DAgger smoke baselines
  train_replay_memory_policy_v2.py      # Generic replay-memory smoke baseline
  vln_dataset_free_smoke.py             # NaVILA artifact/runtime smoke test
  vln_batch_diagnostic.py               # Controlled 640-case VLN diagnostic
  eval_vln_safety_adapter.py            # Paired system-interface smoke test
  plot_margin_phase.py                  # Rule-vs-DEGNAV width-margin phase diagram
  record_habitat_video.py               # Habitat video/keyframe generation
  narrow_passage/models/belief_state.py # Compact feasibility-belief state
  narrow_passage/envs/belief_mode_env.py# Discrete mode wrapper for DEGNAV-RL
  narrow_passage/metrics/strict_metrics.py # Shared clearance diagnostic metrics
  results/narrow_passage_rl/            # CSV/Markdown/LaTeX/figures
```

The Habitat task and sensors used by these scripts live in:

```text
habitat-lab/habitat/tasks/narrow_passage/
```

## VLN/VLA Integration Status

The current NaVILA checkpoint has been evaluated in 640 controlled inferences:
64 immediate-command probes, 64 visual-perturbation cases, and 256 paired R2R
val-unseen instruction/stop-override probes.  It returns parseable actions in
100% of cases, but explicit stop-override compliance is only 1.6%, controlled
left/right/stop command match is 0%, and non-clean visual action consistency is
75%.

The added deterministic safety adapter treats NaVILA as a semantic proposal
model and places trusted runtime directives plus measured geometry risk below
it.  On the same saved outputs, controlled command match changes from 25.0% to
81.3%, trusted stop compliance from 1.5% to 100.0%, and forward output on four
controlled blocked-fixture cases from 100.0% to 0.0%.  Original untrusted R2R
route proposals are preserved in all 256 pairs.  These are smoke-test interface
effects, not trained VLN improvements or formal R2R metrics.
They should not be described as learned VLN gains, standard R2R performance, or
evidence that NaVILA itself learned safe stopping/rejection.

A paired R2R/VLN-CE preflight is now available.  It confirms the NaVILA
checkpoint, R2R `val_unseen` annotations, DDPPO depth encoder, data symlinks,
and Habitat 0.1.7 imports, but reports that the local MP3D scene directory has
0/11 required `val_unseen` `.glb` files and 0/11 `.navmesh` files.  The
standard SR/SPL/NE/nDTW evaluation is therefore still blocked by licensed MP3D
assets rather than code setup.

See the detailed architecture, complete data, limitations, commands, and
priority roadmap in:

```text
examples/narrow_passage_rl/README.md
  -> VLN/VLA Integration: Current Evidence And Improvement Plan
```

## Result Registry

Use the manifest below as the provenance source for paper-facing results:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/results_manifest.yaml
```

Use the paper-ready index below as the human-facing entry point for final
tables and figures:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_ready_results.md
```

It separates main-paper evidence from diagnostic, smoke, mixed-split, and legacy
artifacts.  Start there when deciding what to cite in the manuscript.

The current procedural v2 main table is:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_procedural_v2_main.md
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_procedural_v2_main.md
```

False-feasible outcome decomposition is tracked separately:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/raw/false_feasible_outcomes.csv
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_false_feasible_outcomes.md
```

The decomposition shows that false-feasible traversal success is 0.0% for both
Reactive rule baseline and DEGNAV-Rule / Geometry-FSM, but correct reject is
also 0.0%.  Therefore 0% traversal success must not be interpreted as correct
rejection; it is decomposed into collision, timeout/stuck, and wasted attempts.

Reject is counted only when a controller explicitly emits the Reject mode.  The
default DEGNAV-Rule / Geometry-FSM row should not be described as solving
false-feasible rejection.  A separate diagnostic candidate can be run with
`--methods rule_baseline geometry_fsm feasibility_reject` to test conservative
observable-geometry rejection and report false-reject rate separately.

Deprecated legacy tables such as `paper_table_main.md` and
`paper_table_ablation.md` are stubs only.  Their old contents are preserved under
`examples/narrow_passage_rl/results/narrow_passage_rl/legacy/` and must not be
cited as current main-paper results.

## Mixed-Split Habitat Diagnostic Comparison

`paper_table_formal_baselines.md` is a mixed-split Habitat diagnostic
comparison, not a canonical main-paper ranking:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_formal_baselines.md
```

It contains rows from both HM3D Val set A and mined Val set B, so it cannot be
used for fair method ordering in the main paper.

Headline values from that table:

| Method | Role | Train domain | Eval domain | Seeds | Success |
|---|---|---|---|---:|---:|
| PPO v2 geometry sensor | Direct-control RL baseline | Synthetic v2 -> Habitat | HM3D Val set A | 3 | 2.1% +/- 2.6% |
| SAC v2 geometry sensor | Direct-control RL baseline | Synthetic v2 -> Habitat | HM3D mined Val set B | 1 | 0.0% |
| TD3 synthetic-to-Habitat | Direct-control RL baseline | Synthetic v2 -> Habitat | HM3D mined Val set B | 1 | 2.0% |
| APF+Gap | Classical local baseline | None | HM3D Val set A | deterministic | 93.6% |
| DEGNAV-Rule / Geometry-FSM | Proposed rule controller | None | HM3D mined Val set B | deterministic | 100.0% nominal anchor validation |

The 100.0% DEGNAV-Rule / Geometry-FSM row is **nominal anchor validation under
the current mining protocol**, not a complete robustness claim.  The mined HM3D
starts are mostly well aligned, so this table must be read together with the
stress-validation table.

Habitat experiments should be interpreted in three separate categories:

- Nominal same-split evaluation: unperturbed HM3D anchors evaluated on a single
  split/protocol.
- Perturbation or stress evaluation: yaw, lateral offset, dropout, noise, and
  extreme-narrow stressors used to expose module sensitivity.
- Clearance-aware diagnostic evaluation: depth-derived body-margin diagnostics
  such as clearance-aware strict success, near-collision, and
  success-but-unsafe.

The Habitat clearance-related metrics are derived from depth observations and an approximate robot body-margin model. They are used as clearance-aware diagnostic indicators rather than calibrated contact measurements.

Older single-run PPO mined-val logs reported 6.0% on 151 episodes.  That number
is kept only as legacy/single-run provenance and is not the formal 3-seed
baseline.

## Habitat Stress Validation

Stress validation is the module-sensitivity evidence for DEGNAV-Rule /
Geometry-FSM.  It perturbs yaw, lateral offset, depth-sector dropout, feature
noise, and the extreme-narrow subset.

```bash
python examples/narrow_passage_rl/eval_habitat_stress_validation.py \
    --preset paper \
    --split val \
    --num-episodes -1
```

Outputs:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/habitat_stress_validation.csv
examples/narrow_passage_rl/results/narrow_passage_rl/raw/habitat_stress_all.csv
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_habitat_stress_nominal.md
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_habitat_clearance_diagnostic.md
examples/narrow_passage_rl/results/narrow_passage_rl/tables/paper_table_habitat_stress_key_slices.md
```

## Habitat Experimental Analysis

The analysis below is restricted to the canonical same-split HM3D Val set A
comparison (157 shared episode IDs) and the controlled Habitat stress protocol.
The mixed-split diagnostic table is not used for method ranking.  For the
requested memory ablation, **Ours** denotes DEGNAV-Rule with failure memory and
**Ours w/o Memory** denotes the same Geometry-FSM controller without that
memory.

### Overall Performance

On the shared nominal anchors, Ours and Ours w/o Memory both reach 100.0%
Success, compared with 93.6% for APF+Gap and 10.2% for the PPO direct-control
baseline.  Their mean episode length is also identical at 54.0 steps, whereas
APF+Gap requires 111.5 steps and PPO averages 498.0 steps.  The improvement over
PPO is not explained by a larger learned policy: DEGNAV explicitly decomposes
the problem into entrance alignment, feasibility-aware mode selection, and
mode-conditioned velocity realization.  This inductive structure prevents a
single continuous policy from having to rediscover the coupling among heading,
lateral centering, clearance, and forward progress after a
synthetic-to-Habitat domain shift.  PPO instead maps the 19-D geometry vector
directly to velocity, so errors in entry alignment can compound over a long
rollout.

SPL is not available in the four canonical same-split CSVs because geodesic
shortest distance and executed path length were not logged consistently.
Average steps suggests a large difference in control-time efficiency, but it is
not a substitute for SPL: the methods use continuous actions and may execute
different linear/angular velocities.  A quantitative path-efficiency claim
therefore remains pending a same-split rerun with `path_length`,
`shortest_path_length`, and `spl` recorded for every method.

### Failure Analysis

**Collision.** Ours and Ours w/o Memory record zero simulator collision flags on
the 157 nominal anchors, while the canonical PPO/APF CSVs do not contain a
comparable collision field.  The stress evaluator also reports zero collision
for the listed methods, but near-collision is 100% and the average depth-derived
minimum margin is negative.  The apparent zero-collision result is therefore
best explained by the current boundary-stopping/collision interface and cannot
be interpreted as calibrated physical safety.

**Timeout.** Full DEGNAV has 0% timeout on nominal, yaw-60, and
extreme-yaw-60 stress slices.  Removing heading alignment raises timeout to
16.6% at yaw 60 degrees and 12.5% on extreme-narrow plus yaw 60 degrees.  The
full controller avoids these failures because it corrects orientation before
committing, whereas an unaligned controller can spend the episode making little
longitudinal progress.  PPO's 498/500 mean steps is consistent with frequent
budget exhaustion, but its canonical CSV has no explicit timeout flag, so an
exact PPO timeout rate is not claimed.

**Oscillation.** The canonical Habitat files do not log an episode-level
oscillation count or signed angular-velocity switching frequency.  Long PPO
rollouts and rotating failure videos are useful qualitative diagnostics, but
they are not quantitative oscillation evidence.  A formal claim requires all
methods to log angular sign changes, heading-error reversals, and dwell time
near the entrance under the same protocol.

**Wrong decision.** Wrong decisions are only well defined for controllers with
an explicit mode interface.  Neither Ours variant rejects any nominal passable
anchor, but the nominal set does not contain the repeated false-feasible
encounters needed to evaluate Reject.  PPO has no discrete Commit/Reject mode,
so its failures cannot be relabeled as wrong high-level decisions.  Likewise,
failure or timeout must not be counted as correct rejection.

### Ablation Analysis

| Variant | Success | Mean steps | Collision | Stuck | Memory writes |
|---|---:|---:|---:|---:|---:|
| Ours (DEGNAV-Rule + failure memory) | 100.0% | 54.0 | 0.0% | 0.0% | 0 |
| Ours w/o Memory (DEGNAV-Rule / Geometry-FSM) | 100.0% | 54.0 | 0.0% | 0.0% | 0 |
| PPO direct-control baseline | 10.2% | 498.0 | not logged | not logged | N/A |

The equality between the two Ours rows is itself informative.  These are
one-shot, mostly passable nominal anchors; neither controller triggers recovery
or writes a failure memory item.  The table therefore isolates the contribution
of the geometry controller but does not test the claimed purpose of memory.
Memory must be judged in the separate recurrence/transfer protocol, where prior
failures are available and false rejection on similar feasible passages is also
measured.  PPO is a learning baseline rather than a strict component ablation;
its low transfer Success shows that direct velocity learning does not inherit
the geometry controller's alignment and mode priors.

### Generalization Analysis

The nominal experiment transfers the controller from procedural development to
scanned HM3D geometry, while the stress protocol changes initial yaw, lateral
offset, passage margin, depth-sector availability, and feature noise.  Full
DEGNAV retains 100.0% Success at yaw 60 degrees and on the
extreme-narrow-plus-yaw-60 slice, and reaches 98.0% at a 0.20 m lateral offset.
Its relative geometry features and explicit alignment stage explain this
stability: decisions depend on passage-centered quantities rather than scene
texture, and orientation error is corrected before forward commitment.  The
drop to 76.8% for the no-heading-alignment variant at yaw 60 degrees supports
this mechanism rather than a generic robustness claim.

These results remain bounded to mined HM3D validation anchors and the tested
perturbations.  The nominal anchors are mostly well aligned, the clearance
metric is a depth-derived proxy, and no calibrated hardware contact or unseen
HM3D test-split result is implied.  Generalization should therefore be described
as **module sensitivity and scene-based validation under controlled Habitat
perturbations**, not universal narrow-passage robustness.

## Diagnostic Baselines

Diagnostic baselines are not the main method ranking.  They explain metric
failure modes.  The current key diagnostic is Habitat-native TD3:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_diagnostic_baselines.md
```

| Method | Purpose | Nominal success | Strict success | Success-but-unsafe | Near collision |
|---|---|---:|---:|---:|---:|
| TD3 Habitat-native | Nominal metric exploitation diagnostic | 100.0% | 2.0% | 98.0% | 100.0% |

This row shows that nominal Habitat success can be optimized without preserving
the depth-derived body-margin diagnostic.  It should not be presented as a
calibrated clearance result.

## Smoke / Appendix-Only Baselines

Smoke baselines verify that training/evaluation paths run end to end.  They are
not main paper baselines unless rerun under the full protocol.

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_smoke_baselines.md
```

Examples:

- GRU-PPO lightweight.
- RecurrentPPO smoke.
- BC-FSM.
- DAgger-FSM.
- Replay Memory Policy.

The Habitat-Baselines `ppo_narrow_passage.yaml` config is a task/policy
smoke-test config, not the main paper PPO training pipeline.

## Memory Contribution

Memory does not improve one-shot nominal Habitat success.  Its contribution is
to suppress repeated commitments to previously failed infeasible passages.

```bash
python examples/narrow_passage_rl/eval_repeated_failure_memory.py \
    --n-rounds 5 \
    --n-passable 20 \
    --n-ff 15 \
    --max-steps 220
```

Main output:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_repeated_failure_memory.md
```

For transfer/interference beyond repeated identical passages:

```bash
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset paper
```

Outputs:

```text
examples/narrow_passage_rl/results/narrow_passage_rl/memory_transfer_interference.csv
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_transfer_interference.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_memory_transfer_interference.tex
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_degnav_rl_diagnostic.md
examples/narrow_passage_rl/results/narrow_passage_rl/paper_table_belief_mode_ablation.md
```

Paper-preset interpretation: geometry-guided cross-episode failure memory keeps
passable success at 90.0% +/- 2.2%, reaches 100.0% final false-feasible
rejection, transfers rejection to 93.0% +/- 8.6% of similar new false-feasible
passages, and has 0.0% false rejection on similar feasible passages.  Vanilla
episodic memory and kNN failure memory also reject false-feasible passages, but
they over-generalize and falsely reject 100.0% of similar feasible passages.

## DEGNAV-RL Diagnostic Mode-Selection Baseline

DEGNAV-RL tests `pi(m_t | b_t)` over `Commit`, `Explore`, `Recover`, and
`Reject`.  It does not output direct velocities.  The mode-conditioned velocity
controller is shared with DEGNAV-Rule / Geometry-FSM.
DEGNAV-RL is included as a diagnostic policy rather than a competitive final
method. Under the current reward and action interface, the learned policy
collapses to Commit and Explore and does not demonstrate meaningful Recover or
Reject behavior.

Current 3-seed procedural v2 result, using 1M PPO steps per seed:

| Variant | Strict SR | Collision | Near collision | Reject |
|---|---:|---:|---:|---:|
| DEGNAV-RL full belief | 26.7% | 62.3% | 70.9% | 0.0% |
| no p_feas | 25.9% | 61.9% | 72.5% | 0.0% |
| no delta_var | 28.4% | 58.9% | 70.3% | 0.0% |
| no memory | 28.4% | 58.9% | 70.3% | 0.0% |
| no alignment | 25.8% | 61.5% | 72.2% | 0.0% |
| geometry only | 30.9% | 56.0% | 68.1% | 0.0% |

Mode distribution over 1500 procedural v2 evaluation episodes:
Commit: 31.0%, Explore: 69.0%, Recover: 0.0%, Reject: 0.0%.

Interpretation: DEGNAV-RL is a diagnostic learned mode selector.  In the current
setup it still has high collision/near-collision rates and does not learn to use
`Recover` or `Reject`.  It should not be presented as stronger than
DEGNAV-Rule; instead it supports the paper's argument that explicit feasibility,
risk, and failure memory remain necessary near geometric limits.

Smoke run:

```bash
python examples/narrow_passage_rl/train_belief_mode_ppo.py \
    --total-steps 2048 \
    --num-envs 1 \
    --ctypes straight_only \
    --ablation full \
    --eval-episodes 20 \
    --save-dir data/degnav_rl_belief_mode_smoke
```

Evaluation:

```bash
python examples/narrow_passage_rl/eval_belief_mode_ppo.py \
    --model data/degnav_rl_belief_mode_smoke/belief_mode_ppo.zip \
    --episodes 50 \
    --ctypes full \
    --ablation full \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_smoke_eval.csv
```

Supported belief-state ablations are `full`, `no_p_feas`, `no_delta_var`,
`no_memory`, `no_alignment`, and `geometry_only`.

## Reproduction Commands

Run from the repository root:

```bash
# Procedural v2 benchmark
python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500

# Habitat stress validation
python examples/narrow_passage_rl/eval_habitat_stress_validation.py \
    --preset paper \
    --split val \
    --num-episodes -1

# Repeated failure memory
python examples/narrow_passage_rl/eval_repeated_failure_memory.py \
    --n-rounds 5 \
    --n-passable 20 \
    --n-ff 15 \
    --max-steps 220

# Memory transfer/interference
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset paper

# DEGNAV-RL diagnostic mode-selection run
for seed in 0 1 2; do
  python examples/narrow_passage_rl/train_belief_mode_ppo.py \
      --total-steps 1000000 \
      --num-envs 8 \
      --seed ${seed} \
      --ctypes full \
      --ablation full \
      --eval-episodes 500 \
      --device cuda \
      --save-dir data/degnav_rl_belief_mode_full_seed${seed}

  python examples/narrow_passage_rl/eval_belief_mode_ppo.py \
      --model data/degnav_rl_belief_mode_full_seed${seed}/belief_mode_ppo.zip \
      --episodes 500 \
      --seed $((1000 + seed)) \
      --ctypes full \
      --ablation full \
      --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed${seed}_eval.csv
done

# D_min calibration
python examples/narrow_passage_rl/eval_dmin_calibration.py \
    --priors 0.26 0.31 0.36 0.46 0.56 \
    --true-width 0.36 \
    --episodes 300 \
    --seeds 0 1 2 \
    --output-dir examples/narrow_passage_rl/results/narrow_passage_rl

# Rule-vs-DEGNAV width-margin phase analysis
python examples/narrow_passage_rl/eval_harder_benchmark.py \
    --methods rule_baseline geometry_fsm \
    --episodes 500 \
    --seeds 0 1 2 \
    --log-belief-diagnostics \
    --log-outcome-decomposition \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv

python examples/narrow_passage_rl/plot_margin_phase.py \
    --inputs examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv \
    --output-dir examples/narrow_passage_rl/results/narrow_passage_rl/figures \
    --table-dir examples/narrow_passage_rl/results/narrow_passage_rl/tables \
    --methods rule_baseline geometry_fsm \
    --labels "Reactive rule baseline,DEGNAV-Rule" \
    --bin-width 0.05 \
    --margin-min -0.30 \
    --margin-max 0.30 \
    --min-bin-count 5

# Optional infeasible-side margin probe; keep separate from the main figure
python examples/narrow_passage_rl/eval_harder_benchmark.py \
    --methods rule_baseline geometry_fsm feasibility_reject \
    --episodes 500 \
    --seeds 0 1 2 \
    --width-range 0.30 0.55 \
    --log-belief-diagnostics \
    --log-outcome-decomposition \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_infeasible_probe.csv

# Regenerate paper tables from available CSV artifacts
python examples/narrow_passage_rl/make_paper_tables.py
```

The infeasible-side probe is a sensitivity analysis, not the main paired figure.
It increases negative-margin support and tests the diagnostic
`feasibility_reject` variant.  Current results show that this conservative gate
does not solve false-feasible rejection: it introduces false rejects and leaves
most false-feasible blockers as timeout/wasted-attempt cases.  Use repeated
failure-memory experiments for the memory/rejection claim.

Habitat runs require:

```text
data/scene_datasets/hm3d/
data/datasets/narrow_passage/{split}/{split}.json.gz
```

## Video And Figures

Record Habitat videos with overlays:

```bash
python examples/narrow_passage_rl/record_habitat_video.py \
    --method geometry_fsm \
    --split val \
    --episode-index 0 \
    --output-dir video_dir/narrow_passage_habitat \
    --save-keyframes
```

Core documents:

- `examples/narrow_passage_rl/README.md`
- `examples/narrow_passage_rl/docs/method.md`
- `examples/narrow_passage_rl/docs/baseline_taxonomy.md`
- `examples/narrow_passage_rl/docs/experiment_protocol.md`
- `examples/narrow_passage_rl/docs/reproducibility.md`

## Strict Feasibility Ablation v2 Mechanism Validation

The strict ablation has been repaired after auditing the identical v1 results.
The previous estimator fixed `var_delta=0.0025`; its high-uncertainty branch
tested a strict `>` against exactly that value, while `no_uncertainty` still
read variance indirectly through `p_feas`. Those three variants were therefore
mathematically equivalent, not empirically proven equivalent.

The v2 controller uses a scene-dependent sigma estimated from ray dispersion,
valid/dropout ratio, boundary-fit residual, temporal width variation, and
yaw/pose uncertainty. The temporal history resets at the simulator's 10 m
open-space sentinel boundary. `full` uses LCB/UCB gating; `point_estimate` and
`no_uncertainty` use only the mean for decisions; `fixed_uncertainty` uses a
global sigma; and `no_yaw_prior` removes only the yaw-dependent required-width
mean. Every step logs the belief, interval, counterfactual yaw decision, final
four-mode selection, and linear/angular action.

The 45-episode paired mechanism validation passed every launch gate:

| Gate | Result |
|:---|---:|
| Dynamic sigma | 132 distinct values (0.0157–0.0974 m) |
| Uncertainty branch | 33/180 full steps (18.33%) |
| Full vs point final-mode disagreement | 33/180 (18.33%) |
| Yaw prior changed decision | 106/180 (58.89%) |
| Paired non-ablated inputs/random draws | passed |

Current artifacts and complete audit:

```text
examples/narrow_passage_rl/results/ablation_feasibility/mechanism_20260818_170356/
examples/narrow_passage_rl/docs/feasibility_ablation.md
```

The new 3-seed, five-variant evaluation was run only after these checks passed.
It completed with 10,500 episode rows and 528,873 step rows in
`paper_dynamic_20260818_170356/`:

| Method | Success | Collision | False reject |
|:---|---:|---:|---:|
| full | 0.2 ± 0.2% | 11.2 ± 1.4% | 63.5 ± 4.3% |
| point_estimate | 0.3 ± 0.4% | 10.8 ± 1.4% | 64.4 ± 4.1% |
| no_uncertainty | 0.3 ± 0.4% | 10.8 ± 1.4% | 64.4 ± 4.1% |
| fixed_uncertainty | 0.2 ± 0.2% | 11.1 ± 1.4% | 63.6 ± 4.4% |
| no_yaw_prior | 0.2 ± 0.2% | 11.6 ± 1.4% | 62.5 ± 3.9% |

Mechanisms now differ even though success is saturated at only 4–6 successes
per method: formal full uncertainty gating triggers on 2.32% of steps, full vs
point changes 30 episode outcomes, and full vs no-yaw changes 29. The result is
auditable but not a performance gain; no thresholds were retuned.

## Strict Feasibility Ablation v1 Result (Archived Diagnostic)

The paired four-way procedural ablation is complete at 3 seeds × 7 scene types
× 100 episodes per scene × 4 methods (8,400 rows total):

| Method | Success | Collision | Correct reject | False reject | Timeout/stuck |
|:---|---:|---:|---:|---:|---:|
| DEGNav full | 17.8 ± 1.9% | 15.1 ± 3.1% | 11.6 ± 1.4% | 52.2 ± 3.5% | 3.3 ± 1.2% |
| DEGNav point estimate | 17.8 ± 1.9% | 15.1 ± 3.1% | 11.6 ± 1.4% | 52.2 ± 3.5% | 3.3 ± 1.2% |
| DEGNav w/o uncertainty-aware gating | 17.8 ± 1.9% | 15.1 ± 3.1% | 11.6 ± 1.4% | 52.2 ± 3.5% | 3.3 ± 1.2% |
| DEGNav w/o yaw prior | 17.8 ± 1.9% | 15.3 ± 3.0% | 11.6 ± 1.4% | 51.9 ± 3.7% | 3.4 ± 1.1% |

`full`, `point_estimate`, and `no_uncertainty` are identical episode by episode.
The current estimator fixes `var_delta` at 0.0025, so probability gating is a
monotone reparameterization of mean margin and the explicit high-uncertainty
rule never fires. The paired `full - no_yaw_prior` success difference is -0.05
percentage points with 95% CI `[-0.25, 0.16]`. Thus the current data support
neither a probabilistic/uncertainty advantage nor a success benefit from the
existing secant yaw prior. This is reported as a paper-code limitation rather
than being hidden or retuned.

Full audit, commands, calibration, plots, and exact artifact paths:

```text
examples/narrow_passage_rl/docs/feasibility_ablation.md
examples/narrow_passage_rl/results/ablation_feasibility/paper_20260818/
```

## Recurrent 19-D four-mode selector

An auditable GRU actor-critic, episode-grouped BC collector/trainer, recurrent
PPO runner, fixed-set evaluator, and automatic metric table are now available.
The staged experiment was intentionally stopped after the 100k gate: pure
BC+PPO produced 0/78 correct deterministic rejections, while a training-only
auxiliary diagnostic produced 60/78 correct rejections but 38/122 false
rejections. These are procedural geometry results, not Habitat contact safety
or Lite3 validation.

```text
examples/narrow_passage_rl/docs/mode_selector_ppo_audit.md
examples/narrow_passage_rl/results/mode_selector/EXPERIMENT_REPORT.md
```
