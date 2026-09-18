# Geometry-Guided Failure-Aware Narrow-Passage Navigation

This directory contains the narrow-passage research code added on top of the
Habitat-Lab fork.  It is the main entry point for reproducing the paper's
experiments and tables.

Naming convention:

- `Geometry-FSM` is the implementation name used in scripts and CSV files.
- `DEGNAV-Rule` is the paper-facing name for `Geometry-FSM`.
- `DEGNAV-RL` is a diagnostic policy that tests whether PPO can learn only the
  high-level mode selector `pi(m_t | b_t)` over `Commit`, `Explore`, `Recover`,
  and `Reject`.
- PPO/SAC/TD3 are learning-only direct-control baselines that map geometry
  observations directly to velocity actions.

Geometry-FSM is referred to as DEGNAV-Rule in the paper.

## Latest Formal Habitat / MP3D Queue (2026-09-05)

All requested official baselines have completed the same 495-episode MP3D v1
validation protocol. PointNav PPO is `78.52±0.34%` Success / `0.64±0.01` SPL;
DD-PPO is `93.87±0.42%` / `0.85±0.00` and must be labeled as a released
Gibson-2+ Depth checkpoint evaluated on MP3D. ForwardOnly, Random Agent,
RandomForward, GoalFollower, and ShortestPathFollower are also complete.

The separate MP3D-derived 80-episode narrow-passage protocol reports
Geometry-19D, kNN-memory-19D, and DEGNAV-memory-19D at identical `82.50%`
all-episode Success, `98.44%` feasible Success, and `0%` Correct Reject. The
corrected-history raw-Depth DEGNAV-E2E three-seed run is complete at
`83.33±0.59%` all-episode Success, `99.48±0.74%` feasible Success,
`95.42±1.18%` decision accuracy, and `79.17±2.95%` Correct Reject, with
`2.08±0.74%` False Reject and zero feasible collisions. It is trained by
offline BC and outputs modes to a fixed controller; earlier invalid E2E
artifacts are retained only under `diagnostics/`.

Use these as the current entry points:

- [final method status](results/formal_mp3d_summary_20260905/final_method_status.md)
- [complete formal work report](results/formal_mp3d_summary_20260905/formal_experiment_work_report.md)
- [environment and artifact hashes](results/formal_mp3d_summary_20260905/environment_snapshot.md)
- [official PointNav table](results/official_pointnav_mp3d_v1_20260904/paper_table_official_pointnav.md)
- [derived 19-D table](results/degnav_e2e_mp3d_narrow_v1_20260904/19d_comparison/formal_mp3d_comparison.md)
- [derived E2E table](results/degnav_e2e_mp3d_narrow_v1_20260904/paper_table_degnav_e2e.md)

Official PointNav and derived narrow-passage values must not be placed in one
ranking: their agent morphology, action space, sliding behavior, Success
definition, and episode set differ. New paper-facing tables omit strict
Success.

## Latest Paired Local-Baseline Results (2026-08-24)

The latest unified Procedural-v2 comparison evaluates five method rows on the
same ordered held-out scenarios: evaluation seeds `42/43/44`, 500 episodes per
method and seed, seven corridor types, passage width `[0.45, 0.90] m`, and a
400-step budget.  This is a 19-D analytical local-navigation benchmark, not an
end-to-end Habitat RGB evaluation.

| Method | Success ↑ | Collision ↓ | Near collision ↓ | Timeout/stuck ↓ | Avg steps ↓ |
|:---|---:|---:|---:|---:|---:|
| Geometry rule | 23.3±1.9% | 70.3±2.2% | 38.6±1.6% | 6.3±0.5% | 60.0±2.6 |
| Direct-control PPO | 84.3±0.5% | 15.7±0.5% | 14.9±0.4% | 0.0±0.0% | 68.5±0.1 |
| Recurrent PPO (from scratch) | 0.1±0.1% | 20.3±0.3% | 19.1±0.2% | 79.7±0.4% | 323.8±1.1 |
| Recurrent PPO (Direct-init + PPO fine-tune) | **85.0±0.9%** | **15.0±0.9%** | **14.1±0.7%** | **0.0±0.0%** | **68.3±0.2** |
| DEGNAV + geometry-guided memory | 83.1±0.2% | 0.0±0.0% | 45.1±2.2% | 16.9±0.2% | 188.5±1.7 |

The from-scratch recurrent model fails mainly through low-progress policies and
timeout.  Its default actor/critic LSTMs contain 608,709 policy parameters,
about 55 times the 11,077 parameters of Direct PPO, while the current
observation is already approximately Markov.  The repaired recurrent policy
uses a 64-unit actor LSTM, a feed-forward critic, Direct-PPO behavior
initialization on 300 evaluation-isolated training episodes, and 106,496
low-learning-rate RecurrentPPO fine-tuning interactions.  It exceeds the 50%
success gate, but it is not an independent from-scratch baseline and does not
prove that recurrence itself outperforms Direct PPO.

DEGNAV has zero reported simulator collisions in this run, but its 45.1%
near-collision rate and 16.9% timeout/stuck rate prevent interpreting that
number as complete safety.  Its current unified-evaluation path also records
zero correct rejects, so this result does not yet validate terminal rejection
on false-feasible passages.

Canonical artifacts:

- [complete work report](docs/unified_local_baselines_work_report_20260826.md)
- [paper-facing comparison table](results/narrow_passage_rl/unified_local_baselines_recurrent_updated_20260824/paper_table_unified_local_baselines.md)
- [aggregate CSV](results/narrow_passage_rl/unified_local_baselines_recurrent_updated_20260824/summary.csv)
- [per-seed CSV](results/narrow_passage_rl/unified_local_baselines_recurrent_updated_20260824/summary_by_eval_seed.csv)
- [7,500 paired episode rows](results/narrow_passage_rl/unified_local_baselines_recurrent_updated_20260824/episodes.csv)

## Latest Strict Feasibility Audit (2026-08-18)

The current comparable result uses a shared `0.36 x 0.60 m` oriented-rectangle
collision/label/controller morphology, separate structural and pose-conditioned
margins, and five non-duplicate ablations. The validated three-seed result is:

```text
results/ablation_feasibility/structural_full_20260818_194833_final/
```

The main negative finding is structural: full dynamic uncertainty reaches only
`0.4%` Success on the feasible subset because cross-episode failure memory
causes `96.7%` False Reject. Disabling memory raises feasible Success to `12.2%`
and removes False Reject, but increases feasible Collision to `21.5%` and
Timeout to `66.3%`. These results do not support a claim that the current full
method is superior; they isolate failure-memory overgeneralization as the next
mechanism to redesign. See [docs/feasibility_ablation.md](docs/feasibility_ablation.md)
for formulas, gates, grouped metrics, validity labels, and reproduction commands.

The older `paper_dynamic_20260818_170356` circular-body result remains preserved
for historical diagnosis and is not geometrically comparable to the OBB run.

## Recursive Belief Audit (2026-08-20)

`DynamicFeasibilityEstimator` now exposes an explicit yaw propagation operator
and observation-fusion operator while preserving every selector-facing legacy
field and action exactly. A recursive posterior variance/concentration is logged
in parallel with the existing engineering sigma. The 45-episode validation,
kappa sweep, failure-mode table, and cross-morphology protocol are documented in
[docs/belief_filter_ablation.md](docs/belief_filter_ablation.md). The strongest
failure label remains cross-episode-memory false rejection; no threshold was
tuned and no existing result was overwritten.

The three-seed morphology run covers the preserved 0.36×0.60 m base plus new
0.28×0.55, 0.45×0.65, and 0.36×0.80 m bodies. Full-minus-no-memory feasible
False Reject is +96.30 to +97.41 percentage points with all paired 95% CIs
above zero, while full-minus-no-yaw feasible Success is exactly 0.00 points for
all four bodies. This generalizes the memory failure and the null yaw-success
effect; it does not establish a performance advantage for full DEGNav.

### Complete low-Success diagnosis

The result is now accompanied by an evidence-ranked causal diagnosis at
`results/ablation_feasibility/success_rate_diagnosis_20260820_150459/`.
The low Success has three stacked layers: 360/630 mixed-benchmark episodes are
ground-truth infeasible; memory then false-rejects 261/270 feasible episodes;
after removing memory, an open-space uncertainty → zero-speed Explore → stuck
→ reverse Recover loop produces 66.30% feasible Timeout, while incomplete
scene-specific control produces 21.48% Collision. All 33 no-memory successes
come from `narrow_exit`; straight, L-shaped, S-shaped, and narrow-entry have no
success, and every asymmetric case collides. See
[the complete diagnosis](results/ablation_feasibility/success_rate_diagnosis_20260820_150459/success_rate_diagnosis.md)
for quantified causes, secondary factors, and ruled-out explanations.

## Structural repair status (2026-08-20, not paper-final)

The memory/outcome, ray-state, active sensing, bounded recovery, entry waypoint,
and swept-OBB action paths have now been repaired and audited without changing
`kappa=1.645`, `tau_commit=tau_reject=0.02`, the `0.36 x 0.60 m` OBB, or the
200-step budget. The latest 42-scenario smoke result is preserved at:

```text
results/ablation_feasibility/repaired_stage3_smoke_20260820_164046/
```

On its 18 OBB-labelled feasible episodes, Success is `9/18 = 50.00%`, False
Reject is `0%`, Collision is `0%`, and Timeout is `50.00%`. Narrow-entry and
narrow-exit are both `3/3`, straight is `2/3`, asymmetric is `1/3`, while
L-shaped and S-shaped are both `0/3`. This is a real improvement over the
archived formal run (`0.37%` feasible Success and `96.67%` False Reject), but it
does **not** pass the `Success >= 60%` / `Timeout < 20%` engineering gates.
Consequently no phase-4 uncertainty calibration or three-seed formal run was
started.

The complete current result, mutually exclusive episode labels, all residual
low-Success causes, implementation-validity caveats, gate decision, and exact
reproduction commands are in
[repaired_success_diagnosis.md](results/ablation_feasibility/repaired_diagnosis_20260820_164118/repaired_success_diagnosis.md).
The main residual causes are L/S corner-control failure, OBB safety-projection
saturation, incomplete asymmetric clearance control, sparse/discontinuous ray
observations, and the fact that junction passability/collision is not yet a
full polygon-union OBB reachability proof. Active Explore also ends with zero
translation on `28.69%` of feasible Explore steps, and width observability is
true on `100%` of feasible steps, so both stage-2 mechanisms remain suspect.
The versioned strict `gate_report.json` records
`full_evaluation_permitted=false` and prevents the formal evaluation from being
started with this smoke.

## What This Directory Adds

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
  train_sb3_v2.py                       # PPO/SAC/TD3 direct-control training
  train_recurrent_ppo_v2.py             # RecurrentPPO smoke baseline
  train_bc_dagger_v2.py                 # BC/DAgger smoke baselines
  train_replay_memory_policy_v2.py      # Generic replay-memory smoke baseline
  vln_dataset_free_smoke.py             # NaVILA artifact/runtime smoke test
  vln_batch_diagnostic.py               # Large-batch controlled VLN diagnostics
  eval_vln_safety_adapter.py            # Paired NaVILA + DEGNAV adapter smoke test
  analyze_belief_confidence.py          # Engineering-vs-posterior confidence audit
  sweep_kappa_tau.py                    # Fixed-tau paired kappa smoke sweep
  classify_feasibility_failures.py      # Mutually exclusive failure-mode table
  eval_morphology_generalization.py     # Three-seed cross-morphology protocol
  diagnose_low_success.py               # Evidence-ranked low-Success diagnosis
  record_habitat_video.py               # Habitat video/keyframe generation
  narrow_passage/models/vln_safety_adapter.py # Trusted-directive and geometry gate
  results/narrow_passage_rl/            # CSV/Markdown/LaTeX/figures
```

The Habitat task and sensors used by these scripts live in:

```text
habitat-lab/habitat/tasks/narrow_passage/
```

## Main Claim

The paper focuses on local narrow-passage navigation near geometric feasibility
limits.  The main contribution is not a generic RL navigation policy.  The claim
is that explicit passage geometry, clearance risk, decision mode, and failure
history are more stable and interpretable than learning-only or generic-history
baselines in this regime.

The main method is DEGNAV-Rule / Geometry-FSM.  DEGNAV-RL is retained as a
diagnostic appendix experiment.

## VLN/VLA Integration: Current Evidence And Improvement Plan

### Intended Role In The System

The downloaded NaVILA checkpoint is not treated as a low-level quadruped
controller.  The evidence below shows that directly executing its textual
action output would be unsafe and weakly grounded.  The intended interface is:

```text
Language instruction + RGB history
                |
                v
NaVILA semantic proposal / waypoint / completion proposal
                |
                v
Trusted runtime directive + depth geometry + robot state + failure memory
                |
                v
VLN safety adapter -> Commit / Explore / Recover / Reject
                |
                v
Mode-conditioned local controller -> velocity command
                |
                v
Quadruped locomotion policy using IMU, joint state, and contact
```

This separation gives each module a testable responsibility:

| Layer | Primary inputs | Output | Explicitly not responsible for |
|---|---|---|---|
| NaVILA / VLN | Route instruction, RGB history | Semantic goal, waypoint, action proposal, completion proposal | Contact safety and direct joint control |
| Geometry belief | Depth, passage width, robot envelope, pose | `p_feas`, margin uncertainty, heading/lateral error | Language grounding |
| Failure-aware decision | VLN proposal, geometry belief, failure memory | Commit, Explore, Recover, Reject | Learned locomotion dynamics |
| Local controller | Mode, local goal, clearance | Bounded `(v_x, omega_z)` command | Long-horizon semantic planning |
| Quadruped controller | Velocity, IMU, joints, contacts | Stable body motion | Route-level language reasoning |

Generic route text is never parsed as an immediate command.  Only an explicitly
trusted runtime-control channel may issue a terminal stop or immediate local
turn.  This prevents a future phrase such as "turn left after the sofa" from
being executed at the current location.

### Completed VLN Tests

模型结构、精确参数量、4-bit 推理资源、真实机器人 ROS2/Lite3 分层部署，以及
相机外参与机器人初始位姿变化的训练方案，统一整理在
[NaVILA 模型与实机部署报告](docs/navila_model_real_robot_deployment.md)。

The current checkpoint is:

```text
/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/models/navila-llama3-8b-8f
```

It is loaded with bitsandbytes NF4 4-bit quantization on an NVIDIA RTX A4000.
The NaVILA source commit used by the diagnostics is
`76b98f233dd0fff05dfcd69435eec6740febff9d`.

#### Test A: Model Artifact And Runtime Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/narrow_passage_rl/vln_dataset_free_smoke.py \
  --output-dir examples/narrow_passage_rl/results/vln_dataset_free
```

Results:

| Metric | Result |
|---|---:|
| Model artifact validation | PASS |
| 4-bit model load | PASS |
| Completed inference cases | 7/7 |
| Parseable navigation action | 100.0% |
| Deterministic repeated-output match | 100.0% |
| Mean inference latency | 1.409 s |
| P95 inference latency | 1.696 s |
| Peak GPU memory | 8003.1 MiB |

This test proves that the checkpoint is complete and the language/video/action
interface runs locally.  It does not measure navigation success.

#### Test B: 640-Case Controlled Batch Diagnostic

Command:

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/narrow_passage_rl/vln_batch_diagnostic.py \
  --r2r-samples 256 \
  --output-dir examples/narrow_passage_rl/results/vln_batch_diagnostic
```

The batch contains 64 controlled local-command cases, 64 repository-demo visual
perturbation cases, and 256 R2R val-unseen instructions paired with an original
and explicit-stop-override probe, for 640 inferences total.

| Diagnostic | Cases | Result |
|---|---:|---:|
| Parseable action output | 640 | 100.0% |
| Mean / P95 latency | 640 | 1.466 / 1.485 s |
| Original R2R text probe: move-forward output | 256 | 71.5% |
| Original R2R text probe: stop output | 256 | 1.2% |
| Explicit R2R stop-override compliance | 256 | 1.6% |
| Action changed after explicit stop override | 256 | 12.9% |
| Controlled move-forward command match | 16 | 100.0% |
| Controlled turn-left command match | 16 | 0.0% |
| Controlled turn-right command match | 16 | 0.0% |
| Controlled stop command match | 16 | 0.0% |
| Mild visual-perturbation action consistency | 40 | 77.5% |
| Frame-dropout action consistency | 8 | 75.0% |
| Temporal-reversal action consistency | 8 | 62.5% |
| Center-occlusion exact-response consistency | 8 | 12.5% |

The overall output distribution is 458 move-forward, 144 turn-right, 31
turn-left, and 7 stop actions.  The strongest measured problems are therefore:

1. **Stop failure:** explicit task-completion updates are almost always ignored.
2. **Action-prior collapse:** forward motion dominates output under unmatched
   R2R text probes.
3. **Weak direction grounding:** controlled left/right instructions do not
   override the visual action prior.
4. **Missing obstacle veto:** a forward instruction can dominate a blocked
   synthetic view.
5. **Appearance sensitivity:** lighting, blur, noise, and occlusion flip about
   22.5% of actions in the small controlled perturbation suite.
6. **Temporal sensitivity:** history reversal flips 37.5% of actions.  This is
   diagnostic rather than automatically erroneous because reversal changes the
   motion context.

The R2R instruction probes use real val-unseen instruction text but unmatched
NaVILA repository-demo frames because MP3D scene assets are not installed.
They measure action priors and trusted stop response only.  They are not R2R
episodes and cannot produce SR, SPL, NE, or nDTW.

#### Test C: R2R Text-Only Corpus Diagnostic

Command:

```bash
python examples/narrow_passage_rl/vln_r2r_text_analysis.py
```

This test uses R2R/VLN-CE annotation text only, so it does not require MP3D
scene assets.  It analyzes train, val-seen, and val-unseen instructions and
joins the existing NaVILA text-probe predictions from Test B when available.

Output:

```text
examples/narrow_passage_rl/results/vln_r2r_text_analysis/report.md
examples/narrow_passage_rl/results/vln_r2r_text_analysis/figures/
```

Key corpus statistics:

| Split | Episodes | Unique scenes | Mean words | Median words |
|---|---:|---:|---:|---:|
| train | 10819 | 61 | 26.67 | 25 |
| val_seen | 778 | 53 | 27.03 | 25 |
| val_unseen | 1839 | 11 | 26.65 | 25 |

Val-unseen language cue rates:

| Cue | Episodes containing cue |
|---|---:|
| forward / motion | 98.4% |
| left | 48.1% |
| right | 52.1% |
| stop / goal relation | 81.4% |
| room / place | 92.0% |
| object / landmark | 45.9% |
| spatial relation | 67.9% |

Joined with the 256 saved NaVILA R2R text probes, the model still outputs
`move_forward` in 71.5% of cases and `stop` in only 1.2% of cases.  This
supports the interface argument: real R2R instructions contain rich spatial
and goal cues, but the current direct local-action prompt collapses toward a
forward-motion prior under unmatched visuals.

Figures:

```text
figures/r2r_instruction_length.png
figures/r2r_val_unseen_language_cues.png
figures/r2r_top_tokens.png
figures/navila_action_prior_by_language_cue.png
```

This analysis is paper-safe as a dataset/interface diagnostic only.  It is not
a navigation rollout and must not be described as standard R2R performance.

#### Test D: Open-Nav Installation Preflight

Open-Nav is installed as a planned zero-shot open-source LLM VLN baseline:

```text
/home/xiaotian/vla/Open-Nav
```

Preflight command:

```bash
PYTHONPATH=/home/xiaotian/vla/Open-Nav:/home/xiaotian/vla/habitat-lab-v0.1.7 \
conda run -n navila-eval \
python examples/narrow_passage_rl/opennav_preflight.py
```

Current status:

| Item | Status |
|---|---:|
| Open-Nav repository | cloned, commit `3a8dcef` |
| OpenNav_R2R-CE_100 quick dataset | present, 100 episodes |
| Habitat 0.1.7 imports | pass |
| DDPPO depth encoder | pass |
| RAM source | cloned and installed |
| MP3D scene coverage | 0/10 |
| waypoint predictor checkpoint | missing / partial download only |
| RAM checkpoint | missing |
| SpatialBot3B | gated HF asset, missing |
| LLM backend / API key | missing |

Report:

```text
examples/narrow_passage_rl/results/opennav_preflight/opennav_preflight.md
examples/narrow_passage_rl/docs/opennav_install_status.md
```

Open-Nav is therefore not yet a formal result row.  It is a prepared baseline
whose simulator evaluation can start after the listed assets and API/local LLM
backend are available.

#### Test E: Paired NaVILA + DEGNAV Safety-Adapter Smoke

The implemented adapter is:

```text
narrow_passage/models/vln_safety_adapter.py
```

It applies the following precedence without changing NaVILA weights:

1. A trusted terminal-stop directive stops immediately.
2. Collision or high stuck score requests Recover.
3. High failure-memory risk, low `p_feas`, low measured clearance, or low body
   margin requests Reject.
4. Uncertain but non-rejected geometry requests Explore and caps forward motion.
5. Safe proposals request Commit.
6. Untrusted route instructions pass through unchanged.

Reproduction:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q \
  examples/narrow_passage_rl/tests/test_vln_safety_adapter.py

python examples/narrow_passage_rl/eval_vln_safety_adapter.py \
  --predictions \
    examples/narrow_passage_rl/results/vln_batch_diagnostic/predictions.csv \
  --output-dir \
    examples/narrow_passage_rl/results/vln_safety_adapter_smoke
```

Focused unit tests: **6/6 passed**.  The paired comparison reuses all 640 saved
NaVILA outputs:

| Metric | Direction | Cases | NaVILA direct | + safety adapter |
|---|:---:|---:|---:|---:|
| Controlled command exact match | higher | 64 | 25.0% | 81.3% |
| Non-blocked command exact match | higher | 48 | 25.0% | 100.0% |
| Explicit stop compliance | higher | 272 | 1.5% | 100.0% |
| R2R stop-override compliance | higher | 256 | 1.6% | 100.0% |
| Blocked-fixture forward output | lower | 4 | 100.0% | 0.0% |
| Original R2R proposal preservation | higher | 256 | 100.0% | 100.0% |
| Visual-perturbation action consistency | higher | 56 | 75.0% | 75.0% |

The aggregate command score is 81.3% rather than 100% because safety has
priority: immediate move/turn commands are intentionally rejected in the
controlled blocked fixture.  Per-action adapted command matches are 75.0% for
move-forward, left, and right, and 100.0% for stop.

The blocked-fixture result has only four forward-command cases and uses
synthetic clearance inputs.  It is an interface smoke test, not an obstacle
benchmark.  Likewise, 100% trusted-stop compliance is deterministic control
logic rather than evidence that NaVILA learned stopping.

The adapter preserves the original model proposal on all 256 untrusted R2R
route instructions.  Visual robustness remains 75.0%, correctly showing that a
post-policy safety interface cannot repair the visual representation.

The six deterministic mode-path probes cover safe Commit, uncertain Explore,
stuck Recover, collision Recover, memory-risk Reject, and low-clearance Reject.
They verify code paths only.  They do not establish learned recovery, learned
rejection, or physical safety.

### Recommended Technical Improvements

#### Priority 0: Preserve The Hierarchical Safety Boundary

Status: **implemented as a smoke-tested adapter**.

- Keep NaVILA as a semantic proposal model.
- Keep trusted operator/runtime directives separate from route instructions.
- Give measured collision, stuck, clearance, body margin, and memory risk
  priority over a forward proposal.
- Clamp forward distance and turn angle before execution.
- Log the original proposal, intervention reason, selected mode, and final
  command for every step.
- Never present deterministic adapter gains as learned VLN gains.

Acceptance gate:

- 100% action-format validation.
- 100% trusted emergency-stop handling.
- No change to untrusted route proposals without measured risk evidence.
- Unit coverage for Commit, Explore, Recover, and Reject paths.

#### Priority 1: Install The Paired VLN Benchmark Data

Status: **blocked by missing MP3D scenes**.

The R2R JSON files are present, but formal VLN evaluation requires the matching
MP3D scene assets.  Once available, run NaVILA on `val_seen` and `val_unseen`
with paired simulator observations.

Current preflight status:

| Item | Status |
|---|---:|
| NaVILA checkpoint | pass |
| R2R `val_unseen` annotations | pass, 1839 episodes |
| R2R `val_unseen` unique MP3D scenes | 11 |
| DDPPO depth encoder | pass |
| NaVILA evaluation data symlinks | pass |
| Habitat 0.1.7 Python stack | pass |
| MP3D `.glb` files for `val_unseen` | 0/11 |
| MP3D `.navmesh` files for `val_unseen` | 0/11 |

Preflight command:

```bash
conda run -n navila-eval python examples/narrow_passage_rl/vln_r2r_preflight.py \
  --ensure-symlinks
```

Preflight report:

```text
examples/narrow_passage_rl/results/vln_r2r_eval_preflight/r2r_vlnce_preflight.md
examples/narrow_passage_rl/results/vln_r2r_eval_preflight/r2r_vlnce_preflight.json
```

Expected MP3D scene layout after installing licensed Matterport3D Habitat
assets:

```text
/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data/scene_datasets/mp3d/{scene}/{scene}.glb
/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data/scene_datasets/mp3d/{scene}/{scene}.navmesh
```

VLN-CE expects the official Matterport3D `download_mp.py` script obtained
through the Matterport3D project terms.  After obtaining that script, place or
run it so the output root is:

```bash
cd /media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data
python2.7 download_mp.py --task habitat -o scene_datasets/mp3d/
```

The current `val_unseen` split needs these scene IDs:

```text
2azQ1b91cZZ, 8194nk5LbLH, EU6Fwq7SyZv, QUCTc6BB5sX, TbHJrupSAjP,
X7HyMhZNoso, Z6MFQCViBuw, oLBMNvg9in8, pLe4wQe7qrG, x8F5xyUWy9e,
zsNo4HB9uLZ
```

Run the standard NaVILA paired R2R/VLN-CE evaluation only after the preflight
reports `PASS`:

```bash
cd /home/xiaotian/vla/NaVILA/evaluation
bash scripts/eval/r2r.sh \
  /media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/models/navila-llama3-8b-8f \
  1 0 "0"

python scripts/eval_jsons.py \
  ./eval_out/navila-llama3-8b-8f/VLN-CE-v1/val_unseen \
  1
```

Required formal metrics:

- Success Rate and SPL.
- Navigation Error, Oracle Success, nDTW, and SDTW.
- Stop precision/recall and premature-stop rate.
- Collision, timeout, oscillation, and stuck rate.
- Intervention rate and progress lost to safety interventions.
- Per-instruction-length and per-scene breakdown.

Required comparison:

| Variant | Semantic model | Geometry gate | Failure memory | Local recovery |
|---|---|---|---|---|
| NaVILA direct | yes | no | no | no |
| NaVILA + trusted directive | yes | no | no | no |
| NaVILA + geometry belief | yes | yes | no | no |
| NaVILA + geometry + recovery | yes | yes | no | yes |
| Full hierarchical system | yes | yes | yes | yes |

All variants must use the same episode IDs, sensor configuration, maximum
steps, stop definition, and random seeds.

#### Priority 2: Change The VLN Output From Velocity-Like Actions To Subgoals

Status: **planned**.

The main model output should be a local semantic waypoint rather than an
unfiltered distance/turn command:

```text
z_semantic, waypoint_mean, waypoint_uncertainty, p_complete
```

The waypoint should be transformed into the robot frame and passed to the
geometry controller.  NaVILA should not decide whether the quadruped body fits
through a passage.  That decision belongs to the geometry belief:

```text
b_t = (
  p_feas,
  delta_mean,
  delta_var,
  heading_error,
  lateral_error,
  stuck_score,
  collision_flag,
  memory_risk
)
```

Recommended interface:

- `semantic_goal`: room/object/landmark embedding.
- `waypoint_xy`: short-horizon target in the robot frame.
- `waypoint_covariance`: uncertainty used to trigger Explore.
- `p_complete`: completion probability, calibrated separately from route text.
- `proposal_timestamp`: rejects stale language actions after recovery.

#### Priority 3: Fuse RGB Semantics With Depth Geometry

Status: **geometry branch exists; VLN fusion remains planned**.

Use the existing 19-D narrow-passage geometry feature as the interpretable
branch and retain NaVILA visual tokens as the semantic branch:

```text
z_vln = NaVILA(RGB history, instruction)
z_geo = GeometryEncoder(19-D geometry)
z_fused = MLP([z_vln, z_geo, waypoint, robot state])
```

The geometry branch should include passage width, yaw-aware required width,
left/right clearance, body margin, heading/lateral error, velocity history,
stuck score, and collision/contact evidence.  Report an ablation with the
geometry gate disabled; do not infer its contribution from a mixed-domain
table.

Recommended safety calibration:

- Calibrate depth-derived width against measured doorway widths.
- Report reliability bins for `p_feas`.
- Separate depth-proxy near-collision from physical contact.
- Evaluate missing depth sectors and conservative uncertainty inflation.
- Reject stale or non-finite geometry rather than converting it to zero risk.

#### Priority 4: Repair Stop And Direction Grounding

Status: **problem measured; training not yet performed**.

Construct action-balanced, paired local clips:

- Equal numbers of forward, left, right, and stop targets.
- Same visual clip with counterfactual left/right instructions.
- Same route state before and after goal completion.
- Hard negatives containing future words such as "turn left after the sofa".
- Explicit distinction between route instruction and trusted runtime update.
- Blocked/open visual pairs with identical forward language.

Recommended objectives:

```text
L = lambda_wp   * L_waypoint
  + lambda_act  * CE(action)
  + lambda_stop * BCE(p_complete)
  + lambda_risk * BCE(p_collision_or_infeasible)
  + lambda_cons * KL(action_clean || action_perturbed)
  + lambda_cal  * Brier(p_feas, outcome)
```

Start with LoRA/adapters while freezing the vision tower and most of the
language model.  First train the waypoint/completion heads, then unfreeze a
small number of multimodal projector layers only if val-unseen metrics justify
it.  Longer training alone is not an adequate response to action collapse.

Required reporting:

- Per-action confusion matrix.
- Stop precision, recall, and calibration.
- Left/right counterfactual consistency.
- Action entropy and dominant-action share.
- Seen/unseen split and three or more seeds.

#### Priority 5: Improve Visual And Temporal Robustness

Status: **problem measured; no learned improvement yet**.

The current small suite shows 75.0% action consistency over non-clean visual
perturbations and 62.5% under temporal reversal.

Training augmentations:

- Exposure and white-balance variation.
- Gaussian/shot noise and motion blur.
- Random center/peripheral occlusion.
- Depth dropout and invalid-sector masks.
- Frame dropout, duplicated frames, and variable history length.
- Camera latency and RGB/depth timestamp offsets.

Evaluation must retain clean performance and report each perturbation
separately.  Temporal reversal should remain a diagnostic because it can change
the correct action; matched next-action labels are needed before calling every
flip an error.

#### Priority 6: Connect Geometry-Guided Failure Memory

Status: **memory is validated separately in synthetic recurrence protocols;
not yet connected to NaVILA**.

Store only evidence that can be audited:

```text
{
  geometry_embedding,
  scene_or_place_id,
  semantic_goal,
  passage_width,
  width_body_ratio,
  entry_heading,
  action_mode,
  outcome,
  min_clearance,
  stuck_and_oscillation_count,
  recovery_result,
  confidence
}
```

Retrieve by geometry similarity plus semantic/place compatibility.  Feed the
weighted failure rate into `memory_risk`; do not concatenate an opaque replay
vector and call it failure-aware memory.

Required tests:

- Repeated identical false-feasible passage.
- Similar but new false-feasible passage.
- Similar but feasible interference case.
- Memory retrieval precision.
- Wasted attempts and steps saved.
- False-reject rate on passable passages.

Memory should be claimed to reduce repeated infeasible commitment, not to
increase one-shot nominal Habitat success.

#### Priority 7: Couple To Quadruped Locomotion

Status: **planned integration**.

The local mode controller should output bounded velocity references to the
Isaac Lab locomotion policy:

- Commit: normal tracking with clearance filter.
- Explore: reduced speed and increased sensing/alignment time.
- Recover: backward/rotation primitive with contact-aware stabilization.
- Reject: zero velocity and request global replanning.

Train locomotion with observation latency, IMU noise, contact uncertainty,
friction/mass randomization, actuator delay, and camera-body extrinsic error.
The real Lite3 controller must retain an independent emergency stop and velocity
limit.  Simulated adapter results must not be reported as real-robot safety.

### Evidence Status And Next Decision

| Evidence item | Status | Paper role |
|---|---|---|
| NaVILA checkpoint/runtime | completed | implementation readiness |
| 640-case controlled diagnostic | completed | failure analysis / appendix |
| Deterministic safety adapter | completed smoke test | system-design evidence |
| Standard R2R paired simulator evaluation | not run | required VLN baseline |
| Learned waypoint/completion head | not implemented | future method |
| RGB-depth semantic/geometry fusion | not trained | future method |
| Failure memory connected to VLN | not implemented | future method |
| Isaac Lab locomotion coupling | not evaluated here | separate experiment |
| Lite3 closed-loop VLN experiment | not evaluated here | future hardware test |

The immediate research conclusion is not that the adapter solves VLN.  It is
that the present checkpoint has a reliable runtime interface but exhibits stop,
direction, action-prior, and perturbation failures.  A hierarchical
NaVILA-to-DEGNAV interface removes deterministic control-channel hazards while
preserving semantic proposals, and it defines the correct boundary for future
paired-data training and evaluation.

VLN output artifacts:

```text
results/vln_dataset_free/report.md
results/vln_dataset_free/predictions.csv
results/vln_batch_diagnostic/report.md
results/vln_batch_diagnostic/predictions.csv
results/vln_batch_diagnostic/issues_and_improvements.csv
results/vln_batch_diagnostic/figures/
results/vln_safety_adapter_smoke/report.md
results/vln_safety_adapter_smoke/adapted_predictions.csv
results/vln_safety_adapter_smoke/comparison_summary.csv
results/vln_safety_adapter_smoke/mode_path_smoke.csv
results/vln_safety_adapter_smoke/figures/
```

## Main Experiments

### 1. Procedural v2 Benchmark

Script:

```bash
python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500
```

What it tests:

- Straight corridors.
- L-shaped and S-shaped corridors.
- Narrow entry and narrow exit.
- Asymmetric obstacles.
- False-feasible, physically infeasible passages.

Key result:

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feasible traversal success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reactive rule baseline | 25.4±1.2% | 60.7±3.8% | 8.4±3.1% | 6.4±2.7% | 27.8±5.9% | 24.8±1.6% | 33.6±4.6% | 0.0±0.0% |
| DEGNAV-Rule / Geometry-FSM | **70.3±1.4%** | **92.8±2.6%** | **79.4±7.0%** | **70.3±4.4%** | **96.5±2.0%** | **61.7±9.3%** | **50.3±3.9%** | 0.0±0.0% |

Values are mean±std across seeds 42-44.  False-feasible traversal success is
not bolded because it does not distinguish correct rejection from collision or
timeout/stuck.

Output tables:

- `results/narrow_passage_rl/paper_table_procedural_v2_main.md`
- `results/narrow_passage_rl/tables/paper_table_procedural_v2_main.md`
- `results/narrow_passage_rl/tables/paper_table_procedural_v2_ablation_core.md`
- `results/narrow_passage_rl/tables/paper_table_false_feasible_outcomes.md`

False-feasible outcome decomposition:

| Method | Episodes | Traversal success | Correct reject | Collision | Timeout/stuck | Wasted attempts |
|---|---:|---:|---:|---:|---:|---:|
| Reactive rule baseline | 1500 | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% |
| DEGNAV-Rule / Geometry-FSM | 1500 | 0.0% | 0.0% | 2.8% | 97.2% | 100.0% |

This table prevents a common misreading: 0% traversal success is not equivalent
to correct rejection.  The raw CSV logs `reject`, `correct_reject`,
`false_reject`, `collision`, `timeout`, `stuck`, and `wasted_attempt` per
episode.

Reject is therefore reported as an explicit mode, not inferred from failure.
The default DEGNAV-Rule / Geometry-FSM controller does not currently demonstrate
correct rejection on benchmark-labeled false-feasible passages.  To test an
explicit abstention policy without changing the default main method, the
evaluator includes a separate conservative diagnostic variant:

```bash
python examples/narrow_passage_rl/eval_harder_benchmark.py \
  --methods rule_baseline geometry_fsm feasibility_reject \
  --corridor-types false_feasible \
  --episodes 500 \
  --seeds 0 1 2 \
  --log-belief-diagnostics \
  --log-outcome-decomposition \
  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/false_feasible_reject_probe.csv
```

`feasibility_reject` uses only observable geometry/belief signals and does not
read the environment passability label.  Treat it as a diagnostic candidate, not
as a replacement for the current main DEGNAV-Rule row unless its false-reject
rate on passable corridors is also reported.

### 2. Habitat HM3D Nominal Anchor Validation

Scripts:

```bash
python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py
python examples/narrow_passage_rl/eval_habitat_apf_gap.py
python examples/narrow_passage_rl/eval_habitat_sb3.py --algo td3
```

What it tests:

- Scene-based validation in HM3D using `NarrowPassageNav-v0`.
- Mined anchors from held-out HM3D scenes.
- Direct-control learning baselines and classical APF+Gap baseline.

Important interpretation:

The 100.0% DEGNAV-Rule / Geometry-FSM rows are nominal anchor validation under
the current mining protocol.  The mined anchors are mostly well aligned, so this
result should not be described as complete robustness.  Stress validation is
required to expose module sensitivity.

Diagnostic and provenance tables:

- `results/narrow_passage_rl/paper_table_habitat.md`
- `results/narrow_passage_rl/paper_table_formal_baselines.md` (mixed-split Habitat diagnostic comparison)

### 3. Habitat Stress Validation

Script:

```bash
python examples/narrow_passage_rl/eval_habitat_stress_validation.py \
    --preset paper \
    --split val \
    --num-episodes -1
```

Stress settings:

- Start yaw perturbation: 0, 30, 60 degrees.
- Lateral start offset: 0.0 m, 0.10 m, 0.20 m.
- Depth-sector dropout: 0%, 10%, 30%.
- Feature Gaussian noise: sigma = 0.0, 0.02, 0.05.
- Extreme-narrow subset: `body_margin < 0.05 m`.

Compared methods:

- Full DEGNAV-Rule / Geometry-FSM.
- FSM without heading alignment.
- FSM without recovery.
- FSM without lateral alignment.
- APF+Gap, when the interface is available.

Output:

- `results/narrow_passage_rl/habitat_stress_validation.csv`
- `results/narrow_passage_rl/raw/habitat_stress_all.csv`
- `results/narrow_passage_rl/tables/paper_table_habitat_stress_nominal.md`
- `results/narrow_passage_rl/tables/paper_table_habitat_clearance_diagnostic.md`
- `results/narrow_passage_rl/tables/paper_table_habitat_stress_key_slices.md`

### Habitat Experimental Analysis

The analysis below is restricted to the canonical same-split HM3D Val set A
comparison (157 shared episode IDs) and the controlled Habitat stress protocol.
The mixed-split diagnostic table is not used for method ranking.  For the
requested memory ablation, **Ours** denotes DEGNAV-Rule with failure memory and
**Ours w/o Memory** denotes the same Geometry-FSM controller without that
memory.

#### Overall Performance

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

#### Failure Analysis

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

#### Ablation Analysis

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

#### Generalization Analysis

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

### 4. Clearance-Aware RL Diagnostic

This is a diagnostic table, not the main method ranking.

TD3 trained directly in Habitat reaches 100% nominal success but only 2.0%
clearance-aware strict success.  This demonstrates that nominal goal-reaching
success can be exploited by RL and must be reported together with clearance
diagnostics.

The Habitat clearance-related metrics are derived from depth observations and an approximate robot body-margin model. They are used as clearance-aware diagnostic indicators rather than calibrated contact measurements.

Output:

- `results/narrow_passage_rl/paper_table_diagnostic_baselines.md`

### 5. Repeated Failure Memory

Script:

```bash
python examples/narrow_passage_rl/eval_repeated_failure_memory.py \
    --n-rounds 5 \
    --n-passable 20 \
    --n-ff 15 \
    --max-steps 220
```

Claim tested:

Memory does not improve one-shot nominal Habitat success.  Its contribution is
to suppress repeated commitments to previously failed infeasible passages.

Key result:

| Method | Passable SR | Passable false reject | Final FF reject | Wasted FF steps | Steps saved | Precision |
|---|---:|---:|---:|---:|---:|---:|
| no_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 | - |
| local_intra_episode_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 | - |
| kNN Failure Memory | 0.860 | 0.060 | 1.000 | 5500 | 11000 | 0.916 |
| Vanilla Episodic Memory | 0.820 | 0.130 | 1.000 | 8140 | 8360 | 0.736 |
| Geometry-Guided Cross-Episode Failure Memory | 0.900 | 0.030 | 1.000 | 4400 | 12100 | 0.963 |

Output:

- `results/narrow_passage_rl/repeated_failure_memory.csv`
- `results/narrow_passage_rl/paper_table_repeated_failure_memory.md`
- `results/narrow_passage_rl/paper_table_repeated_failure_memory.tex`

### 6. Memory Transfer And Interference

Script:

```bash
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset smoke
```

Claim tested:

This experiment tests whether geometry-guided failure memory transfers from
repeated false-feasible passages to similar new false-feasible passages, while
avoiding false rejection on similar feasible passages.  This is distinct from
the repeated identical false-feasible table above.

Output:

- `results/narrow_passage_rl/memory_transfer_interference.csv`
- `results/narrow_passage_rl/paper_table_memory_transfer_interference.md`
- `results/narrow_passage_rl/paper_table_memory_transfer_interference.tex`

Paper-preset result:

| Method | Passable SR | Final FF reject | Transfer reject new FF | Interference false reject |
|---|---:|---:|---:|---:|
| no_memory | 90.0 +/- 2.2 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| vanilla episodic memory | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| kNN failure memory | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| geometry-guided cross-episode failure memory | 90.0 +/- 2.2 | 100.0 +/- 0.0 | 93.0 +/- 8.6 | 0.0 +/- 0.0 |

Interpretation: generic episodic/kNN memory can suppress repeated infeasible
commitments, but in this protocol it also over-generalizes and rejects all
similar feasible passages.  Geometry-guided failure memory preserves passable
success while transferring rejection to similar new false-feasible passages.

### 7. D_min Calibration

Script:

```bash
python examples/narrow_passage_rl/eval_dmin_calibration.py \
    --priors 0.26 0.31 0.36 0.46 0.56 \
    --true-width 0.36 \
    --episodes 300 \
    --seeds 0 1 2 \
    --output-dir examples/narrow_passage_rl/results/narrow_passage_rl
```

What it tests:

- Online calibration of the effective robot width / minimum traversable
  clearance threshold under under-conservative, oracle, and over-conservative
  priors.
- Directional reliability of `p_feas` from raw per-episode predictions.

Output:

- `results/narrow_passage_rl/tables/paper_table_calibration_extended.md`
- `results/narrow_passage_rl/raw/calibration_prior_sweep.csv`
- `results/narrow_passage_rl/raw/calibration_episode_predictions.csv`
- `results/narrow_passage_rl/tables/calibration_reliability_bins.csv`

## Diagnostic And Smoke Baselines

Use these tables for paper organization:

```text
results/narrow_passage_rl/paper_table_formal_baselines.md  # mixed-split Habitat diagnostic comparison
results/narrow_passage_rl/paper_table_diagnostic_baselines.md
results/narrow_passage_rl/paper_table_smoke_baselines.md
```

Mixed-split Habitat diagnostic comparison:

- PPO v2 geometry sensor, 3 seeds, synthetic-to-Habitat direct-control transfer.
- SAC v2 geometry sensor direct-control baseline.
- TD3 synthetic-to-Habitat direct-control baseline.
- APF+Gap classical baseline.
- DEGNAV-Rule / Geometry-FSM.

Clearance-aware diagnostic evaluation:

- TD3 Habitat-native nominal success vs clearance-aware strict success.

Smoke / appendix-only baselines:

- GRU-PPO lightweight.
- RecurrentPPO smoke.
- BC-FSM.
- DAgger-FSM.
- Replay Memory Policy.

Legacy PPO runs with the v1 `obs[10]` yaw/heading mismatch are provenance only
and are excluded from formal baseline tables.

## Current RL Interpretation

RL is currently used in four clearly separated roles:

1. Direct-control learning baselines: PPO/SAC/TD3 policies with the same
   19-D geometry input.
2. Clearance-aware diagnostic evaluation: Habitat-native TD3 nominal success vs clearance-aware strict success.
3. Smoke/appendix paths: recurrent, imitation, and replay-memory policies.
4. DEGNAV-RL: a diagnostic policy for high-level mode selection over the
   explicit belief state, not a direct velocity policy.

The important diagnostic result is:

| Method | Nominal SR | Strict SR | Success-but-unsafe | Near collision |
|---|---:|---:|---:|---:|
| TD3 Habitat-native | 1.000 | 0.020 | 0.980 | 1.000 |

This means the policy learned to satisfy the nominal Habitat success condition,
but not the depth-derived body-margin diagnostic.  It should be reported as
nominal-metric exploitation, not as a calibrated clearance result.

The AAAI-facing framing is:

- learning-only transfer is weak near geometric feasibility boundaries;
- nominal success can hide low body-margin behavior;
- clearance-aware strict success, near-collision, and minimum clearance must be
  reported for RL baselines;
- DEGNAV-RL is intended to test only the high-level mode selector
  `pi(m_t | b_t)`, while the geometry/risk decision variables remain explicit.

DEGNAV-RL formulation:

```text
b_t = (p_feas, E[Delta], Var[Delta], heading_error, lateral_error,
       stuckness, contact, memory_risk)
m_t ~ pi(m_t | b_t),  m_t in {Commit, Explore, Recover, Reject}
u_t = mode_conditioned_controller(m_t, geometry, local_goal)
```

Under this framing, PPO/SAC/TD3 remain direct-control baselines, while
DEGNAV-RL is a diagnostic policy that tests whether the mode selector can be
learned without replacing DEGNAV-Rule's explicit geometry/risk logic.

## DEGNAV-RL: Diagnostic Belief-Guided Mode-Selection PPO

DEGNAV-RL trains PPO on the compact feasibility-belief state `b_t` instead of
the raw 19-D geometry observation.  The policy is intended to output a
high-level mode:

```text
0 = COMMIT
1 = EXPLORE
2 = RECOVER
3 = REJECT
```

The wrapper then realizes that mode with the shared mode-conditioned controller.
This is not the same as the PPO/SAC/TD3 direct-control baselines in
`train_sb3_v2.py`, which output velocity actions directly.
DEGNAV-RL is included as a diagnostic policy rather than a competitive final
method. Under the current reward and action interface, the learned policy
collapses to Commit and Explore and does not demonstrate meaningful Recover or
Reject behavior.

Current 3-seed result on procedural v2, using 1M PPO steps per seed:

| Variant | Overall SR | Strict SR | Collision | Near collision | Reject | Mode usage |
|---|---:|---:|---:|---:|---:|---|
| DEGNAV-RL full belief | 26.7% | 26.7% | 62.3% | 70.9% | 0.0% | Commit 31.0%, Explore 69.0%, Recover 0.0%, Reject 0.0% |
| no p_feas | 25.9% | 25.9% | 61.9% | 72.5% | 0.0% | Commit 62%, Explore 38% |
| no delta_var | 28.4% | 28.4% | 58.9% | 70.3% | 0.0% | Commit 29%, Explore 71% |
| no memory | 28.4% | 28.4% | 58.9% | 70.3% | 0.0% | Commit 29%, Explore 71% |
| no alignment | 25.8% | 25.8% | 61.5% | 72.2% | 0.0% | Commit 37%, Explore 63% |
| geometry only | 30.9% | 30.9% | 56.0% | 68.1% | 0.0% | Explore 100% |

Interpretation:

- DEGNAV-RL is no longer `not run`, but it should remain a diagnostic learning
  variant rather than the main method.
- It is far below DEGNAV-Rule / Geometry-FSM and still has high collision and
  near-collision rates.
- The learned mode selector did not learn `Reject` or `Recover` under the
  current reward/controller interface.  This is useful evidence that simply
  learning `pi(m_t | b_t)` is not enough; the explicit rule-based feasibility,
  risk, and failure-memory checks remain important.
- The negative result supports five design conclusions: sparse reward alone is
  insufficient, mode semantics are not learned automatically, explicit failure
  memory may still be necessary, Recover/Reject require dedicated reward or
  supervision, and longer training alone may not solve mode collapse.
- The belief-state ablation does not show a clean advantage for the full belief
  vector.  `geometry_only` performs best in this run, so the paper should not
  claim that the current DEGNAV-RL policy successfully exploits uncertainty or
  memory risk.

Smoke training:

```bash
python examples/narrow_passage_rl/train_belief_mode_ppo.py \
    --total-steps 2048 \
    --num-envs 1 \
    --ctypes straight_only \
    --ablation full \
    --eval-episodes 20 \
    --save-dir data/degnav_rl_belief_mode_smoke
```

Paper-scale training:

```bash
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
done
```

Evaluation:

```bash
for seed in 0 1 2; do
  python examples/narrow_passage_rl/eval_belief_mode_ppo.py \
      --model data/degnav_rl_belief_mode_full_seed${seed}/belief_mode_ppo.zip \
      --episodes 500 \
      --seed $((1000 + seed)) \
      --ctypes full \
      --ablation full \
      --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed${seed}_eval.csv
done
```

Belief-state ablations use the same scripts:

```bash
python examples/narrow_passage_rl/train_belief_mode_ppo.py \
    --total-steps 1000000 \
    --num-envs 8 \
    --ctypes full \
    --ablation no_p_feas \
    --eval-episodes 500 \
    --save-dir data/degnav_rl_belief_mode_no_p_feas

python examples/narrow_passage_rl/eval_belief_mode_ppo.py \
    --model data/degnav_rl_belief_mode_no_p_feas/belief_mode_ppo.zip \
    --episodes 500 \
    --ctypes full \
    --ablation no_p_feas \
    --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_p_feas_eval.csv
```

Supported ablations are `full`, `no_p_feas`, `no_delta_var`, `no_memory`,
`no_alignment`, and `geometry_only`.  `geometry_only` changes the policy
observation to six features: `d_hat`, `body_margin`, `clearance_left`,
`clearance_right`, `heading_error`, and `lateral_error`.

To regenerate the DEGNAV-RL ablation table from one or more eval CSVs:

```bash
python examples/narrow_passage_rl/make_paper_tables.py \
    --belief-mode-inputs \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed0_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed1_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed2_eval.csv \
    --belief-mode-ablation-inputs \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_full_seed*_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_p_feas_seed*_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_delta_var_seed*_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_memory_seed*_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_no_alignment_seed*_eval.csv \
      examples/narrow_passage_rl/results/narrow_passage_rl/belief_mode_geometry_only_seed*_eval.csv
```

Training outputs include `belief_mode_ppo.zip`,
`belief_mode_ppo_config.json`, `belief_mode_ppo_quick_eval.csv`, and
`belief_mode_ppo_quick_summary.md`.

## What Not To Overclaim

- HM3D nominal 100.0% is not a complete robustness proof.  It is nominal anchor
  validation under the current mining protocol on mined, mostly well-aligned
  starts.
- Habitat stress validation is the evidence for module sensitivity under yaw,
  lateral, dropout, noise, and extreme-clearance perturbations.
- Failure memory is not for improving one-shot passable-anchor success.  It
  reduces repeated infeasible commitment and wasted attempts.
- Smoke RL baselines are not main baselines.  They verify code paths and belong
  in appendix/status tables unless rerun under the full protocol.

## Reproduction Commands

Run from the repository root.

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

# Memory transfer/interference smoke test
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset smoke

# Memory transfer/interference paper run
python examples/narrow_passage_rl/eval_memory_transfer_interference.py \
    --preset paper

# D_min calibration
python examples/narrow_passage_rl/eval_dmin_calibration.py \
    --priors 0.26 0.31 0.36 0.46 0.56 \
    --true-width 0.36 \
    --episodes 300 \
    --seeds 0 1 2 \
    --output-dir examples/narrow_passage_rl/results/narrow_passage_rl

# Regenerate paper tables from available CSV artifacts
python examples/narrow_passage_rl/make_paper_tables.py

```

## Rule vs DEGNAV Margin-Phase Analysis

This is the paper-facing margin-phase analysis for the main method.  It compares
the reactive rule baseline against DEGNAV-Rule / Geometry-FSM on paired
procedural v2 scenarios.  `--episodes 500` means 500 episodes per method per
seed.

```bash
python examples/narrow_passage_rl/eval_harder_benchmark.py \
  --methods rule_baseline geometry_fsm \
  --episodes 500 \
  --seeds 0 1 2 \
  --log-belief-diagnostics \
  --log-outcome-decomposition \
  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv
```

The evaluator writes the paired raw CSV and metadata:

```text
results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv
results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.meta.json
results/narrow_passage_rl/tables/margin_phase_raw_validation.md
```

Generate the publication figures and regime summary:

```bash
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
```

Outputs:

```text
results/narrow_passage_rl/figures/margin_phase_rule_vs_degnav_rule.{pdf,png}
results/narrow_passage_rl/figures/margin_phase_near_boundary_zoom.{pdf,png}
results/narrow_passage_rl/tables/margin_phase_rule_vs_degnav_rule.csv
results/narrow_passage_rl/tables/paper_table_margin_phase_summary.{md,tex}
```

Margin is computed at the shared pre-divergence decision snapshot.  Belief
diagnostics for the reactive rule baseline are used only for stratification and
do not influence its actions.  `Reject` denotes an explicit Reject mode, not
generic failure.

The default sampled benchmark has limited support on the infeasible side of the
margin axis.  To stress negative margins directly, run a separate probe instead
of over-interpreting the low-support bins in the main figure:

```bash
python examples/narrow_passage_rl/eval_harder_benchmark.py \
  --methods rule_baseline geometry_fsm feasibility_reject \
  --episodes 500 \
  --seeds 0 1 2 \
  --width-range 0.30 0.55 \
  --log-belief-diagnostics \
  --log-outcome-decomposition \
  --output-csv examples/narrow_passage_rl/results/narrow_passage_rl/raw/margin_phase_infeasible_probe.csv
```

This probe is for sensitivity analysis.  Keep it separate from the main paired
Rule-vs-DEGNAV figure unless the paper explicitly labels it as a deliberately
harder width-prior slice.

Current infeasible-side probe outputs:

```text
results/narrow_passage_rl/raw/margin_phase_infeasible_probe.csv
results/narrow_passage_rl/figures/infeasible_probe/margin_phase_rule_vs_degnav_rule.{pdf,png}
results/narrow_passage_rl/figures/infeasible_probe/margin_phase_near_boundary_zoom.{pdf,png}
results/narrow_passage_rl/tables/infeasible_probe/paper_table_margin_phase_summary.{md,tex}
```

This probe fixes the low-support issue for negative margins: it contains 381
episodes per method with `delta_mean < -0.05` and 872 episodes per method in
`[-0.10, 0.10]`.  It also shows why the conservative reject gate remains
diagnostic rather than part of the main method: it triggers Reject in 10.2% of
episodes, but only 0.1% are correct rejects overall and 10.1% are false rejects.
On false-feasible corridors specifically, correct reject is 1.3% and timeout
remains 98.7%.  This supports the current paper interpretation: false-feasible
rejection needs failure memory or an explicit blockage/occlusion model, not just
a static width-margin gate.

Habitat runs require:

```text
data/scene_datasets/hm3d/
data/datasets/narrow_passage/{split}/{split}.json.gz
```

## Dataset Generation

Mine anchors from HM3D and generate `NarrowPassageNav-v0` episode files:

```bash
python examples/narrow_passage_rl/mine_habitat_passages.py \
    --scenes-dir data/scene_datasets/hm3d/val \
    --target-episodes 800 \
    --out-train data/datasets/narrow_passage/anchors_train.csv \
    --out-val data/datasets/narrow_passage/anchors_val.csv

python examples/narrow_passage_rl/generate_habitat_episodes.py \
    --anchors data/datasets/narrow_passage/anchors_val.csv \
    --split val \
    --output data/datasets/narrow_passage/val/val.json.gz
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

Render top-down synthetic trajectories:

```bash
python examples/narrow_passage_rl/render_trajectories.py \
    --panels l_shaped s_shaped false_feasible narrow_entry \
    --dpi 150
```

## Result Files

Start with the paper-ready result index:

```text
results/narrow_passage_rl/paper_ready_results.md
```

This index is the intended entry point for the final effective data and tables.
It separates main-paper results from diagnostic, smoke, mixed-split, and legacy
artifacts.

Main paper table/figure outputs:

```text
results/narrow_passage_rl/paper_table_procedural_v2_main.md
results/narrow_passage_rl/tables/paper_table_procedural_v2_ablation_core.md
results/narrow_passage_rl/tables/paper_table_false_feasible_outcomes.md
results/narrow_passage_rl/paper_table_habitat_same_split.md
results/narrow_passage_rl/tables/paper_table_habitat_stress_nominal.md
results/narrow_passage_rl/tables/paper_table_habitat_stress_key_slices.md
results/narrow_passage_rl/paper_table_repeated_failure_memory.md
results/narrow_passage_rl/paper_table_memory_transfer_interference.md
results/narrow_passage_rl/tables/paper_table_calibration_extended.md
results/narrow_passage_rl/tables/paper_table_margin_phase_summary.md
results/narrow_passage_rl/figures/margin_phase_rule_vs_degnav_rule.{pdf,png}
```

Diagnostic or appendix-only outputs:

```text
results/narrow_passage_rl/tables/paper_table_habitat_clearance_diagnostic.md
results/narrow_passage_rl/paper_table_diagnostic_baselines.md
results/narrow_passage_rl/paper_table_degnav_rl_diagnostic.md
results/narrow_passage_rl/paper_table_belief_mode_ablation.md
results/narrow_passage_rl/paper_table_formal_baselines.md  # mixed-split diagnostic, not a main ranking
results/narrow_passage_rl/paper_table_smoke_baselines.md
```

Important raw CSV outputs:

```text
results/narrow_passage_rl/harder_benchmark_episodes.csv
results/narrow_passage_rl/raw/procedural_ablation_core.csv
results/narrow_passage_rl/raw/false_feasible_outcomes.csv
results/narrow_passage_rl/raw/margin_phase_rule_vs_degnav_episodes.csv
results/narrow_passage_rl/habitat_stress_validation.csv
results/narrow_passage_rl/raw/habitat_stress_all.csv
results/narrow_passage_rl/repeated_failure_memory.csv
results/narrow_passage_rl/memory_transfer_interference.csv
results/narrow_passage_rl/raw/calibration_prior_sweep.csv
results/narrow_passage_rl/raw/calibration_episode_predictions.csv
results/narrow_passage_rl/belief_mode_full_seed*_eval.csv
results/narrow_passage_rl/belief_mode_<ablation>_seed*_eval.csv
results/narrow_passage_rl/habitat_td3_habitat_native_strict_eval.csv
```

## Implementation Notes

The 19-D geometry feature vector contains depth sectors, left/right clearance,
passage width, body margin, heading error, lateral offset, distance to goal,
previous action, stuck score, and collision flag.  Habitat exposes this through
`NarrowPassageGeometrySensor`.

The Geometry-FSM / DEGNAV-Rule outputs local velocity commands `(v_x, omega_z)`
through interpretable modes: align, commit, explore, recover, reject, and
follow-space for L/S-shaped turns.  DEGNAV-RL is only a diagnostic test of
mode-selection learning, and should not be described as replacing this interface
with direct velocity regression or as the current final method.

The real-robot interface should reproduce the same 19-D feature vector from
depth, localization, and proprioception.  Hardware experiments should be
reported as pilot validation unless accompanied by full videos, logs, bags, and
statistics.

## Strict feasibility ablation v2 (current)

The strict feasibility audit has been repaired and revalidated without
overwriting the old result. The root cause of the earlier identical success
rates was structural: `var_delta` was fixed at `0.0025`, the full selector used
`var_delta > 0.0025` (therefore never true), and `no_uncertainty` still consumed
`p_feas`, which indirectly contained the same fixed variance. Point thresholds
were a monotone mapping of those probability boundaries, so full,
point-estimate, and no-uncertainty necessarily produced identical actions.

The repaired strict call chain is:

```text
19-D observation
 -> scene-dependent DynamicFeasibilityEstimator
 -> mu_delta, sigma_delta, p_feas, LCB, UCB, yaw widths
 -> strict feasibility selector
 -> shared Recover/alignment/follow-space overrides
 -> Commit/Explore/Recover/Reject selected_mode
 -> shared linear/angular controller
 -> per-step audit CSV
```

The dynamic uncertainty estimator combines ray dispersion, valid/dropout ratio,
boundary-fit residual, temporal width variation, and yaw/lateral pose
uncertainty. Its temporal window resets across the simulator's 10 m open-space
sentinel boundary, so that sentinel is never mixed with passage widths. The
common interval rule is
`LCB=mu_delta-kappa*sigma_delta`, `UCB=mu_delta+kappa*sigma_delta`, with Commit
when `LCB>0.02 m`, Reject when `UCB<-0.02 m`, Explore on interval overlap, and
Recover after a failed prior commitment/collision/stuck event.

| Variant | Strict change | Mode decision |
|:---|:---|:---|
| `full` | dynamic mean and sigma, yaw correction on | dynamic LCB/UCB |
| `point_estimate` | mean-only information boundary | deterministic mean threshold |
| `no_uncertainty` | estimator/logging retained; sigma forbidden in selector | deterministic mean threshold |
| `fixed_uncertainty` | global `sigma_delta=0.05 m` | fixed-sigma LCB/UCB |
| `no_yaw_prior` | only yaw-dependent width mean removed | dynamic LCB/UCB |

The rectangle projection is
`A*abs(cos(yaw)) + B*abs(sin(yaw)) + 2*safety_margin`, using the explicitly
declared strict decision envelope `A=0.36 m`, `B=0.60 m`, and margin `0.03 m`
per side. The subtractive candidate formula was rejected by geometry and unit
test: it can produce a negative required width at 45 degrees. The procedural
collision shape remains a 0.18 m-radius circle, so the rectangle is documented
as the decision-model envelope rather than simulator morphology.

All 35 narrow-passage unit tests pass. A 45-episode paired mechanism validation
was then run over a deterministic subset of the complete 225-cell stress grid
(`margin x yaw x noise/dropout x lateral offset`). It records four steps for all
five methods with identical paired observation and random-draw hashes.

| Gate | Result |
|:---|---:|
| Dynamic full `sigma_delta` | 132 distinct values; range 0.0157–0.0974 m |
| Full uncertainty branch | 33/180 steps (18.33%) |
| Full vs point final-mode disagreement | 33/180 steps (18.33%) |
| Yaw prior changed feasibility decision | 106/180 steps (58.89%) |
| Paired non-ablated inputs/randomness | passed |

Mechanism artifacts are under
`results/ablation_feasibility/mechanism_20260818_170356/`; the complete
per-step audit is `steps.csv` and the fail-fast result is
`mechanism_report.json`. The full three-seed protocol was run only after every
gate passed and completed in the separate timestamped
`results/ablation_feasibility/paper_dynamic_20260818_170356/` directory: 10,500
episode rows, 528,873 step rows, and exactly 100 episodes in each of the 105
method/seed/scene cells.

| Method | Success | Collision | Correct reject | False reject | Timeout/stuck |
|:---|---:|---:|---:|---:|---:|
| full | 0.2 ± 0.2% | 11.2 ± 1.4% | 13.9 ± 0.5% | 63.5 ± 4.3% | 11.2 ± 3.0% |
| point_estimate | 0.3 ± 0.4% | 10.8 ± 1.4% | 13.9 ± 0.5% | 64.4 ± 4.1% | 10.6 ± 2.8% |
| no_uncertainty | 0.3 ± 0.4% | 10.8 ± 1.4% | 13.9 ± 0.5% | 64.4 ± 4.1% | 10.6 ± 2.8% |
| fixed_uncertainty | 0.2 ± 0.2% | 11.1 ± 1.4% | 13.9 ± 0.5% | 63.6 ± 4.4% | 11.2 ± 3.0% |
| no_yaw_prior | 0.2 ± 0.2% | 11.6 ± 1.4% | 13.9 ± 0.5% | 62.5 ± 3.9% | 11.8 ± 2.7% |

The low and nearly equal success rates do not mean the mechanisms are still
dead. In the formal run, full used 17,294 distinct sigma values (0.0184–0.1297
m), triggered uncertainty gating on 2.32% of its steps, and changed 30 episode
outcomes relative to point-estimate; the yaw counterfactual changed 1.43% of
full step decisions and full vs no-yaw changed 29 episode outcomes. Success is
saturated at only 4–6 successes per 2,100 episodes, so it is too coarse to
resolve those changes. Point-estimate and no-uncertainty remain behaviorally
identical by definition because both use the same deterministic mean-only
selector; no-uncertainty's estimator is retained for instrumentation, while
sigma cannot affect either mode decision.

This is not a positive performance result. The 62–64% false-reject rates also
show that the explicitly declared rectangular decision envelope is much more
conservative than the procedural simulator's circular collision body. No
threshold was retuned after observing these outcomes.

Reproduce the staged validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xiaotian/miniconda3/envs/navila/bin/python -m pytest -q \
  examples/narrow_passage_rl/tests

/home/xiaotian/miniconda3/envs/navila/bin/python \
  examples/narrow_passage_rl/eval_feasibility_stress.py \
  --episodes 45 \
  --output-dir examples/narrow_passage_rl/results/ablation_feasibility/mechanism_NEW_TIMESTAMP

# Run only if mechanism_report.json has every gate=true.
/home/xiaotian/miniconda3/envs/navila/bin/python \
  examples/narrow_passage_rl/eval_feasibility_ablation.py \
  --all-ablations --seeds 0 1 2 --episodes-per-scene 100 \
  --output-dir examples/narrow_passage_rl/results/ablation_feasibility/paper_dynamic_NEW_TIMESTAMP
```

See [`docs/feasibility_ablation.md`](docs/feasibility_ablation.md) for the
complete variable call-chain audit, equations, output schema, mechanism table,
and exact reproduction commands.

## Strict feasibility ablation v1 (archived diagnostic)

The controlled four-way comparison (`full`, `point_estimate`,
`no_uncertainty`, and `no_yaw_prior`) now has a separate paired runner, tested
selector interfaces, non-overwriting result layout, paper tables, calibration,
and margin/yaw/uncertainty plots.

The completed run uses seeds `0, 1, 2`, all seven procedural-v2 scene types,
and 100 episodes per scene per seed. This is 700 episodes per method per seed,
2,100 episodes per method, and 8,400 CSV rows. All 84
method/seed/scene cells contain exactly 100 episodes, and paired rows have
identical episode IDs, environment seeds, available widths, initial yaw, and
lateral offsets.

| Method | Success | Collision | Correct reject | False reject | Timeout/stuck | Steps | Time | Oscillation |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEGNav full | 17.8 ± 1.9% | 15.1 ± 3.1% | 11.6 ± 1.4% | 52.2 ± 3.5% | 3.3 ± 1.2% | 36.47 ± 5.88 | 9.12 ± 1.47 | 0.56 ± 0.15 |
| DEGNav point estimate | 17.8 ± 1.9% | 15.1 ± 3.1% | 11.6 ± 1.4% | 52.2 ± 3.5% | 3.3 ± 1.2% | 36.47 ± 5.88 | 9.12 ± 1.47 | 0.56 ± 0.15 |
| DEGNav w/o uncertainty-aware gating | 17.8 ± 1.9% | 15.1 ± 3.1% | 11.6 ± 1.4% | 52.2 ± 3.5% | 3.3 ± 1.2% | 36.47 ± 5.88 | 9.12 ± 1.47 | 0.56 ± 0.15 |
| DEGNav w/o yaw prior | 17.8 ± 1.9% | 15.3 ± 3.0% | 11.6 ± 1.4% | 51.9 ± 3.7% | 3.4 ± 1.1% | 36.59 ± 5.84 | 9.15 ± 1.46 | 0.53 ± 0.15 |

Values are mean ± sample standard deviation across three seeds. The paired
success differences are:

- `full - point_estimate`: 0.00 percentage points, 95% CI `[0.00, 0.00]`;
- `full - no_uncertainty`: 0.00 percentage points, 95% CI `[0.00, 0.00]`;
- `full - no_yaw_prior`: -0.05 percentage points, 95% CI `[-0.25, 0.16]`.

The first three methods are identical episode by episode because the current
belief uses a constant
`var_delta = 0.04^2 + 0.03^2 = 0.0025`. Consequently,
`p_feas = Phi(mu_delta / 0.05)` is only a monotone transformation of the mean
margin. The point-estimate thresholds are the fair analytic mappings of the
same probability boundaries (`tau_commit = 0.026220 m`,
`tau_reject = -0.070254 m`), and the full method's explicit uncertainty rule
checks `var_delta > 0.0025`, which never occurs. The three selectors therefore
produce identical modes, trajectories, memory histories, and outcomes. This is
an estimator limitation, not evidence that probability distributions or
uncertainty are generally unnecessary.

The yaw-prior success interval includes zero, so the run also does not support
a benefit for the current `A/cos(|theta|)` prior. It does not test the paper's
`A|cos(theta)| + B|sin(theta)| + ...` equation because the repository has no
configured body length `B`, sensor margin, or roll/body-height parameters.

Probability calibration against the benchmark `passable_label` is
`Brier/ECE/NLL = 0.4932/0.5011/5.7931` for full and no-uncertainty, and
`0.2563/0.2920/2.5452` for no-yaw-prior. Point estimate is not assigned
probability calibration metrics; its classification accuracy/precision/recall
are `0.4419/0.8285/0.4400`. These values include the documented mismatch
between entry-width feasibility and downstream false-feasible blockers.

Canonical artifacts:

```text
results/ablation_feasibility/paper_20260818/episodes.csv
results/ablation_feasibility/paper_20260818/run_metadata.json
results/ablation_feasibility/paper_20260818/analysis/overall_ablation_table.md
results/ablation_feasibility/paper_20260818/analysis/overall_ablation_table.tex
results/ablation_feasibility/paper_20260818/analysis/paired_differences.csv
results/ablation_feasibility/paper_20260818/analysis/margin_phase.{csv,png,pdf}
results/ablation_feasibility/paper_20260818/analysis/yaw_sensitivity.{csv,png,pdf}
results/ablation_feasibility/paper_20260818/analysis/uncertainty_behavior.{csv,png,pdf}
results/ablation_feasibility/paper_20260818/analysis/calibration_metrics.csv
results/ablation_feasibility/paper_20260818/analysis/reliability_diagram.{png,pdf}
results/ablation_feasibility/paper_20260818/analysis/objective_conclusion.md
```

See
[`docs/feasibility_ablation.md`](docs/feasibility_ablation.md) for the audit,
known paper-code mismatches, exact commands, smoke test, and complete result
interpretation. The requested sample scale is complete, but the results do not
support a causal superiority claim for any of the three factors.

## Strict 19-D recurrent four-mode selector (staged)

The new selector pipeline keeps historical `BeliefModeEnv`/MLP-PPO artifacts
unchanged and adds a separate, auditable interface with the fixed action map
`COMMIT=0, EXPLORE=1, RECOVER=2, REJECT=3`. Its Actor receives exactly the
19-D geometry/execution vector; optional 4-D failure memory is passed through a
separate argument. Reject is an explicit terminal outcome rather than timeout.

The implementation, feature/units audit, known Habitat/procedural unit
differences, reward definition, staged commands, and domain limitations are in
[`docs/mode_selector_ppo_audit.md`](docs/mode_selector_ppo_audit.md). New runs
belong under `results/mode_selector/`. A run is not a paper result merely
because a checkpoint exists: use `eval_mode_selector.py` on fixed held-out
episodes and generate comparison tables from its `metrics.json` using
`summarize_mode_selector.py`.

The completed staged results through the 100k acceptance gate are documented
in the [Chinese work report](results/mode_selector/实验工作报告.md) and the
[English report](results/mode_selector/EXPERIMENT_REPORT.md).
The gate did not authorize longer training: pure BC+PPO returned zero
deterministic Reject on the fixed set, while the auxiliary diagnostic raised
false rejection to 31.15%. The 1M configurations are therefore templates, not
claims of completed experiments.
