# VLN Integration Status

This document summarizes the current NaVILA/VLN evidence and the next
engineering requirements for integrating language navigation with DEGNAV and a
quadruped controller.

## Current Scope

NaVILA is used as a semantic proposal layer, not as a direct low-level
controller.  The current repository has not run standard R2R/RxR navigation
metrics because the matching MP3D scene assets are not available in the local
setup.  The available evidence is therefore diagnostic:

- model artifact and runtime validation;
- controlled local-command behavior;
- R2R val-unseen instruction-text probes with unmatched demo frames;
- visual perturbation sensitivity;
- deterministic NaVILA-to-DEGNAV safety-adapter replay.

Do not report SR, SPL, Navigation Error, nDTW, SDTW, or scene generalization
from these diagnostics.

## System Interface

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
Quadruped locomotion policy
```

The important design decision is that route instructions and trusted immediate
control directives are separated.  A route such as "turn left after the sofa"
must not be parsed as a current low-level turn.  A trusted runtime update such
as "the task is complete; stop immediately" can safely override the semantic
proposal.

## Current Results

### Dataset-Free Smoke

Source:

```text
examples/narrow_passage_rl/results/vln_dataset_free/report.md
```

Key outcomes:

| Metric | Value |
|---|---:|
| Inference cases | 7 |
| Parseable action rate | 100.0% |
| Deterministic repeat match | 100.0% |
| Mean latency | 1.409 s |
| P95 latency | 1.696 s |
| Peak GPU memory | 8003.1 MiB |

### 640-Case Controlled Diagnostic

Source:

```text
examples/narrow_passage_rl/results/vln_batch_diagnostic/report.md
```

Key outcomes:

| Metric | Cases | Value |
|---|---:|---:|
| Parseable action rate | 640 | 100.0% |
| Original R2R text probe: move-forward output | 256 | 71.5% |
| Original R2R text probe: stop output | 256 | 1.2% |
| R2R stop-override compliance | 256 | 1.6% |
| Controlled move-forward command match | 16 | 100.0% |
| Controlled turn-left command match | 16 | 0.0% |
| Controlled turn-right command match | 16 | 0.0% |
| Controlled stop command match | 16 | 0.0% |
| Non-clean visual perturbation action consistency | 56 | 75.0% |

Measured problems:

| Problem | Evidence | Immediate implication |
|---|---:|---|
| Explicit stop failure | 1.6% stop-override compliance | Add trusted stop channel and train completion head. |
| Action-prior bias | 71.5% move-forward output on R2R text probes | Use balanced action data and per-action calibration. |
| Weak left/right grounding | 0.0% controlled left/right match | Add direction-contrastive local clips. |
| Missing obstacle veto | blocked synthetic forward cases still move forward | Gate forward actions with depth/body-margin risk. |
| Visual perturbation sensitivity | 75.0% non-clean action consistency | Add visual/depth robustness training. |

### R2R Text-Only Corpus Diagnostic

Source:

```text
examples/narrow_passage_rl/results/vln_r2r_text_analysis/report.md
```

This diagnostic reads only R2R/VLN-CE annotations and does not require MP3D
scene assets.  It analyzes 13436 total instructions from train, val-seen, and
val-unseen.

| Split | Episodes | Unique scenes | Mean words | Median words |
|---|---:|---:|---:|---:|
| train | 10819 | 61 | 26.67 | 25 |
| val_seen | 778 | 53 | 27.03 | 25 |
| val_unseen | 1839 | 11 | 26.65 | 25 |

Val-unseen instructions contain dense navigation cues: 98.4% include
forward/motion terms, 48.1% include left, 52.1% include right, 81.4% include a
stop/goal relation, 92.0% include room/place references, 45.9% include
object/landmark references, and 67.9% include spatial-relation terms.

When joined with the existing 256 NaVILA text-probe predictions, the output
prior remains strongly forward-biased: 71.5% `move_forward` and only 1.2%
`stop`.  This supports using VLN as a semantic proposal layer below a trusted
geometry/risk/memory gate rather than as a direct velocity controller.

### Paired Safety-Adapter Smoke

Source:

```text
examples/narrow_passage_rl/results/vln_safety_adapter_smoke/report.md
```

The adapter replays the exact same 640 saved NaVILA outputs.

| Metric | NaVILA direct | + safety adapter |
|---|---:|---:|
| Controlled command exact match | 25.0% | 81.3% |
| Non-blocked command exact match | 25.0% | 100.0% |
| Explicit stop compliance | 1.5% | 100.0% |
| R2R stop-override compliance | 1.6% | 100.0% |
| Blocked-fixture forward output | 100.0% | 0.0% |
| Original R2R proposal preservation | 100.0% | 100.0% |
| Visual perturbation action consistency | 75.0% | 75.0% |

Interpretation:

- The adapter improves trusted-control and geometry-gated interface behavior.
- It does not improve NaVILA weights or visual robustness.
- It must not be described as learned VLN improvement or standard R2R
  performance.
- The blocked-fixture result uses only four controlled cases and synthetic
  clearance values; treat it as a smoke test.
- Recover/Reject paths are deterministic code paths, not learned policies.

## Required Next Experiments

1. Install MP3D scenes and run paired R2R/VLN-CE evaluation on `val_seen` and
   `val_unseen`.
2. Report SR, SPL, NE, Oracle Success, nDTW, SDTW, stop precision/recall,
   collision, timeout, and intervention rate.
3. Compare `NaVILA direct`, `NaVILA + trusted directive`, `NaVILA + geometry
   gate`, `NaVILA + geometry + recovery`, and the full hierarchical system on
   identical episode IDs.
4. Replace direct action execution with waypoint/completion outputs:
   `semantic_goal`, `waypoint_xy`, `waypoint_covariance`, and `p_complete`.
5. Add action-balanced and direction-counterfactual supervised data.
6. Add visual/depth robustness training and evaluate each perturbation
   separately.
7. Connect geometry-guided failure memory to `memory_risk` and evaluate
   repeated false-feasible, similar-new false-feasible, and similar-feasible
   interference cases.

## R2R/VLN-CE Paired Evaluation Preflight

Source:

```text
examples/narrow_passage_rl/results/vln_r2r_eval_preflight/r2r_vlnce_preflight.md
```

Current status: **blocked by missing licensed MP3D scene files**, not by the
NaVILA code path.

| Item | Status |
|---|---:|
| NaVILA checkpoint | pass |
| R2R `val_unseen` annotations | pass, 1839 episodes |
| Unique MP3D scenes in `val_unseen` | 11 |
| DDPPO depth encoder | pass |
| NaVILA evaluation data symlinks | pass |
| Habitat 0.1.7 imports | pass |
| MP3D `.glb` coverage | 0/11 |
| MP3D `.navmesh` coverage | 0/11 |

Missing `val_unseen` scenes:

```text
2azQ1b91cZZ, 8194nk5LbLH, EU6Fwq7SyZv, QUCTc6BB5sX, TbHJrupSAjP,
X7HyMhZNoso, Z6MFQCViBuw, oLBMNvg9in8, pLe4wQe7qrG, x8F5xyUWy9e,
zsNo4HB9uLZ
```

After installing licensed Matterport3D Habitat assets under
`/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data/scene_datasets/mp3d/`,
run:

```bash
conda run -n navila-eval python examples/narrow_passage_rl/vln_r2r_preflight.py \
  --ensure-symlinks

cd /home/xiaotian/vla/NaVILA/evaluation
bash scripts/eval/r2r.sh \
  /media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/models/navila-llama3-8b-8f \
  1 0 "0"

python scripts/eval_jsons.py \
  ./eval_out/navila-llama3-8b-8f/VLN-CE-v1/val_unseen \
  1
```

## Current Paper-Safe Wording

Recommended:

> The current NaVILA checkpoint runs locally and produces parseable local action
> proposals, but controlled diagnostics reveal strong stop failure, forward
> action bias, weak left/right grounding, and visual perturbation sensitivity.
> We therefore use NaVILA as a semantic proposal module and place explicit
> geometry, failure-memory, and trusted-control gates below it.

Avoid:

- "NaVILA solves VLN on our robot."
- "The safety adapter improves VLN benchmark performance."
- "The model learned safe stopping/rejection."
- "The current R2R text probes demonstrate unseen-scene generalization."
