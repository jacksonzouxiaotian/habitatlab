# EAGOR current-episode spherical direction belief

Latest matched stop diagnosis: [`STOP_DIAGNOSIS_10SCENES_CN.md`](STOP_DIAGNOSIS_10SCENES_CN.md).
All 100 episodes were rerun with frozen navigation. Oracle Stop: GT 6/10,
EAGOR/Grid/Centroid 4/10 each, exploration 2/10; common-task comparisons do
not show an EAGOR first-reach efficiency advantage. GT pose and Oracle Semantic remain in use.

Latest online-map/frontier/A* development and fixed-ten-task results (2026-09-11):
[`ONLINE_NAVIGATION_10SCENES_CN.md`](ONLINE_NAVIGATION_10SCENES_CN.md).
GT direction + online planner + Oracle stop reaches 6/10; with the unchanged
Area stop, EAGOR remains 0/10 (Centroid 1/10). These are distinct conditions.
The report includes actual videos, figures, privilege boundaries and rerun commands.

This directory is an isolated, runnable reproduction of the episode-local
Spherical Harmonic Belief Field (SH-BF) in
[EAGOR: Embodied Reasoning in Omni-direction](https://arxiv.org/abs/2607.06165)
(arXiv:2607.06165v1). It does not import or modify the repository's
narrow-passage, PPO, or failure-memory implementations.

For a Chinese group-meeting and paper-ready report with explicit figure and
video insertion points, see
[`README_GROUP_MEETING_CN.md`](README_GROUP_MEETING_CN.md).

For the completed MP3D Phase 1-4 direction/planner/controller/stop attribution,
including the A-I matrix and Oracle-navmesh upper bound, see
[`FAILURE_ATTRIBUTION_PHASE1_4_CN.md`](FAILURE_ATTRIBUTION_PHASE1_4_CN.md).

The current corrected 10-unique-scene attribution is documented in
[`FAILURE_ATTRIBUTION_10SCENES_CN.md`](FAILURE_ATTRIBUTION_10SCENES_CN.md).
With GT direction and GT stop, direct and single-frame depth-local navigation
remain at 0/10, whereas Habitat's discrete navmesh follower reaches 10/10
(SPL 0.978, zero collisions). This supersedes the older heading-only row-I
upper-bound number; rows A-H are unaffected.

The implementation targets the audited local stack:

- Python 3.9.19 in conda environment `habitat`
- Habitat-Lab 0.3.3 (editable checkout)
- Habitat-Sim 0.3.3, native `EquirectangularSensorSpec` available
- Hydra 1.3.2 and OmegaConf 2.3.0
- SciPy 1.13.1, NumPy 1.26.4, PyTorch 2.8.0+cu128
- ObjectNav discrete `stop`, `move_forward`, `turn_left`, `turn_right` actions

The installed Habitat-Sim build reports `cuda_enabled=False`; all math and
Oracle code therefore works on CPU. Rendering still succeeded through the
machine's OpenGL RTX A4000 context.

## Architecture

```text
native ERP RGB + semantic / recorded / optional local Qwen
                         |
                  likelihood [0,1]
                         |
             ERP pixels -> unit sphere + dOmega
                         |
       log likelihood -> cached real-SH projection (L=7)
                         |
 previous posterior -> viewer-frame pose propagation
                         |
             paper/decayed coefficient update
                         |
             paper/probability direction decode
                         |
       shared discrete fixed-step Habitat controller
                         |
        CSV + summary JSON + debug MP4 + metrics
```

The four matched policies are `centroid`, `circular_centroid`, `grid`, and
`eagor`. They receive the same panorama/likelihood and use the same controller.

## Coordinates and rotation

The internal viewer frame is right-handed: `+x` is front, `+y` is left, and
`+z` is up. Positive azimuth and positive yaw point/turn left. The ERP equations
are the paper's Eq. (1):

```text
theta = 2*pi*u/W - pi
phi   = pi/2 - pi*v/H
omega = (cos(phi)cos(theta), cos(phi)sin(theta), sin(phi))
```

Grid samples use pixel centers. Exact latitude-band solid angles are cached;
their sum is exactly `4*pi`, and polar pixels receive less area than equatorial
pixels. Habitat-Sim 0.3.3's native ERP has increasing `u` toward viewer-right.
RGB and semantic observations are therefore horizontally flipped once so their
increasing `u` agrees with EAGOR's positive-left convention. This was checked
against a co-located perspective sensor on the bundled van-gogh-room scene.

Habitat local `-z/-x/+y` (front/left/up) is converted to EAGOR `+x/+y/+z`.
Given world-from-body rotations, prior propagation uses exactly
`R_current_from_previous = R_WB,t.T @ R_WB,t-1`. Thus, after an agent left turn
of 90 degrees, a previously front target appears at current right (`-y`).

Planar yaw uses exact real-SH coefficient block rotations. The general SO(3)
interface rotates spherical sampling directions and reprojects them; see the
reproduction-completion notes below.

## Real spherical-harmonic convention

`spherical/spherical_harmonics.py` uses SciPy's orthonormal complex SH including
the Condon-Shortley phase, with polar angle `pi/2-elevation`. It converts this to
a real basis as follows:

- `m < 0`: sine component, `sqrt(2)*(-1)^m*Im(Y_l^|m|)`
- `m = 0`: zonal component
- `m > 0`: cosine component, `sqrt(2)*(-1)^m*Re(Y_l^m)`
- order: `(0,0), (1,-1), (1,0), (1,1), ...`

Consequently degree one aligns with `+y`, `+z`, `+x` at `m=-1,0,+1` up to the
common normalization. The cached projection is
`Y.T @ (field.reshape(-1) * area_weights)`.

## What follows the paper, and what does not

Implemented directly from the paper:

- full-sphere ERP lifting with the stated azimuth/elevation equations;
- solid-angle correction in SH projection;
- `[0,1]` likelihood normalization, `log(likelihood + epsilon)`;
- real SH belief with default bandlimit `L=7` (64 coefficients);
- viewer-frame pose propagation and additive coefficient update;
- first observation initializes the posterior;
- degree-one/spherical-moment direction decode;
- no explicit translation correction, matching the paper's statement that its
  fixed steps keep parallax within observation noise.

The v1 PDF repeatedly points to supplementary sections, but its downloaded
12-page arXiv artifact contains no supplementary material. It does not specify
the real-SH signs/order, Wigner-D Euler convention, VLM response-map extraction,
confidence calibration, controller, stop rule, grid interpolation, or Habitat
sensor convention. The following are explicit reproduction completions:

- the real-SH convention above;
- exact analytic yaw blocks and an interpolation/reprojection SO(3) fallback
  instead of claiming an undocumented e3nn/Wigner-D convention;
- native Habitat ERP horizontal normalization;
- Oracle semantic, recorded, and JSON-box/point Qwen adapters;
- Gaussian smoothing, controller thresholds, target-area stop heuristic, CSV,
  metrics, video layout, and baseline definitions;
- grid baseline uses bilinear interpolation and circular horizontal padding;
- absent observations default to `propagate_only`;
- `decayed` update is an engineering ablation and never the paper default.

There is also an ambiguity in paper Eq. (6): `f_t` is described as a signed
log-posterior, but the paper divides its moment by `integral(f_t)` and claims a
resultant in `[0,1]`. A signed field cannot generally satisfy that claim, and a
spherical mean is not generally a MAP estimate. `decode_mode: paper` keeps the
signed degree-one moment but uses the minimally shifted non-negative field only
for confidence normalization. `decode_mode: probability` instead computes the
mathematically standard softmax probability with log-max stabilization and
solid-angle mass. Both modes are separate and logged.

Oracle is debugging/upper-bound perception only. It must not be described as a
zero-shot result. Qwen scores are marked uncalibrated when the model omits an
explicit confidence; equal weights are then used rather than inventing scores.

## Verified commands

Run from the repository root. These commands use the environment that was
actually inspected and tested.

```bash
cd /home/xiaotian/navigation/habitat-lab
conda run -n habitat python -m eagor_repro.scripts.run_tests
```

If running from another directory, set the repository on `PYTHONPATH` instead:

```bash
conda run -n habitat env PYTHONPATH=/home/xiaotian/navigation/habitat-lab \
  python -m eagor_repro.scripts.run_tests
```

The environment lacks `pytest`, so `run_tests` executes the same plain assertion
functions without installing or changing packages. If pytest is later installed,
the files are standard pytest-discoverable tests under `eagor_repro/tests/`.

Run synthetic seam/occlusion comparisons plus a native-ERP closed loop on the
bundled Habitat scene:

```bash
conda run -n habitat python -m eagor_repro.scripts.run_smoke_test \
  --output-root /tmp/eagor_results/smoke --max-steps 50
```

The bundled scene has no semantic annotations. This smoke therefore uses a
clearly labelled synthetic waypoint-oracle likelihood and is not an HM3D
ObjectNav or zero-shot result. It also resets and steps a real Habitat-Lab Env
with the configured native ERP sensor.

Run the requested waypoint initial-heading sweep (`0,45,90,135,179` degrees):

```bash
conda run -n habitat python -m eagor_repro.scripts.evaluate_waypoints \
  --output-root /tmp/eagor_results/waypoints --max-steps 50
```

Generate the longer object-search video from the locally bundled ReplicaCAD
apartment:

```bash
conda run -n habitat python -m eagor_repro.scripts.run_long_object_search \
  --output-root eagor_outputs/long_object_search
```

This episode searches for the real ReplicaCAD television semantic IDs `82/87`
using native 512x256 RGB and semantic ERP observations.  It follows a fixed,
target-independent five-waypoint coverage route through both apartment levels;
only after a reliable target mask is observed does control switch to the EAGOR
SH direction.  The default route is about 41 m and the output dashboard is
1280x720 at 10 FPS.  A furniture-aware navmesh is rebuilt at startup so the
agent does not clip through sofas, cabinets, or the television.  Outputs are:

- `apt_0_television_long_search.mp4`: perspective RGB, ERP target overlay,
  current likelihood, SH belief, phase/action telemetry, and trajectory;
- `apt_0_television_steps.csv`: one row per discrete action;
- `apt_0_television_summary.json`: episode settings and aggregate results.

This is an Oracle semantic debugging upper bound, not a zero-shot or HM3D
ObjectNav score.  No target coordinate enters either the patrol or EAGOR
controller.

The verified local default run produced 333 frames (33.3 s), detected the TV at
step 324 after 39.21 m of exploration, stopped after 39.93 m, and recorded zero
collisions.  The final MP4 is 1280x720 and about 5.8 MiB.

### MP3D multi-scene ObjectNav diagnostic

The external drive now contains all 90 Habitat-format MP3D scenes and the
ObjectNav MP3D v1 train/val/val_mini episodes. The scene archive passed a full
ZIP CRC check before extraction. Machine-specific paths, three-unique-scene
selection, 300-step episodes, Oracle semantics, and output location are defined
in `configs/eagor/mp3d_oracle.yaml`.

Check readiness and run the matched four-policy batch with:

```bash
conda run -n habitat python -m eagor_repro.scripts.evaluate_eagor \
  --config configs/eagor/base.yaml \
  --overlay configs/eagor/mp3d_oracle.yaml --preflight

conda run -n habitat python -m eagor_repro.scripts.evaluate_baselines \
  --config configs/eagor/base.yaml \
  --overlay configs/eagor/mp3d_oracle.yaml \
  --policies centroid circular_centroid grid eagor
```

The verified preflight reports `ready: true`, `scene_dataset: mp3d`, and 90
scenes. The batch samples one episode from each of three distinct val scenes:

- `EU6Fwq7SyZv`, target `cabinet`;
- `2azQ1b91cZZ`, target `cabinet`;
- `8194nk5LbLH`, target `gym_equipment`.

Each policy writes to its own `centroid/`, `circular_centroid/`, `grid/`, or
`eagor/` subtree under `eagor_outputs/mp3d_objectnav_multiscene`, and episode
filenames contain both scene and episode ID. This prevents policies or scenes
with repeated `episode_id=0` from overwriting each other. The 1280x720 video
dashboard shows forward RGB, native ERP, current likelihood, policy belief,
trajectory, and action telemetry. Representative EAGOR videos are:

- [`EU6Fwq7SyZv / cabinet`](../eagor_outputs/mp3d_objectnav_multiscene/eagor/videos/episode_EU6Fwq7SyZv_0.mp4);
- [`2azQ1b91cZZ / cabinet`](../eagor_outputs/mp3d_objectnav_multiscene/eagor/videos/episode_2azQ1b91cZZ_0.mp4);
- [`8194nk5LbLH / gym_equipment`](../eagor_outputs/mp3d_objectnav_multiscene/eagor/videos/episode_8194nk5LbLH_0.mp4).

Ten of the twelve policy/scene videos contain 300 frames (30 s at 10 FPS).
Grid and circular-centroid incorrectly call STOP on the `gym_equipment`
episode, so those two failure videos end at 24 frames (2.4 s) and 40 frames
(4.0 s), respectively.

The three-scene aggregate is:

| Policy | SR | Mean steps | MAE (deg) | Temporal inconsistency (deg/step) | Mean collisions | Latency (ms/step) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Centroid | 0.0 | 300.0 | 54.79 | 4.79 | 23.33 | 7.09 |
| Circular centroid | 0.0 | 213.33 | 47.40 | 4.02 | 16.67 | 7.57 |
| Grid belief | 0.0 | 208.0 | 38.43 | 2.02 | 16.33 | 9.45 |
| EAGOR SH-BF | 0.0 | 300.0 | 52.15 | 2.37 | 28.33 | 23.95 |

![MP3D four-policy Oracle diagnostic](../eagor_outputs/mp3d_objectnav_multiscene/summaries/baseline_comparison.png)

Machine-readable and Markdown results are
[`baseline_comparison.csv`](../eagor_outputs/mp3d_objectnav_multiscene/summaries/baseline_comparison.csv)
and
[`baseline_comparison.md`](../eagor_outputs/mp3d_objectnav_multiscene/summaries/baseline_comparison.md).

This is a useful negative result, not a paper benchmark reproduction. All four
policies receive perfect semantic category masks, but none succeeds on these
three episodes. The videos isolate several limitations:

- ObjectNav accepts any target instance, but a category likelihood can contain
  multiple widely separated instances. A centroid or spherical mean may lie
  between them rather than point to a navigable instance. Directional GT is
  therefore scored against the minimum angular error over all valid instances,
  strictly in the evaluation branch.
- A target direction does not encode walls, doors, traversability, or a route.
  The direction-only fixed-step controller repeatedly collides even when the
  semantic direction itself is stable.
- The target-area stop heuristic can trigger on a large visible mask while the
  agent remains far from Habitat's success region, explaining the two short
  false-STOP videos.
- EAGOR is much more temporally stable than the naive centroid here, but does
  not beat the grid baseline in MAE on this very small sample and costs more
  update time. No positive claim should be made from three scenes.

When confidence is low, the MP3D overlay enables a target-independent
forward/collision-turn exploration fallback shared by all policies. This is an
explicit engineering completion for long episodes, not an EAGOR paper method
and not a substitute for obstacle-aware local planning.

Check HM3D v2 readiness without opening an environment:

```bash
conda run -n habitat python -m eagor_repro.scripts.evaluate_eagor --preflight
```

Once HM3D v2 scenes and episodes are mounted at the configured paths, run one
Oracle episode and then the matched four-policy batch:

```bash
conda run -n habitat python -m eagor_repro.scripts.evaluate_eagor \
  --config configs/eagor/base.yaml --overlay configs/eagor/oracle_debug.yaml \
  --policy eagor

conda run -n habitat python -m eagor_repro.scripts.evaluate_baselines \
  --config configs/eagor/base.yaml \
  --policies centroid circular_centroid grid eagor \
  --override evaluation.num_episodes=10
```

Switch reproduction/ablation modes without editing code:

```bash
conda run -n habitat python -m eagor_repro.scripts.evaluate_eagor \
  --override sh_belief.update_mode=decayed \
  --override sh_belief.belief_decay=0.97 \
  --override sh_belief.decode_mode=probability
```

For local Qwen, set `likelihood.qwen.model_name_or_path` to an already-installed
model directory and use `configs/eagor/qwen_eval.yaml`. No model or network API
is imported unless that backend is selected. Generation has `max_time`, bounded
tokens, and fail-open weak observations; model errors are preserved in metadata.

## Metrics and outputs

The configured output is isolated under
`<output.root>/<policy>/{raw,summaries,videos,plots,logs}`. Each episode writes
step CSV, summary JSON, and optionally MP4. Videos contain perspective RGB,
normalized ERP, likelihood, belief, predicted direction, evaluation-only GT
direction, action/confidence/error, and top-down x-z trajectory.

Direction metrics are angular error, mean angular error, seam-only error,
occlusion-only error, confidence/error/visibility/success rows, and temporal
consistency. Here temporal consistency is explicitly
`abs(angle(pred_t,pred_t-1) - angle(gt_t,gt_t-1))` in degrees per step.
Navigation summaries contain success, SPL, steps, path length, final distance,
collision count, stop correctness, seam-crossing success, and mean perception +
belief-update latency. GT direction is used only for metrics/video; the policy
and controller receive only likelihood-derived prediction and confidence.

## Two current SH-BF weaknesses

The following are controlled synthetic characterizations of the current
paper-style update, **not EAGOR benchmark results**. They isolate belief-filter
behavior from Habitat scene layout, VLM quality, and navigation policy. Rebuild
the figures and their underlying CSV data with:

```bash
conda run -n habitat python -m eagor_repro.scripts.analyze_belief_weaknesses
```

### Weakness 1: temporally correlated evidence is over-counted

The paper update, also implemented in `spherical/belief_filter.py`, is
`c_t = rotate(c_{t-1}) + b_t`. It has no observation-correlation model, evidence
discount, or bound on coefficient magnitude. Applying the exact same broad
single-frame likelihood for 60 steps therefore multiplies its contribution 60
times even though no independent information has been added.

In the diagnostic, coefficient L2 norm grows from `23.335` to `1400.108`
(exactly `60x`) and probability-decode confidence rises from `0.770` to `0.988`.
The control that observes the frame once and then only propagates remains at
`0.770`. The clean target direction remains correct: this demonstrates an
uncalibrated certainty/state-magnitude failure, not a directional error in this
particular example. Highly correlated adjacent video frames are less extreme
than exact duplicates, but the filter has no mechanism that distinguishes the
two cases.

![Repeated correlated observations make the coefficients and confidence grow](../eagor_outputs/belief_weaknesses/weakness_1_repeated_evidence.png)

Raw data:
[CSV](../eagor_outputs/belief_weaknesses/weakness_1_repeated_evidence.csv).
Candidate engineering ablations include correlation-aware update weights,
effective-sample-size or magnitude caps, evidence decay, and calibrated
confidence. None is part of the paper-default reproduction.

### Weakness 2: rotation-only propagation cannot correct translation parallax

The stored SH field represents direction only. Propagation receives relative
rotation but no agent translation or target range/depth. If the camera keeps a
fixed orientation while translating during an occlusion, the coefficients stay
unchanged although the correct egocentric bearing changes.

The diagnostic initializes a stationary target 4 m in front, hides it, and
moves the agent laterally by 4 m without rotating. The true bearing changes from
`0 deg` to `-45 deg`; SH-BF continues predicting `0 deg`, producing `45 deg`
error, while the maximum coefficient change is exactly `0`. This deliberately
stresses the model beyond the paper's fixed-step assumption that parallax stays
within single-frame observation noise. A translation-aware filter would need
range/depth or a 3D target-position belief, likely fused across sensors.

![Translation changes bearing while rotation-only SH propagation stays fixed](../eagor_outputs/belief_weaknesses/weakness_2_translation_parallax.png)

Raw data:
[CSV](../eagor_outputs/belief_weaknesses/weakness_2_translation_parallax.csv).
Both diagnostics also write machine-readable aggregate values to
[`belief_weaknesses_summary.json`](../eagor_outputs/belief_weaknesses/belief_weaknesses_summary.json).

Weakness 2 agrees with a limitation explicitly acknowledged by the authors.
Weakness 1 is an additional characterization found in this reproduction; the
paper does not report it as a failure case.

## Paper-reported failure cases and limitations

The following values reproduce Table 4 of the
[EAGOR arXiv v1 paper](https://arxiv.org/abs/2607.06165), evaluated on HOS with
Qwen2.5-VL-7B. They are author-reported results, not outputs of this repository:

| Failure mode | Episodes | Success rate (%) |
| --- | ---: | ---: |
| OCR / fine-grained text | 18.5 | 13.2 |
| Rare target | 16.9 | 3.6 |
| Multi-instance confusion | 12.5 | 33.3 |
| VLM false detection | 11.9 | 13.6 |
| Overall | 59.9 | 40.1 |

The authors attribute the central limitation to dependence on VLM directional
likelihoods: semantic ambiguity and perception errors directly contaminate the
belief. Their named future directions are grounded detection for distinguishing
multiple instances, translation-aware multi-sensor estimation for parallax, and
more efficient bio-inspired perception. Two additional disclosed trade-offs
are:

- the default `L=7` bandlimit gives about `25 deg` angular resolution; increasing
  `L` sharpens the peak but increases sidelobes/Gibbs ringing;
- reported runtime is about `1.6 s` versus `1.0 s` for the baseline with
  Qwen2.5-VL-7B, approximately `0.6x` additional inference time.

These paper failures mostly originate in perception or target ambiguity. The
two diagnostics above instead probe state-estimation assumptions, so the two
sets should not be merged into one benchmark claim.

## Current blocker and scope

The repository `data` symlink still points to the unavailable
`/mnt/sda1/habitat_data`. The MP3D overlay avoids that link by using explicit
paths on `/media/xiaotian/ACD525D1B7A093D9`; MP3D evaluation is ready and has
been run. The default HM3D configuration remains unavailable because HM3D v2
scenes and episodes have not been mounted. No HM3D result is fabricated.

This module intentionally contains no cross-episode failure memory, PPO
training, narrow-passage strategy, or ROS 2/Lite3 control. Those can consume the
stable episode-local `DirectionPrediction` interface in a later stage without
changing the spherical math.
