# 19-D Four-Mode PPO Audit and Reproduction Contract

Status: implementation and staged smoke validation. This document separates
historical results from the new strict-19D selector and does not claim a final
trained result before the corresponding CSV/JSON exists.

## Code facts and historical mismatch

The canonical Habitat vector is constructed in
`habitat-lab/habitat/tasks/narrow_passage/geometry.py` by the literal
`np.asarray([...], dtype=np.float32)` call. Its sensor space is `(19,)` in
`habitat-lab/habitat/tasks/narrow_passage/sensors.py`.

The historical learned selector is not this architecture. `BeliefModeEnv`
converts the 19-D vector to a 14-D belief vector (or a 6-D geometry ablation),
and `train_belief_mode_ppo.py` trains Stable-Baselines3 `MlpPolicy`, not a
recurrent network. Those old files and results are retained for provenance.
The existing three-seed diagnostic reports 26.7% success and mode usage
Commit 31.0%, Explore 69.0%, Recover 0.0%, Reject 0.0%. It must not be renamed
to “PPO-19D-GRU”.

The new path is deliberately separate:

```text
raw 19-D observation -> running mean/std -> Linear(19,128)+ELU
                     -> Linear(128,128)+ELU -> GRU(128,128)
                     -> Actor Linear(128 + memory_dim, 4)
                     -> Critic Linear(128 + privileged_dim, 1)
```

The base Actor observation remains 19-D. The optional memory vector is passed
through a separate model argument and is exactly
`[geometry_similarity, supported_failure_count, supported_success_count,
recurrence_risk]`. Simulator feasibility is used only as a training/evaluation
label and for reward/termination.

## Actual 19-D index table

No fixed manual feature scaling is applied before the new policy. The training
split supplies the running mean and variance; normalized values are therefore
unbounded before the implementation's numerical clip to `[-10,10]`.

| index | feature | source | unit | raw range | normalized range |
|---:|:---|:---|:---|:---|:---|
| 0 | d_left_near | 10th percentile, near-left depth ROI | m | [0,5] | running mean/std, clipped [-10,10] |
| 1 | d_center_near | 10th percentile, near-center depth ROI | m | [0,5] | same |
| 2 | d_right_near | 10th percentile, near-right depth ROI | m | [0,5] | same |
| 3 | d_left_far | 10th percentile, far-left depth ROI | m | [0,5] | same |
| 4 | d_center_far | 10th percentile, far-center depth ROI | m | [0,5] | same |
| 5 | d_right_far | 10th percentile, far-right depth ROI | m | [0,5] | same |
| 6 | clearance_left | min(left near, left far) | m | Habitat [0,5], procedural [-1,5] | same |
| 7 | clearance_right | min(right near, right far) | m | Habitat [0,5], procedural [-1,5] | same |
| 8 | passage_width | clearance_left + clearance_right | m | Habitat [0,10], procedural typically [0,10] | same |
| 9 | body_margin | min(clearance left/right) - radius | m | Habitat [-radius, 5-radius] | same |
| 10 | heading_error | goal/path bearing minus yaw, wrapped | rad | [-pi,pi] | same |
| 11 | lateral_offset | signed start-goal/path-line offset | m | unbounded | same |
| 12 | distance_to_local_goal | planar Euclidean distance | m | [0,+inf) | same |
| 13 | current_vx | Habitat measured planar displacement per task step; procedural commanded velocity | m/step or m/s | Habitat [0,+inf), procedural [-0.15,0.35] | same |
| 14 | current_wz | Habitat currently hard-coded 0; procedural commanded angular velocity | rad/s | Habitat {0}, procedural [-0.8,0.8] | same |
| 15 | stuck_score | translation-stuck counter / threshold | ratio | [0,1] | same |
| 16 | collision_flag | Habitat-Sim `collided` or procedural OBB contact | Boolean | {0,1} | same |
| 17 | previous_action_vx | Habitat normalized requested action; procedural prior velocity | dimensionless or m/s | Habitat [-1,1], procedural [-0.15,0.35] | same |
| 18 | previous_action_wz | Habitat normalized requested action; procedural prior angular velocity | dimensionless or rad/s | Habitat [-1,1], procedural [-0.8,0.8] | same |

Habitat normalized depth is converted back to metric units using its configured
physical `[0,10] m` interval and then clipped at 5 m before ROI extraction.
The near ROI uses rows 55–90%, the far ROI 25–55%, and columns are thirds.

Two cross-domain inconsistencies are now explicit rather than hidden:

- Habitat index 13 is displacement per task step, while procedural index 13 is
  the current command in m/s.
- Habitat index 14 is always zero. Habitat indices 17–18 cache normalized
  action arguments, while procedural indices 17–18 use physical commands.

These fields require a later versioned feature-contract migration before a
single checkpoint can be interpreted as having identical physical units in
both domains.

## Stuck, collision, previous action, and memory

Habitat translation stuck increments only when requested translation is above
0.02, planar motion is below 1 mm, and goal progress is below 1 mm. It decays
otherwise and is normalized by 20. Collision reads Habitat-Sim's previous
observation `collided` flag. The task caches requested normalized velocity
arguments before stepping. Rotation has a separate diagnostic counter and does
not contaminate translation stuck.

The original Habitat memory sensor is episode-local and returns
`[failure_count, failed_state_count, should_recover, should_reject]`. The
cross-episode implementation retrieves morphology-aware continuous neighbors
and stores only confirmed geometry-positive/negative evidence, but its legacy
four-vector is `[d_hat, n_similar_norm, prior_sr, cautious_flag]`. Neither is
silently presented as the new memory contract. The new learned selector uses
its own explicit four-field context, initially zero for the no-memory stage.

## Four modes, reward, and termination

The action mapping is fixed and asserted at startup:

| index | mode | deterministic realization |
|---:|:---|:---|
| 0 | COMMIT | forward motion plus bounded goal correction |
| 1 | EXPLORE | probing motion capped at 0.10 m/s plus clearance/goal steering |
| 2 | RECOVER | reverse motion plus re-alignment |
| 3 | REJECT | no low-level step; terminal selective-navigation outcome |

Reject is always distinct from timeout: it returns `terminated=True`,
`truncated=False`, does not call the wrapped environment's `step`, and records
`correct_reject`, `false_reject`, or `reject_unknown`. Recover remains a mode
decision and is not relabeled as collision recovery.

The initial selector reward is the requested simple scale: progress ×1,
success +10, collision -5, stuck -2, mode switch -0.05, step -0.01, correct
infeasible Reject +3, feasible Reject -5, justified Recover +1, unnecessary
Recover -0.5, and Commit after supported failures -1. Every step records the
raw mode, executed mode, selected probability, override reason, training-only
feasibility label, estimated margin, alignment readiness, memory support,
collision, stuck, timeout, and success.

## Why the historical selector learned 0% Reject and unreliable Recover

The failure is supported by code and logged behavior, not proof that PPO cannot
learn selective navigation:

1. Reject was registered as action 3 and no action mask existed, so registration
   was not the cause.
2. The old policy received 14-D instantaneous belief, not 19-D history. Its
   `memory_risk` stayed at the default zero in procedural-v2.
3. Only about 10% of the default environment distribution was false-feasible.
   With old rewards (+25 correct, -35 false), an indiscriminate entry-time
   Reject had strongly negative expected return.
4. False-feasible scenes intentionally look feasible at the entrance; the
   hidden blocker becomes distinguishable later. A memoryless selector lacks
   the requested temporal evidence.
5. Ideal kinematic motion means commanded movement normally occurs. Collision
   terminates immediately, so the stuck signal rarely creates pre-terminal
   Recover examples. Recover also moves backward and can lose progress reward.
6. There was no class-balanced BC initialization. The observed 31/69/0/0 mode
   distribution is therefore consistent with data/reward collapse.

## Staged commands

Use the existing `habitat` environment; it contains PyTorch, Gymnasium,
Stable-Baselines3, and sb3-contrib. The new lightweight runner only needs
PyTorch, NumPy, Gymnasium/Gym, and PyYAML. Current pytest is available in the
`navila-eval` environment, so tests are run with plugin auto-loading disabled.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xiaotian/miniconda3/envs/navila-eval/bin/python -m pytest -q \
  examples/narrow_passage_rl/tests/test_four_mode_selector.py \
  examples/narrow_passage_rl/tests/test_belief_state.py

/home/xiaotian/miniconda3/envs/habitat/bin/python \
  examples/narrow_passage_rl/collect_mode_selector_bc.py \
  --episodes 400 --max-steps 160 \
  --output-dir examples/narrow_passage_rl/results/mode_selector/bc_dataset_v1

/home/xiaotian/miniconda3/envs/habitat/bin/python \
  examples/narrow_passage_rl/train_mode_selector_bc.py \
  --dataset examples/narrow_passage_rl/results/mode_selector/bc_dataset_v1/mode_selector_bc.npz \
  --output-dir examples/narrow_passage_rl/results/mode_selector/bc_gru_seed1701 \
  --epochs 20 --batch-size 1024 --learning-rate 3e-4 \
  --weight-decay 1e-5 --early-stop-patience 3 --hidden-size 128 --history 8

/home/xiaotian/miniconda3/envs/habitat/bin/python \
  examples/narrow_passage_rl/train_mode_selector_ppo.py \
  --config examples/narrow_passage_rl/configs/mode_selector/ppo_20k_smoke.yaml \
  --bc-checkpoint examples/narrow_passage_rl/results/mode_selector/bc_gru_seed1701/bc_best.pt \
  --output-dir examples/narrow_passage_rl/results/mode_selector/bc_ppo_20k_seed1701
```

Only if the 20k run is numerically sound should the 100k gate be run. If one
mode exceeds 95% after 100k, stop before 300k/1M and inspect
`training_summary.json`, `steps.csv`, per-mode raw advantages, immediate
rewards, terminal reasons, and actor-head gradient norms.

## Domain scope

- The selector training and results in this directory are **procedural geometry
  simulation** unless a Habitat evaluator explicitly says otherwise.
- Habitat is a **scene-level navigation/stress evaluation** and is not evidence
  of quadruped joint/contact safety.
- Lite3 results must be labeled **physical validation** and cannot be inferred
  from either simulator.
