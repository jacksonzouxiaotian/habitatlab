# Table: Habitat HM3D Anchor Validation
# Original val set (A): 157 episodes, 20 held-out scenes (NarrowPassageNav-v0)
# Mined val set (B):    151 episodes automatically mined from 20 held-out HM3D val scenes
#   Mining: geodesic/clearance-based; 73 narrow / 55 normal / 23 wide (608 train / 151 val)
# PPO results: mean (std) across 3 independent training seeds
# FSM/APF results: single run (deterministic policies — no variance)
# Interpretation: unperturbed mined anchors are mostly well aligned.  The 100%
# FSM rows are nominal anchor-validation results, not a blanket robustness
# claim.  Stress validation isolates module sensitivity.

# --- Inference speed (CPU, n=10k calls, separate from accuracy table) ---
# fsm_action:       20.9 µs  (7.1× faster than SB3 PPO v1)
# fsm_turn_commit:  21.5 µs  (6.9× faster)
# fsm_cross_memory: 26.0 µs  (5.7× faster, includes memory lookup)
# ppo_sb3_v1:      148.0 µs  (baseline)

## Val set A (original 157 episodes)

| Method | SR | Narrow | Normal | Wide | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| PPO-SB3 baseline (fixed formula) | 0.0% | 0.0% | 0.0% | 0.0% | Trained on synthetic 2D v1 env; obs[10] mismatch vs. Habitat |
| PPO v2 (geometry sensor, re-trained) | 5.7% | 5.1% | 9.1% | 0.0% | Single seed, v2 env + heading fix, 5M steps |
| **PPO w/ geometry sensor** | **2.1% (±2.6%)** | **1.7% (±1.6%)** | **3.0% (±4.3%)** | **1.4% (±2.0%)** | 3 seeds: 5.7%, 0.0%, 0.6% |
| APF+Gap (Khatib 1986) | 93.6% | 92.4% | 100% | 82.6% | Depth + GPS only |
| **Geometry-FSM (ours)** | **100%** | **100%** | **100%** | **100%** | Nominal anchors |
| **FSM + Failure Memory (ours)** | **100%** | **100%** | **100%** | **100%** | Memory gain is not visible on one-shot passable anchors |

## Val set B (mined 151 episodes, 20 held-out HM3D scenes)

| Method | SR | Narrow (73) | Normal (55) | Wide (23) | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| PPO v2 (geometry sensor) | 6.0% | 5.5% | 9.1% | 0.0% | 9/151; consistent with val set A (5.7%) |
| SAC v2 (geometry sensor) | 0.0% | 0.0% | 0.0% | 0.0% | 0/151; same observation interface as PPO v2 |
| TD3 v2 synthetic transfer | 2.0% | 4.1% | 0.0% | 0.0% | 3/151; action-mapped synthetic-to-Habitat transfer remains poor |
| TD3 Habitat-native | 100% nominal / 2.0% strict | 100% / 0.0% | 100% / 0.0% | 100% / 13.0% | 1M Habitat steps; 98.0% success-but-unsafe, near-collision 100% |
| **Geometry-FSM (ours)** | **100%** | **100%** | **100%** | **100%** | 151/151 |
| FSM w/o recovery | **100%** | **100%** | **100%** | **100%** | Extreme-narrow subset: 24/24 |
| FSM w/o alignment | **100%** | **100%** | **100%** | **100%** | Extreme-narrow subset: 24/24 |

## Habitat FSM Stress Ablation (+60° start-heading perturb, 24 extreme-narrow episodes)

| Variant | SR | Successes | Avg steps | Notes |
|:---|:---:|:---:|:---:|:---|
| **Full FSM** | **100%** | **24/24** | 55.3 | Heading alignment recovers the perturbation |
| FSM w/o recovery | **100%** | **24/24** | 55.3 | Recovery is not triggered on these anchors |
| FSM w/o all alignment | 0.0% | 0/24 | 124.7 | Cannot reorient to the corridor/goal |
| FSM w/o heading alignment | 0.0% | 0/24 | 1.0 | Zero heading correction causes immediate failed stop |
| FSM w/o lateral alignment | **100%** | **24/24** | 55.3 | Lateral centering is not the bottleneck here |

## Experiment ①: Cross-Episode Memory (multi-agent, 8 agents × 100 corridors)

Setup: 60 passable (STRAIGHT/L/S/NARROW_EXIT/NARROW_ENTRY/ASYMMETRIC) + 40 false_feasible (impassable)
corridors; same seeds repeated across 8 rounds (agents). CrossEpisodeMemory with exact fingerprint
matching (fp_radius=0, min_similar_for_reject=2), rejection checked pre-episode only.

| Condition | R1 wasted | R2 wasted | R3-R8 wasted | FF reject R1 | FF reject R2 | FF reject R3+ | Pass SR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cross-episode memory** | 7,500 | 900 | **0** | 37.5% | 92.5% | **100%** | **90%** |
| Local (intra-episode) memory | 12,000 | 12,000 | 12,000 | 0% | 0% | 0% | 90% |
| No memory (baseline) | 12,000 | 12,000 | 12,000 | 0% | 0% | 0% | 90% |

Cumulative steps saved by cross_memory vs no_memory over 8 rounds: **87,600 steps**
(R1: 4,500 saved; R2 cumulative: 15,600; R3+: 12,000/round).

Cross_memory passable SR stays at 90% throughout all rounds (no false rejections of passable corridors).
R1 rejection of 37.5% FF occurs because FF seeds processed later in R1 already find matching failures
from earlier FF seeds in the same round (sequential within-round learning).

Notes:
- PPO-SB3 0.0% with fixed heading formula.  Pre-fix results used a buggy
  atan2 sign convention that inverted turn direction and inflated success rate
  artificially; do not report those legacy numbers.
  Root cause of 0%: obs[10] format mismatch — SB3 trained on v1 env (absolute yaw),
  Habitat sensor emits path-relative heading error → policy receives garbage input.
- PPO v2 (5.7%) confirms heading fix alone does not close the sim-to-real gap;
  high variance (0–9.1% across difficulties) reflects sparse reward in scanned scenes.
- SAC v2 also fails to transfer on mined Habitat val (0/151), despite matching the v2
  observation format. TD3 trained in synthetic v2 reaches high synthetic SR but
  transfers to only 2.0% HM3D SR. Habitat-native TD3 can optimize the nominal
  distance/alignment success measure (100% SR), but strict clearance-aware SR is
  only 2.0%, with 98.0% success-but-unsafe and 100% near-collision.
- PPO w/ geometry sensor 3-seed SR: 5.7%, 0.0%, 0.6% — high variance confirms
  training instability and sim-to-real collapse (99%+ train SR → <6% Habitat SR).
- collision_rate=0% for all methods due to allow_sliding=False (navmesh boundary stop).
- min_clearance<0 is a depth-sensor artifact at valid navmesh positions near walls,
  not physical wall penetration.
- FSM ablation (full / no_recovery / no_alignment): all 100% on unperturbed val;
  the +60° Habitat stress ablation isolates heading alignment as the critical module.
- Failure memory should be argued primarily from repeated false-feasible /
  cross-episode memory experiments, not from the one-shot Habitat passable-anchor
  table where Geometry-FSM and FSM+Memory both reach 100%.
