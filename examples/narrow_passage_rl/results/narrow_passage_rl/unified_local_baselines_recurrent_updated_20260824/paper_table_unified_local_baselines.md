# Paired Procedural-v2 Local-Baseline Rerun

| Method | Success ↑ | Collision ↓ | Near collision ↓ | Timeout/stuck ↓ | Correct reject ↑ | False reject ↓ | Avg steps ↓ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Geometry rule | 23.3±1.9% | 70.3±2.2% | 38.6±1.6% | 6.3±0.5% | — | — | 60.0±2.6 |
| Direct-control PPO | 84.3±0.5% | 15.7±0.5% | 14.9±0.4% | 0.0±0.0% | — | — | 68.5±0.1 |
| Recurrent PPO | 0.1±0.1% | 20.3±0.3% | 19.1±0.2% | 79.7±0.4% | — | — | 323.8±1.1 |
| Recurrent PPO (Direct-init + PPO fine-tune) | 85.0±0.9% | 15.0±0.9% | 14.1±0.7% | 0.0±0.0% | — | — | 68.3±0.2 |
| DEGNAV + geometry-guided memory | 83.1±0.2% | 0.0±0.0% | 45.1±2.2% | 16.9±0.2% | 0.0±0.0% | 0.0±0.0% | 188.5±1.7 |

## Success by corridor type

| Method | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False feasible |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Geometry rule | 59.6±0.3% | 1.8±0.6% | 0.9±0.6% | 29.0±1.3% | 34.1±2.7% | 29.8±5.6% | 0.0±0.0% |
| Direct-control PPO | 96.7±1.1% | 100.0±0.0% | 100.0±0.0% | 97.1±1.0% | 91.7±1.5% | 61.8±4.6% | 0.0±0.0% |
| Recurrent PPO | 0.0±0.0% | 0.0±0.0% | 0.0±0.0% | 0.0±0.0% | 0.6±0.8% | 0.0±0.0% | 0.0±0.0% |
| Recurrent PPO (Direct-init + PPO fine-tune) | 98.4±1.2% | 100.0±0.0% | 100.0±0.0% | 97.0±0.3% | 93.8±3.1% | 63.8±4.5% | 0.0±0.0% |
| DEGNAV + geometry-guided memory | 100.0±0.0% | 100.0±0.0% | 98.9±0.9% | 100.0±0.0% | 99.4±0.9% | 32.9±8.7% | 0.0±0.0% |

## Protocol

- Environment: current `HarderNarrowPassageEnv` / Procedural v2.
- Paired evaluation seeds: `[42, 43, 44]`; `500` episodes per evaluation seed and method.
- All methods receive the same ordered episode seeds, width range, maximum step budget, and seven corridor types.
- Values are mean±population-std across evaluation-scene seeds, not across independent RL training seeds.
- `Strict success` means goal success without simulator collision or body overlap; near collision is the procedural body-margin diagnostic.
- `Correct reject` is conditioned on `false_feasible` episodes; `False reject` is conditioned on all other corridor types.
- Direct and recurrent policies have no explicit Commit/Explore/Recover/Reject interface; their mode-specific fields are unavailable.
- This is a new paired rerun and does not overwrite the existing canonical tables.
- Direct-control PPO checkpoint steps: `3006464`; one training seed.
- Recurrent PPO checkpoint steps: `3006464`; one training seed.
- Recurrent PPO (Direct-init + PPO fine-tune) checkpoint steps: `606496`; one training seed.
- Recurrent training note: Direct PPO behavior distillation on 300 evaluation-isolated training episodes (seed 1701), followed by 106,496 RecurrentPPO interactions at learning rate 1e-5; cumulative counter 606,496.
