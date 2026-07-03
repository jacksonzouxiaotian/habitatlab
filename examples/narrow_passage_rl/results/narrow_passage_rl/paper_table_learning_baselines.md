| Method | Domain | Train budget | Eval episodes | Success ↑ | Collision ↓ | Notes |
|---|---|---:|---:|---:|---:|---|
| SB3 PPO synthetic-to-Habitat geometry | Habitat HM3D mined-val | 5M synthetic steps | 151 | 0.060 | - | Synthetic v2 checkpoint transferred to HM3D |
| SAC v2 geometry | Habitat HM3D mined-val | 2M steps | 151 | 0.000 | - | Existing SB3 checkpoint |
| GRU-PPO lightweight | Synthetic v2 | 5k steps | 50 | 0.000 | 0.980 | PyTorch fallback; SB3-Contrib unavailable in current env |
| RecurrentPPO | Synthetic v2 | 3M steps | 500 | 0.130 | 0.456 | SB3-Contrib MlpLstmPolicy + fair reward |
| BC-FSM | Synthetic v2 | 5179 expert transitions | 40 | 0.500 | 0.475 | Supervised imitation smoke |
| DAgger-FSM | Synthetic v2 | 6346 transitions | 40 | 0.300 | 0.650 | One DAgger iteration smoke |
| Replay Memory Policy | Synthetic v2 | 1024 steps | 40 | 0.000 | 0.025 | PPO + generic replay embedding smoke |
