| Method | Domain | Train budget | Eval episodes | Success ↑ | Collision ↓ | Notes |
|---|---|---:|---:|---:|---:|---|
| PPO v2 geometry | Habitat HM3D mined-val | 5M steps | 151 | 0.060 | - | Existing SB3 checkpoint |
| SAC v2 geometry | Habitat HM3D mined-val | 2M steps | 151 | 0.000 | - | Existing SB3 checkpoint |
| GRU-PPO lightweight | Synthetic v2 | 5k steps | 50 | 0.000 | 0.980 | PyTorch fallback; SB3-Contrib unavailable in current env |
