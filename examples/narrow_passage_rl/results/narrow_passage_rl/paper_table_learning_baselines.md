# Table: Learning Baselines and Diagnostic Smoke Runs

Rows labeled smoke/diagnostic verify code paths and failure modes. They should
be reported in the appendix or baseline-status section unless rerun with the
full multi-seed protocol. The main paper claim is that learning-only and generic
history baselines transfer poorly near narrow-passage feasibility boundaries.

| Method | Domain | Train budget | Eval episodes | Success ↑ | Collision ↓ | Notes |
|---|---|---:|---:|---:|---:|---|
| SB3 PPO synthetic-to-Habitat geometry | Habitat HM3D mined-val | 5M synthetic steps | 151 | 0.060 | - | Synthetic v2 checkpoint transferred to HM3D |
| SAC v2 geometry | Habitat HM3D mined-val | 2M steps | 151 | 0.000 | - | Existing SB3 checkpoint |
| TD3 synthetic-to-Habitat geometry | Habitat HM3D mined-val | 3M synthetic steps | 151 | 0.020 | 0.000 | Synthetic TD3 transfers poorly despite 77.8% synthetic v2 SR |
| TD3 Habitat-native | Habitat HM3D mined-val | 1M Habitat steps | 151 | 1.000 nominal / 0.020 strict | 0.000 | 98.0% success-but-unsafe; near-collision 100% |
| GRU-PPO lightweight | Synthetic v2 | 5k steps | 50 | 0.000 | 0.980 | PyTorch fallback; SB3-Contrib unavailable in current env |
| RecurrentPPO | Synthetic v2 | 3M steps | 500 | 0.130 | 0.456 | SB3-Contrib MlpLstmPolicy + fair reward |
| BC-FSM | Synthetic v2 | 5179 expert transitions | 40 | 0.500 | 0.475 | Diagnostic smoke; appendix only |
| DAgger-FSM | Synthetic v2 | 6346 transitions | 40 | 0.300 | 0.650 | Diagnostic smoke; appendix only |
| Replay Memory Policy | Synthetic v2 | 1024 steps | 40 | 0.000 | 0.025 | Generic replay embedding smoke; appendix only |
