| Baseline | Train data / budget | Eval episodes | Success ↑ | Collision ↓ | Notes |
|---|---:|---:|---:|---:|---|
| FSM expert trajectories | 40 episodes / 5179 transitions | 40 | 0.750 | 0.100 | Data source for BC/DAgger smoke |
| BC-FSM | 5179 expert transitions / 5 epochs | 40 | 0.500 | 0.475 | Supervised imitation of TurnCommitFSM |
| DAgger-FSM | 6346 transitions / 1 DAgger iter | 40 | 0.300 | 0.650 | Smoke run only; needs longer aggregation |
| RecurrentPPO | 1024 env steps | 40 | 0.000 | 0.000 | SB3-Contrib MlpLstmPolicy smoke |
| Replay Memory Policy | 1024 env steps | 40 | 0.000 | 0.025 | PPO + generic 8-D replay embedding smoke |
