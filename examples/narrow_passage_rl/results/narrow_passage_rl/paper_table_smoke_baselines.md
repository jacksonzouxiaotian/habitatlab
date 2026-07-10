# Table: Smoke / Appendix-Only Baselines

These runs verify code paths and provide preliminary negative controls.  They
should stay in the appendix or artifact status section unless rerun with the
same multi-seed, same-eval-domain protocol as the formal main baselines.

| Method | Purpose | Train domain | Eval domain | Train budget | Seeds | Eval episodes | Success metric | SR | Collision | Status / Notes |
|:---|:---|:---|:---|:---|:---:|---:|:---|---:|---:|:---|
| GRU-PPO lightweight | Generic recurrent-memory negative control | Synthetic v2 | Synthetic v2 | 5k steps | 1 | 50 | Synthetic v2 success | 0.0% | 98.0% | Lightweight PyTorch fallback; appendix only |
| RecurrentPPO | SB3-Contrib recurrent policy smoke | Synthetic v2 | Synthetic v2 | 3M steps | 1 | 500 | Synthetic v2 success | 13.0% | 45.6% | MlpLstmPolicy + fair reward; runs end-to-end but not a final multi-seed baseline |
| BC-FSM | Imitation baseline smoke | Synthetic v2 FSM expert | Synthetic v2 | 5,179 expert transitions | 1 | 40 | Synthetic v2 success | 50.0% | 47.5% | Appendix only; needs full expert dataset and multi-seed rerun before main table use |
| DAgger-FSM | Interactive imitation baseline smoke | Synthetic v2 FSM expert + DAgger rollout | Synthetic v2 | 6,346 transitions | 1 | 40 | Synthetic v2 success | 30.0% | 65.0% | Appendix only; preliminary diagnostic |
| Replay Memory Policy | Generic replay-input baseline smoke | Synthetic v2 + replay observation wrapper | Synthetic v2 | 1,024 steps | 1 | 40 | Synthetic v2 success | 0.0% | 2.5% | Tests "history input only"; not geometry-guided failure memory |

Interpretation:
- These baselines are useful for appendix discussion but should not be mixed
  into the formal Habitat main table.
- Recurrent hidden state, imitation from FSM, and generic replay input are
  different from the proposed geometry-guided failure memory mechanism.
