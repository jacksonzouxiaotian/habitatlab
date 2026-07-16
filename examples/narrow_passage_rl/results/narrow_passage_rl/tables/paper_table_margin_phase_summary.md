# Table: Margin-Phase Summary

Regime summary by `delta_mean = d_hat - w_req_cons`. Episode counts are rows inside the plotted margin range [-0.30, 0.30] m.

| Method | Regime | Episodes | Success | Collision | Near collision | Reject | Correct reject | Timeout/stuck |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reactive rule baseline | Infeasible-side | 2 | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| Reactive rule baseline | Near-boundary | 136 | 12.5% | 83.1% | 97.8% | 0.0% | 0.0% | 4.4% |
| Reactive rule baseline | Feasible-side | 786 | 18.7% | 70.6% | 79.3% | 0.0% | 0.0% | 10.7% |
| DEGNAV-Rule | Infeasible-side | 2 | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| DEGNAV-Rule | Near-boundary | 136 | 25.0% | 67.6% | 92.6% | 0.0% | 0.0% | 7.4% |
| DEGNAV-Rule | Feasible-side | 786 | 65.9% | 16.5% | 44.0% | 0.0% | 0.0% | 17.6% |

Notes:
- Margin is computed at the shared pre-divergence decision snapshot.
- Belief diagnostics for the rule baseline are used only for stratification and do not influence its actions.
- Bins with fewer than 5 episodes are marked low support.
- Reject denotes an explicit Reject mode, not generic failure.
