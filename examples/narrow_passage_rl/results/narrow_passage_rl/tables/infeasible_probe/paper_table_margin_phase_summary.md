# Table: Margin-Phase Summary

Regime summary by `delta_mean = d_hat - w_req_cons`. Episode counts are rows inside the plotted margin range [-0.30, 0.30] m.

| Method | Regime | Episodes | Success | Collision | Near collision | Reject | Correct reject | Timeout/stuck |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reactive rule baseline | Infeasible-side | 381 | 2.1% | 97.4% | 98.2% | 0.0% | 0.0% | 0.5% |
| Reactive rule baseline | Near-boundary | 453 | 6.2% | 91.8% | 99.1% | 0.0% | 0.0% | 2.0% |
| Reactive rule baseline | Feasible-side | 449 | 17.4% | 78.8% | 82.0% | 0.0% | 0.0% | 3.8% |
| DEGNAV-Rule | Infeasible-side | 381 | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| DEGNAV-Rule | Near-boundary | 453 | 12.4% | 83.4% | 97.6% | 0.0% | 0.0% | 4.2% |
| DEGNAV-Rule | Feasible-side | 449 | 43.9% | 27.6% | 57.2% | 0.0% | 0.0% | 28.5% |
| Conservative reject gate | Infeasible-side | 381 | 0.0% | 99.0% | 100.0% | 1.0% | 0.0% | 0.0% |
| Conservative reject gate | Near-boundary | 453 | 12.1% | 70.2% | 97.6% | 13.7% | 0.0% | 4.0% |
| Conservative reject gate | Feasible-side | 449 | 39.6% | 15.8% | 55.0% | 17.8% | 0.4% | 26.7% |

Notes:
- Margin is computed at the shared pre-divergence decision snapshot.
- Belief diagnostics for the rule baseline are used only for stratification and do not influence its actions.
- Bins with fewer than 5 episodes are marked low support.
- Reject denotes an explicit Reject mode, not generic failure.
