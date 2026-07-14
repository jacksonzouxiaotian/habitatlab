# Table: Margin-Phase Summary

Binned by `delta_mean = d_hat - w_req_cons`. Episode counts are rows inside the plotted margin range [-0.30, 0.30] m. Low-support bins are bins with fewer than 5 episodes.

| Method | Episodes | Success | Collision | Near collision | Reject | Near-boundary episodes | Bins | Low-support bins |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reactive rule baseline | 880 | 16.1% | 83.9% | 88.4% | 0.0% | 265 | 9 | 3 |
| DEGNAV-Rule | 724 | 51.4% | 30.0% | 55.7% | 0.0% | 306 | 7 | 1 |

Note: the reactive rule baseline does not consume a feasibility-belief state; its margin is logged as an environment-derived diagnostic for common x-axis binning.
