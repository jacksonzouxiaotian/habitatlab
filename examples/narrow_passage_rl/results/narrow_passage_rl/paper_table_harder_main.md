# Table: Harder Synthetic Benchmark v2 — Main Comparison
# 500 episodes × 3 seeds (seeds 42–44), no entry jitter
# Metrics: mean (std across seeds in parentheses)

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feas. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Rule baseline | 25.4 (1.0) | 60.5 | 8.2 | 6.5 | 27.2 | 24.8 | 33.5 | 0.0 |
| **Geometry-FSM (ours)** | **70.3 (1.1)** | **92.8** | **79.6** | **70.0** | **96.7** | **61.4** | **50.6** | **0.0** |
| FSM + local memory | 70.3 (1.1) | 92.8 | 79.6 | 70.0 | 96.7 | 61.4 | 50.6 | 0.0 |
| FSM + cross memory | 70.3 (1.1) | 92.8 | 79.6 | 70.0 | 96.7 | 61.4 | 50.6 | 0.0 |

Notes:
- false_feasible SR=0% is correct: body-margin gating rejects impassable corridors before entry.
- Memory variants identical to base FSM in single-run eval; differentiation requires multi-round
  experiment with repeated false_feasible passages (see eval_memory_differentiation.py).
- collision_rate=0% for all methods; tight_passage_rate (bm<0.05m) ~34% for FSM variants.
