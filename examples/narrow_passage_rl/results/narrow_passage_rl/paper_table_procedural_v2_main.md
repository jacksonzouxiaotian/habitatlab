# Table: Harder Synthetic Benchmark v2 - Main Comparison
# 500 episodes x 3 seeds (seeds 42-44), no entry jitter
# Metrics: success rate mean (std across seeds)

| Method | Overall | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False-feas. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Rule baseline | 25.4 (1.0) | 60.7 | 8.4 | 6.4 | 27.8 | 24.8 | 33.6 | 0.0 |
| **Geometry-FSM (ours)** | **70.3 (1.1)** | **92.8** | **79.4** | **70.3** | **96.5** | **61.7** | **50.3** | **0.0** |
| FSM + local memory | 70.3 (1.1) | 92.8 | 79.4 | 70.3 | 96.5 | 61.7 | 50.3 | 0.0 |
| FSM + cross memory | 70.3 (1.1) | 92.8 | 79.4 | 70.3 | 96.5 | 61.7 | 50.3 | 0.0 |
