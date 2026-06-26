# Table: Habitat HM3D Generalization (157 val episodes, 20 held-out scenes)
# PPO results: mean (std) across 3 independent training seeds
# FSM/APF results: single run (deterministic policies — no variance)

| Method | SR | Narrow | Normal | Wide | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| PPO-SB3 baseline | 10.2% | 3.8% | 18.2% | 13.0% | Trained on synthetic 2D env, single seed |
| **PPO w/ geometry sensor** | **2.1% (2.6%)** | **1.7% (1.6%)** | **3.0% (4.3%)** | **1.4% (2.0%)** | 3 seeds: 5.7%, 0.0%, 0.6% |
| APF+Gap (Khatib 1986) | 93.6% | 92.4% | 100% | 82.6% | Depth + GPS only |
| **Geometry-FSM (ours)** | **100%** | **100%** | **100%** | **100%** | |
| **FSM + Failure Memory (ours)** | **100%** | **100%** | **100%** | **100%** | |

Notes:
- PPO w/ geometry sensor 3-seed SR: 5.7%, 0.0%, 0.6% — high variance confirms
  training instability and sim-to-real collapse (99%+ train SR → <6% Habitat SR)
- collision_rate=0% for all methods due to allow_sliding=False (navmesh boundary stop)
- min_clearance<0 is a depth-sensor artifact at valid navmesh positions near walls,
  not physical wall penetration
- FSM ablation (full / no_recovery / no_alignment): all 100% on this val set;
  differentiation shown on harder synthetic benchmark (Table 2)
