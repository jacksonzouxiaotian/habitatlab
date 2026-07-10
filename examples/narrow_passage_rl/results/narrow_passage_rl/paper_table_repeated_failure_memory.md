# Table: Repeated Failure Memory

Failure memory is evaluated by repeated exposure to passable and false-feasible passages. The contribution is reduced repeated infeasible commitment and wasted attempts, not higher one-shot nominal Habitat success.

| Method | Passable SR ↑ | Passable false reject ↓ | Final false-feasible reject ↑ | Wasted FF steps ↓ | Steps saved vs no memory ↑ | Retrieval precision ↑ |
|:---|---:|---:|---:|---:|---:|---:|
| no_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 | - |
| local_intra_episode_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 | - |
| knn_failure_memory | 0.860 | 0.060 | 1.000 | 5500 | 11000 | 0.916 |
| vanilla_episodic_memory | 0.820 | 0.130 | 1.000 | 8140 | 8360 | 0.736 |
| geometry_guided_cross_episode_failure_memory | 0.900 | 0.030 | 1.000 | 4400 | 12100 | 0.963 |
