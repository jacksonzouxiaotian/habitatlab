# Table: Memory Baselines

This legacy table is kept for compatibility.  The preferred paper table is
`paper_table_repeated_failure_memory.md`, which adds local intra-episode memory
and retrieval precision.

| Method | Passable SR ↑ | Passable false reject ↓ | Final FF reject ↑ | Wasted FF steps ↓ | Steps saved vs no memory ↑ | Retrieval precision ↑ |
|:---|---:|---:|---:|---:|---:|---:|
| no_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 | - |
| local_intra_episode_memory | 0.900 | 0.000 | 0.000 | 16500 | 0 | - |
| knn_failure_memory | 0.860 | 0.060 | 1.000 | 5500 | 11000 | 0.916 |
| vanilla_episodic_memory | 0.820 | 0.130 | 1.000 | 8140 | 8360 | 0.736 |
| geometry_guided_cross_episode_failure_memory | 0.900 | 0.030 | 1.000 | 4400 | 12100 | 0.963 |

Interpretation: memory does not improve one-shot nominal Habitat success. Its
contribution is to suppress repeated commitments to previously failed infeasible
passages while avoiding false rejection of passable corridors.
