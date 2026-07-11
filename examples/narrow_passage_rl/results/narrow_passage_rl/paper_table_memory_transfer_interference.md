# Table: Memory Transfer and Interference

This table tests whether failure memory transfers from repeated false-feasible passages to similar new false-feasible passages, and whether that transfer causes false rejection on similar feasible passages.

Preset: `smoke`. Use `--preset paper` for the paper-scale run; `smoke` is a quick reproducibility check.

| Method | Seeds | Passable SR (up) | Passable false reject (down) | FF reject R1 (up) | FF reject final (up) | Wasted FF attempts R1 (down) | Wasted FF attempts final (down) | Transfer reject on new FF (up) | Interference false reject on feasible (down) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_memory | 1 | 75.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 8.0 +/- 0.0 | 8.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| local_intra_episode_memory | 1 | 75.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 8.0 +/- 0.0 | 8.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| vanilla_episodic_memory | 1 | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 75.0 +/- 0.0 | 100.0 +/- 0.0 | 2.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| knn_failure_memory | 1 | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 75.0 +/- 0.0 | 100.0 +/- 0.0 | 2.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| geometry_guided_cross_episode_failure_memory | 1 | 75.0 +/- 0.0 | 0.0 +/- 0.0 | 25.0 +/- 0.0 | 100.0 +/- 0.0 | 6.0 +/- 0.0 | 0.0 +/- 0.0 | 50.0 +/- 0.0 | 0.0 +/- 0.0 |
