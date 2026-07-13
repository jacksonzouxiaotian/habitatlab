# Table: Memory Transfer and Interference

This table tests whether failure memory transfers from repeated false-feasible passages to similar new false-feasible passages, and whether that transfer causes false rejection on similar feasible passages.

Preset: `paper`. Use `--preset paper` for the paper-scale run; `smoke` is a quick reproducibility check.

| Method | Seeds | Passable SR (up) | Passable false reject (down) | FF reject R1 (up) | FF reject final (up) | Wasted FF attempts R1 (down) | Wasted FF attempts final (down) | Transfer reject on new FF (up) | Interference false reject on feasible (down) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_memory | 5 | 90.0 +/- 2.2 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 40.0 +/- 0.0 | 40.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| local_intra_episode_memory | 5 | 90.0 +/- 2.2 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 40.0 +/- 0.0 | 40.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| vanilla_episodic_memory | 5 | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 95.0 +/- 0.0 | 100.0 +/- 0.0 | 2.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| knn_failure_memory | 5 | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 95.0 +/- 0.0 | 100.0 +/- 0.0 | 2.0 +/- 0.0 | 0.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| geometry_guided_cross_episode_failure_memory | 5 | 90.0 +/- 2.2 | 0.0 +/- 0.0 | 42.5 +/- 5.2 | 100.0 +/- 0.0 | 23.0 +/- 2.1 | 0.0 +/- 0.0 | 93.0 +/- 8.6 | 0.0 +/- 0.0 |
