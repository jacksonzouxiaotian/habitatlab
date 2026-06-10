#!/usr/bin/env python3

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class FailureMemoryConfig:
    passage_width_bin: float = 0.15
    clearance_bin: float = 0.05
    lateral_bin: float = 0.10
    heading_bin: float = 0.25
    trigger_count: int = 1
    reject_count: int = 3


class PassageFailureMemory:
    """Episode-local passage-centric failure memory.

    The key is a coarse local passage/risk state. It intentionally avoids
    storing exact pose so that nearby failures generalize to similar states.
    """

    def __init__(self, config: FailureMemoryConfig = None):
        self.config = config or FailureMemoryConfig()
        self._counts: Dict[Tuple[int, int, int, int, int], int] = defaultdict(int)

    def reset(self):
        self._counts.clear()

    def key(self, obs) -> Tuple[int, int, int, int, int]:
        cfg = self.config
        min_clearance = min(float(obs[6]), float(obs[7]))
        passage_width = float(obs[8])
        lateral_offset = float(obs[11])
        heading_error = float(obs[10])
        side = -1 if lateral_offset < 0.0 else 1

        return (
            int(passage_width / cfg.passage_width_bin),
            int(min_clearance / cfg.clearance_bin),
            int(abs(lateral_offset) / cfg.lateral_bin),
            int(abs(heading_error) / cfg.heading_bin),
            side,
        )

    def add_failure(self, obs):
        self._counts[self.key(obs)] += 1

    def count(self, obs) -> int:
        return self._counts.get(self.key(obs), 0)

    def should_recover(self, obs) -> bool:
        return self.count(obs) >= self.config.trigger_count

    def should_reject(self, obs) -> bool:
        return self.count(obs) >= self.config.reject_count

    @property
    def num_failed_states(self) -> int:
        return len(self._counts)

    @property
    def total_failures(self) -> int:
        return sum(self._counts.values())
