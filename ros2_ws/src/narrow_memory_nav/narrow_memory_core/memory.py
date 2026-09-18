from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Tuple

from narrow_memory_core.features import PassageFeatures


@dataclass(frozen=True)
class FailureMemoryConfig:
    passage_width_bin: float = 0.15
    clearance_bin: float = 0.05
    lateral_bin: float = 0.10
    heading_bin: float = 0.25
    trigger_count: int = 1
    reject_count: int = 3


class PassageFailureMemory:
    """Coarse passage-local memory for repeated stuck/collision states."""

    def __init__(self, config: FailureMemoryConfig | None = None) -> None:
        self.config = config or FailureMemoryConfig()
        self._counts: DefaultDict[Tuple[int, int, int, int, int], int] = defaultdict(int)

    def reset(self) -> None:
        self._counts.clear()

    def key(self, features: PassageFeatures) -> Tuple[int, int, int, int, int]:
        cfg = self.config
        side = -1 if features.lateral_offset < 0.0 else 1
        return (
            int(features.passage_width / cfg.passage_width_bin),
            int(features.min_clearance / cfg.clearance_bin),
            int(abs(features.lateral_offset) / cfg.lateral_bin),
            int(abs(features.heading_error) / cfg.heading_bin),
            side,
        )

    def add_failure(self, features: PassageFeatures) -> None:
        self._counts[self.key(features)] += 1

    def count(self, features: PassageFeatures) -> int:
        return self._counts.get(self.key(features), 0)

    def should_recover(self, features: PassageFeatures) -> bool:
        return self.count(features) >= self.config.trigger_count

    def should_reject(self, features: PassageFeatures) -> bool:
        return self.count(features) >= self.config.reject_count

    @property
    def num_failed_states(self) -> int:
        return len(self._counts)

    @property
    def total_failures(self) -> int:
        return sum(self._counts.values())
