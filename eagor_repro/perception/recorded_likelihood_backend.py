"""Deterministic likelihood playback for tests and ablations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import imageio.v3 as iio
import numpy as np

from eagor_repro.perception.likelihood_backend import (
    LikelihoodBackend,
    LikelihoodResult,
)


class RecordedLikelihoodBackend(LikelihoodBackend):
    backend_name = "recorded"

    def __init__(
        self,
        sources: str | Path | np.ndarray | Sequence[str | Path | np.ndarray],
        loop: bool = False,
        visible_threshold: float = 1e-5,
    ) -> None:
        if isinstance(sources, (str, Path, np.ndarray)):
            if isinstance(sources, np.ndarray) and sources.ndim == 3:
                self.sources = [frame for frame in sources]
            else:
                self.sources = [sources]
        else:
            self.sources = list(sources)
        if not self.sources:
            raise ValueError("At least one recorded likelihood source is required")
        self.loop = loop
        self.visible_threshold = float(visible_threshold)
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    @staticmethod
    def _load(source: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source.copy()
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path, allow_pickle=False)
        if suffix == ".npz":
            archive = np.load(path, allow_pickle=False)
            key = "likelihood" if "likelihood" in archive else archive.files[0]
            return archive[key]
        image = np.asarray(iio.imread(path))
        if image.ndim == 3:
            image = image[..., :3].mean(axis=-1)
        return image

    def infer(
        self,
        panorama: np.ndarray,
        target_query: str,
        observations: Optional[Dict[str, Any]] = None,
        simulator_state: Any = None,
    ) -> LikelihoodResult:
        start = time.perf_counter()
        if self.index >= len(self.sources):
            if not self.loop:
                raise IndexError("Recorded likelihood sequence is exhausted")
            self.index = 0
        source_index = self.index
        likelihood = np.asarray(self._load(self.sources[source_index]), np.float32)
        if likelihood.ndim == 3 and likelihood.shape[0] > 1:
            # Expand a .npy/.npz T,H,W source once so subsequent calls advance
            # through every recorded frame rather than replaying only frame 0.
            frames = [frame.copy() for frame in likelihood]
            self.sources[source_index : source_index + 1] = frames
            likelihood = np.asarray(self.sources[source_index], np.float32)
        self.index += 1
        maximum = float(likelihood.max(initial=0.0))
        if maximum > 1.0:
            likelihood /= maximum
        likelihood = np.clip(likelihood, 0.0, 1.0)
        visible = maximum > self.visible_threshold
        return LikelihoodResult(
            likelihood=likelihood,
            target_visible=visible,
            raw_detections=[{"source_index": source_index}],
            latency_ms=(time.perf_counter() - start) * 1000.0,
            backend_name=self.backend_name,
            confidence_calibrated=False,
        )
