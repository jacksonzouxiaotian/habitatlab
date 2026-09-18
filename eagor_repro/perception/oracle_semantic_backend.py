"""Habitat ground-truth semantic mask backend for debugging and upper bounds."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set

import numpy as np
from scipy.ndimage import gaussian_filter

from eagor_repro.perception.likelihood_backend import (
    LikelihoodBackend,
    LikelihoodResult,
)
from eagor_repro.sensors.panorama_sensor import normalize_native_equirectangular


def _canonical_category(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


class OracleSemanticBackend(LikelihoodBackend):
    """Create likelihood from native equirectangular semantic observations.

    This backend is evaluation/debug-only.  It must never be reported as a
    zero-shot VLM result.  Habitat semantic pixels contain object/instance IDs;
    those IDs are matched to ``sim.semantic_scene.objects[*].category.name()``.
    """

    backend_name = "oracle_semantic"

    def __init__(
        self,
        semantic_sensor_uuid: str = "equirect_semantic_sensor",
        sigma_azimuth_deg: float = 8.0,
        sigma_elevation_deg: float = 8.0,
        weak_observation: float = 1e-6,
    ) -> None:
        self.semantic_sensor_uuid = semantic_sensor_uuid
        self.sigma_azimuth_deg = float(sigma_azimuth_deg)
        self.sigma_elevation_deg = float(sigma_elevation_deg)
        self.weak_observation = float(weak_observation)

    @staticmethod
    def _object_semantic_ids(
        semantic_scene: Any, target_query: str
    ) -> Set[int]:
        wanted = _canonical_category(target_query)
        ids: Set[int] = set()
        if semantic_scene is None:
            return ids
        for fallback_index, obj in enumerate(
            getattr(semantic_scene, "objects", ()) or ()
        ):
            if obj is None:
                continue
            category = getattr(obj, "category", None)
            try:
                category_name = category.name()
            except (AttributeError, TypeError):
                category_name = str(category or "")
            canonical = _canonical_category(category_name)
            wanted_tokens = set(wanted.split())
            category_tokens = set(canonical.split())
            if (
                canonical != wanted
                and not wanted_tokens.issubset(category_tokens)
                and not category_tokens.issubset(wanted_tokens)
            ):
                continue
            candidates: Iterable[Any] = (
                getattr(obj, "semantic_id", None),
                getattr(obj, "id", None),
                fallback_index,
            )
            for candidate in candidates:
                if candidate is None:
                    continue
                try:
                    ids.add(int(candidate))
                    break
                except (TypeError, ValueError):
                    pass
                match = re.search(r"(-?\d+)$", str(candidate))
                if match:
                    ids.add(int(match.group(1)))
                    break
        return ids

    def infer(
        self,
        panorama: np.ndarray,
        target_query: str,
        observations: Optional[Dict[str, Any]] = None,
        simulator_state: Any = None,
    ) -> LikelihoodResult:
        start = time.perf_counter()
        observations = observations or {}
        if self.semantic_sensor_uuid not in observations:
            raise KeyError(
                f"Oracle backend needs observation {self.semantic_sensor_uuid!r}"
            )
        semantic = np.asarray(
            normalize_native_equirectangular(
                observations[self.semantic_sensor_uuid]
            )
        ).squeeze()
        if semantic.ndim != 2:
            raise ValueError(f"Expected HxW semantic image, got {semantic.shape}")

        sim = simulator_state
        if sim is not None and hasattr(sim, "semantic_scene"):
            semantic_scene = sim.semantic_scene
        else:
            semantic_scene = getattr(sim, "semantic_scene", None)
        ids = self._object_semantic_ids(semantic_scene, target_query)
        mask = np.isin(semantic, list(ids)) if ids else np.zeros_like(semantic, bool)
        target_visible = bool(mask.any())
        if target_visible:
            height, width = semantic.shape
            sigma = (
                self.sigma_elevation_deg / 180.0 * height,
                self.sigma_azimuth_deg / 360.0 * width,
            )
            # Vertical reflection and horizontal wrap preserve poles/seam.
            likelihood = gaussian_filter(
                mask.astype(np.float32), sigma=sigma, mode=("reflect", "wrap")
            )
            maximum = float(likelihood.max(initial=0.0))
            if maximum > 0.0:
                likelihood /= maximum
        else:
            likelihood = np.full(semantic.shape, self.weak_observation, np.float32)

        elapsed = (time.perf_counter() - start) * 1000.0
        return LikelihoodResult(
            likelihood=likelihood,
            target_visible=target_visible,
            raw_detections=[
                {
                    "semantic_ids": sorted(ids),
                    "pixel_count": int(mask.sum()),
                    "target_query": target_query,
                }
            ],
            latency_ms=elapsed,
            backend_name=self.backend_name,
            confidence_calibrated=True,
            metadata={"debug_oracle": True},
        )


class SemanticIdOracleBackend(LikelihoodBackend):
    """Oracle adapter for datasets that render semantic IDs without a scene graph.

    ReplicaCAD's rigid object templates carry valid ``semantic_id`` values, but
    Habitat-Sim 0.3.3 does not expose those objects through
    ``sim.semantic_scene.objects``.  This adapter therefore accepts an explicit
    query-to-ID table while still deriving every online observation from the
    native semantic ERP.  The table is dataset metadata, not a detector, so this
    remains an Oracle/debug upper bound rather than a zero-shot result.

    ``min_visible_pixels`` suppresses isolated sub-pixel rasterization at long
    range.  Pixels below that threshold are reported as not visible and never
    inject a false SH peak.
    """

    backend_name = "semantic_id_oracle"

    def __init__(
        self,
        semantic_ids: Mapping[str, Sequence[int]] | Sequence[int],
        semantic_sensor_uuid: str = "equirect_semantic_sensor",
        sigma_azimuth_deg: float = 5.0,
        sigma_elevation_deg: float = 5.0,
        weak_observation: float = 1e-6,
        min_visible_pixels: int = 1,
    ) -> None:
        if isinstance(semantic_ids, Mapping):
            self.semantic_ids = {
                _canonical_category(query): tuple(int(value) for value in values)
                for query, values in semantic_ids.items()
            }
        else:
            self.semantic_ids = {"*": tuple(int(value) for value in semantic_ids)}
        self.semantic_sensor_uuid = semantic_sensor_uuid
        self.sigma_azimuth_deg = float(sigma_azimuth_deg)
        self.sigma_elevation_deg = float(sigma_elevation_deg)
        self.weak_observation = float(weak_observation)
        self.min_visible_pixels = int(min_visible_pixels)
        if self.min_visible_pixels < 1:
            raise ValueError("min_visible_pixels must be positive")

    def _ids_for_query(self, target_query: str) -> tuple[int, ...]:
        canonical = _canonical_category(target_query)
        if canonical in self.semantic_ids:
            return self.semantic_ids[canonical]
        if "*" in self.semantic_ids:
            return self.semantic_ids["*"]
        raise KeyError(
            f"No semantic IDs configured for target query {target_query!r}; "
            f"available={sorted(self.semantic_ids)}"
        )

    def infer(
        self,
        panorama: np.ndarray,
        target_query: str,
        observations: Optional[Dict[str, Any]] = None,
        simulator_state: Any = None,
    ) -> LikelihoodResult:
        del panorama, simulator_state
        start = time.perf_counter()
        observations = observations or {}
        if self.semantic_sensor_uuid not in observations:
            raise KeyError(
                f"Semantic-ID Oracle needs observation "
                f"{self.semantic_sensor_uuid!r}"
            )
        semantic = np.asarray(
            normalize_native_equirectangular(
                observations[self.semantic_sensor_uuid]
            )
        ).squeeze()
        if semantic.ndim != 2:
            raise ValueError(f"Expected HxW semantic image, got {semantic.shape}")

        ids = self._ids_for_query(target_query)
        mask = np.isin(semantic, ids)
        pixel_count = int(mask.sum())
        target_visible = pixel_count >= self.min_visible_pixels
        if target_visible:
            height, width = semantic.shape
            sigma = (
                self.sigma_elevation_deg / 180.0 * height,
                self.sigma_azimuth_deg / 360.0 * width,
            )
            likelihood = gaussian_filter(
                mask.astype(np.float32), sigma=sigma, mode=("reflect", "wrap")
            )
            maximum = float(likelihood.max(initial=0.0))
            if maximum > 0.0:
                likelihood /= maximum
        else:
            likelihood = np.full(semantic.shape, self.weak_observation, np.float32)

        elapsed = (time.perf_counter() - start) * 1000.0
        return LikelihoodResult(
            likelihood=likelihood,
            target_visible=target_visible,
            raw_detections=[
                {
                    "semantic_ids": list(ids),
                    "pixel_count": pixel_count,
                    "min_visible_pixels": self.min_visible_pixels,
                    "target_query": target_query,
                }
            ],
            latency_ms=elapsed,
            backend_name=self.backend_name,
            confidence_calibrated=True,
            metadata={
                "debug_oracle": True,
                "explicit_dataset_semantic_ids": True,
            },
        )
