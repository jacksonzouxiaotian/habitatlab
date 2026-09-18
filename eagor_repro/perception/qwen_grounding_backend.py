"""Optional local Qwen2.5-VL grounding adapter.

This backend is deliberately lazy: importing EAGOR does not require
``transformers`` or a GPU.  Qwen's text output is parsed conservatively for
boxes/points; unless an explicit score is present the generated peaks use equal
weights and are labelled uncalibrated.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from eagor_repro.perception.likelihood_backend import (
    LikelihoodBackend,
    LikelihoodResult,
)


class QwenGroundingBackend(LikelihoodBackend):
    backend_name = "qwen_grounding"

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        inference_callable: Optional[Callable[[np.ndarray, str], str]] = None,
        sigma_azimuth_deg: float = 8.0,
        sigma_elevation_deg: float = 8.0,
        absence_floor: float = 1e-6,
        timeout_s: float = 30.0,
        max_new_tokens: int = 256,
        fail_open: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.inference_callable = inference_callable
        self.sigma_azimuth_deg = sigma_azimuth_deg
        self.sigma_elevation_deg = sigma_elevation_deg
        self.absence_floor = absence_floor
        self.timeout_s = float(timeout_s)
        self.max_new_tokens = int(max_new_tokens)
        self.fail_open = bool(fail_open)
        self._model = None
        self._processor = None

    @staticmethod
    def parse_detections(text: str, width: int, height: int) -> List[Dict[str, float]]:
        """Parse JSON-ish Qwen points or boxes in pixels or normalized 0..1000."""

        detections: List[Dict[str, float]] = []
        candidates: List[Any] = []
        try:
            parsed = json.loads(text)
            candidates = parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            for match in re.findall(r"\[[^\[\]]+\]", str(text)):
                try:
                    candidates.append(json.loads(match))
                except json.JSONDecodeError:
                    continue
        for candidate in candidates:
            score = 1.0
            score_calibrated = False
            if isinstance(candidate, dict):
                score_calibrated = "score" in candidate or "confidence" in candidate
                score = float(candidate.get("score", candidate.get("confidence", 1.0)))
                coords = candidate.get("bbox", candidate.get("box", candidate.get("point")))
            else:
                coords = candidate
            if not isinstance(coords, (list, tuple)) or len(coords) not in (2, 4):
                continue
            values = np.asarray(coords, dtype=np.float64)
            if values.max(initial=0.0) > max(width, height) and values.max() <= 1000:
                values[0::2] *= width / 1000.0
                values[1::2] *= height / 1000.0
            if len(values) == 4:
                x = float((values[0] + values[2]) / 2.0)
                y = float((values[1] + values[3]) / 2.0)
            else:
                x, y = map(float, values)
            detections.append(
                {
                    "u": x % width,
                    "v": np.clip(y, 0, height - 1),
                    "score": score,
                    "score_calibrated": score_calibrated,
                }
            )
        return detections

    def _run_local_model(self, panorama: np.ndarray, target_query: str) -> str:
        if self.inference_callable is not None:
            return self.inference_callable(panorama, target_query)
        if not self.model_name_or_path:
            raise RuntimeError("Qwen backend needs model_name_or_path or inference_callable")
        if self._model is None:
            try:
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            except ImportError as exc:
                raise RuntimeError(
                    "Install a Qwen-compatible transformers build to enable this optional backend"
                ) from exc
            self._processor = AutoProcessor.from_pretrained(self.model_name_or_path)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name_or_path, device_map="auto", torch_dtype="auto"
            )
        from PIL import Image

        prompt = (
            "Ground every visible instance matching the target query "
            f"{target_query!r} in this equirectangular panorama. Return only a "
            "JSON array. Each item must contain either pixel bbox [x1,y1,x2,y2] "
            "or pixel point [x,y]. Add score only if the model has an explicit "
            "confidence; otherwise omit it. Return [] when absent."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(panorama[..., :3])},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[chat_text],
            images=[Image.fromarray(panorama[..., :3])],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        generated = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            max_time=self.timeout_s,
            do_sample=False,
        )
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def infer(
        self,
        panorama: np.ndarray,
        target_query: str,
        observations: Optional[Dict[str, Any]] = None,
        simulator_state: Any = None,
    ) -> LikelihoodResult:
        start = time.perf_counter()
        height, width = panorama.shape[:2]
        try:
            response = self._run_local_model(panorama, target_query)
        except Exception as exc:
            if not self.fail_open:
                raise
            return LikelihoodResult(
                likelihood=np.full((height, width), self.absence_floor, np.float32),
                target_visible=False,
                raw_detections=[],
                latency_ms=(time.perf_counter() - start) * 1000.0,
                backend_name=self.backend_name,
                confidence_calibrated=False,
                metadata={
                    "model_error": f"{type(exc).__name__}: {exc}",
                    "fail_open": True,
                },
            )
        detections = self.parse_detections(response, width, height)
        likelihood = np.full((height, width), self.absence_floor, np.float32)
        yy, xx = np.mgrid[:height, :width]
        sigma_u = max(self.sigma_azimuth_deg / 360.0 * width, 1e-3)
        sigma_v = max(self.sigma_elevation_deg / 180.0 * height, 1e-3)
        for detection in detections:
            du = np.abs(xx - detection["u"])
            du = np.minimum(du, width - du)
            peak = np.exp(
                -0.5 * (du / sigma_u) ** 2
                - 0.5 * ((yy - detection["v"]) / sigma_v) ** 2
            )
            likelihood += float(detection["score"]) * peak
        maximum = float(likelihood.max(initial=0.0))
        if maximum > 0.0:
            likelihood /= maximum
        model_scores_provided = bool(detections) and all(
            bool(item.get("score_calibrated", False)) for item in detections
        )
        return LikelihoodResult(
            likelihood=likelihood,
            target_visible=bool(detections),
            raw_detections=detections,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            backend_name=self.backend_name,
            # Raw generative scores are not treated as calibrated probabilities,
            # even when the model emits a numeric score.
            confidence_calibrated=False,
            metadata={
                "raw_response": response,
                "model_scores_provided": model_scores_provided,
                "scores_are_uncalibrated": True,
            },
        )
