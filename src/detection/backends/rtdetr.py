"""RT-DETRv2 detection backend via Ultralytics (FR-005)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.detection.backends.yolo import _auto_device, _parse_ultralytics_result
from src.utils.contracts import DetectionResult

logger = logging.getLogger(__name__)


class RTDETRBackend:
    """RT-DETRv2 backend using Ultralytics RTDETR wrapper.

    End-to-end detection without NMS; the Ultralytics RT-DETR integration
    exposes the same predict() interface as YOLO, making this backend
    structurally identical to YOLOBackend.

    Args:
        model_name: Human-readable variant name (e.g. 'rtdetrv2_s').
    """

    def __init__(self, model_name: str = "rtdetrv2_s") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._class_names: list[str] = []
        self._device: str = "cpu"

    def load(self, weights_path: str, config: dict[str, Any]) -> None:
        """Load RT-DETRv2 weights.

        Args:
            weights_path: Path to .pt weights or Ultralytics model identifier.
            config: Detection config dict. Relevant keys: device.
        """
        from ultralytics import RTDETR

        self._model = RTDETR(weights_path)
        self._device = config.get("device", "auto")
        if self._device == "auto":
            self._device = _auto_device()
        self._class_names = list(self._model.names.values()) if self._model.names else []
        logger.info(
            "Loaded RT-DETRv2 model '%s' from '%s' on device '%s'",
            self._model_name,
            weights_path,
            self._device,
        )

    def predict(
        self,
        image: np.ndarray,
        conf_thresh: float = 0.3,
        iou_thresh: float = 0.45,
    ) -> DetectionResult:
        """Predict detections for a single image.

        Args:
            image: (H, W, 3) float32 preprocessed image.
            conf_thresh: Minimum detection confidence.
            iou_thresh: Unused for RT-DETR (no NMS); kept for API compatibility.

        Returns:
            DetectionResult in input image pixel coordinates.
        """
        return self.predict_batch([image], conf_thresh, iou_thresh)[0]

    def predict_batch(
        self,
        images: list[np.ndarray],
        conf_thresh: float = 0.3,
        iou_thresh: float = 0.45,
    ) -> list[DetectionResult]:
        """Predict detections for a batch of images.

        Args:
            images: List of (H, W, 3) float32 arrays.
            conf_thresh: Minimum detection confidence.
            iou_thresh: Unused for RT-DETR.

        Returns:
            List of DetectionResult.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded; call load() first")

        uint8_images = [(img * 255).clip(0, 255).astype(np.uint8) for img in images]
        t0 = time.perf_counter()
        results = self._model.predict(
            uint8_images,
            conf=conf_thresh,
            verbose=False,
            device=self._device,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_image_ms = elapsed_ms / max(len(images), 1)

        output: list[DetectionResult] = []
        for result in results:
            boxes, scores, class_ids, class_names = _parse_ultralytics_result(
                result, self._class_names
            )
            output.append(
                DetectionResult(
                    image_id="",
                    model_name=self._model_name,
                    model_backend="rtdetrv2",
                    inference_latency_ms=per_image_ms,
                    boxes=boxes,
                    scores=scores,
                    class_ids=class_ids,
                    class_names=class_names,
                )
            )
        return output
