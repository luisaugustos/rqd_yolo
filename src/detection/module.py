"""DetectionModule — orchestrates backend loading and inference (FR-005, FR-006)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.detection import registry as _reg
from src.detection.backends import YOLOBackend  # noqa: F401 — triggers registration
from src.utils.contracts import DetectionResult, ProcessedImage

logger = logging.getLogger(__name__)


class DetectionModule:
    """Loads a detection backend and runs fragment/fracture detection.

    Backends are selected by name from the config (model.backend key). The
    module normalises all output box coordinates to the original image pixel
    space using the preprocessing inversion metadata stored in ProcessedImage.

    Args:
        config: Detection configuration dictionary (from yolo_train.yaml or
            rtdetr_train.yaml). Must contain 'model.backend' and
            'model.weights' keys.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        model_cfg = config.get("model", {})
        self._backend_name: str = model_cfg.get("backend", "yolov12")
        self._weights: str = model_cfg.get("weights", "yolov12m.pt")
        self._conf_thresh: float = config.get("inference", {}).get("conf_thresh", 0.25)
        self._iou_thresh: float = config.get("inference", {}).get("iou_thresh", 0.45)
        self._backend: Any | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the backend and load model weights.

        Raises:
            KeyError: When the configured backend name is not registered.
            RuntimeError: When weight loading fails.
        """
        backend_cls = _reg.get(self._backend_name)
        variant = self._config.get("model", {}).get("variant", "m")
        self._backend = backend_cls(model_name=f"{self._backend_name}{variant}")
        self._backend.load(self._weights, self._config)
        logger.info(
            "DetectionModule loaded backend '%s' with weights '%s'",
            self._backend_name,
            self._weights,
        )

    def detect(self, processed: ProcessedImage) -> DetectionResult:
        """Detect objects in a single preprocessed image.

        Coordinates in the returned DetectionResult are in original image
        pixel space (preprocessing inversion applied automatically).

        Args:
            processed: ProcessedImage from the Preprocessing module.

        Returns:
            DetectionResult with boxes in original pixel coordinates.
        """
        results = self.detect_batch([processed])
        result = results[0]
        result.image_id = processed.image_id
        return result

    def detect_batch(self, images: list[ProcessedImage]) -> list[DetectionResult]:
        """Detect objects in a batch of preprocessed images.

        Args:
            images: List of ProcessedImage objects.

        Returns:
            List of DetectionResult, one per image.
        """
        if self._backend is None:
            raise RuntimeError("Backend not loaded; call load() first")

        tensors = [img.tensor for img in images]
        raw_results = self._backend.predict_batch(tensors, self._conf_thresh, self._iou_thresh)

        from src.preprocessing.preprocessor import Preprocessor

        pp = Preprocessor()
        final: list[DetectionResult] = []
        for proc_img, raw in zip(images, raw_results):
            raw.image_id = proc_img.image_id
            # Invert coordinates from preprocessed space to original image space
            if raw.num_detections > 0:
                coords = np.array([[b.x1, b.y1, b.x2, b.y2] for b in raw.boxes], dtype=np.float32)
                inv_coords = pp.invert_coords(coords, proc_img.metadata)
                from src.utils.contracts import BBox

                raw.boxes = [
                    BBox(x1=inv_coords[i, 0], y1=inv_coords[i, 1],
                         x2=inv_coords[i, 2], y2=inv_coords[i, 3])
                    for i in range(len(raw.boxes))
                ]
            final.append(raw)
        return final
