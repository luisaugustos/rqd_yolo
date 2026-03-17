"""YOLOv11 / YOLOv12 detection backend using the Ultralytics library (FR-005)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.utils.contracts import BBox, DetectionResult

logger = logging.getLogger(__name__)


class YOLOBackend:
    """Ultralytics YOLO detection backend.

    Supports YOLOv11 and YOLOv12 via the ``ultralytics`` library.
    The same class handles both families since their inference API is identical.

    Args:
        model_name: Human-readable model variant name, e.g. 'yolov12m'.
    """

    def __init__(self, model_name: str = "yolov12m") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._class_names: list[str] = []

    def load(self, weights_path: str, config: dict[str, Any]) -> None:
        """Load YOLO weights from disk or download from Ultralytics hub.

        Args:
            weights_path: Path to .pt file or model identifier (e.g. 'yolov11n.pt').
            config: Detection config dict. Relevant keys: device, amp.
        """
        from pathlib import Path
        from ultralytics import YOLO
        from ultralytics.utils.downloads import attempt_download_asset

        model_identifier = weights_path
        path_obj = Path(weights_path)

        if path_obj.exists():
            # Local file exists, use as-is
            model_identifier = str(path_obj.resolve())
        elif isinstance(weights_path, str) and weights_path.endswith(".pt"):
            # File doesn't exist locally; try to download from Ultralytics hub
            logger.debug("Attempting to download model '%s' from Ultralytics hub", weights_path)
            try:
                model_identifier = attempt_download_asset(weights_path)
                logger.debug("Successfully downloaded to '%s'", model_identifier)
            except Exception as e:
                logger.warning("Failed to download '%s' from hub: %s", weights_path, e)
                # Fall through to let YOLO() attempt to handle it

        self._model = YOLO(model_identifier, verbose=False)
        device = config.get("device", "auto")
        if device == "auto":
            device = _auto_device()
        logger.info(
            "Loaded YOLO model '%s' from '%s' on device '%s'",
            self._model_name,
            weights_path,
            device,
        )
        self._device = device
        self._class_names = list(self._model.names.values()) if self._model.names else []

    def predict(
        self,
        image: np.ndarray,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> DetectionResult:
        """Run inference on a single image.

        Args:
            image: (H, W, 3) float32 array. The YOLO library handles rescaling
                internally; the image is passed as-is.
            conf_thresh: Confidence score threshold.
            iou_thresh: NMS IoU threshold.

        Returns:
            DetectionResult with coordinates in the input image pixel space.
        """
        return self.predict_batch([image], conf_thresh, iou_thresh)[0]

    def predict_batch(
        self,
        images: list[np.ndarray],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> list[DetectionResult]:
        """Run batched inference.

        Args:
            images: List of (H, W, 3) float32 arrays.
            conf_thresh: Confidence threshold.
            iou_thresh: NMS IoU threshold.

        Returns:
            List of DetectionResult, one per input image.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded; call load() first")

        # Convert to uint8 for Ultralytics (expects uint8 or file paths)
        uint8_images = [(img * 255).clip(0, 255).astype(np.uint8) for img in images]
        t0 = time.perf_counter()
        results = self._model.predict(
            uint8_images,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
            device=self._device,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_image_ms = elapsed_ms / max(len(images), 1)

        detection_results: list[DetectionResult] = []
        for result, img in zip(results, images):
            h, w = img.shape[:2]
            boxes, scores, class_ids, class_names = _parse_ultralytics_result(
                result, self._class_names
            )
            detection_results.append(
                DetectionResult(
                    image_id="",
                    model_name=self._model_name,
                    model_backend="yolo",
                    inference_latency_ms=per_image_ms,
                    boxes=boxes,
                    scores=scores,
                    class_ids=class_ids,
                    class_names=class_names,
                )
            )
        return detection_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ultralytics_result(
    result: Any, class_names: list[str]
) -> tuple[list[BBox], list[float], list[int], list[str]]:
    """Extract lists from an Ultralytics result object."""
    boxes: list[BBox] = []
    scores: list[float] = []
    class_ids: list[int] = []
    names: list[str] = []

    if result.boxes is None or len(result.boxes) == 0:
        return boxes, scores, class_ids, names

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3])
        boxes.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2))
        scores.append(float(conf[i]))
        class_ids.append(int(cls[i]))
        cname = class_names[int(cls[i])] if int(cls[i]) < len(class_names) else str(int(cls[i]))
        names.append(cname)

    return boxes, scores, class_ids, names


def _auto_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
