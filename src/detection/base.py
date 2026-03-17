"""DetectorBackend Protocol — interface every detection backend must satisfy (NFR-002, NFR-012)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.utils.contracts import DetectionResult


@runtime_checkable
class DetectorBackend(Protocol):
    """Interface that all detection model backends must implement.

    Backends are loaded via the DetectorRegistry using a string name that
    matches the ``model.backend`` key in the detection config YAML.
    Adding a new backend requires only:
    1. Implementing this protocol.
    2. Calling ``DetectorRegistry.register(name, BackendClass)``.
    """

    def load(self, weights_path: str, config: dict[str, Any]) -> None:
        """Load model weights and prepare for inference.

        Args:
            weights_path: Path to the model weight file or HuggingFace model ID.
            config: Backend-specific configuration dictionary.
        """
        ...

    def predict(
        self,
        image: np.ndarray,
        conf_thresh: float,
        iou_thresh: float,
    ) -> DetectionResult:
        """Run inference on a single image.

        Args:
            image: (H, W, 3) float32 preprocessed image tensor.
            conf_thresh: Minimum confidence score to retain a detection.
            iou_thresh: IoU threshold for non-maximum suppression.

        Returns:
            DetectionResult in original image pixel coordinates.
        """
        ...

    def predict_batch(
        self,
        images: list[np.ndarray],
        conf_thresh: float,
        iou_thresh: float,
    ) -> list[DetectionResult]:
        """Run inference on a batch of images.

        Args:
            images: List of (H, W, 3) float32 preprocessed image tensors.
            conf_thresh: Minimum confidence threshold.
            iou_thresh: NMS IoU threshold.

        Returns:
            List of DetectionResult, one per image.
        """
        ...
