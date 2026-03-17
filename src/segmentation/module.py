"""SegmentationModule — orchestrates backend selection and mask prediction (FR-007)."""

from __future__ import annotations

import logging
from typing import Any

from src.segmentation import registry as _reg
from src.segmentation.backends import BBoxFallbackBackend  # noqa: F401 — triggers registration
from src.segmentation.base import PromptBox
from src.utils.contracts import DetectionResult, ProcessedImage, SegmentationResult

logger = logging.getLogger(__name__)

# Class ID for intact fragments (IC-007, label index 1)
_FRAGMENT_CLASS_ID = 1


class SegmentationModule:
    """Load a segmentation backend and produce masks from detection prompts.

    The backend is selected via the ``backend`` key in the segmentation config.
    When ``backend == 'none'`` the BBoxFallbackBackend is used transparently,
    ensuring downstream modules always receive SegmentationResult objects.

    Args:
        config: Segmentation config dictionary (from segmentation.yaml).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._backend_name: str = config.get("backend", "none")
        self._weights: str = config.get(self._backend_name, {}).get("weights", "")
        self._backend: Any | None = None

    def load(self) -> None:
        """Instantiate the backend and load any required model weights.

        Raises:
            KeyError: When the configured backend is not registered.
        """
        backend_cls = _reg.get(self._backend_name)
        self._backend = backend_cls()
        weights = self._weights or ""
        self._backend.load(weights, self._config)
        logger.info("SegmentationModule loaded backend '%s'", self._backend_name)

    def segment(
        self,
        processed: ProcessedImage,
        detections: DetectionResult,
    ) -> list[SegmentationResult]:
        """Produce instance masks for all intact_fragment detections.

        Only detections with class_id == 1 (intact_fragment) are segmented.
        Masks are returned in original image pixel coordinates.

        Args:
            processed: ProcessedImage (tensor in preprocessed space).
            detections: DetectionResult from the Detection module (coordinates
                already inverted to original image space).

        Returns:
            List of SegmentationResult in original image pixel space.
        """
        if self._backend is None:
            raise RuntimeError("Backend not loaded; call load() first")

        fragment_detections = detections.filter_by_class(_FRAGMENT_CLASS_ID)
        if fragment_detections.num_detections == 0:
            logger.debug("No intact_fragment detections; skipping segmentation for %s", processed.image_id)
            return []

        prompts = [
            PromptBox(fragment_id=i, bbox=box)
            for i, box in enumerate(fragment_detections.boxes)
        ]

        # Pass the original image (not the preprocessed tensor) to the backend
        original_image = processed.original_sample.image.astype(float) / 255.0
        results = self._backend.segment(original_image, prompts)

        for r in results:
            r.image_id = processed.image_id

        return results
