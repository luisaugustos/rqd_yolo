"""Bounding-box fallback segmentation backend (FR-007, AC-007-3).

When segmentation is disabled (backend == 'none'), this backend synthesises
a SegmentationResult by filling the detection bounding box with 1s.
This ensures downstream modules always receive a SegmentationResult without
special-casing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.segmentation.base import PromptBox
from src.utils.contracts import BBox, SegmentationResult

logger = logging.getLogger(__name__)


class BBoxFallbackBackend:
    """Synthesise binary masks from bounding boxes (no model required).

    This backend creates a rectangular binary mask the same shape as the
    bounding box. It is used when segmentation is disabled and serves as the
    baseline for the measurement engine.
    """

    def __init__(self) -> None:
        self._image_shape: tuple[int, int] = (0, 0)

    def load(self, weights_path: str, config: dict[str, Any]) -> None:
        """No-op; this backend requires no weights.

        Args:
            weights_path: Ignored.
            config: Ignored.
        """
        logger.info("BBoxFallbackBackend: no model weights required")

    def segment(
        self,
        image: np.ndarray,
        prompts: list[PromptBox],
    ) -> list[SegmentationResult]:
        """Return rectangular masks filled from the bounding box prompts.

        Args:
            image: (H, W, 3) float32 image (used only to read dimensions).
            prompts: List of PromptBox objects.

        Returns:
            List of SegmentationResult with rectangular binary masks.
        """
        h, w = image.shape[:2]
        results: list[SegmentationResult] = []
        for prompt in prompts:
            b = prompt.bbox
            x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
            x2, y2 = min(w, int(b.x2)), min(h, int(b.y2))
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            area = int(mask.sum())
            # Recompute tight bbox from the filled rectangle
            refined = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            results.append(
                SegmentationResult(
                    image_id="",
                    fragment_id=prompt.fragment_id,
                    model_name="bbox_fallback",
                    mask=mask,
                    mask_score=1.0,
                    refined_bbox=refined,
                    prompt_bbox=prompt.bbox,
                    mask_area_px=area,
                    inference_latency_ms=0.0,
                )
            )
        return results
