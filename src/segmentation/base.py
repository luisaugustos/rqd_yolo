"""SegmentorBackend Protocol — interface every segmentation backend must satisfy (NFR-012)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.utils.contracts import BBox, SegmentationResult


class PromptBox:
    """A bounding box prompt for segmentation.

    Args:
        fragment_id: Index into the parent DetectionResult.boxes list.
        bbox: Bounding box to use as the segmentation prompt.
    """

    def __init__(self, fragment_id: int, bbox: BBox) -> None:
        self.fragment_id = fragment_id
        self.bbox = bbox


@runtime_checkable
class SegmentorBackend(Protocol):
    """Interface that all segmentation backends must implement."""

    def load(self, weights_path: str, config: dict[str, Any]) -> None:
        """Load model weights.

        Args:
            weights_path: Path or identifier for the segmentation model.
            config: Backend-specific configuration dictionary.
        """
        ...

    def segment(
        self,
        image: np.ndarray,
        prompts: list[PromptBox],
    ) -> list[SegmentationResult]:
        """Produce instance segmentation masks for each prompt.

        Args:
            image: (H, W, 3) float32 preprocessed image.
            prompts: List of bounding box prompts, one per candidate fragment.

        Returns:
            List of SegmentationResult, one per prompt.
        """
        ...
