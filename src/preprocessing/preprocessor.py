"""Image preprocessing: resize, normalize, pad and invert coordinates (FR-002).

Produces ProcessedImage objects that carry enough metadata to map detected
bounding boxes back to original image pixel coordinates.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.utils.contracts import (
    BBox,
    ImageSample,
    PreprocessMetadata,
    ProcessedImage,
)

logger = logging.getLogger(__name__)

# Normalization statistics
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Preprocessor:
    """Prepare images for model input (FR-002).

    Applies letterbox resizing, normalization, and optional augmentation.
    Records all transform parameters so that output coordinates can be
    inverted back to original image space.

    Args:
        config: Preprocessing configuration dict. Keys:
            - target_size (int | list[int]): Target (width, height). Default 640.
            - normalization (str): 'imagenet' | 'zero_one' | 'none'. Default 'imagenet'.
            - interpolation (str): 'linear' | 'nearest'. Default 'linear'.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        target = cfg.get("target_size", 640)
        if isinstance(target, int):
            self._target_w = target
            self._target_h = target
        else:
            self._target_w, self._target_h = int(target[0]), int(target[1])
        self._normalization: str = cfg.get("normalization", "imagenet")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, sample: ImageSample) -> ProcessedImage:
        """Preprocess a single image sample.

        Args:
            sample: Raw ImageSample with uint8 RGB image.

        Returns:
            ProcessedImage with float32 tensor and inversion metadata.
        """
        tensor, meta = self._letterbox_and_normalize(sample.image)
        return ProcessedImage(
            image_id=sample.image_id,
            tensor=tensor,
            metadata=meta,
            original_sample=sample,
        )

    def process_batch(self, samples: list[ImageSample]) -> list[ProcessedImage]:
        """Preprocess a list of image samples.

        Args:
            samples: List of raw ImageSamples.

        Returns:
            List of ProcessedImage objects.
        """
        return [self.process(s) for s in samples]

    def invert_coords(self, coords: np.ndarray, metadata: PreprocessMetadata) -> np.ndarray:
        """Map bounding box coordinates from preprocessed space to original image space.

        Args:
            coords: Array of shape (N, 4) with columns [x1, y1, x2, y2]
                in preprocessed image coordinates.
            metadata: PreprocessMetadata from the corresponding ProcessedImage.

        Returns:
            Array of shape (N, 4) in original image pixel coordinates.
        """
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"coords must be (N, 4), got {coords.shape}")
        inverted = coords.copy().astype(np.float32)
        # Remove padding offsets
        inverted[:, 0] -= metadata.pad_left
        inverted[:, 2] -= metadata.pad_left
        inverted[:, 1] -= metadata.pad_top
        inverted[:, 3] -= metadata.pad_top
        # Reverse scaling
        inverted[:, [0, 2]] /= metadata.scale_x
        inverted[:, [1, 3]] /= metadata.scale_y
        # Clip to original image bounds
        inverted[:, [0, 2]] = inverted[:, [0, 2]].clip(0, metadata.original_width)
        inverted[:, [1, 3]] = inverted[:, [1, 3]].clip(0, metadata.original_height)
        return inverted

    def invert_bbox(self, bbox: BBox, metadata: PreprocessMetadata) -> BBox:
        """Invert a single BBox from preprocessed to original image coordinates.

        Args:
            bbox: BBox in preprocessed image coordinates.
            metadata: PreprocessMetadata from the corresponding ProcessedImage.

        Returns:
            BBox in original image pixel coordinates.
        """
        arr = np.array([[bbox.x1, bbox.y1, bbox.x2, bbox.y2]], dtype=np.float32)
        inv = self.invert_coords(arr, metadata)
        return BBox(x1=inv[0, 0], y1=inv[0, 1], x2=inv[0, 2], y2=inv[0, 3])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _letterbox_and_normalize(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, PreprocessMetadata]:
        """Apply letterbox resize followed by normalization.

        The image is scaled uniformly so the longer edge fits the target,
        then padded symmetrically with grey (114, 114, 114).

        Args:
            image: (H, W, 3) uint8 RGB array.

        Returns:
            Tuple of (float32 tensor, PreprocessMetadata).
        """
        import cv2

        orig_h, orig_w = image.shape[:2]
        scale = min(self._target_w / orig_w, self._target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_top = (self._target_h - new_h) // 2
        pad_bottom = self._target_h - new_h - pad_top
        pad_left = (self._target_w - new_w) // 2
        pad_right = self._target_w - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        tensor = padded.astype(np.float32) / 255.0
        if self._normalization == "imagenet":
            tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
        elif self._normalization == "none":
            tensor = (tensor * 255.0).astype(np.float32)
        # 'zero_one' is the default (already divided by 255)

        meta = PreprocessMetadata(
            original_width=orig_w,
            original_height=orig_h,
            scale_x=new_w / orig_w,
            scale_y=new_h / orig_h,
            pad_top=pad_top,
            pad_left=pad_left,
            target_width=self._target_w,
            target_height=self._target_h,
            normalization=self._normalization,
        )
        return tensor, meta
