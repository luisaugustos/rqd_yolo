"""SAM2 segmentation backend (FR-007, Phase 5 Model Spec §4).

Uses the SAM2 predictor in box-prompt mode. The model is loaded once and
reused across all prompts for an image to amortise the image-encoder cost.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.segmentation.base import PromptBox
from src.utils.contracts import BBox, SegmentationResult

logger = logging.getLogger(__name__)


class SAM2Backend:
    """SAM2 (Segment Anything Model 2) instance segmentation backend.

    Requires the ``sam2`` package (``pip install segment-anything-2``).

    Args:
        model_name: Descriptive name for logging (e.g. 'sam2_vit_b').
    """

    def __init__(self, model_name: str = "sam2_vit_b") -> None:
        self._model_name = model_name
        self._predictor: Any = None
        self._device: str = "cpu"

    def load(self, weights_path: str, config: dict[str, Any]) -> None:
        """Load SAM2 model checkpoint.

        Args:
            weights_path: Path to SAM2 .pt checkpoint.
            config: Segmentation config dict. Keys: device, sam2.model_cfg.
        """
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_config = config.get("sam2", {})
            model_cfg = sam2_config.get("model_cfg", "sam2_hiera_b+.yaml")
            self._device = config.get("device", "auto")
            if self._device == "auto":
                self._device = _auto_device()

            sam_model = build_sam2(model_cfg, weights_path, device=self._device)
            self._predictor = SAM2ImagePredictor(sam_model)
            logger.info(
                "Loaded SAM2 backend '%s' from '%s' on device '%s'",
                self._model_name,
                weights_path,
                self._device,
            )
        except ImportError:
            raise ImportError(
                "SAM2 backend requires 'sam2' package. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

    def segment(
        self,
        image: np.ndarray,
        prompts: list[PromptBox],
    ) -> list[SegmentationResult]:
        """Produce instance masks for each bounding-box prompt.

        The image encoder is run once per image; all prompts are batched.

        Args:
            image: (H, W, 3) float32 image in [0, 1] range.
            prompts: List of bounding box prompts.

        Returns:
            List of SegmentationResult, one per prompt.
        """
        if self._predictor is None:
            raise RuntimeError("SAM2 model not loaded; call load() first")

        uint8_image = (image * 255).clip(0, 255).astype(np.uint8)
        self._predictor.set_image(uint8_image)

        results: list[SegmentationResult] = []
        for prompt in prompts:
            b = prompt.bbox
            box_np = np.array([b.x1, b.y1, b.x2, b.y2], dtype=np.float32)
            t0 = time.perf_counter()
            masks, scores, _ = self._predictor.predict(
                box=box_np[None, :],
                multimask_output=True,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Select highest scoring mask
            best_idx = int(scores.argmax())
            mask = masks[best_idx].astype(np.uint8)
            score = float(scores[best_idx])
            area = int(mask.sum())

            ys, xs = np.where(mask)
            if len(xs) > 0:
                refined = BBox(
                    x1=float(xs.min()),
                    y1=float(ys.min()),
                    x2=float(xs.max()) + 1,
                    y2=float(ys.max()) + 1,
                )
            else:
                refined = prompt.bbox

            results.append(
                SegmentationResult(
                    image_id="",
                    fragment_id=prompt.fragment_id,
                    model_name=self._model_name,
                    mask=mask,
                    mask_score=score,
                    refined_bbox=refined,
                    prompt_bbox=prompt.bbox,
                    mask_area_px=area,
                    inference_latency_ms=latency_ms,
                )
            )
        return results


def _auto_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
