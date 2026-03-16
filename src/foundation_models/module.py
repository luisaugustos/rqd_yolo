"""FoundationModelModule — exploratory zero-shot detection via Florence-2 / DINO (Phase 5 §7-8)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.utils.contracts import BBox, DetectionResult, ProcessedImage

logger = logging.getLogger(__name__)

# Default text prompts for core fragment detection
_DEFAULT_PROMPTS = {
    "intact_fragment": "intact rock core fragment",
    "fracture": "fracture break in rock core",
    "scale_marker": "ruler scale marker",
}


class FoundationModelModule:
    """Load and run foundation models (Florence-2, Grounding DINO) for zero-shot detection.

    These models are exploratory — they do not require domain annotations but
    typically produce lower mAP than fine-tuned detection models.

    Args:
        config: Foundation model configuration. Keys:
            - model (str): 'florence2' | 'grounding_dino'
            - model_id (str): HuggingFace model ID or local path.
            - device (str): 'auto' | 'cpu' | 'cuda'.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._model_type: str = config.get("model", "florence2")
        self._model_id: str = config.get("model_id", "microsoft/Florence-2-base")
        self._device: str = config.get("device", "auto")
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Load the foundation model from HuggingFace or local cache."""
        if self._model_type == "florence2":
            self._load_florence2()
        elif self._model_type == "grounding_dino":
            self._load_grounding_dino()
        else:
            raise ValueError(f"Unknown foundation model type: '{self._model_type}'")

    def detect(self, image: ProcessedImage, prompt: str) -> DetectionResult:
        """Run zero-shot object detection with a natural language prompt.

        Args:
            image: ProcessedImage from the Preprocessing module.
            prompt: Natural language description (e.g. 'intact rock core fragment').

        Returns:
            DetectionResult with coordinates in original image pixel space.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded; call load() first")
        if self._model_type == "florence2":
            return self._florence2_detect(image, prompt)
        raise NotImplementedError(f"detect() not implemented for '{self._model_type}'")

    def describe_region(self, image: ProcessedImage, box: list[float]) -> str:
        """Generate a text description of a region in the image.

        Args:
            image: ProcessedImage.
            box: [x1, y1, x2, y2] region in original image pixels.

        Returns:
            Caption string.
        """
        if self._model_type != "florence2":
            return ""
        return self._florence2_caption(image, box)

    # ------------------------------------------------------------------
    # Florence-2 helpers
    # ------------------------------------------------------------------

    def _load_florence2(self) -> None:
        """Load Florence-2 model and processor from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            device = self._device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            self._processor = AutoProcessor.from_pretrained(
                self._model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id, trust_remote_code=True
            ).to(device)
            self._device = device
            logger.info("Loaded Florence-2 model '%s' on device '%s'", self._model_id, device)
        except ImportError:
            raise ImportError(
                "Florence-2 requires 'transformers>=4.38'. Install with: pip install transformers"
            )

    def _florence2_detect(self, image: ProcessedImage, prompt: str) -> DetectionResult:
        """Run Florence-2 OD or grounding task."""
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image.original_sample.image)
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self._processor(text=task, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.original_sample.width, image.original_sample.height),
        )

        boxes: list[BBox] = []
        scores: list[float] = []
        class_ids: list[int] = []
        class_names_out: list[str] = []

        for bbox_coords in parsed.get(task, {}).get("bboxes", []):
            x1, y1, x2, y2 = bbox_coords
            try:
                boxes.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2))
                scores.append(1.0)
                class_ids.append(1)
                class_names_out.append("intact_fragment")
            except Exception:
                continue

        return DetectionResult(
            image_id=image.image_id,
            model_name="florence2",
            model_backend="florence2",
            inference_latency_ms=0.0,
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names_out,
        )

    def _florence2_caption(self, image: ProcessedImage, box: list[float]) -> str:
        """Generate a region caption with Florence-2."""
        return ""  # Placeholder; full implementation via <REGION_CAPTION> task

    # ------------------------------------------------------------------
    # Grounding DINO helpers
    # ------------------------------------------------------------------

    def _load_grounding_dino(self) -> None:
        """Load Grounding DINO model (placeholder)."""
        raise NotImplementedError(
            "Grounding DINO backend not yet implemented. "
            "Use 'florence2' or a fine-tuned YOLO/RT-DETR backend instead."
        )
