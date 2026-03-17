"""Segmentation backend implementations with auto-registration."""

from src.segmentation.backends.bbox_fallback import BBoxFallbackBackend
from src.segmentation.backends.sam2 import SAM2Backend
from src.segmentation import registry

registry.register("none", BBoxFallbackBackend)
registry.register("bbox_fallback", BBoxFallbackBackend)
registry.register("sam2", SAM2Backend)
