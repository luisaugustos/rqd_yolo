"""Tests for detection and segmentation backend registries (NFR-012)."""

from __future__ import annotations

import pytest

import src.detection.registry as det_reg
import src.segmentation.registry as seg_reg
from src.detection.backends import YOLOBackend
from src.segmentation.backends import BBoxFallbackBackend


class TestDetectorRegistry:
    """Tests for detection backend registry."""

    def test_yolov12_registered(self) -> None:
        cls = det_reg.get("yolov12")
        assert cls is YOLOBackend

    def test_yolov11_registered(self) -> None:
        cls = det_reg.get("yolov11")
        assert cls is YOLOBackend

    def test_rtdetrv2_registered(self) -> None:
        from src.detection.backends.rtdetr import RTDETRBackend
        cls = det_reg.get("rtdetrv2")
        assert cls is RTDETRBackend

    def test_unknown_backend_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="nonexistent_backend_xyz"):
            det_reg.get("nonexistent_backend_xyz")

    def test_list_backends_includes_known(self) -> None:
        backends = det_reg.list_backends()
        assert "yolov12" in backends
        assert "rtdetrv2" in backends

    def test_register_custom_backend(self) -> None:
        class MockBackend:
            def load(self, w, c): pass
            def predict(self, i, ct, it): pass
            def predict_batch(self, imgs, ct, it): pass

        det_reg.register("mock_test_backend", MockBackend)
        assert det_reg.get("mock_test_backend") is MockBackend
        # Cleanup
        del det_reg._registry["mock_test_backend"]


class TestSegmentorRegistry:
    """Tests for segmentation backend registry."""

    def test_none_backend_registered(self) -> None:
        cls = seg_reg.get("none")
        assert cls is BBoxFallbackBackend

    def test_sam2_registered(self) -> None:
        from src.segmentation.backends.sam2 import SAM2Backend
        cls = seg_reg.get("sam2")
        assert cls is SAM2Backend

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            seg_reg.get("nonexistent_seg_backend")

    def test_list_backends(self) -> None:
        backends = seg_reg.list_backends()
        assert "none" in backends
        assert "sam2" in backends
