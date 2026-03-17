"""Tests for the BBoxFallbackBackend segmentation backend."""

from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.backends.bbox_fallback import BBoxFallbackBackend
from src.segmentation.base import PromptBox
from src.utils.contracts import BBox


class TestBBoxFallbackBackend:
    """Tests for BBoxFallbackBackend."""

    def _make_image(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.float32)

    def test_returns_one_result_per_prompt(self) -> None:
        backend = BBoxFallbackBackend()
        backend.load("", {})
        image = self._make_image()
        prompts = [
            PromptBox(0, BBox(x1=10.0, y1=10.0, x2=100.0, y2=60.0)),
            PromptBox(1, BBox(x1=200.0, y1=50.0, x2=400.0, y2=100.0)),
        ]
        results = backend.segment(image, prompts)
        assert len(results) == 2

    def test_mask_is_binary_uint8(self) -> None:
        backend = BBoxFallbackBackend()
        backend.load("", {})
        image = self._make_image()
        prompt = PromptBox(0, BBox(x1=0.0, y1=0.0, x2=200.0, y2=100.0))
        results = backend.segment(image, [prompt])
        mask = results[0].mask
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})

    def test_mask_area_matches_bbox(self) -> None:
        backend = BBoxFallbackBackend()
        backend.load("", {})
        image = self._make_image()
        prompt = PromptBox(0, BBox(x1=10.0, y1=20.0, x2=110.0, y2=70.0))
        results = backend.segment(image, [prompt])
        expected_area = 100 * 50  # 100px wide × 50px tall
        assert results[0].mask_area_px == expected_area

    def test_fragment_id_preserved(self) -> None:
        backend = BBoxFallbackBackend()
        backend.load("", {})
        image = self._make_image()
        prompts = [PromptBox(7, BBox(x1=0.0, y1=0.0, x2=50.0, y2=50.0))]
        results = backend.segment(image, prompts)
        assert results[0].fragment_id == 7

    def test_empty_prompts_returns_empty_list(self) -> None:
        backend = BBoxFallbackBackend()
        backend.load("", {})
        results = backend.segment(self._make_image(), [])
        assert results == []

    def test_refined_bbox_matches_prompt(self) -> None:
        backend = BBoxFallbackBackend()
        backend.load("", {})
        image = self._make_image()
        bbox = BBox(x1=10.0, y1=20.0, x2=110.0, y2=70.0)
        results = backend.segment(image, [PromptBox(0, bbox)])
        refined = results[0].refined_bbox
        assert refined.x1 == pytest.approx(10.0)
        assert refined.y1 == pytest.approx(20.0)
