"""Shared pytest fixtures for the rqd-ai-lab test suite."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.contracts import (
    Annotation,
    BBox,
    CalibrationInfo,
    DetectionResult,
    FragmentMeasurement,
    ImageSample,
    PreprocessMetadata,
    ProcessedImage,
    RQDResult,
    SegmentationResult,
    TrayRow,
)


# ---------------------------------------------------------------------------
# BBox fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_bbox() -> BBox:
    """A 100×50 px bounding box."""
    return BBox(x1=10.0, y1=20.0, x2=110.0, y2=70.0)


@pytest.fixture
def qualifying_bbox() -> BBox:
    """A 600×50 px bounding box — large enough to qualify at 5 px/mm."""
    return BBox(x1=0.0, y1=0.0, x2=600.0, y2=50.0)


# ---------------------------------------------------------------------------
# CalibrationInfo fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manual_calibration() -> CalibrationInfo:
    """Manual calibration at 5 pixels per mm."""
    return CalibrationInfo(image_id="test", pixels_per_mm=5.0, source="manual")


# ---------------------------------------------------------------------------
# ImageSample fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def blank_image_sample() -> ImageSample:
    """A 640×480 blank white image sample."""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    return ImageSample(
        image_id="test_img",
        file_path="/fake/test_img.jpg",
        image=image,
        width=640,
        height=480,
    )


# ---------------------------------------------------------------------------
# TrayRow fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def full_width_row() -> TrayRow:
    """A tray row spanning the full 640px width of the test image."""
    return TrayRow(
        row_id=0,
        image_id="test_img",
        bbox=BBox(x1=0.0, y1=0.0, x2=640.0, y2=480.0),
        row_length_px=640.0,
        row_length_mm=128.0,  # 640 px / 5 px_per_mm
    )


# ---------------------------------------------------------------------------
# DetectionResult fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def detection_result_with_fragments(qualifying_bbox: BBox) -> DetectionResult:
    """Two intact_fragment detections — one qualifying, one not."""
    small_box = BBox(x1=0.0, y1=0.0, x2=400.0, y2=30.0)  # 400px -> 80mm @ 5px/mm
    return DetectionResult(
        image_id="test_img",
        model_name="yolov12m",
        model_backend="yolov12",
        inference_latency_ms=10.0,
        boxes=[qualifying_bbox, small_box],
        scores=[0.9, 0.85],
        class_ids=[1, 1],
        class_names=["intact_fragment", "intact_fragment"],
    )


# ---------------------------------------------------------------------------
# FragmentMeasurement fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qualifying_measurement(qualifying_bbox: BBox) -> FragmentMeasurement:
    """A 120mm qualifying fragment."""
    return FragmentMeasurement(
        image_id="test_img",
        row_id=0,
        fragment_id=0,
        length_mm=120.0,
        orientation_deg=0.0,
        measurement_method="bbox",
        qualifies_rqd=True,
        rqd_threshold_mm=100.0,
        bbox_px=qualifying_bbox,
        mask_present=False,
        calibration_source="manual",
    )


@pytest.fixture
def non_qualifying_measurement() -> FragmentMeasurement:
    """An 80mm non-qualifying fragment."""
    return FragmentMeasurement(
        image_id="test_img",
        row_id=0,
        fragment_id=1,
        length_mm=80.0,
        orientation_deg=0.0,
        measurement_method="bbox",
        qualifies_rqd=False,
        rqd_threshold_mm=100.0,
        bbox_px=BBox(x1=0.0, y1=0.0, x2=400.0, y2=30.0),
        mask_present=False,
        calibration_source="manual",
    )


# ---------------------------------------------------------------------------
# SegmentationResult fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def bbox_mask_result(qualifying_bbox: BBox) -> SegmentationResult:
    """A simple rectangular mask from the qualifying bbox."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[0:50, 0:600] = 1
    return SegmentationResult(
        image_id="test_img",
        fragment_id=0,
        model_name="bbox_fallback",
        mask=mask,
        mask_score=1.0,
        refined_bbox=qualifying_bbox,
        prompt_bbox=qualifying_bbox,
        mask_area_px=int(mask.sum()),
        inference_latency_ms=0.0,
    )
