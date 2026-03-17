"""Unit tests for data contracts (IC-001 to IC-010)."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.contracts import (
    BBox,
    CalibrationInfo,
    DetectionResult,
    FragmentMeasurement,
    ImageSample,
    PreprocessMetadata,
    RQDResult,
    SegmentationResult,
    TrayRow,
)


class TestBBox:
    """Tests for BBox (IC-003)."""

    def test_basic_properties(self) -> None:
        b = BBox(x1=0.0, y1=0.0, x2=100.0, y2=50.0)
        assert b.width == 100.0
        assert b.height == 50.0
        assert b.area == 5000.0
        assert b.center == (50.0, 25.0)

    def test_invalid_x2_le_x1(self) -> None:
        with pytest.raises(ValueError):
            BBox(x1=50.0, y1=0.0, x2=10.0, y2=20.0)

    def test_invalid_y2_le_y1(self) -> None:
        with pytest.raises(ValueError):
            BBox(x1=0.0, y1=40.0, x2=10.0, y2=20.0)

    def test_from_xywh(self) -> None:
        b = BBox.from_xywh(10.0, 20.0, 30.0, 40.0)
        assert b.x1 == 10.0
        assert b.y1 == 20.0
        assert b.x2 == 40.0
        assert b.y2 == 60.0

    def test_from_yolo_normalised(self) -> None:
        b = BBox.from_yolo(0.5, 0.5, 1.0, 1.0, 100, 100)
        assert b.x1 == pytest.approx(0.0)
        assert b.y1 == pytest.approx(0.0)
        assert b.x2 == pytest.approx(100.0)
        assert b.y2 == pytest.approx(100.0)

    def test_to_list(self) -> None:
        b = BBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        assert b.to_list() == [1.0, 2.0, 3.0, 4.0]


class TestCalibrationInfo:
    """Tests for CalibrationInfo (IC-005)."""

    def test_valid_manual(self) -> None:
        c = CalibrationInfo(image_id="x", pixels_per_mm=5.0, source="manual")
        assert c.pixels_per_mm == 5.0

    def test_invalid_zero_ppm(self) -> None:
        with pytest.raises(ValueError):
            CalibrationInfo(image_id="x", pixels_per_mm=0.0, source="manual")

    def test_invalid_negative_ppm(self) -> None:
        with pytest.raises(ValueError):
            CalibrationInfo(image_id="x", pixels_per_mm=-1.0, source="manual")

    def test_invalid_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            CalibrationInfo(image_id="x", pixels_per_mm=5.0, source="auto", confidence=1.5)


class TestImageSample:
    """Tests for ImageSample (IC-001)."""

    def test_shape_mismatch_raises(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            ImageSample(
                image_id="bad", file_path="/x", image=img, width=300, height=100
            )

    def test_depth_order_enforced(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            ImageSample(
                image_id="bad",
                file_path="/x",
                image=img,
                width=200,
                height=100,
                depth_from_m=10.0,
                depth_to_m=5.0,
            )

    def test_valid_construction(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        s = ImageSample(image_id="ok", file_path="/x", image=img, width=200, height=100)
        assert s.width == 200
        assert s.height == 100
        assert s.annotations == []


class TestDetectionResult:
    """Tests for DetectionResult (IC-007)."""

    def test_mismatched_lists_raise(self) -> None:
        with pytest.raises(ValueError):
            DetectionResult(
                image_id="x",
                model_name="m",
                model_backend="b",
                inference_latency_ms=0.0,
                boxes=[BBox(x1=0, y1=0, x2=1, y2=1)],
                scores=[0.9, 0.8],  # length mismatch
                class_ids=[0],
                class_names=["a"],
            )

    def test_filter_by_class(self) -> None:
        dr = DetectionResult(
            image_id="x",
            model_name="m",
            model_backend="b",
            inference_latency_ms=0.0,
            boxes=[BBox(x1=0, y1=0, x2=1, y2=1), BBox(x1=2, y1=2, x2=3, y2=3)],
            scores=[0.9, 0.8],
            class_ids=[1, 0],
            class_names=["intact_fragment", "fracture"],
        )
        fragments = dr.filter_by_class(1)
        assert fragments.num_detections == 1
        assert fragments.class_names[0] == "intact_fragment"

    def test_filter_by_score(self) -> None:
        dr = DetectionResult(
            image_id="x",
            model_name="m",
            model_backend="b",
            inference_latency_ms=0.0,
            boxes=[BBox(x1=0, y1=0, x2=1, y2=1), BBox(x1=2, y1=2, x2=3, y2=3)],
            scores=[0.9, 0.3],
            class_ids=[1, 1],
            class_names=["intact_fragment", "intact_fragment"],
        )
        high_conf = dr.filter_by_score(0.5)
        assert high_conf.num_detections == 1


class TestFragmentMeasurement:
    """Tests for FragmentMeasurement (IC-009)."""

    def test_qualifies_consistency_enforced(self) -> None:
        with pytest.raises(ValueError):
            FragmentMeasurement(
                image_id="x",
                row_id=0,
                fragment_id=0,
                length_mm=120.0,
                qualifies_rqd=False,  # should be True for 120mm >= 100mm threshold
                rqd_threshold_mm=100.0,
                bbox_px=BBox(x1=0, y1=0, x2=10, y2=5),
            )

    def test_non_positive_length_raises(self) -> None:
        with pytest.raises(ValueError):
            FragmentMeasurement(
                image_id="x",
                row_id=0,
                fragment_id=0,
                length_mm=-5.0,
                qualifies_rqd=False,
                rqd_threshold_mm=100.0,
                bbox_px=BBox(x1=0, y1=0, x2=10, y2=5),
            )


class TestRQDResult:
    """Tests for RQDResult (IC-010)."""

    def test_qualifying_exceeds_total_raises(self) -> None:
        with pytest.raises(ValueError):
            RQDResult(
                image_id="x",
                scope="row",
                row_id=0,
                total_run_length_mm=100.0,
                qualifying_length_mm=200.0,  # > total
                rqd_pct=200.0,
                num_fragments_total=2,
                num_fragments_qualifying=1,
            )

    def test_rqd_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            RQDResult(
                image_id="x",
                scope="row",
                row_id=0,
                total_run_length_mm=100.0,
                qualifying_length_mm=50.0,
                rqd_pct=110.0,  # > 100
                num_fragments_total=1,
                num_fragments_qualifying=0,
            )

    def test_image_scope_with_row_id_raises(self) -> None:
        with pytest.raises(ValueError):
            RQDResult(
                image_id="x",
                scope="image",
                row_id=0,  # must be None for image scope
                total_run_length_mm=100.0,
                qualifying_length_mm=50.0,
                rqd_pct=50.0,
                num_fragments_total=1,
                num_fragments_qualifying=0,
            )
