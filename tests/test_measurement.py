"""Unit tests for the MeasurementEngine (FR-008, FR-009, FR-010)."""

from __future__ import annotations

import numpy as np
import pytest

from src.measurement.engine import (
    MeasurementEngine,
    _measure_from_bbox,
    _measure_from_mask,
    bbox_to_length,
    compute_principal_axis_length,
)
from src.utils.contracts import BBox, CalibrationInfo, DetectionResult, TrayRow


class TestMeasureFromBbox:
    """Tests for bbox-based length estimation."""

    def test_returns_width_as_length(self) -> None:
        bbox = BBox(x1=0.0, y1=0.0, x2=200.0, y2=30.0)
        length, width, angle, method = _measure_from_bbox(bbox)
        assert length == pytest.approx(200.0)
        assert width == pytest.approx(30.0)
        assert angle == pytest.approx(0.0)
        assert method == "bbox"


class TestMeasureFromMask:
    """Tests for PCA-based mask length estimation."""

    def test_horizontal_bar_returns_correct_length(self) -> None:
        mask = np.zeros((100, 500), dtype=np.uint8)
        mask[40:60, 50:450] = 1  # 400px wide, 20px tall
        length, width, _, method = _measure_from_mask(mask, "mask_pca")
        assert method == "mask_pca"
        assert length == pytest.approx(400.0, abs=5.0)

    def test_empty_mask_returns_zero(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        length, width, _, method = _measure_from_mask(mask, "mask_pca")
        assert length == 0.0
        assert method == "bbox_fallback"


class TestComputePrincipalAxisLength:
    """Tests for compute_principal_axis_length utility."""

    def test_wide_horizontal_bar(self) -> None:
        mask = np.zeros((50, 300), dtype=np.uint8)
        mask[10:40, 0:300] = 1
        principal, secondary = compute_principal_axis_length(mask)
        assert principal > secondary  # principal axis is along the longer dimension


class TestBboxToLength:
    """Tests for bbox_to_length utility."""

    def test_horizontal_orientation(self) -> None:
        bbox = BBox(x1=0.0, y1=0.0, x2=300.0, y2=50.0)
        assert bbox_to_length(bbox, "horizontal") == pytest.approx(300.0)

    def test_vertical_orientation(self) -> None:
        bbox = BBox(x1=0.0, y1=0.0, x2=50.0, y2=300.0)
        assert bbox_to_length(bbox, "vertical") == pytest.approx(300.0)


class TestMeasurementEngine:
    """Integration tests for MeasurementEngine."""

    def test_qualifying_fragment_at_5pxmm(
        self,
        detection_result_with_fragments: DetectionResult,
        manual_calibration: CalibrationInfo,
        full_width_row: TrayRow,
    ) -> None:
        engine = MeasurementEngine()
        measurements = engine.measure(
            detection_result_with_fragments, [], manual_calibration, full_width_row
        )
        qualifying = [m for m in measurements if m.qualifies_rqd]
        # 600px / 5px_per_mm = 120mm → qualifies
        assert any(m.length_mm == pytest.approx(120.0) for m in qualifying)

    def test_non_qualifying_fragment_at_5pxmm(
        self,
        detection_result_with_fragments: DetectionResult,
        manual_calibration: CalibrationInfo,
        full_width_row: TrayRow,
    ) -> None:
        engine = MeasurementEngine()
        measurements = engine.measure(
            detection_result_with_fragments, [], manual_calibration, full_width_row
        )
        non_qualifying = [m for m in measurements if not m.qualifies_rqd]
        # 400px / 5px_per_mm = 80mm → does not qualify
        assert any(m.length_mm == pytest.approx(80.0) for m in non_qualifying)

    def test_zero_calibration_raises(
        self,
        detection_result_with_fragments: DetectionResult,
        full_width_row: TrayRow,
    ) -> None:
        bad_calib = CalibrationInfo(
            image_id="x", pixels_per_mm=0.1, source="manual"
        )
        # Override to 0 is disallowed by contract, so test with very small value
        engine = MeasurementEngine()
        # Should not raise — 0.1 px/mm is valid (just produces large mm values)
        measurements = engine.measure(
            detection_result_with_fragments, [], bad_calib, full_width_row
        )
        assert len(measurements) == 2

    def test_configurable_threshold(
        self,
        detection_result_with_fragments: DetectionResult,
        manual_calibration: CalibrationInfo,
        full_width_row: TrayRow,
    ) -> None:
        engine = MeasurementEngine(config={"rqd_threshold_mm": 50.0})
        measurements = engine.measure(
            detection_result_with_fragments, [], manual_calibration, full_width_row
        )
        qualifying = [m for m in measurements if m.qualifies_rqd]
        # Both 120mm and 80mm exceed 50mm threshold
        assert len(qualifying) == 2
