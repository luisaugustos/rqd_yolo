"""Unit tests for the RQD engine (FR-011, FR-012)."""

from __future__ import annotations

import pytest

from src.rqd.engine import RQDEngine, _safe_rqd, _row_length_mm
from src.utils.contracts import (
    BBox,
    CalibrationInfo,
    FragmentMeasurement,
    TrayRow,
)


def _make_measurement(row_id: int, length_mm: float, threshold: float = 100.0) -> FragmentMeasurement:
    """Helper to build a minimal FragmentMeasurement."""
    return FragmentMeasurement(
        image_id="img1",
        row_id=row_id,
        fragment_id=0,
        length_mm=length_mm,
        qualifies_rqd=length_mm >= threshold,
        rqd_threshold_mm=threshold,
        bbox_px=BBox(x1=0.0, y1=0.0, x2=length_mm * 5, y2=30.0),
        measurement_method="bbox",
    )


def _make_row(row_id: int = 0, row_length_mm: float = 1500.0) -> TrayRow:
    return TrayRow(
        row_id=row_id,
        image_id="img1",
        bbox=BBox(x1=0.0, y1=0.0, x2=7500.0, y2=100.0),
        row_length_px=7500.0,
        row_length_mm=row_length_mm,
    )


def _make_calibration() -> CalibrationInfo:
    return CalibrationInfo(image_id="img1", pixels_per_mm=5.0, source="manual")


class TestSafeRqd:
    """Tests for the _safe_rqd helper."""

    def test_perfect_quality(self) -> None:
        assert _safe_rqd(1500.0, 1500.0) == pytest.approx(100.0)

    def test_zero_qualifying(self) -> None:
        assert _safe_rqd(0.0, 1500.0) == pytest.approx(0.0)

    def test_clamped_at_100(self) -> None:
        # qualifying > total should be caught by RQDResult invariants,
        # but _safe_rqd clamps to 100 defensively
        assert _safe_rqd(2000.0, 1500.0) == pytest.approx(100.0)

    def test_zero_total_returns_zero(self) -> None:
        assert _safe_rqd(0.0, 0.0) == pytest.approx(0.0)


class TestRowLengthMm:
    """Tests for _row_length_mm helper."""

    def test_uses_row_length_mm_when_set(self) -> None:
        row = _make_row(row_length_mm=1500.0)
        calib = _make_calibration()
        assert _row_length_mm(row, calib) == pytest.approx(1500.0)

    def test_falls_back_to_pixel_conversion(self) -> None:
        row = TrayRow(
            row_id=0,
            image_id="img1",
            bbox=BBox(x1=0.0, y1=0.0, x2=7500.0, y2=100.0),
            row_length_px=7500.0,
            row_length_mm=None,
        )
        calib = _make_calibration()
        assert _row_length_mm(row, calib) == pytest.approx(1500.0)


class TestRQDEngineRowRqd:
    """Tests for RQDEngine.compute_row_rqd."""

    def test_all_qualifying(self) -> None:
        engine = RQDEngine()
        row = _make_row(row_length_mm=1500.0)
        measurements = [_make_measurement(0, 500.0), _make_measurement(0, 400.0), _make_measurement(0, 300.0)]
        result = engine.compute_row_rqd(measurements, row, _make_calibration())
        assert result.rqd_pct == pytest.approx(100.0 * 1200.0 / 1500.0, abs=0.01)
        assert result.num_fragments_qualifying == 3

    def test_no_measurements_returns_zero_rqd(self) -> None:
        engine = RQDEngine()
        row = _make_row(row_length_mm=1500.0)
        result = engine.compute_row_rqd([], row, _make_calibration())
        assert result.rqd_pct == pytest.approx(0.0)
        assert result.num_fragments_total == 0

    def test_mixed_qualifying(self) -> None:
        engine = RQDEngine()
        row = _make_row(row_length_mm=1000.0)
        measurements = [
            _make_measurement(0, 300.0),   # qualifying
            _make_measurement(0, 50.0),    # not qualifying
            _make_measurement(0, 150.0),   # qualifying
        ]
        result = engine.compute_row_rqd(measurements, row, _make_calibration())
        expected_rqd = (300.0 + 150.0) / 1000.0 * 100.0
        assert result.rqd_pct == pytest.approx(expected_rqd, abs=0.01)
        assert result.num_fragments_qualifying == 2

    def test_zero_run_length_raises(self) -> None:
        """Engine must raise when _row_length_mm resolves to zero/negative."""
        from unittest.mock import patch

        engine = RQDEngine()
        row = _make_row(row_length_mm=1000.0)
        with patch("src.rqd.engine._row_length_mm", return_value=0.0):
            with pytest.raises(ValueError, match="must be positive"):
                engine.compute_row_rqd([], row, _make_calibration())

    def test_rqd_clamped_to_100(self) -> None:
        engine = RQDEngine()
        row = _make_row(row_length_mm=100.0)
        # qualifying length (200mm) > run length (100mm) → clamp to 100%
        measurements = [_make_measurement(0, 200.0)]
        result = engine.compute_row_rqd(measurements, row, _make_calibration())
        assert result.rqd_pct == pytest.approx(100.0)


class TestRQDEngineImageRqd:
    """Tests for RQDEngine.compute_image_rqd."""

    def test_aggregate_uses_sum_not_average(self) -> None:
        """AC-012-2: aggregate RQD must use raw lengths, not average row percentages."""
        engine = RQDEngine()
        calib = _make_calibration()

        row0 = _make_row(0, row_length_mm=1000.0)
        row1 = _make_row(1, row_length_mm=500.0)

        m0 = [_make_measurement(0, 600.0)]   # row0 RQD = 60%
        m1 = [_make_measurement(1, 500.0)]   # row1 RQD = 100%

        r0 = engine.compute_row_rqd(m0, row0, calib)
        r1 = engine.compute_row_rqd(m1, row1, calib)
        image_result = engine.compute_image_rqd([r0, r1])

        # Correct: (600 + 500) / (1000 + 500) = 1100/1500 ≈ 73.33%
        # Wrong average: (60 + 100) / 2 = 80%
        expected = (600.0 + 500.0) / 1500.0 * 100.0
        assert image_result.rqd_pct == pytest.approx(expected, abs=0.01)
        assert image_result.scope == "image"
        assert image_result.row_id is None

    def test_empty_row_results_raises(self) -> None:
        engine = RQDEngine()
        with pytest.raises(ValueError):
            engine.compute_image_rqd([])

    def test_rqd_formula_matches_isrm(self) -> None:
        """ISRM: RQD = (qualifying / total) × 100, clamped to [0, 100]."""
        engine = RQDEngine()
        calib = _make_calibration()
        row = _make_row(row_length_mm=1500.0)
        measurements = [_make_measurement(0, 200.0), _make_measurement(0, 300.0)]
        r = engine.compute_row_rqd(measurements, row, calib)
        image_r = engine.compute_image_rqd([r])
        # 500mm qualifying / 1500mm total × 100 ≈ 33.33%
        assert image_r.rqd_pct == pytest.approx(500.0 / 1500.0 * 100.0, abs=0.01)
