"""Tests for the EvaluationModule (FR-014)."""

from __future__ import annotations

import pytest

from src.evaluation.module import (
    EvaluationModule,
    _bbox_iou,
    _f1,
    _safe_rqd_for_test,
)
from src.utils.contracts import (
    Annotation,
    BBox,
    DetectionResult,
    FragmentMeasurement,
    RQDResult,
)


# Helper to avoid importing private function - redefine locally
def _make_rqd_result(image_id: str, rqd_pct: float) -> RQDResult:
    return RQDResult(
        image_id=image_id,
        scope="image",
        row_id=None,
        total_run_length_mm=1000.0,
        qualifying_length_mm=rqd_pct * 10.0,
        rqd_pct=rqd_pct,
        num_fragments_total=3,
        num_fragments_qualifying=1,
    )


def _make_ann(bbox: BBox) -> Annotation:
    return Annotation(
        annotation_id=0, image_id="x", class_id=1, class_name="intact_fragment", bbox=bbox
    )


class TestBboxIou:
    """Tests for _bbox_iou."""

    def test_identical_boxes(self) -> None:
        b = BBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        assert _bbox_iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        a = BBox(x1=0.0, y1=0.0, x2=50.0, y2=50.0)
        b = BBox(x1=100.0, y1=100.0, x2=150.0, y2=150.0)
        assert _bbox_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        a = BBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        b = BBox(x1=50.0, y1=0.0, x2=150.0, y2=100.0)
        # Intersection = 50×100 = 5000; Union = 10000+10000-5000 = 15000
        assert _bbox_iou(a, b) == pytest.approx(5000.0 / 15000.0, abs=0.001)


class TestF1:
    """Tests for _f1 helper."""

    def test_perfect(self) -> None:
        assert _f1(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        assert _f1(1.0, 0.0) == pytest.approx(0.0)

    def test_balanced(self) -> None:
        assert _f1(0.8, 0.8) == pytest.approx(0.8)


class TestEvaluationModuleRqd:
    """Tests for EvaluationModule.evaluate_rqd."""

    def test_zero_error_when_perfect(self) -> None:
        module = EvaluationModule({})
        preds = [_make_rqd_result("img1", 75.0)]
        gt = {"img1": 75.0}
        metrics = module.evaluate_rqd(preds, gt)
        assert metrics.mean_absolute_error_pct == pytest.approx(0.0)

    def test_correct_mae(self) -> None:
        module = EvaluationModule({})
        preds = [
            _make_rqd_result("img1", 80.0),
            _make_rqd_result("img2", 60.0),
        ]
        gt = {"img1": 70.0, "img2": 70.0}
        metrics = module.evaluate_rqd(preds, gt)
        assert metrics.mean_absolute_error_pct == pytest.approx(10.0)

    def test_missing_gt_skipped(self) -> None:
        module = EvaluationModule({})
        preds = [_make_rqd_result("img1", 75.0), _make_rqd_result("img2", 50.0)]
        gt = {"img1": 75.0}  # img2 missing
        metrics = module.evaluate_rqd(preds, gt)
        assert metrics.num_images_evaluated == 1

    def test_per_image_breakdown_populated(self) -> None:
        module = EvaluationModule({})
        preds = [_make_rqd_result("imgA", 80.0)]
        gt = {"imgA": 60.0}
        metrics = module.evaluate_rqd(preds, gt)
        assert len(metrics.per_image) == 1
        assert metrics.per_image[0]["image_id"] == "imgA"
        assert metrics.per_image[0]["abs_error"] == pytest.approx(20.0)


class TestEvaluationModuleMeasurement:
    """Tests for EvaluationModule.evaluate_measurement."""

    def _make_measurement(self, length_mm: float) -> FragmentMeasurement:
        return FragmentMeasurement(
            image_id="x",
            row_id=0,
            fragment_id=0,
            length_mm=length_mm,
            qualifies_rqd=length_mm >= 100.0,
            rqd_threshold_mm=100.0,
            bbox_px=BBox(x1=0, y1=0, x2=length_mm * 5, y2=30),
            measurement_method="bbox",
        )

    def test_zero_error_perfect_predictions(self) -> None:
        module = EvaluationModule({})
        preds = [self._make_measurement(120.0), self._make_measurement(80.0)]
        gt = [120.0, 80.0]
        metrics = module.evaluate_measurement(preds, gt)
        assert metrics.mae_length_mm == pytest.approx(0.0)
        assert metrics.rmse_length_mm == pytest.approx(0.0)

    def test_length_mismatch_raises(self) -> None:
        module = EvaluationModule({})
        with pytest.raises(ValueError):
            module.evaluate_measurement([self._make_measurement(100.0)], [100.0, 200.0])
