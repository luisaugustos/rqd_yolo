"""Tests for annotation utilities (format conversion and validation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.annotation_utils import (
    ValidationError,
    annotations_from_yolo_file,
    compute_class_distribution,
    validate_annotations,
)
from src.utils.contracts import Annotation, BBox


class TestAnnotationsFromYolo:
    """Tests for YOLO label file parsing."""

    def test_parse_single_annotation(self, tmp_path: Path) -> None:
        label = tmp_path / "img.txt"
        # class 1, centre at (0.5, 0.5), 0.1w × 0.05h in a 640×480 image
        label.write_text("1 0.5 0.5 0.1 0.05\n")
        anns = annotations_from_yolo_file(label, "img", 640, 480, ["fracture", "intact_fragment"])
        assert len(anns) == 1
        assert anns[0].class_id == 1
        assert anns[0].class_name == "intact_fragment"
        assert anns[0].bbox.x1 == pytest.approx(288.0)
        assert anns[0].bbox.y1 == pytest.approx(228.0)

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        label = tmp_path / "empty.txt"
        label.write_text("")
        anns = annotations_from_yolo_file(label, "x", 640, 480, ["a"])
        assert anns == []

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        anns = annotations_from_yolo_file(tmp_path / "nonexistent.txt", "x", 640, 480, [])
        assert anns == []


class TestValidateAnnotations:
    """Tests for annotation geometry validation."""

    def _make_ann(self, x1: float, y1: float, x2: float, y2: float) -> Annotation:
        return Annotation(
            annotation_id=0,
            image_id="img",
            class_id=0,
            class_name="fracture",
            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        )

    def test_valid_annotation_no_errors(self) -> None:
        ann = self._make_ann(10.0, 10.0, 100.0, 50.0)
        errors = validate_annotations([ann], (480, 640))
        assert errors == []

    def test_out_of_bounds_x2_flagged(self) -> None:
        ann = self._make_ann(0.0, 0.0, 700.0, 50.0)  # x2=700 > width=640
        errors = validate_annotations([ann], (480, 640))
        rules = [e.rule for e in errors]
        assert "DQ-004" in rules

    def test_out_of_bounds_y2_flagged(self) -> None:
        ann = self._make_ann(0.0, 0.0, 100.0, 500.0)  # y2=500 > height=480
        errors = validate_annotations([ann], (480, 640))
        rules = [e.rule for e in errors]
        assert "DQ-004" in rules


class TestComputeClassDistribution:
    """Tests for class distribution computation."""

    def test_single_class(self) -> None:
        anns = [
            Annotation(annotation_id=i, image_id="x", class_id=1, class_name="intact_fragment", bbox=BBox(x1=0, y1=0, x2=1, y2=1))
            for i in range(3)
        ]
        dist = compute_class_distribution(anns)
        assert dist == {"intact_fragment": 3}

    def test_multiple_classes(self) -> None:
        anns = [
            Annotation(annotation_id=0, image_id="x", class_id=0, class_name="fracture", bbox=BBox(x1=0, y1=0, x2=1, y2=1)),
            Annotation(annotation_id=1, image_id="x", class_id=1, class_name="intact_fragment", bbox=BBox(x1=0, y1=0, x2=1, y2=1)),
            Annotation(annotation_id=2, image_id="x", class_id=1, class_name="intact_fragment", bbox=BBox(x1=0, y1=0, x2=1, y2=1)),
        ]
        dist = compute_class_distribution(anns)
        assert dist["fracture"] == 1
        assert dist["intact_fragment"] == 2
