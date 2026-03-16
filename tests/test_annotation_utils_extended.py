"""Extended tests for annotation_utils — format conversion (FR-001)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.annotation_utils import coco_to_yolo, yolo_to_coco


class TestYoloToCoco:
    """Tests for yolo_to_coco conversion."""

    def test_output_has_expected_keys(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "img1.txt").write_text("1 0.5 0.5 0.2 0.1\n")
        output = tmp_path / "coco.json"
        yolo_to_coco(label_dir, output, ["fracture", "intact_fragment"])
        data = json.loads(output.read_text())
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

    def test_annotation_count_matches(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "img1.txt").write_text("0 0.1 0.1 0.05 0.05\n1 0.9 0.9 0.05 0.05\n")
        (label_dir / "img2.txt").write_text("1 0.5 0.5 0.2 0.2\n")
        output = tmp_path / "coco.json"
        yolo_to_coco(label_dir, output, ["fracture", "intact_fragment"])
        data = json.loads(output.read_text())
        assert len(data["annotations"]) == 3
        assert len(data["images"]) == 2

    def test_empty_label_file_creates_image_no_annotations(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "img1.txt").write_text("")
        output = tmp_path / "coco.json"
        yolo_to_coco(label_dir, output, ["fracture"])
        data = json.loads(output.read_text())
        assert len(data["images"]) == 1
        assert len(data["annotations"]) == 0


class TestCocoToYolo:
    """Tests for coco_to_yolo conversion."""

    def test_produces_correct_label_files(self, tmp_path: Path) -> None:
        coco_data = {
            "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 50, 200, 100],
                 "area": 20000, "iscrowd": 0}
            ],
            "categories": [{"id": 0, "name": "fracture"}, {"id": 1, "name": "intact_fragment"}],
        }
        coco_json = tmp_path / "coco.json"
        coco_json.write_text(json.dumps(coco_data))
        out_dir = tmp_path / "labels"
        coco_to_yolo(coco_json, out_dir)
        label_file = out_dir / "img1.txt"
        assert label_file.exists()
        lines = label_file.read_text().strip().split("\n")
        assert len(lines) == 1
        parts = lines[0].split()
        assert int(parts[0]) == 1  # category_id
        # cx = (100 + 100) / 640 ≈ 0.3125
        assert float(parts[1]) == pytest.approx((100 + 100) / 640, abs=0.001)

    def test_roundtrip_yolo_coco_yolo(self, tmp_path: Path) -> None:
        """YOLO → COCO → YOLO should preserve class IDs."""
        label_dir = tmp_path / "labels_orig"
        label_dir.mkdir()
        (label_dir / "img1.txt").write_text("1 0.5 0.5 0.2 0.1\n")

        coco_path = tmp_path / "coco.json"
        yolo_to_coco(label_dir, coco_path, ["fracture", "intact_fragment"])

        # Inject image dimensions into COCO (needed for coco_to_yolo)
        data = json.loads(coco_path.read_text())
        data["images"][0]["width"] = 640
        data["images"][0]["height"] = 480
        coco_path.write_text(json.dumps(data))

        out_dir = tmp_path / "labels_out"
        coco_to_yolo(coco_path, out_dir)

        out_labels = list(out_dir.glob("*.txt"))
        assert len(out_labels) == 1
        lines = out_labels[0].read_text().strip().split("\n")
        parts = lines[0].split()
        assert int(parts[0]) == 1  # class_id intact_fragment preserved
