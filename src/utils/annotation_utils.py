"""Annotation format conversion and validation utilities.

Supports conversions between YOLO, COCO and internal Annotation types,
as well as per-class statistics and geometry validation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils.contracts import Annotation, BBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format conversions
# ---------------------------------------------------------------------------


def yolo_to_coco(
    yolo_dir: Path,
    output_path: Path,
    class_names: list[str],
    image_dir: Path | None = None,
) -> None:
    """Convert a YOLO-format annotation directory to a COCO JSON file.

    Each image must have a corresponding .txt label file with the same basename.
    Images without a label file are included as images with zero annotations.

    Args:
        yolo_dir: Directory containing YOLO .txt label files.
        output_path: Destination COCO JSON file path.
        class_names: Ordered list of class name strings (index = class_id).
        image_dir: Optional directory of image files; used to resolve resolution.
            When None, width and height are set to 0 in the COCO JSON.
    """
    from datetime import datetime, timezone

    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    label_files = sorted(yolo_dir.glob("*.txt"))
    for img_id, label_path in enumerate(label_files, start=1):
        w, h = _resolve_image_size(label_path.stem, image_dir)
        images.append({"id": img_id, "file_name": label_path.stem, "width": w, "height": h})

        text = label_path.read_text().strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                logger.warning("Skipping malformed YOLO line in %s: %s", label_path, line)
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), *[float(p) for p in parts[1:5]]
            if w > 0 and h > 0:
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                abs_w, abs_h = bw * w, bh * h
            else:
                x1, y1, abs_w, abs_h = cx, cy, bw, bh
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x1, y1, abs_w, abs_h],
                    "segmentation": [],
                    "area": abs_w * abs_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "info": {
            "version": "1.0",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2))
    logger.info("Wrote COCO JSON with %d images, %d annotations to %s", len(images), len(annotations), output_path)


def coco_to_yolo(coco_json: Path, output_dir: Path) -> None:
    """Convert a COCO JSON annotation file to per-image YOLO .txt files.

    Args:
        coco_json: Path to the COCO format JSON file.
        output_dir: Destination directory for YOLO label .txt files.
    """
    data: dict[str, Any] = json.loads(coco_json.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    id_to_image: dict[int, dict[str, Any]] = {img["id"]: img for img in data["images"]}
    lines_per_image: dict[int, list[str]] = {img["id"]: [] for img in data["images"]}

    for ann in data.get("annotations", []):
        img = id_to_image[ann["image_id"]]
        iw, ih = img["width"], img["height"]
        if iw == 0 or ih == 0:
            logger.warning("Image %s has zero dimension; skipping annotation %d", img["file_name"], ann["id"])
            continue
        x, y, bw, bh = ann["bbox"]
        cx = (x + bw / 2) / iw
        cy = (y + bh / 2) / ih
        nw = bw / iw
        nh = bh / ih
        lines_per_image[ann["image_id"]].append(
            f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )

    for img_id, img in id_to_image.items():
        stem = Path(img["file_name"]).stem
        out_file = output_dir / f"{stem}.txt"
        out_file.write_text("\n".join(lines_per_image[img_id]))

    logger.info("Wrote %d YOLO label files to %s", len(id_to_image), output_dir)


# ---------------------------------------------------------------------------
# Internal annotation objects
# ---------------------------------------------------------------------------


def annotations_from_yolo_file(
    label_path: Path,
    image_id: str,
    image_width: int,
    image_height: int,
    class_names: list[str],
) -> list[Annotation]:
    """Parse a YOLO label .txt file and return a list of Annotation objects.

    Args:
        label_path: Path to the YOLO .txt label file.
        image_id: Identifier of the parent image.
        image_width: Width of the parent image in pixels.
        image_height: Height of the parent image in pixels.
        class_names: Ordered class name list (index = class_id).

    Returns:
        List of Annotation objects. Returns empty list for empty/missing files.
    """
    if not label_path.exists():
        return []
    text = label_path.read_text().strip()
    if not text:
        return []

    anns: list[Annotation] = []
    for ann_id, line in enumerate(text.splitlines()):
        parts = line.split()
        if len(parts) < 5:
            logger.warning("Malformed YOLO line %d in %s", ann_id, label_path)
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        bbox = BBox.from_yolo(cx, cy, bw, bh, image_width, image_height)
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        anns.append(
            Annotation(
                annotation_id=ann_id,
                image_id=image_id,
                class_id=cls_id,
                class_name=cls_name,
                bbox=bbox,
                area=bbox.area,
            )
        )
    return anns


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationError:
    """A single annotation geometry validation error.

    Args:
        annotation_id: ID of the offending annotation.
        rule: Rule code that was violated.
        message: Human-readable description.
    """

    def __init__(self, annotation_id: int, rule: str, message: str) -> None:
        self.annotation_id = annotation_id
        self.rule = rule
        self.message = message

    def __repr__(self) -> str:
        return f"ValidationError(ann={self.annotation_id}, rule={self.rule}: {self.message})"


def validate_annotations(
    anns: list[Annotation], image_shape: tuple[int, int]
) -> list[ValidationError]:
    """Validate annotation geometry against the image bounds.

    Checks:
    - DQ-004: bbox coordinates in valid pixel range.
    - DQ-005: bbox has positive dimensions.
    - DQ-014: segmentation polygon has >= 3 points.

    Args:
        anns: List of Annotation objects to validate.
        image_shape: (height, width) of the parent image in pixels.

    Returns:
        List of ValidationError objects (empty if all pass).
    """
    h, w = image_shape
    errors: list[ValidationError] = []
    for ann in anns:
        b = ann.bbox
        if b.x1 < 0 or b.y1 < 0 or b.x2 > w or b.y2 > h:
            errors.append(
                ValidationError(
                    ann.annotation_id,
                    "DQ-004",
                    f"bbox [{b.x1:.1f},{b.y1:.1f},{b.x2:.1f},{b.y2:.1f}] out of image bounds [{w}x{h}]",
                )
            )
        if ann.segmentation:
            for poly in ann.segmentation:
                if len(poly) < 6:
                    errors.append(
                        ValidationError(ann.annotation_id, "DQ-014", "Polygon has < 3 points")
                    )
    return errors


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_class_distribution(annotations: list[Annotation]) -> dict[str, int]:
    """Count annotations per class name.

    Args:
        annotations: List of Annotation objects.

    Returns:
        Mapping from class_name to annotation count.
    """
    dist: dict[str, int] = {}
    for ann in annotations:
        dist[ann.class_name] = dist.get(ann.class_name, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_image_size(stem: str, image_dir: Path | None) -> tuple[int, int]:
    """Attempt to read image dimensions from disk; return (0, 0) if unavailable."""
    if image_dir is None:
        return 0, 0
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            try:
                from PIL import Image

                with Image.open(candidate) as im:
                    return im.width, im.height
            except Exception:
                pass
    return 0, 0
