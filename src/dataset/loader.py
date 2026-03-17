"""Dataset loading, split management and validation (FR-001, NFR-015).

Loads images and YOLO-format annotations, applies versioned splits, and
enforces data quality rules DQ-001 to DQ-014.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.annotation_utils import (
    ValidationError,
    annotations_from_yolo_file,
    compute_class_distribution,
    validate_annotations,
)
from src.utils.contracts import Annotation, ImageSample

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class ValidationReport:
    """Summary of dataset validation results.

    Attributes:
        errors: List of hard-failure ValidationError objects.
        warnings: List of soft-failure ValidationError objects.
        num_images_checked: Total images evaluated.
    """

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    num_images_checked: int = 0

    @property
    def has_errors(self) -> bool:
        """True when there are any hard-failure errors."""
        return len(self.errors) > 0


@dataclass
class DatasetStats:
    """Descriptive statistics for a loaded dataset split.

    Attributes:
        num_images: Number of images in the split.
        num_annotations: Total annotations across all images.
        class_distribution: Mapping from class name to annotation count.
        mean_width: Mean image width in pixels.
        mean_height: Mean image height in pixels.
    """

    num_images: int = 0
    num_annotations: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    mean_width: float = 0.0
    mean_height: float = 0.0


class DatasetLoader:
    """Load, validate and split a drill-core image dataset (FR-001, NFR-015).

    Args:
        config: Dataset configuration dictionary (from dataset.yaml).
        repo_root: Repository root directory. All relative paths are resolved
            against this directory.
    """

    def __init__(self, config: dict[str, Any], repo_root: Path | None = None) -> None:
        self._config = config
        self._root = repo_root or Path.cwd()
        self._class_names = [lbl["name"] for lbl in config.get("labels", [])]
        self._split_data: dict[str, list[str]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_split(self, split: str) -> list[ImageSample]:
        """Load all images for the requested split.

        Args:
            split: One of 'train', 'val', or 'test'.

        Returns:
            List of ImageSample objects for the split.

        Raises:
            ValueError: When an unrecognised split name is provided.
            FileNotFoundError: When the split definition file is missing.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split '{split}'; must be one of train/val/test")

        split_entries = self._get_split_entries(split)
        samples: list[ImageSample] = []
        for rel_path in split_entries:
            sample = self._load_one(rel_path)
            if sample is not None:
                samples.append(sample)
        logger.info("Loaded %d samples for split '%s'", len(samples), split)
        return samples

    def validate(self) -> ValidationReport:
        """Run all data quality checks across all splits.

        Returns:
            ValidationReport with errors and warnings.
        """
        report = ValidationReport()
        for split in ("train", "val", "test"):
            entries = self._get_split_entries(split)
            for rel_path in entries:
                report.num_images_checked += 1
                errs, warns = self._validate_entry(rel_path)
                report.errors.extend(errs)
                report.warnings.extend(warns)
        logger.info(
            "Validation complete: %d errors, %d warnings across %d images",
            len(report.errors),
            len(report.warnings),
            report.num_images_checked,
        )
        return report

    def stats(self, split: str) -> DatasetStats:
        """Compute descriptive statistics for a split without loading images into memory.

        Args:
            split: One of 'train', 'val', 'test'.

        Returns:
            DatasetStats for the split.
        """
        samples = self.load_split(split)
        all_anns: list[Annotation] = []
        widths: list[float] = []
        heights: list[float] = []
        for s in samples:
            all_anns.extend(s.annotations)
            widths.append(float(s.width))
            heights.append(float(s.height))
        return DatasetStats(
            num_images=len(samples),
            num_annotations=len(all_anns),
            class_distribution=compute_class_distribution(all_anns),
            mean_width=float(np.mean(widths)) if widths else 0.0,
            mean_height=float(np.mean(heights)) if heights else 0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_split_entries(self, split: str) -> list[str]:
        """Load and cache the split definition YAML."""
        if self._split_data is None:
            import yaml

            split_file = self._root / self._config["split_file"]
            if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}")
            raw = yaml.safe_load(split_file.read_text())
            self._split_data = raw.get("splits", {})
        return self._split_data.get(split, [])

    def _load_one(self, rel_path: str) -> ImageSample | None:
        """Load a single image and its annotations.

        Returns None when the image cannot be read (logs error).
        """
        img_path = self._root / rel_path
        if not img_path.exists():
            logger.error("DQ-001: Image not found: %s", img_path)
            return None
        if img_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            logger.error("DQ-001: Unsupported format for %s", img_path)
            return None

        image = _read_image(img_path)
        if image is None:
            logger.error("DQ-001: Could not read image: %s", img_path)
            return None

        h, w = image.shape[:2]
        min_w = self._config.get("validation", {}).get("min_width", 1024)
        min_h = self._config.get("validation", {}).get("min_height", 768)
        if w < min_w or h < min_h:
            logger.warning("DQ-002: Image %s resolution %dx%d below minimum %dx%d", rel_path, w, h, min_w, min_h)

        image_id = _path_to_id(rel_path)
        label_dir = self._root / self._config["paths"]["annotations_yolo"] / "labels"
        label_path = label_dir / f"{img_path.stem}.txt"
        anns = annotations_from_yolo_file(label_path, image_id, w, h, self._class_names)

        return ImageSample(
            image_id=image_id,
            file_path=str(img_path),
            image=image,
            width=w,
            height=h,
            annotations=anns,
        )

    def _validate_entry(self, rel_path: str) -> tuple[list[ValidationError], list[ValidationError]]:
        """Run DQ checks for a single image entry without storing pixel data.

        Returns:
            Tuple of (errors, warnings).
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        img_path = self._root / rel_path

        if not img_path.exists():
            errors.append(ValidationError(-1, "DQ-001", f"Missing: {img_path}"))
            return errors, warnings

        image = _read_image(img_path)
        if image is None:
            errors.append(ValidationError(-1, "DQ-001", f"Corrupt: {img_path}"))
            return errors, warnings

        h, w = image.shape[:2]
        image_id = _path_to_id(rel_path)
        label_dir = self._root / self._config["paths"]["annotations_yolo"] / "labels"
        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            errors.append(ValidationError(-1, "DQ-003", f"Missing label file for {img_path.name}"))
            return errors, warnings

        anns = annotations_from_yolo_file(label_path, image_id, w, h, self._class_names)
        geom_errors = validate_annotations(anns, (h, w))
        for ge in geom_errors:
            if ge.rule in ("DQ-004", "DQ-005"):
                errors.append(ge)
            else:
                warnings.append(ge)

        return errors, warnings


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _read_image(path: Path) -> np.ndarray | None:
    """Read an image from disk as an RGB uint8 array.

    Args:
        path: Path to the image file.

    Returns:
        (H, W, 3) uint8 array in RGB order, or None on failure.
    """
    try:
        from PIL import Image

        with Image.open(path) as im:
            rgb = im.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)
    except Exception as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return None


def _path_to_id(rel_path: str) -> str:
    """Derive a stable image ID from a relative file path."""
    return Path(rel_path).stem
