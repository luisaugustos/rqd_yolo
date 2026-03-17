"""Data contracts (IC-001 to IC-011) shared across all pipeline modules.

These Pydantic models define the authoritative schema for every inter-module
data transfer object. All pipeline modules must accept and return these types
at their public boundaries.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# IC-003 — BBox
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Axis-aligned bounding box in absolute pixel coordinates (IC-003).

    Args:
        x1: Left edge in pixels (origin at top-left of image).
        y1: Top edge in pixels.
        x2: Right edge in pixels.
        y2: Bottom edge in pixels.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_positive_dimensions(self) -> "BBox":
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) must be > x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) must be > y1 ({self.y1})")
        return self

    @property
    def width(self) -> float:
        """Width of the bounding box in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of the bounding box in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Area of the bounding box in pixels squared."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Centre point (cx, cy) of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return box as (x, y, width, height) — COCO format."""
        return (self.x1, self.y1, self.width, self.height)

    def to_list(self) -> list[float]:
        """Return [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BBox":
        """Construct from COCO (x, y, width, height) format."""
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)

    @classmethod
    def from_yolo(
        cls, cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
    ) -> "BBox":
        """Construct from YOLO normalised (cx, cy, w, h) format."""
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)


# ---------------------------------------------------------------------------
# IC-002 — Annotation
# ---------------------------------------------------------------------------


class Annotation(BaseModel):
    """A single annotated object within an image (IC-002).

    Args:
        annotation_id: Unique annotation ID within the image.
        image_id: Parent image identifier.
        class_id: Integer class ID from the configured label set.
        class_name: Human-readable class label string.
        bbox: Axis-aligned bounding box in absolute pixel coordinates.
        segmentation: Optional COCO polygon [[x1,y1,x2,y2,...]] or None.
        area: Mask or bbox area in pixels squared; computed from bbox if None.
        annotator_id: Optional annotator identifier.
        annotation_date: Optional ISO 8601 date string.
    """

    annotation_id: int
    image_id: str
    class_id: int
    class_name: str
    bbox: BBox
    segmentation: list[list[float]] | None = None
    area: float | None = None
    annotator_id: str | None = None
    annotation_date: str | None = None

    @field_validator("segmentation")
    @classmethod
    def _check_polygon_points(
        cls, v: list[list[float]] | None
    ) -> list[list[float]] | None:
        if v is not None:
            for poly in v:
                if len(poly) < 6:
                    raise ValueError("Each segmentation polygon needs >= 3 points (6 values).")
        return v

    def effective_area(self) -> float:
        """Return area; falls back to bbox area when annotation area is None."""
        return self.area if self.area is not None else self.bbox.area


# ---------------------------------------------------------------------------
# IC-001 — ImageSample  (numpy array stored outside Pydantic)
# ---------------------------------------------------------------------------


class ImageSample:
    """A single raw image with its associated metadata (IC-001).

    Stores the numpy image array outside of Pydantic to avoid serialisation
    issues with large arrays.

    Args:
        image_id: Unique identifier derived from the file path.
        file_path: Absolute path to the source image.
        image: Raw RGB image array of shape (H, W, 3), dtype uint8.
        width: Image width in pixels.
        height: Image height in pixels.
        project_id: Originating project identifier (optional).
        borehole_id: Originating borehole identifier (optional).
        depth_from_m: Core run start depth in metres (optional).
        depth_to_m: Core run end depth in metres (optional).
        run_length_mm: Known physical run length in mm (optional).
        annotations: Ground truth annotations (may be empty).
        metadata: Additional free-form metadata.
    """

    def __init__(
        self,
        image_id: str,
        file_path: str,
        image: np.ndarray,
        width: int,
        height: int,
        project_id: str | None = None,
        borehole_id: str | None = None,
        depth_from_m: float | None = None,
        depth_to_m: float | None = None,
        run_length_mm: float | None = None,
        annotations: list[Annotation] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if image.shape != (height, width, 3):
            raise ValueError(
                f"image.shape {image.shape} does not match (height={height}, width={width}, 3)"
            )
        if depth_from_m is not None and depth_to_m is not None:
            if depth_to_m <= depth_from_m:
                raise ValueError("depth_to_m must be > depth_from_m")
        self.image_id = image_id
        self.file_path = file_path
        self.image = image
        self.width = width
        self.height = height
        self.project_id = project_id
        self.borehole_id = borehole_id
        self.depth_from_m = depth_from_m
        self.depth_to_m = depth_to_m
        self.run_length_mm = run_length_mm
        self.annotations: list[Annotation] = annotations or []
        self.metadata: dict[str, Any] = metadata or {}


# ---------------------------------------------------------------------------
# IC-004 — ProcessedImage
# ---------------------------------------------------------------------------


class PreprocessMetadata(BaseModel):
    """Preprocessing transform parameters needed to invert coordinates (IC-004).

    Args:
        original_width: Width of the source image before preprocessing.
        original_height: Height of the source image before preprocessing.
        scale_x: Horizontal scale factor applied (processed_w / original_w).
        scale_y: Vertical scale factor applied.
        pad_top: Pixels of top-padding added.
        pad_left: Pixels of left-padding added.
        target_width: Width of the output tensor.
        target_height: Height of the output tensor.
        normalization: Normalization scheme applied (e.g. 'imagenet', 'zero_one').
    """

    original_width: int
    original_height: int
    scale_x: float
    scale_y: float
    pad_top: int
    pad_left: int
    target_width: int
    target_height: int
    normalization: str

    @model_validator(mode="after")
    def _check_positive_scales(self) -> "PreprocessMetadata":
        if self.scale_x <= 0:
            raise ValueError("scale_x must be positive")
        if self.scale_y <= 0:
            raise ValueError("scale_y must be positive")
        return self


class ProcessedImage:
    """A preprocessed image tensor ready for model input (IC-004).

    Args:
        image_id: Source image identifier.
        tensor: Float32 array of shape (H, W, 3).
        metadata: Preprocessing metadata for coordinate inversion.
        original_sample: Reference to the source ImageSample.
    """

    def __init__(
        self,
        image_id: str,
        tensor: np.ndarray,
        metadata: PreprocessMetadata,
        original_sample: ImageSample,
    ) -> None:
        if tensor.dtype != np.float32:
            raise ValueError(f"tensor dtype must be float32, got {tensor.dtype}")
        self.image_id = image_id
        self.tensor = tensor
        self.metadata = metadata
        self.original_sample = original_sample


# ---------------------------------------------------------------------------
# IC-005 — CalibrationInfo
# ---------------------------------------------------------------------------


class CalibrationInfo(BaseModel):
    """Pixel-to-millimetre calibration for one image (IC-005).

    Args:
        image_id: Corresponding image identifier.
        pixels_per_mm: Conversion factor: pixels that equal 1 mm.
        source: How the calibration was obtained.
        confidence: Model confidence in [0, 1] when source is 'auto'.
        scale_marker_bbox: Detected scale marker location (auto only).
        reference_length_mm: Known physical length of the reference object.
        reference_length_px: Measured pixel length of the reference object.
        warning: Non-empty string if calibration is uncertain.
    """

    image_id: str
    pixels_per_mm: float
    source: Literal["auto", "manual", "metadata"]
    confidence: float | None = None
    scale_marker_bbox: BBox | None = None
    reference_length_mm: float | None = None
    reference_length_px: float | None = None
    warning: str | None = None

    @model_validator(mode="after")
    def _check_positive_factor(self) -> "CalibrationInfo":
        if self.pixels_per_mm <= 0:
            raise ValueError(f"pixels_per_mm must be positive, got {self.pixels_per_mm}")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
        return self


# ---------------------------------------------------------------------------
# IC-006 — TrayRow
# ---------------------------------------------------------------------------


class TrayRow(BaseModel):
    """A detected tray row region within a core box image (IC-006).

    Args:
        row_id: Zero-indexed row number, ordered top-to-bottom.
        image_id: Parent image identifier.
        bbox: Bounding box of the row in original image pixel coordinates.
        row_length_px: Pixel span of the row along its principal axis.
        row_length_mm: Physical length in mm after calibration (optional).
        orientation: 'horizontal' (default) or 'vertical'.
    """

    row_id: int
    image_id: str
    bbox: BBox
    row_length_px: float
    row_length_mm: float | None = None
    orientation: Literal["horizontal", "vertical"] = "horizontal"

    @field_validator("row_id")
    @classmethod
    def _check_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("row_id must be >= 0")
        return v

    @field_validator("row_length_px")
    @classmethod
    def _check_positive_length(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("row_length_px must be positive")
        return v


# ---------------------------------------------------------------------------
# IC-007 — DetectionResult
# ---------------------------------------------------------------------------


class DetectionResult:
    """All detected objects in a single image from one model pass (IC-007).

    Args:
        image_id: Source image identifier.
        model_name: Model variant name (e.g. 'yolov12m').
        model_backend: Backend registry key (e.g. 'yolov12').
        inference_latency_ms: Wall-clock time for detection in milliseconds.
        boxes: List of detected bounding boxes.
        scores: Confidence scores, same order as boxes.
        class_ids: Integer class IDs, same order as boxes.
        class_names: Class label strings, same order as boxes.
        raw_output: Raw model tensor output for debugging (optional).
    """

    def __init__(
        self,
        image_id: str,
        model_name: str,
        model_backend: str,
        inference_latency_ms: float,
        boxes: list[BBox],
        scores: list[float],
        class_ids: list[int],
        class_names: list[str],
        raw_output: Any = None,
    ) -> None:
        n = len(boxes)
        if not (len(scores) == n and len(class_ids) == n and len(class_names) == n):
            raise ValueError(
                "boxes, scores, class_ids and class_names must have the same length"
            )
        self.image_id = image_id
        self.model_name = model_name
        self.model_backend = model_backend
        self.inference_latency_ms = inference_latency_ms
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        self.class_names = class_names
        self.raw_output = raw_output

    @property
    def num_detections(self) -> int:
        """Number of detected objects."""
        return len(self.boxes)

    def filter_by_class(self, class_id: int) -> "DetectionResult":
        """Return a new DetectionResult containing only detections with the given class ID."""
        indices = [i for i, c in enumerate(self.class_ids) if c == class_id]
        return DetectionResult(
            image_id=self.image_id,
            model_name=self.model_name,
            model_backend=self.model_backend,
            inference_latency_ms=self.inference_latency_ms,
            boxes=[self.boxes[i] for i in indices],
            scores=[self.scores[i] for i in indices],
            class_ids=[self.class_ids[i] for i in indices],
            class_names=[self.class_names[i] for i in indices],
        )

    def filter_by_score(self, min_score: float) -> "DetectionResult":
        """Return a new DetectionResult containing only detections above min_score."""
        indices = [i for i, s in enumerate(self.scores) if s >= min_score]
        return DetectionResult(
            image_id=self.image_id,
            model_name=self.model_name,
            model_backend=self.model_backend,
            inference_latency_ms=self.inference_latency_ms,
            boxes=[self.boxes[i] for i in indices],
            scores=[self.scores[i] for i in indices],
            class_ids=[self.class_ids[i] for i in indices],
            class_names=[self.class_names[i] for i in indices],
        )


# ---------------------------------------------------------------------------
# IC-008 — SegmentationResult
# ---------------------------------------------------------------------------


class SegmentationResult:
    """Instance segmentation result for a single detected fragment (IC-008).

    Args:
        image_id: Source image identifier.
        fragment_id: Index matching DetectionResult.boxes.
        model_name: Segmentation model name.
        mask: Binary mask of shape (H, W), dtype uint8, values {0, 1}.
        mask_score: Model-reported mask quality score in [0, 1].
        refined_bbox: Tight bounding box around the foreground mask pixels.
        prompt_bbox: Detection bounding box used as the segmentation prompt.
        mask_area_px: Count of foreground pixels (must equal mask.sum()).
        inference_latency_ms: Per-instance segmentation time in ms.
    """

    def __init__(
        self,
        image_id: str,
        fragment_id: int,
        model_name: str,
        mask: np.ndarray,
        mask_score: float,
        refined_bbox: BBox,
        prompt_bbox: BBox,
        mask_area_px: int,
        inference_latency_ms: float,
    ) -> None:
        if mask.dtype != np.uint8:
            raise ValueError(f"mask dtype must be uint8, got {mask.dtype}")
        if not set(np.unique(mask)).issubset({0, 1}):
            raise ValueError("mask values must be in {0, 1}")
        self.image_id = image_id
        self.fragment_id = fragment_id
        self.model_name = model_name
        self.mask = mask
        self.mask_score = mask_score
        self.refined_bbox = refined_bbox
        self.prompt_bbox = prompt_bbox
        self.mask_area_px = mask_area_px
        self.inference_latency_ms = inference_latency_ms


# ---------------------------------------------------------------------------
# IC-009 — FragmentMeasurement
# ---------------------------------------------------------------------------


class FragmentMeasurement(BaseModel):
    """Measured length (in mm) of a single core fragment (IC-009).

    Args:
        image_id: Source image identifier.
        row_id: Parent tray row index.
        fragment_id: Matches SegmentationResult.fragment_id.
        length_mm: Principal axis length in millimetres.
        width_mm: Fragment width perpendicular to principal axis (optional).
        orientation_deg: Angle from horizontal in degrees in [-90, 90].
        measurement_method: How the length was computed.
        qualifies_rqd: True when length_mm >= rqd_threshold_mm.
        rqd_threshold_mm: Threshold used (default 100.0 mm).
        bbox_px: Source bounding box in original image pixels.
        mask_present: True when a segmentation mask was available.
        calibration_source: How calibration was obtained.
    """

    image_id: str
    row_id: int
    fragment_id: int
    length_mm: float
    width_mm: float | None = None
    orientation_deg: float = 0.0
    measurement_method: Literal["bbox", "mask_pca", "skeleton", "bbox_fallback"] = "bbox"
    qualifies_rqd: bool
    rqd_threshold_mm: float = 100.0
    bbox_px: BBox
    mask_present: bool = False
    calibration_source: Literal["auto", "manual", "metadata"] = "manual"

    @model_validator(mode="after")
    def _check_qualifies_consistency(self) -> "FragmentMeasurement":
        expected = self.length_mm >= self.rqd_threshold_mm
        if self.qualifies_rqd != expected:
            raise ValueError(
                f"qualifies_rqd={self.qualifies_rqd} inconsistent with "
                f"length_mm={self.length_mm} and threshold={self.rqd_threshold_mm}"
            )
        if self.length_mm <= 0:
            raise ValueError(f"length_mm must be positive, got {self.length_mm}")
        return self


# ---------------------------------------------------------------------------
# IC-010 — RQDResult
# ---------------------------------------------------------------------------


class RQDResult(BaseModel):
    """Computed RQD for a tray row or full image (IC-010).

    Args:
        image_id: Source image identifier.
        scope: 'row' for per-row result; 'image' for the aggregate.
        row_id: Set for scope=='row'; None for scope=='image'.
        depth_from_m: Start depth in metres (optional).
        depth_to_m: End depth in metres (optional).
        total_run_length_mm: Denominator — total measured run length.
        qualifying_length_mm: Numerator — sum of qualifying fragment lengths.
        rqd_pct: RQD percentage in [0, 100].
        num_fragments_total: Total fragment count.
        num_fragments_qualifying: Qualifying fragment count.
        rqd_threshold_mm: Threshold used.
        fragment_measurements: Constituent fragment measurements.
        calibration_source: How calibration was obtained.
    """

    image_id: str
    scope: Literal["row", "image"]
    row_id: int | None = None
    depth_from_m: float | None = None
    depth_to_m: float | None = None
    total_run_length_mm: float
    qualifying_length_mm: float
    rqd_pct: float
    num_fragments_total: int
    num_fragments_qualifying: int
    rqd_threshold_mm: float = 100.0
    fragment_measurements: list[FragmentMeasurement] = []
    calibration_source: Literal["auto", "manual", "metadata"] = "manual"

    @model_validator(mode="after")
    def _check_invariants(self) -> "RQDResult":
        if self.total_run_length_mm <= 0:
            raise ValueError("total_run_length_mm must be positive")
        if self.qualifying_length_mm < 0:
            raise ValueError("qualifying_length_mm must be >= 0")
        if self.qualifying_length_mm > self.total_run_length_mm:
            raise ValueError("qualifying_length_mm cannot exceed total_run_length_mm")
        if not (0.0 <= self.rqd_pct <= 100.0):
            raise ValueError(f"rqd_pct must be in [0, 100], got {self.rqd_pct}")
        if self.num_fragments_qualifying > self.num_fragments_total:
            raise ValueError("num_fragments_qualifying cannot exceed num_fragments_total")
        if self.scope == "image" and self.row_id is not None:
            raise ValueError("row_id must be None when scope is 'image'")
        return self
