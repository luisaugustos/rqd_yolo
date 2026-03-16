# Interface Contracts

**Project:** rqd-ai-lab
**Phase:** 7 — Interface Contracts
**Version:** 1.0.0
**Date:** 2026-03-16
**Status:** Draft

---

## Overview

This document defines the explicit data contracts between all major modules in the rqd-ai-lab pipeline. Every module boundary has a named schema. These schemas are the authoritative definition of what data flows between modules. Implementations must conform exactly.

Contracts are expressed in Pydantic-style schema notation. All schemas are implemented as Python dataclasses or Pydantic `BaseModel` subclasses in `src/utils/contracts.py`.

---

## IC-001 — ImageSample

**Flows between:** Dataset Module → Preprocessing Module

**Description:** Represents a single raw image and its associated metadata loaded from disk.

```python
class ImageSample(BaseModel):
    image_id: str                      # Unique identifier derived from file path
    file_path: Path                    # Absolute path to the source image
    image: np.ndarray                  # Raw image array, shape (H, W, 3), dtype uint8, RGB
    width: int                         # Image width in pixels
    height: int                        # Image height in pixels
    project_id: str | None             # Originating project identifier
    borehole_id: str | None            # Originating borehole identifier
    depth_from_m: float | None         # Core run start depth (meters)
    depth_to_m: float | None           # Core run end depth (meters)
    run_length_mm: float | None        # Known physical run length (mm); None if not provided
    annotations: List[Annotation]      # Ground truth annotations (may be empty)
    metadata: Dict[str, Any]           # Additional freeform metadata
```

**Invariants:**
- `image.shape == (height, width, 3)`
- `width >= 1024` and `height >= 768` (Warning if violated; not a hard error)
- `image_id` is unique within a dataset split
- If `depth_to_m` is not None, `depth_to_m > depth_from_m`

**Error Conditions:**
- `FileNotFoundError` if `file_path` does not exist at load time
- `ImageReadError` if the file is corrupt or format unsupported

---

## IC-002 — Annotation

**Flows between:** Dataset Module ↔ Evaluation Module (ground truth); all annotation utilities

**Description:** A single annotated object within an image.

```python
class Annotation(BaseModel):
    annotation_id: int                 # Unique annotation ID within an image
    image_id: str                      # Parent image identifier
    class_id: int                      # Integer class ID (0=fracture, 1=intact_fragment, 2=tray_row, 3=scale_marker, ...)
    class_name: str                    # Human-readable class label
    bbox: BBox                         # Bounding box (see IC-003)
    segmentation: List[List[float]] | None   # COCO polygon [[x1,y1,x2,y2,...]] or None
    area: float | None                 # Mask or bbox area in pixels squared
    annotator_id: str | None
    annotation_date: str | None        # ISO 8601 date
```

**Invariants:**
- `class_id` must be in the configured label set
- `bbox.width > 0` and `bbox.height > 0`
- If `segmentation` is not None, each polygon has ≥ 6 values (≥ 3 points)

---

## IC-003 — BBox

**Flows between:** Annotation, DetectionResult, SegmentationResult, FragmentMeasurement

**Description:** An axis-aligned bounding box in absolute pixel coordinates.

```python
class BBox(BaseModel):
    x1: float    # Left edge (pixels from image left)
    y1: float    # Top edge (pixels from image top)
    x2: float    # Right edge
    y2: float    # Bottom edge

    @property
    def width(self) -> float: return self.x2 - self.x1
    @property
    def height(self) -> float: return self.y2 - self.y1
    @property
    def area(self) -> float: return self.width * self.height
    @property
    def center(self) -> Tuple[float, float]: return ((self.x1+self.x2)/2, (self.y1+self.y2)/2)
```

**Invariants:**
- `x2 > x1`
- `y2 > y1`
- All values non-negative
- All values ≤ image width/height (checked at module boundaries)

---

## IC-004 — ProcessedImage

**Flows between:** Preprocessing Module → Detection Module, Segmentation Module, Foundation Model Module

**Description:** A preprocessed image ready for model input, together with metadata needed to invert coordinates back to the original image space.

```python
class PreprocessMetadata(BaseModel):
    original_width: int
    original_height: int
    scale_x: float              # x scale factor applied (processed_width / original_width)
    scale_y: float              # y scale factor applied
    pad_top: int                # pixels of padding added to top
    pad_left: int               # pixels of padding added to left
    target_width: int           # width of processed tensor
    target_height: int          # height of processed tensor
    normalization: str          # e.g. "imagenet", "zero_one", "minus_one_one"

class ProcessedImage(BaseModel):
    image_id: str
    tensor: np.ndarray          # shape (H, W, 3) or (3, H, W) per backend convention; float32
    metadata: PreprocessMetadata
    original_sample: ImageSample
```

**Invariants:**
- `tensor.dtype == np.float32`
- `scale_x > 0.0` and `scale_y > 0.0`
- `pad_top >= 0` and `pad_left >= 0`

---

## IC-005 — CalibrationInfo

**Flows between:** Scale Marker Detection (part of Preprocessing) → Measurement Engine → RQD Engine

**Description:** Pixel-to-millimeter calibration information for an image.

```python
class CalibrationInfo(BaseModel):
    image_id: str
    pixels_per_mm: float           # Conversion factor: how many pixels = 1 mm
    source: Literal["auto", "manual", "metadata"]
    confidence: float | None       # [0,1]; None if source=="manual" or "metadata"
    scale_marker_bbox: BBox | None # Detected scale marker location; None if manual
    reference_length_mm: float | None  # Known physical length of reference object
    reference_length_px: float | None  # Measured pixel length of reference object
    warning: str | None            # Non-empty if calibration is uncertain
```

**Invariants:**
- `pixels_per_mm > 0.0`
- If `source == "auto"`, `scale_marker_bbox` must not be None
- `confidence` in [0.0, 1.0] when not None

**Error Conditions:**
- `CalibrationError` if `pixels_per_mm` is missing, zero, or negative

---

## IC-006 — TrayRow

**Flows between:** Detection Module (tray row detection) → RQD Engine, Measurement Engine

**Description:** A detected tray row region within a core box image.

```python
class TrayRow(BaseModel):
    row_id: int                   # 0-indexed, top-to-bottom ordering
    image_id: str
    bbox: BBox                    # Bounding box of the row in original image coordinates
    row_length_px: float          # Pixel length of the row (horizontal span)
    row_length_mm: float | None   # Physical length in mm; computed from calibration if available
    orientation: Literal["horizontal", "vertical"]  # Default: horizontal
```

**Invariants:**
- `row_id >= 0`
- `row_length_px > 0.0`
- Rows within the same image have non-overlapping bounding boxes

---

## IC-007 — DetectionResult

**Flows between:** Detection Module → Segmentation Module, Measurement Engine, Evaluation Module, Visualization Module

**Description:** All detected objects in a single image, from a single detection model pass.

```python
class DetectionResult(BaseModel):
    image_id: str
    model_name: str                    # e.g. "yolov12m"
    model_backend: str                 # e.g. "yolov12"
    inference_latency_ms: float
    boxes: List[BBox]                  # One per detected object
    scores: List[float]                # Confidence scores, same order as boxes
    class_ids: List[int]               # Class IDs, same order as boxes
    class_names: List[str]             # Class name strings, same order as boxes
    raw_output: Any | None             # Raw model tensor output (optional; for debugging)

    @property
    def num_detections(self) -> int: return len(self.boxes)

    def filter_by_class(self, class_id: int) -> "DetectionResult": ...
    def filter_by_score(self, min_score: float) -> "DetectionResult": ...
```

**Invariants:**
- `len(boxes) == len(scores) == len(class_ids) == len(class_names)`
- All `scores` in [0.0, 1.0]
- `inference_latency_ms >= 0.0`

**Error Conditions:**
- Mismatched list lengths → `ContractViolationError` at construction

---

## IC-008 — SegmentationResult

**Flows between:** Segmentation Module → Measurement Engine, Evaluation Module, Visualization Module

**Description:** Instance segmentation result for a single detected fragment.

```python
class SegmentationResult(BaseModel):
    image_id: str
    fragment_id: int               # Index corresponding to DetectionResult box index
    model_name: str                # e.g. "sam2_vit_b"
    mask: np.ndarray               # Binary mask, shape (H, W), dtype uint8, values {0,1}
    mask_score: float              # Model-reported mask quality score [0,1]
    refined_bbox: BBox             # Tight bounding box around the mask
    prompt_bbox: BBox              # Detection box used as prompt
    mask_area_px: int              # Count of foreground pixels in mask
    inference_latency_ms: float
```

**Invariants:**
- `mask.dtype == np.uint8`
- `mask` values are in {0, 1}
- `mask_area_px == mask.sum()`
- `mask_score` in [0.0, 1.0]
- `refined_bbox` is tightly fit to the mask (no empty rows/cols between bbox edge and mask)

**Error Conditions:**
- `mask_area_px == 0` → logged as WARNING; fragment skipped in measurement

---

## IC-009 — FragmentMeasurement

**Flows between:** Measurement Engine → RQD Engine, Evaluation Module, Visualization Module

**Description:** The measured length of a single core fragment after pixel-to-mm conversion.

```python
class FragmentMeasurement(BaseModel):
    image_id: str
    row_id: int                        # Parent tray row
    fragment_id: int                   # Matches SegmentationResult.fragment_id
    length_mm: float                   # Principal axis length in millimeters
    width_mm: float | None             # Fragment width (perpendicular to principal axis)
    orientation_deg: float             # Angle from horizontal, degrees, range [-90, 90]
    measurement_method: Literal["bbox", "mask_pca", "skeleton", "bbox_fallback"]
    qualifies_rqd: bool                # True if length_mm >= rqd_threshold_mm
    rqd_threshold_mm: float            # Threshold used (default 100.0)
    bbox_px: BBox                      # Source bounding box in original image pixels
    mask_present: bool                 # True if a segmentation mask was used
    calibration_source: Literal["auto", "manual", "metadata"]
```

**Invariants:**
- `length_mm > 0.0`
- `qualifies_rqd == (length_mm >= rqd_threshold_mm)`
- `rqd_threshold_mm > 0.0`

**Error Conditions:**
- `length_mm <= 0.0` → `MeasurementError`
- Missing `CalibrationInfo` → `CalibrationError`

---

## IC-010 — RQDResult

**Flows between:** RQD Engine → Evaluation Module, Visualization Module, Reporting

**Description:** Computed RQD value for a tray row or for a full image.

```python
class RQDResult(BaseModel):
    image_id: str
    scope: Literal["row", "image"]     # "row" = per tray row; "image" = aggregate
    row_id: int | None                 # Set for scope=="row"; None for scope=="image"
    depth_from_m: float | None
    depth_to_m: float | None
    total_run_length_mm: float         # Denominator: total measured run length
    qualifying_length_mm: float        # Numerator: sum of fragment lengths >= threshold
    rqd_pct: float                     # RQD percentage [0, 100]
    num_fragments_total: int
    num_fragments_qualifying: int
    rqd_threshold_mm: float
    fragment_measurements: List[FragmentMeasurement]
    calibration_source: Literal["auto", "manual", "metadata"]
```

**Invariants:**
- `rqd_pct == (qualifying_length_mm / total_run_length_mm) * 100`, clamped to [0, 100]
- `total_run_length_mm > 0.0`
- `qualifying_length_mm <= total_run_length_mm`
- `num_fragments_qualifying <= num_fragments_total`
- For `scope=="image"`: `row_id is None`

**Error Conditions:**
- `total_run_length_mm <= 0.0` → `RQDComputationError`

---

## IC-011 — EvaluationReport

**Flows between:** Evaluation Module → Experiment Tracker, Reporting

**Description:** Full evaluation metrics for a model run on a dataset split.

```python
class DetectionMetrics(BaseModel):
    model_name: str
    split: Literal["train", "val", "test"]
    map_50: float              # mAP at IoU=0.50
    map_50_95: float           # mAP at IoU=0.50:0.95
    precision: float           # At detection threshold
    recall: float
    f1: float
    per_class: Dict[str, Dict[str, float]]   # class_name -> {map_50, precision, recall, f1}
    iou_threshold: float
    conf_threshold: float
    num_images: int
    num_gt_annotations: int

class SegmentationMetrics(BaseModel):
    model_name: str
    split: Literal["train", "val", "test"]
    mean_mask_iou: float
    per_class_mask_iou: Dict[str, float]
    num_instances_evaluated: int

class MeasurementMetrics(BaseModel):
    mae_length_mm: float       # Mean absolute error of fragment length (mm)
    rmse_length_mm: float      # Root mean squared error of fragment length (mm)
    num_fragments_evaluated: int

class RQDMetrics(BaseModel):
    mean_absolute_error_pct: float     # Mean |predicted_rqd - ground_truth_rqd|
    mean_relative_error_pct: float     # Mean |(predicted - truth) / truth| × 100
    rmse_pct: float
    per_image: List[Dict[str, Any]]    # image_id, pred_rqd, gt_rqd, abs_error
    num_images_evaluated: int

class EvaluationReport(BaseModel):
    run_id: str
    timestamp_utc: str                   # ISO 8601
    config_hash: str                     # SHA-256 of experiment config YAML
    git_commit: str | None
    model_name: str
    detection_metrics: DetectionMetrics | None
    segmentation_metrics: SegmentationMetrics | None
    measurement_metrics: MeasurementMetrics | None
    rqd_metrics: RQDMetrics | None
    inference_latency_ms_mean: float
    inference_latency_ms_std: float
    peak_vram_mb: float | None
    notes: str | None
```

**Invariants:**
- All metric values are finite floats (no NaN or Inf)
- `run_id` is non-empty
- At least one of `detection_metrics`, `segmentation_metrics`, `rqd_metrics` is not None

---

## IC-012 — Pipeline Contract: Preprocessing → Detection

**Boundary:** `PreprocessingModule.process()` output → `DetectionModule.detect()` input

| Property          | Requirement                                                               |
|-------------------|---------------------------------------------------------------------------|
| Input type        | `ProcessedImage`                                                          |
| Tensor dtype      | `float32`                                                                 |
| Tensor shape      | `(H, W, 3)` (HWC format); backends convert to `(3, H, W)` internally     |
| Normalization     | As specified by `PreprocessMetadata.normalization`; backend verifies match|
| Coordinate system | Model output boxes are in **processed image space**; caller must invert   |

**Post-condition:** Detection module inverts all output box coordinates to **original image space** using `PreprocessMetadata` before returning `DetectionResult`.

---

## IC-013 — Pipeline Contract: Detection → Measurement

**Boundary:** `DetectionModule.detect()` output → `MeasurementEngine.measure()` input

| Property             | Requirement                                                          |
|----------------------|----------------------------------------------------------------------|
| Input type           | `DetectionResult` + `List[SegmentationResult]` + `CalibrationInfo`  |
| Coordinate system    | All boxes in **original image pixel coordinates**                    |
| Class filter         | Measurement engine processes only `class_id == 1` (intact_fragment) |
| Segmentation pairing | `SegmentationResult.fragment_id` matches index in `DetectionResult.boxes` |
| Calibration required | `CalibrationInfo.pixels_per_mm > 0`; error raised otherwise         |

---

## IC-014 — Pipeline Contract: Segmentation → Measurement

**Boundary:** `SegmentationModule.segment()` output → `MeasurementEngine.measure()` input

| Property          | Requirement                                                               |
|-------------------|---------------------------------------------------------------------------|
| Input type        | `List[SegmentationResult]`                                                |
| Mask alignment    | Masks are in **original image pixel coordinates** at original resolution  |
| Empty mask        | `mask_area_px == 0` → fragment skipped; WARNING logged                   |
| Fallback          | If segmentation disabled, `SegmentationResult` is synthesized from bbox  |

---

## IC-015 — Pipeline Contract: Measurement → RQD Engine

**Boundary:** `MeasurementEngine.measure()` output → `RQDEngine.compute_row_rqd()` input

| Property              | Requirement                                                     |
|-----------------------|-----------------------------------------------------------------|
| Input type            | `List[FragmentMeasurement]` filtered to one `row_id`           |
| Length units          | All `length_mm` values are in millimeters                       |
| Row denominator       | `TrayRow.row_length_mm` or `TrayRow.row_length_px / pixels_per_mm` |
| Empty measurement list| Returns `RQDResult` with `rqd_pct=0.0`, `qualifying_length_mm=0.0` |

---

## IC-016 — Pipeline Contract: RQD Engine → Evaluation

**Boundary:** `RQDEngine` output → `EvaluationModule.evaluate_rqd()` input

| Property              | Requirement                                                     |
|-----------------------|-----------------------------------------------------------------|
| Input type            | `List[RQDResult]` (one per image) + `List[RQDGroundTruth]`     |
| Matching key          | `image_id` used to pair predictions with ground truth           |
| Missing ground truth  | Image skipped; WARNING logged                                   |
| Scope                 | Evaluation at `scope=="image"` level; row-level comparison is optional |

---

## IC-017 — Pipeline Contract: Evaluation → Reporting

**Boundary:** `EvaluationModule.generate_report()` output → Report generation scripts

| Property       | Requirement                                             |
|----------------|---------------------------------------------------------|
| Input type     | `EvaluationReport`                                      |
| Output formats | Markdown (human-readable), CSV (machine-readable), JSON (archival) |
| Artifact paths | All output files saved under `results/reports/<run_id>/` |
| Completeness   | Report must include run_id, config_hash, all requested metrics |
