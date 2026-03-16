# Functional Specification

**Project:** rqd-ai-lab
**Phase:** 2 — Functional Spec
**Version:** 1.0.0
**Date:** 2026-03-16
**Status:** Draft

---

## Overview

This document defines all functional requirements for the rqd-ai-lab system. Each requirement is assigned a unique identifier in the format `FR-NNN`. Requirements are grouped by functional area.

---

## FR-001 — Image Ingestion

**Title:** Accept drill core box images as input

**Description:** The system must accept digital photographs of drill core boxes in common image formats and load them for downstream processing.

**Rationale:** The primary input to the system is a photographic image. The system must robustly handle varied image sources and formats produced by different cameras and scanners.

**Inputs:**
- File path or directory path pointing to one or more image files
- Supported formats: JPEG, PNG, TIFF
- Minimum resolution: 1024 × 768 pixels (assumption A-7)

**Outputs:**
- Loaded image tensor / array (H × W × C, uint8 or float32)
- Image metadata: filename, resolution, bit depth, color space

**Acceptance Criteria:**
- AC-001-1: System loads JPEG, PNG, and TIFF files without error.
- AC-001-2: System raises a descriptive error for unsupported formats.
- AC-001-3: System handles corrupt or unreadable files gracefully and logs the error.
- AC-001-4: System reports image resolution and bit depth in the metadata output.

**Dependencies:** None

---

## FR-002 — Image Preprocessing

**Title:** Normalize and prepare images for model inference

**Description:** The system must apply configurable preprocessing steps to input images prior to model inference, including resizing, normalization, color space conversion, and padding.

**Rationale:** Different model backends require different input formats. A consistent preprocessing pipeline ensures reproducibility and prevents silent data mismatch errors.

**Inputs:**
- Raw image array from FR-001
- Preprocessing configuration (target size, normalization parameters, interpolation method)

**Outputs:**
- Preprocessed image tensor ready for model input
- Preprocessing metadata (scale factors, padding applied)

**Acceptance Criteria:**
- AC-002-1: Resized images match the configured target resolution.
- AC-002-2: Pixel values are normalized to the configured range (e.g., [0, 1] or [0, 255]).
- AC-002-3: Preprocessing metadata is recorded and accessible for downstream coordinate inversion.
- AC-002-4: Preprocessing is deterministic given the same input and configuration.

**Dependencies:** FR-001

---

## FR-003 — Tray Row Detection

**Title:** Detect and segment individual tray rows within a core box image

**Description:** The system must identify the distinct rows (channels) within a core tray image, separating the image into per-row regions for independent analysis.

**Rationale:** A single core box image typically contains multiple rows of core. RQD must be computed per row (per core run interval). Row detection enables per-row measurement and avoids merging fragments across rows.

**Inputs:**
- Preprocessed image from FR-002
- Tray row detection model or heuristic configuration

**Outputs:**
- List of tray row bounding regions (bounding box or polygon per row)
- Row index and ordering (top-to-bottom)

**Acceptance Criteria:**
- AC-003-1: All visible tray rows in an image are detected.
- AC-003-2: Rows are ordered from top to bottom consistently.
- AC-003-3: Row regions do not overlap.
- AC-003-4: System handles images with 1 to 6 rows (assumption based on typical tray design).

**Dependencies:** FR-002

---

## FR-004 — Scale Marker Detection

**Title:** Detect scale markers or reference objects for pixel-to-mm calibration

**Description:** The system must detect known-size reference objects (rulers, color cards, scale bars) within core box images to determine the pixel-to-millimeter conversion factor.

**Rationale:** Fragment lengths in pixels are meaningless without a conversion to physical units. Scale marker detection enables automatic calibration. Manual calibration must be supported as a fallback (assumption A-3).

**Inputs:**
- Preprocessed image from FR-002
- Scale marker class definition and known physical size (mm)
- Optional: manual calibration override (pixels per mm value)

**Outputs:**
- Detected scale marker bounding box (if found)
- Pixel-to-mm conversion factor
- Calibration source flag: `auto` | `manual`
- Calibration confidence score (if auto)

**Acceptance Criteria:**
- AC-004-1: Scale marker detected in images where one is clearly visible.
- AC-004-2: Manual calibration override is accepted and used when auto detection fails.
- AC-004-3: Calibration factor is propagated to all downstream measurement operations.
- AC-004-4: System logs a warning when calibration source is manual.

**Dependencies:** FR-002

---

## FR-005 — Fragment Detection

**Title:** Detect intact core fragment bounding boxes within each tray row

**Description:** The system must detect all intact core fragments within each tray row using a configurable detection model backend (YOLOv12, YOLOv11, RT-DETRv2, or other).

**Rationale:** Fragment detection is the primary computer vision task. The detected bounding boxes define the candidate objects from which fragment lengths are estimated.

**Inputs:**
- Preprocessed image or per-row crop from FR-003
- Detection model configuration (model family, weights path, confidence threshold, NMS IoU threshold)

**Outputs:**
- List of detection results per row: `[{box: [x1,y1,x2,y2], score: float, class: "intact_fragment"}]`
- Raw detection logits (optional, for analysis)

**Acceptance Criteria:**
- AC-005-1: All fragments with confidence above threshold are returned.
- AC-005-2: Non-maximum suppression is applied with configurable IoU threshold.
- AC-005-3: Output bounding boxes are in original image pixel coordinates (not normalized).
- AC-005-4: Detection model backend is swappable via configuration without changing calling code.
- AC-005-5: Detection results include confidence scores.

**Dependencies:** FR-002, FR-003

---

## FR-006 — Fracture Detection

**Title:** Detect fractures and breaks in core fragments

**Description:** The system must detect fractures (discontinuities) within core tray images. Fractures define boundaries between fragments and are used to validate or refine fragment detection.

**Rationale:** Fractures are the primary geological feature separating fragments. Their detection enables both fragment boundary validation and optional fracture-type classification.

**Inputs:**
- Preprocessed image or per-row crop
- Fracture detection model configuration

**Outputs:**
- List of detected fractures: `[{box: [x1,y1,x2,y2], score: float, class: "fracture"}]`
- Optional: fracture type classification (`natural_fracture`, `mechanical_fracture`, `uncertain_fracture`)

**Acceptance Criteria:**
- AC-006-1: Detected fractures are returned with bounding boxes and confidence scores.
- AC-006-2: Fracture class labels match the configured label set.
- AC-006-3: Fracture detections are in original image pixel coordinates.
- AC-006-4: System can operate in fragment-only mode (fracture detection disabled) via configuration.

**Dependencies:** FR-002, FR-003

---

## FR-007 — Segmentation Refinement

**Title:** Refine fragment boundaries using instance segmentation

**Description:** The system must optionally refine fragment bounding box detections into precise pixel-level segmentation masks, improving the accuracy of length measurements.

**Rationale:** Bounding boxes overestimate fragment extent when fragments are not perfectly rectangular or when fragments partially overlap. Segmentation masks enable more accurate length estimation.

**Inputs:**
- Preprocessed image
- Detection results from FR-005 (used as prompts where applicable)
- Segmentation model configuration (SAM2, Mask R-CNN, U-Net, or none)

**Outputs:**
- Per-fragment instance segmentation mask (binary mask, same resolution as input image crop)
- Refined bounding box (tight box around mask)
- Mask confidence score

**Acceptance Criteria:**
- AC-007-1: Segmentation mask is returned for each detected fragment.
- AC-007-2: Masks are binary (0 or 1) and aligned to the image coordinate system.
- AC-007-3: System falls back to bounding box measurement if segmentation is disabled.
- AC-007-4: Segmentation backend is swappable via configuration.

**Dependencies:** FR-005

---

## FR-008 — Pixel-to-Millimeter Conversion

**Title:** Convert all pixel measurements to millimeter units

**Description:** The system must apply the calibration factor from FR-004 to convert all spatial measurements (fragment lengths, bounding box dimensions) from pixels to millimeters.

**Rationale:** RQD is defined in physical units (mm). All measurements must be expressed in mm for the RQD formula to be applied correctly.

**Inputs:**
- Fragment bounding boxes or segmentation masks in pixel coordinates
- Pixel-to-mm conversion factor from FR-004

**Outputs:**
- Fragment length in millimeters
- Fragment width in millimeters (optional, for quality checks)
- Measurement method: `bbox_length` | `mask_length` | `oriented_bbox_length`

**Acceptance Criteria:**
- AC-008-1: All output fragment lengths are in mm.
- AC-008-2: Conversion is applied consistently using the calibration factor.
- AC-008-3: Measurement method is recorded in the output.
- AC-008-4: System raises an error if calibration factor is missing or zero.

**Dependencies:** FR-004, FR-005, FR-007

---

## FR-009 — Fragment Length Extraction

**Title:** Extract the principal axis length of each detected fragment

**Description:** The system must compute the length of each detected core fragment along its principal axis (longest dimension), accounting for fragment orientation within the tray.

**Rationale:** Core fragments are measured along their longest axis (parallel to the borehole axis, which corresponds to the row direction). Naive bounding box height or width may misestimate length for tilted fragments.

**Inputs:**
- Fragment segmentation mask or bounding box
- Tray row orientation (assumed horizontal by default)
- Pixel-to-mm factor

**Outputs:**
- Principal axis length (mm)
- Orientation angle (degrees from horizontal)
- Length estimation method: `bbox` | `mask_pca` | `skeleton`

**Acceptance Criteria:**
- AC-009-1: Length is computed along the principal axis, not always the vertical or horizontal bounding box dimension.
- AC-009-2: For horizontal fragments (< 10° tilt), length equals the horizontal bounding box span.
- AC-009-3: Orientation angle is recorded for each fragment.
- AC-009-4: Length estimation method is configurable.

**Dependencies:** FR-007, FR-008

---

## FR-010 — Fragment Filtering (≥ 100 mm)

**Title:** Filter fragments meeting the RQD threshold criterion

**Description:** The system must classify each measured fragment as qualifying or non-qualifying for RQD inclusion based on the ISRM threshold of 100 mm.

**Rationale:** The RQD formula only counts fragments with length ≥ 100 mm. This filtering step separates the qualifying sum from total run length.

**Inputs:**
- List of fragment lengths in mm from FR-009
- RQD threshold (default: 100 mm, configurable)

**Outputs:**
- List of qualifying fragments (length ≥ threshold) with lengths
- List of non-qualifying fragments with lengths
- Total qualifying length (mm)

**Acceptance Criteria:**
- AC-010-1: All fragments with length ≥ 100 mm (or configured threshold) are included in qualifying list.
- AC-010-2: Fragments below threshold are excluded from the qualifying sum but retained in output.
- AC-010-3: Threshold is configurable; default is 100 mm.

**Dependencies:** FR-009

---

## FR-011 — Per-Row RQD Calculation

**Title:** Compute RQD for each tray row independently

**Description:** The system must compute RQD independently for each detected tray row, using the qualifying fragment sum and the detected row length as the denominator.

**Rationale:** Each tray row represents a distinct core run interval. Per-row RQD is needed for depth-based geotechnical profiling.

**Inputs:**
- Qualifying fragment lengths per row from FR-010
- Total detected row length (in mm, from tray row detection FR-003 and calibration FR-004)

**Outputs:**
- RQD value per row (percentage, 0–100)
- Qualifying length sum per row (mm)
- Total row length (mm)
- Row index

**Acceptance Criteria:**
- AC-011-1: RQD is computed for each row independently.
- AC-011-2: RQD is clamped to the range [0, 100].
- AC-011-3: Output includes both qualifying and total lengths for transparency.
- AC-011-4: RQD formula matches the ISRM definition.

**Dependencies:** FR-010, FR-003, FR-004

---

## FR-012 — Full-Image RQD Calculation

**Title:** Compute aggregate RQD for the full core box image

**Description:** The system must compute a single aggregate RQD value for the full image by summing qualifying lengths and total lengths across all rows.

**Rationale:** A full-image RQD provides a quick overview and enables comparison with manual tray-level logging when per-row breakdown is not available.

**Inputs:**
- Per-row RQD results from FR-011

**Outputs:**
- Aggregate RQD value (percentage, 0–100)
- Total qualifying length across all rows (mm)
- Total run length across all rows (mm)

**Acceptance Criteria:**
- AC-012-1: Aggregate RQD is computed as the weighted sum of qualifying lengths over total length.
- AC-012-2: Aggregate RQD is not an average of per-row RQD values (must use raw lengths).
- AC-012-3: Output is reported to two decimal places.

**Dependencies:** FR-011

---

## FR-013 — Visualization Overlays

**Title:** Generate annotated visualization images with detection and measurement overlays

**Description:** The system must produce annotated output images overlaying detected fragments, fractures, segmentation masks, scale markers, row boundaries, and RQD values.

**Rationale:** Visual output is essential for geologist review and QA. Overlays enable rapid assessment of detection quality and model errors.

**Inputs:**
- Original image from FR-001
- Detection results from FR-005, FR-006
- Segmentation masks from FR-007
- RQD results from FR-011, FR-012
- Visualization configuration (colors, font size, overlay elements to include)

**Outputs:**
- Annotated image (PNG or JPEG)
- Overlay elements: fragment boxes, fragment lengths (mm), fracture markers, row boundaries, scale marker, per-row RQD text, aggregate RQD text

**Acceptance Criteria:**
- AC-013-1: Each detected fragment is annotated with its length in mm.
- AC-013-2: Qualifying fragments (≥ 100 mm) are visually distinguished from non-qualifying ones.
- AC-013-3: Per-row RQD values are overlaid on the respective rows.
- AC-013-4: Visualization elements are configurable (enable/disable per element).
- AC-013-5: Output image resolution matches input image resolution.

**Dependencies:** FR-005, FR-006, FR-007, FR-011, FR-012

---

## FR-014 — Evaluation Reports

**Title:** Generate quantitative evaluation reports comparing automated results to ground truth

**Description:** The system must compute and report evaluation metrics comparing automated detections, measurements, and RQD values to annotated ground truth.

**Rationale:** Quantitative evaluation is required to validate the system against expert measurements and to compare model families fairly.

**Inputs:**
- Automated detection/segmentation/RQD results
- Ground truth annotations (COCO or YOLO format) and expert RQD values
- Evaluation configuration (metric list, IoU thresholds)

**Outputs:**
- Detection metrics: mAP@0.5, mAP@0.5:0.95, precision, recall, F1 per class
- Segmentation metrics: mask IoU per instance
- Measurement metrics: MAE and RMSE of fragment lengths
- RQD metrics: absolute error, relative error, per-image breakdown
- Summary table (Markdown + CSV)

**Acceptance Criteria:**
- AC-014-1: All metrics are computed on held-out test data only.
- AC-014-2: Report includes per-class and overall metrics.
- AC-014-3: Reports are saved to a configurable output directory.
- AC-014-4: Report format is both human-readable (Markdown) and machine-readable (CSV/JSON).

**Dependencies:** FR-005, FR-006, FR-007, FR-011, FR-012

---

## FR-015 — Experiment Tracking

**Title:** Log all experiments with parameters, metrics, and artifacts

**Description:** The system must log all training and evaluation experiments with full configuration, metrics, and output artifacts to an experiment tracking system.

**Rationale:** Scientific reproducibility requires that every result be traceable to exact parameters, code version, and data snapshot.

**Inputs:**
- Experiment configuration (YAML)
- Training / evaluation metrics
- Output artifacts (model weights, reports, visualizations)

**Outputs:**
- Experiment log entry with: run ID, timestamp, config hash, all metrics, artifact paths
- Compatible with MLflow (primary) or alternative trackers

**Acceptance Criteria:**
- AC-015-1: Every training run creates a unique, timestamped experiment entry.
- AC-015-2: Configuration YAML is logged as an artifact.
- AC-015-3: All evaluation metrics are logged as scalar values.
- AC-015-4: Model checkpoints are linked to the experiment entry.
- AC-015-5: Experiment logs are queryable to compare runs.

**Dependencies:** FR-014

---

## FR-016 — CLI Execution

**Title:** Provide a command-line interface for all major system operations

**Description:** The system must expose all major operations (preprocessing, training, inference, evaluation, RQD computation, report generation) via a CLI.

**Rationale:** CLI access enables automation, scripting, server-side execution, and integration into geotechnical workflows without requiring notebook environments.

**Inputs:**
- Command name and arguments
- Configuration file path (YAML)
- Optional overrides as key=value pairs

**Outputs:**
- Operation results written to configured output directory
- Exit code 0 on success, non-zero on error
- Progress logs to stdout/stderr

**Acceptance Criteria:**
- AC-016-1: CLI provides subcommands for: `ingest`, `preprocess`, `train`, `infer`, `evaluate`, `compute-rqd`, `report`.
- AC-016-2: All options are configurable via YAML file; CLI flags override YAML values.
- AC-016-3: `--help` is available for every subcommand with descriptive documentation.
- AC-016-4: Non-zero exit code is returned on error with a human-readable message.

**Dependencies:** FR-001 through FR-014

---

## FR-017 — Notebook-Based Analysis

**Title:** Provide Jupyter notebooks for interactive exploration and analysis

**Description:** The system must include Jupyter notebooks that demonstrate dataset exploration, model benchmarking, and RQD visualization in an interactive, reproducible format.

**Rationale:** Geologists and researchers may prefer interactive analysis over CLI. Notebooks serve as documentation and reproducible research artifacts.

**Inputs:**
- Dataset directories
- Experiment results
- Model outputs

**Outputs:**
- Rendered notebooks with embedded visualizations
- Exported figures (PNG/SVG) and tables (CSV)

**Acceptance Criteria:**
- AC-017-1: Notebooks execute end-to-end without errors when data is present.
- AC-017-2: Notebooks include narrative text explaining each analysis step.
- AC-017-3: All figures are saved to the results directory.
- AC-017-4: Notebooks are kernel-independent (no hardcoded absolute paths; all paths from config).

**Dependencies:** FR-001 through FR-014
