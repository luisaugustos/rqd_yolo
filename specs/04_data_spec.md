# Data Specification

**Project:** rqd-ai-lab
**Phase:** 4 — Data Spec
**Version:** 1.0.0
**Date:** 2026-03-16
**Status:** Draft

---

## 1. Input Data Types

| Property         | Specification                                                               |
|------------------|-----------------------------------------------------------------------------|
| Primary format   | JPEG (.jpg, .jpeg), PNG (.png), TIFF (.tif, .tiff)                         |
| Color space      | RGB (8-bit per channel); 16-bit TIFF accepted and converted to 8-bit        |
| Minimum resolution | 1024 × 768 pixels                                                         |
| Recommended resolution | ≥ 2048 × 1536 pixels for accurate sub-centimeter fragment detection  |
| Aspect ratio     | Unconstrained; landscape orientation preferred                              |
| Scene content    | One drill core tray per image, photographed from directly above            |
| Depth of field   | Core surface must be in focus; blurred images are flagged during validation |

**Assumption A-1:** Images are photographed orthographically from above. Perspective distortion is assumed negligible for standard core box photography setups.

---

## 2. Directory Conventions

```
data/
├── raw/                        # Original, unmodified images (immutable)
│   ├── <project_id>/
│   │   ├── <borehole_id>/
│   │   │   ├── <depth_from>_<depth_to>.jpg
│   │   │   └── ...
│   └── ...
├── interim/                    # Intermediate products (resized, normalized)
│   └── <project_id>/
├── processed/                  # Final preprocessed images ready for model input
│   └── <project_id>/
└── annotations/
    ├── yolo/                   # YOLO-format label files (.txt)
    │   ├── images/
    │   └── labels/
    ├── coco/                   # COCO-format JSON annotation files
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    ├── splits/                 # Dataset split definition files
    │   └── split_v1.yaml
    └── measurements/           # Expert fragment measurement tables
        └── <project_id>_measurements.csv
```

**Rules:**
- Raw data is read-only. No script may modify files under `data/raw/`.
- Processed data is reproducible from raw data + configuration; it may be regenerated at any time.
- All paths in code reference roots defined in `configs/dataset.yaml`, never hardcoded.

---

## 3. Supported Annotation Formats

### 3.1 YOLO Format

One `.txt` label file per image, same basename as image file.

Each line: `<class_id> <x_center> <y_center> <width> <height>`

- All coordinates are normalized to [0, 1] relative to image width and height.
- Bounding boxes are axis-aligned.
- One object per line.
- Empty label file = image with no annotations (valid for negative samples).

### 3.2 COCO Format

Single JSON file per split with the following top-level keys:

```json
{
  "info": { "version": "1.0", "date_created": "<ISO8601>" },
  "licenses": [],
  "categories": [ { "id": <int>, "name": "<label>", "supercategory": "<super>" } ],
  "images": [ { "id": <int>, "file_name": "<path>", "width": <int>, "height": <int> } ],
  "annotations": [
    {
      "id": <int>,
      "image_id": <int>,
      "category_id": <int>,
      "bbox": [x, y, width, height],   // COCO format: top-left origin, absolute pixels
      "segmentation": [[x1,y1,x2,y2,...]] | null,
      "area": <float>,
      "iscrowd": 0
    }
  ]
}
```

### 3.3 Fragment Measurement Table

CSV file with one row per measured fragment:

```
image_id, row_index, fragment_id, length_mm, width_mm, qualifies_rqd, annotator_id, annotation_date
```

### 3.4 RQD Result Table

CSV file with one row per core tray row:

```
image_id, row_index, depth_from_m, depth_to_m, run_length_mm, qualifying_length_mm, rqd_pct, source, annotator_id, annotation_date
```

- `source`: `manual` | `automated` | `automated_reviewed`

---

## 4. Required Labels

| Class ID | Label Name        | Supercategory | Description                                                           |
|----------|-------------------|---------------|-----------------------------------------------------------------------|
| 0        | `fracture`        | discontinuity | Any visible break or crack separating two fragments                   |
| 1        | `intact_fragment` | fragment      | A continuous, unbroken section of core                                |
| 2        | `tray_row`        | structure     | A single linear channel / row within the core tray                   |
| 3        | `scale_marker`    | reference     | A reference object of known physical size for pixel-to-mm calibration|

These four labels are the **minimum required** for the RQD pipeline to function.

---

## 5. Optional Labels

| Class ID | Label Name             | Supercategory | Description                                                          |
|----------|------------------------|---------------|----------------------------------------------------------------------|
| 4        | `natural_fracture`     | discontinuity | Fracture of geological origin                                        |
| 5        | `mechanical_fracture`  | discontinuity | Fracture induced by the drilling process                             |
| 6        | `uncertain_fracture`   | discontinuity | Fracture whose origin cannot be confidently determined               |

**Note:** If optional labels are present in annotations, the system uses them to refine fracture classification. If absent, all fractures default to the `fracture` class.

---

## 6. Dataset Splits

Splits are defined by a versioned YAML file committed to the repository.

### 6.1 Split Schema

```yaml
version: "1.0"
seed: 42
split_date: "2026-03-16"
ratios:
  train: 0.70
  val: 0.15
  test: 0.15
splits:
  train:
    - images/project_A/bh01/0.0_1.5.jpg
    - ...
  val:
    - ...
  test:
    - ...
```

### 6.2 Split Rules

- Splits are stratified by borehole / project to prevent data leakage between train and test sets.
- Images from the same borehole must not appear in more than one split.
- The test split is held out and never used for model selection or hyperparameter tuning.
- Split files are versioned (`split_v1.yaml`, `split_v2.yaml`, ...); earlier versions are not deleted.

---

## 7. Metadata Expectations

Each image should be accompanied by a sidecar metadata entry (YAML or CSV row):

```yaml
image_id: "project_A_bh01_0.0_1.5"
file_path: "raw/project_A/bh01/0.0_1.5.jpg"
project_id: "project_A"
borehole_id: "bh01"
depth_from_m: 0.0
depth_to_m: 1.5
run_length_mm: 1500.0          # known from drilling; used as denominator fallback
num_tray_rows: 2
scale_marker_present: true
scale_marker_type: "ruler_30cm"
camera_model: "Canon EOS R5"   # optional
image_date: "2025-11-01"
annotator_id: "GE_001"
annotation_status: "complete"  # pending | in_progress | complete | reviewed
```

**[AMBIGUOUS]** — Not all datasets will have complete sidecar metadata. Missing fields are permitted but must be flagged during validation. `run_length_mm` is required for RQD denominator computation; if missing, the tray row pixel span × calibration factor is used as a fallback.

---

## 8. Augmentation Policy

Augmentations are applied during training only. They are disabled during validation and test evaluation.

| Augmentation              | Default Parameters                                         | Rationale                                                  |
|---------------------------|------------------------------------------------------------|------------------------------------------------------------|
| Horizontal flip           | p=0.5                                                      | Core rows can be photographed in either orientation        |
| Random brightness/contrast| brightness ±0.2, contrast ±0.2, p=0.5                    | Compensates for variable lighting conditions               |
| Hue/Saturation shift      | hue ±10, saturation ±20, p=0.3                            | Handles color variation in different core types            |
| Random crop               | min_area_ratio=0.8, p=0.3                                  | Simulates partially visible trays                          |
| Gaussian blur             | kernel 3–7, sigma 0.1–2.0, p=0.2                          | Simulates out-of-focus images                              |
| JPEG compression noise    | quality 60–100, p=0.2                                      | Simulates compression artifacts from field cameras         |
| Vertical flip             | **disabled** (p=0.0)                                       | Core depth direction must be preserved                     |
| Random rotation           | ±5°, p=0.2                                                 | Minor camera tilt compensation                             |
| Mosaic                    | p=0.0 (disabled by default)                                | [AMBIGUOUS] May help with small fragment detection; opt-in |

All augmentation parameters are configurable via `configs/dataset.yaml`. Augmentation is seeded from the global experiment seed (NFR-016).

---

## 9. Data Quality Rules

The following rules are applied during dataset validation (`scripts/validate_dataset.py`):

| Rule ID | Rule Description                                                                          | Severity  |
|---------|-------------------------------------------------------------------------------------------|-----------|
| DQ-001  | Image file is readable and not corrupt                                                    | Error     |
| DQ-002  | Image resolution ≥ 1024 × 768 pixels                                                     | Warning   |
| DQ-003  | Annotation file exists for every image in the split                                       | Error     |
| DQ-004  | All bounding box coordinates are within [0, 1] (YOLO) or [0, W/H] (COCO)                | Error     |
| DQ-005  | No bounding box has zero or negative width or height                                      | Error     |
| DQ-006  | All class IDs in annotation files are present in the configured label set                 | Error     |
| DQ-007  | At least one `intact_fragment` annotation is present per image (if not a negative sample) | Warning   |
| DQ-008  | At least one `tray_row` annotation is present per image                                   | Warning   |
| DQ-009  | Scale marker present in image OR manual calibration value provided in metadata            | Warning   |
| DQ-010  | No duplicate image IDs within a split                                                     | Error     |
| DQ-011  | No image ID appears in more than one split                                                | Error     |
| DQ-012  | Fragment measurement table `length_mm` values are positive                                | Error     |
| DQ-013  | RQD values in ground truth table are in range [0, 100]                                   | Error     |
| DQ-014  | Segmentation polygons (if present) are valid (≥ 3 points, no self-intersections)         | Warning   |

**Behavior:** Errors halt processing for the affected image with a logged message. Warnings are logged but processing continues. A validation summary report is produced listing all violations.

---

## 10. Annotation QA Process

### 10.1 Annotation Workflow

1. Images are distributed to annotators via a configured annotation tool (LabelImg, CVAT, or equivalent).
2. Annotators label all four required classes per image using bounding boxes. Segmentation polygons for `intact_fragment` are optional but preferred.
3. Completed annotations are exported in YOLO format and stored under `data/annotations/yolo/`.
4. A second annotator reviews a random 10% sample of each annotator's work (inter-annotator QA).

### 10.2 Inter-Annotator Agreement

- IoU threshold for agreement: ≥ 0.5 for bounding boxes.
- Disagreement on fragment count ≥ 2 fragments triggers full re-annotation of the image.
- Inter-annotator agreement (IAA) is computed per class and reported in the data validation report.
- Target IAA: ≥ 0.75 (Cohen's Kappa equivalent on detection task).

**[AMBIGUOUS]** — Formal inter-annotator agreement protocol depends on available annotator resources. The target of 10% review sample is an assumption and should be adjusted to available resources.

### 10.3 Annotation Tool Configuration

- Scale: annotators must annotate the `scale_marker` class first to anchor their spatial sense.
- Fragment boundaries: annotators draw bounding boxes tight to the visible fragment extent, not to the tray channel boundary.
- Fracture classification: annotators use `fracture` unless they are confident in distinguishing `natural_fracture` vs. `mechanical_fracture`.

### 10.4 Version Control for Annotations

- Annotation files are committed to the repository (or a DVC-tracked store) with version tags.
- Re-annotation of a previously annotated image creates a new annotation version; old versions are retained.
- The active annotation version used for each experiment is recorded in the split file and experiment log.
