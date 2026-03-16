# Product / Research Specification

**Project:** rqd-ai-lab
**Phase:** 1 — Product / Research Spec
**Version:** 1.0.0
**Date:** 2026-03-16
**Status:** Draft

---

## 1. Problem Statement

Rock Quality Designation (RQD) is a fundamental geotechnical index used to characterize the quality of a rock mass based on drill core recovery. It is defined as:

```
RQD = (sum of intact core fragment lengths >= 100 mm / total core run length) × 100
```

Currently, RQD is estimated manually by trained geologists who physically measure core fragments in trays. This process is:

- **Slow**: A single borehole campaign may involve hundreds of core boxes requiring hours of manual logging.
- **Subjective**: Different geologists may disagree on fracture identification, leading to inter-analyst variability.
- **Error-prone**: Manual measurement accumulates small errors, especially under field conditions.
- **Not reproducible**: No digital audit trail is created for the measurement decisions made.

This project addresses these limitations by building a research-grade automated system that estimates RQD directly from digital photographs of drill core boxes using state-of-the-art computer vision.

---

## 2. Research Motivation

Automated core analysis is an emerging field with limited openly available benchmarks or reusable systems. Key research gaps addressed by this project:

- Lack of reproducible baseline comparisons across modern detection and segmentation architectures (YOLO, RT-DETR, SAM2, Mask R-CNN, Florence-2) applied to drill core imagery.
- No standardized pixel-to-millimeter calibration pipeline for core box images.
- Insufficient quantitative validation of automated RQD against expert measurements in the literature.
- No modular, configurable, open-source system for this task suitable for further academic research.

This project is designed to produce publishable benchmarks and a reusable research artifact.

---

## 3. Target Users

### 3.1 Geotechnical Engineers
- Use RQD as a primary input for rock mass classification (RMR, Q-system).
- Need reliable, auditable RQD values per core run.
- Require output compatible with existing geotechnical logging workflows.

### 3.2 Geologists
- Responsible for core description and fracture classification.
- Need visual validation overlays to verify automated results.
- May provide expert annotations for ground truth.

### 3.3 ML / Computer Vision Researchers
- Interested in benchmarking detection and segmentation models on a novel domain.
- Require modular, configurable code and reproducible training pipelines.
- Need standard evaluation metrics and experiment tracking.

---

## 4. Primary User Goals

| ID   | User Role               | Goal                                                                 |
|------|-------------------------|----------------------------------------------------------------------|
| UG-1 | Geotechnical Engineer   | Compute RQD automatically from a core box photograph                |
| UG-2 | Geotechnical Engineer   | Export RQD results as a table per core run / tray row               |
| UG-3 | Geologist               | Visually review and verify detected fragments and fractures          |
| UG-4 | Geologist               | Compare automated RQD against manually logged values                 |
| UG-5 | ML Researcher           | Train and evaluate multiple detection/segmentation model families    |
| UG-6 | ML Researcher           | Reproduce published benchmark results from configuration only        |
| UG-7 | ML Researcher           | Extend the system with new model backends without modifying core logic|

---

## 5. Project Scope

The following capabilities are **in scope**:

- Ingestion of drill core box images (JPEG, PNG, TIFF).
- Automatic or calibration-assisted scale determination (pixels per mm).
- Detection of tray rows within a core box image.
- Detection of intact core fragments and fractures.
- Segmentation / boundary refinement of detected fragments.
- Estimation of fragment lengths in millimeters.
- Computation of RQD per tray row and for the full image.
- Comparison of at least three detection model families (YOLOv12, YOLOv11, RT-DETRv2).
- Integration of at least two segmentation / refinement approaches (SAM2, Mask R-CNN).
- Optional foundation model integration (Florence-2, Grounding DINO).
- Quantitative evaluation against expert-annotated ground truth.
- Visualization overlays and result reports.
- Experiment tracking with reproducible configuration.
- CLI interface for all major operations.
- Jupyter notebook interface for exploratory analysis.

---

## 6. Out-of-Scope Items

The following are **explicitly out of scope** for this project:

- 3D core scanning or CT reconstruction.
- Real-time video stream analysis from core scanners.
- Full geotechnical report generation (beyond RQD tables).
- Rock type classification or lithological description.
- Automated borehole depth assignment (assumed to be provided externally).
- Deployment to a production web service or API.
- Integration with commercial borehole logging software.
- Support for non-photographic core data (point clouds, wireline logs).

---

## 7. Success Criteria

| ID   | Criterion                                                                                      | Target                              |
|------|-----------------------------------------------------------------------------------------------|-------------------------------------|
| SC-1 | Reproducible training and inference from configuration files                                  | Seed-locked; results match within ±0.5% RQD across runs |
| SC-2 | Automatic RQD computation from a single input image                                           | End-to-end pipeline functional      |
| SC-3 | Comparison of at least 3 detection model families with standard metrics                       | mAP, Precision, Recall, F1 reported |
| SC-4 | Quantitative validation of automated RQD against expert measurements                         | Mean absolute RQD error < 5 percentage points |
| SC-5 | Pixel-to-mm calibration verified against known scale markers                                  | Calibration error < 2%              |
| SC-6 | Fragment length MAE < 10 mm on held-out test set                                              | Measured on annotated fragments     |
| SC-7 | System runs on a single GPU (≥8 GB VRAM) without modification                                | Verified on standard hardware       |
| SC-8 | All experiments logged and traceable to configuration                                         | MLflow or equivalent tracking active|
| SC-9 | Test coverage ≥ 80% for core logic modules (measurement, RQD engine, calibration)             | CI-enforced                         |

---

## 8. Risks

| ID   | Risk                                                                 | Likelihood | Impact | Mitigation                                                              |
|------|----------------------------------------------------------------------|------------|--------|-------------------------------------------------------------------------|
| R-1  | Insufficient labeled data for training                               | High       | High   | Use pre-trained weights, data augmentation, semi-supervised approaches  |
| R-2  | Scale marker not present or not detectable in all images             | Medium     | High   | Support manual calibration input as fallback                            |
| R-3  | Model overfitting to specific core types or lighting conditions      | Medium     | High   | Cross-validation, diverse augmentation, held-out test sets              |
| R-4  | Inter-annotator disagreement reduces ground truth reliability        | Medium     | Medium | Define annotation protocol; compute and report inter-annotator agreement|
| R-5  | Occlusion or overlap of core fragments in tray                       | Medium     | Medium | Use segmentation refinement; flag uncertain detections                  |
| R-6  | Pixel-to-mm calibration error propagates to RQD error               | Low        | High   | Validate calibration independently; report calibration uncertainty      |
| R-7  | Foundation model (Florence-2) API or license constraints             | Low        | Medium | Document dependencies; provide fallback detection-only pipeline         |
| R-8  | Reproducibility broken by non-deterministic CUDA operations          | Medium     | Medium | Use deterministic mode; document known non-deterministic ops            |

---

## 9. Assumptions

| ID   | Assumption                                                                                              |
|------|---------------------------------------------------------------------------------------------------------|
| A-1  | Input images are photographed from directly above the core tray (orthographic-like view).              |
| A-2  | Each image contains one core tray with one or more rows of core.                                       |
| A-3  | A scale marker (ruler, color card, or known-size reference) is present in at least some images.        |
| A-4  | Core fragments lie horizontally within tray rows (no vertical stacking visible in top-down images).    |
| A-5  | The minimum meaningful fragment length for RQD is 100 mm (ISRM standard).                             |
| A-6  | Ground truth RQD values are provided by a qualified geologist for validation images.                   |
| A-7  | Training and evaluation are performed on a system with at least one GPU with ≥ 8 GB VRAM.             |
| A-8  | The annotation format follows YOLO and/or COCO conventions.                                            |
| A-9  | Depth (borehole from/to) information is provided externally and is not derived from images.            |
| A-10 | Lighting conditions in training images are representative of inference-time conditions.                |

---

## 10. Glossary

| Term                  | Definition                                                                                                   |
|-----------------------|--------------------------------------------------------------------------------------------------------------|
| RQD                   | Rock Quality Designation — a core recovery index defined as the ratio of intact pieces ≥ 100 mm to total run length |
| Core run              | A single drilling interval from which a core sample is extracted                                             |
| Core box / tray       | A physical or photographed container holding aligned core segments from one or more core runs               |
| Tray row              | A single linear arrangement of core fragments within a core box                                             |
| Fragment              | A continuous, unbroken piece of drill core                                                                   |
| Fracture              | A discontinuity or break in the core, separating two fragments                                              |
| Scale marker          | A reference object of known physical size present in the image for pixel-to-mm calibration                 |
| Calibration           | The process of determining the conversion factor between image pixels and physical millimeters              |
| mAP                   | Mean Average Precision — a standard object detection evaluation metric                                      |
| IoU                   | Intersection over Union — overlap metric for bounding boxes and segmentation masks                          |
| MAE                   | Mean Absolute Error                                                                                          |
| RMSE                  | Root Mean Squared Error                                                                                      |
| YOLO                  | You Only Look Once — a family of real-time object detection models                                          |
| RT-DETR               | Real-Time Detection Transformer — a transformer-based detection model                                       |
| SAM2                  | Segment Anything Model 2 — a foundation model for promptable image segmentation                             |
| Mask R-CNN            | Region-based CNN with instance segmentation capability                                                      |
| U-Net                 | Encoder-decoder architecture for semantic segmentation                                                      |
| Florence-2            | Microsoft's vision foundation model supporting detection and segmentation via prompting                     |
| Grounding DINO        | Open-vocabulary detection model using text-image grounding                                                  |
| SDD                   | Spec-Driven Development — the methodology used in this project                                              |
| ISRM                  | International Society for Rock Mechanics — defines the standard RQD measurement protocol                    |
