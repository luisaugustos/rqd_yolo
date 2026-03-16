# Model Specification

**Project:** rqd-ai-lab
**Phase:** 5 — Model Spec
**Version:** 1.0.0
**Date:** 2026-03-16
**Status:** Draft

---

## Overview

This document specifies all model families supported by the rqd-ai-lab system. Models are organized by role: detection, segmentation/refinement, and foundation. For each model, the purpose, task type, required input/output format, training and inference modes, strengths, limitations, and expected evaluation metrics are defined.

---

## Model Selection Policy

| Tier       | Models                          | Purpose                                                           |
|------------|---------------------------------|-------------------------------------------------------------------|
| Baseline   | YOLOv11n                        | Fast, established detection baseline; used in all comparisons    |
| Primary    | YOLOv12m, RT-DETRv2-S           | Main benchmark models for detection comparison                   |
| Refinement | SAM2 (ViT-B), Mask R-CNN        | Segmentation refinement of detection results                     |
| Exploratory| Florence-2, Grounding DINO      | Zero/few-shot foundation model exploration; optional             |
| Semantic   | U-Net                           | Pixel-level semantic segmentation; alternative to instance seg   |

**Policy rules:**
- All Primary and Baseline models must be evaluated on the same test split.
- Exploratory models are opt-in and not required for the main benchmark.
- Results must report the model tier to avoid misleading comparisons.

---

## 1. YOLOv11

### Purpose
Serve as the detection baseline. Provides a well-tested, broadly comparable reference point for mAP and inference latency.

### Task Type
Object detection (bounding box regression + classification)

### Required Input Format
- Image tensor: `(B, 3, H, W)`, float32, pixel values in [0, 255]
- H and W must be multiples of 32 (default: 640 × 640)
- Letterbox padding is applied automatically by the YOLO pipeline

### Output Contract
```
List[DetectionResult] where DetectionResult:
  boxes:   Tensor[N, 4]   # (x1, y1, x2, y2) absolute pixel coordinates
  scores:  Tensor[N]      # confidence scores in [0, 1]
  classes: Tensor[N]      # integer class IDs
```

### Training Mode
- Fine-tune from `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt` pretrained COCO weights.
- Configurable: epochs, batch size, learning rate, augmentation, mosaic.
- Training via `ultralytics` library (`model.train(...)`).
- Freeze backbone layers configurable for low-data regimes.

### Inference Mode
- Single image or batched inference.
- Configurable confidence threshold (default: 0.25) and NMS IoU threshold (default: 0.45).
- TensorRT export supported for production acceleration.

### Strengths
- Fast inference (real-time capable).
- Well-maintained library with active community support.
- Built-in augmentation, mixed precision, and distributed training.
- Native COCO and YOLO annotation format support.

### Limitations
- Bounding box output only (no native instance segmentation in detection-only mode).
- May miss small fragments at low resolution.
- Less accurate than transformer-based models on complex scenes.

### Expected Evaluation Metrics
| Metric       | Expected Range (Assumption) |
|--------------|-----------------------------|
| mAP@0.5      | 0.65 – 0.85                 |
| mAP@0.5:0.95 | 0.45 – 0.65                 |
| Precision    | 0.70 – 0.90                 |
| Recall       | 0.65 – 0.85                 |
| FPS (GPU)    | 50 – 150                    |

---

## 2. YOLOv12

### Purpose
Primary detection model. Represents the most recent YOLO generation with improved attention-based backbone; used as the main comparison target.

### Task Type
Object detection (bounding box regression + classification)

### Required Input Format
- Same as YOLOv11: `(B, 3, H, W)`, float32, letterboxed to multiples of 32

### Output Contract
Same structure as YOLOv11 `DetectionResult`.

### Training Mode
- Fine-tune from `yolov12n.pt`, `yolov12s.pt`, `yolov12m.pt`.
- Training via `ultralytics` library (same API as YOLOv11).
- Gradient checkpointing supported for large model variants.

### Inference Mode
- Identical to YOLOv11 inference interface.

### Strengths
- Improved small-object detection over YOLOv11 (area attention mechanism).
- Same ecosystem as YOLOv11; minimal code changes required to switch.

### Limitations
- Newer model; less community validation at time of spec.
- Pretrained weights availability may be limited for non-standard input sizes.

### Expected Evaluation Metrics
| Metric       | Expected Range (Assumption) |
|--------------|-----------------------------|
| mAP@0.5      | 0.68 – 0.88                 |
| mAP@0.5:0.95 | 0.48 – 0.68                 |
| Precision    | 0.72 – 0.92                 |
| Recall       | 0.67 – 0.87                 |
| FPS (GPU)    | 40 – 120                    |

---

## 3. RT-DETRv2

### Purpose
Primary benchmark model representing the transformer-based detection family. Provides end-to-end detection without NMS.

### Task Type
Object detection (transformer decoder with bipartite matching; no NMS required)

### Required Input Format
- Image tensor: `(B, 3, H, W)`, float32, normalized to [0, 1] with ImageNet mean/std
- Default resolution: 640 × 640; supports arbitrary resolutions via dynamic positional encodings

### Output Contract
```
List[DetectionResult] where DetectionResult:
  boxes:   Tensor[N, 4]   # (x1, y1, x2, y2) absolute pixel coordinates
  scores:  Tensor[N]
  classes: Tensor[N]
```

### Training Mode
- Fine-tune from `rtdetrv2_s.pth`, `rtdetrv2_m.pth`, `rtdetrv2_l.pth` pretrained COCO weights.
- Training via `ultralytics` or official RT-DETR repository.
- Configurable: decoder layers, number of queries, learning rate schedule (cosine with warmup).
- Higher VRAM requirement than YOLO models; gradient accumulation recommended for small batches.

### Inference Mode
- Batched inference; no NMS step required.
- Confidence threshold applied post-decoder.

### Strengths
- No NMS; fewer hyperparameters to tune.
- Strong performance on dense and overlapping objects.
- Variable resolution at inference without retraining.

### Limitations
- Higher memory and compute requirement than YOLO.
- Slower inference than YOLO at standard resolutions.
- Less mature tooling for domain fine-tuning.

### Expected Evaluation Metrics
| Metric       | Expected Range (Assumption) |
|--------------|-----------------------------|
| mAP@0.5      | 0.70 – 0.88                 |
| mAP@0.5:0.95 | 0.50 – 0.70                 |
| Precision    | 0.72 – 0.90                 |
| Recall       | 0.68 – 0.86                 |
| FPS (GPU)    | 20 – 60                     |

---

## 4. SAM2 (Segment Anything Model 2)

### Purpose
Prompted instance segmentation for refinement of detected fragment bounding boxes. Used downstream of a detection model to produce precise pixel-level masks.

### Task Type
Promptable instance segmentation (point, box, or mask prompt)

### Required Input Format
- Image encoder input: `(1, 3, H, W)`, float32, normalized (SAM-specific normalization)
- Prompts: bounding box `[x1, y1, x2, y2]` or point `(x, y, label)` per fragment
- Prompts are derived from FR-005 detection results

### Output Contract
```
List[SegmentationResult] where SegmentationResult:
  mask:       np.ndarray[H, W]  # binary mask, uint8 {0, 1}
  score:      float             # mask quality score in [0, 1]
  bbox:       List[4]           # tight bbox around mask [x1, y1, x2, y2]
  prompt_box: List[4]           # original detection prompt
```

### Training Mode
- SAM2 is used in **inference-only** mode with frozen weights (no fine-tuning in Phase 1).
- Optional: fine-tuning of the mask decoder on domain-specific data in future phases.
- Weights: `sam2_hiera_base_plus.pt` or `sam2_hiera_small.pt` (configurable).

### Inference Mode
- Batch processing of detection prompts per image.
- Multimask output (3 candidates); highest-scored mask is selected.
- GPU required for practical inference speed.

### Strengths
- State-of-the-art zero-shot segmentation from box prompts.
- No fine-tuning required for initial experiments.
- High mask quality for clear object boundaries.

### Limitations
- High memory usage (~4–6 GB VRAM for ViT-B backbone).
- Slow per-image compared to detection-only pipeline.
- Quality degrades on low-contrast or touching fragments.

### Expected Evaluation Metrics
| Metric     | Expected Range (Assumption) |
|------------|-----------------------------|
| Mask IoU   | 0.70 – 0.88                 |
| Score      | 0.80 – 0.95 (self-reported) |

---

## 5. Mask R-CNN

### Purpose
Instance segmentation model trained end-to-end on annotated core imagery. Alternative to SAM2 for refinement, with the advantage of being trained on domain data.

### Task Type
End-to-end instance segmentation (detection + segmentation in single model)

### Required Input Format
- Image list: `List[Tensor[3, H, W]]`, float32, normalized to [0, 1] with ImageNet stats
- Variable resolution supported; internally resized to min/max dimensions

### Output Contract
```
List[SegmentationResult] (same schema as SAM2 output)
plus:
  class_id: int
  class_score: float
```

### Training Mode
- Fine-tune from `maskrcnn_resnet50_fpn_v2` pretrained COCO weights (torchvision).
- Configurable: backbone (ResNet-50, ResNet-101), FPN levels, anchor sizes, mask head resolution.
- Training via PyTorch detection reference scripts or Detectron2.

### Inference Mode
- Batched inference with configurable score threshold and NMS IoU threshold.
- Returns both detection boxes and segmentation masks.

### Strengths
- End-to-end trained on domain data; can outperform zero-shot SAM2 when data is sufficient.
- Mature, well-understood architecture with abundant documentation.
- Produces both detection and segmentation from a single forward pass.

### Limitations
- Requires sufficient annotated segmentation masks for training (polygon annotations).
- Fixed-resolution feature pyramid may miss very small fragments.
- Slower than detection-only YOLO pipeline.

### Expected Evaluation Metrics
| Metric     | Expected Range (Assumption) |
|------------|-----------------------------|
| Box mAP@0.5 | 0.65 – 0.85               |
| Mask IoU   | 0.65 – 0.83                 |

---

## 6. U-Net

### Purpose
Semantic segmentation of the core image to produce per-pixel class maps. Used as an alternative segmentation approach and as a basis for fracture probability maps.

### Task Type
Semantic segmentation (per-pixel classification; no instance separation)

### Required Input Format
- Image tensor: `(B, 3, H, W)`, float32, normalized to [0, 1]
- Input resolution padded/resized to multiples of 32 (default: 512 × 512 or 1024 × 1024)

### Output Contract
```
SegmentationMap:
  logits:       Tensor[B, C, H, W]   # raw logits per class
  probabilities: Tensor[B, C, H, W]  # softmax probabilities
  class_map:    Tensor[B, H, W]      # argmax class prediction (int)
```
Where C = number of semantic classes (at minimum: background, intact_fragment, fracture).

### Training Mode
- Train from scratch or from ImageNet-pretrained encoder (ResNet-34 or EfficientNet-B3 backbone).
- Binary cross-entropy or Dice loss (configurable).
- Training resolution: 512 × 512 with sliding window inference for large images.

### Inference Mode
- Sliding window with overlap-tile strategy for large images.
- Post-processing: connected component analysis to extract instances from semantic map.

### Strengths
- Excellent for dense, touching fragments where instance boundaries are ambiguous.
- Can produce fracture probability heatmaps useful for analysis.
- Computationally efficient for pixel-level tasks.

### Limitations
- Semantic only; requires post-processing to separate individual fragment instances.
- Does not natively produce bounding boxes or instance IDs.
- Instance separation accuracy depends on post-processing quality.

### Expected Evaluation Metrics
| Metric         | Expected Range (Assumption) |
|----------------|-----------------------------|
| mIoU           | 0.65 – 0.82                 |
| Fracture IoU   | 0.55 – 0.75                 |
| Fragment IoU   | 0.70 – 0.85                 |

---

## 7. Florence-2

### Purpose
Exploratory foundation model for open-vocabulary detection and grounding. Used to test zero-shot and few-shot detection of core fragments and fractures without task-specific fine-tuning.

### Task Type
Multimodal vision-language model; supports open-vocabulary detection, region description, and referring expression comprehension.

### Required Input Format
- Image tensor + text prompt: `(image: PIL.Image, task_prompt: str, text_input: str | None)`
- Standard Florence-2 processor handles tokenization and image preprocessing

### Output Contract
```
Florence2Result:
  task:     str
  boxes:    List[[x1, y1, x2, y2]]   # absolute pixel coordinates
  labels:   List[str]                 # predicted class labels (text)
  scores:   List[float] | None        # not always available
```

### Training Mode
- Phase 1: inference-only with frozen weights.
- Optional fine-tuning of task-specific heads on annotated data in future phases.
- Model: `microsoft/Florence-2-base` or `microsoft/Florence-2-large`.

### Inference Mode
- Task prompt: `<OD>` (object detection) or `<CAPTION_TO_PHRASE_GROUNDING>`.
- Text input: natural language description of target objects (e.g., "intact core fragment", "fracture line").

### Strengths
- Zero-shot capability without any annotated data.
- Flexible prompting enables rapid prototyping.
- Can describe regions and generate captions for qualitative analysis.

### Limitations
- Slower than specialized detectors.
- Output format varies by task prompt; requires adapter layer to normalize to `DetectionResult`.
- License: MIT; model weights via Hugging Face (requires network access).
- May not generalize to specialized geotechnical imagery without fine-tuning.

### Expected Evaluation Metrics
| Metric       | Expected Range (Assumption — zero-shot) |
|--------------|------------------------------------------|
| mAP@0.5      | 0.30 – 0.60 (zero-shot; highly variable) |

---

## 8. Grounding DINO (Optional)

### Purpose
Open-vocabulary object detection via text-guided grounding. Optional exploratory model for zero-shot fragment and fracture detection using natural language prompts.

### Task Type
Open-vocabulary object detection (bounding box regression guided by text queries)

### Required Input Format
- `(image: PIL.Image | Tensor, text_prompt: str)` where text_prompt is dot-separated category names
- Example: `"intact fragment . fracture . scale marker ."`

### Output Contract
```
DetectionResult (same schema as YOLO/RT-DETR DetectionResult):
  boxes:   Tensor[N, 4]
  scores:  Tensor[N]
  labels:  List[str]   # text labels (not integer class IDs)
```
Adapter converts text labels to integer class IDs using a configurable label map.

### Training Mode
- Phase 1: inference-only.
- Fine-tuning via DINO-based detection fine-tuning pipeline (optional future phase).

### Inference Mode
- Confidence threshold for text-box association (default: 0.3).
- Text-IoU threshold for deduplication (default: 0.45).

### Strengths
- True open-vocabulary; no retraining required for new classes.
- Strong grounding of spatial predictions to natural language.

### Limitations
- Heavy model (~1.5 GB); slower inference than specialized models.
- License: Apache 2.0; weights via Hugging Face.
- Output confidence scores reflect text-image alignment, not standard classification confidence.
- Integration complexity higher than YOLO backends.

### Expected Evaluation Metrics
| Metric       | Expected Range (Assumption — zero-shot) |
|--------------|------------------------------------------|
| mAP@0.5      | 0.25 – 0.55 (zero-shot; highly variable) |

---

## 9. Model Interface Summary

All detection models must implement the `DetectorBackend` interface:

```python
class DetectorBackend(Protocol):
    def load(self, weights_path: str, config: dict) -> None: ...
    def predict(self, image: np.ndarray, conf_thresh: float, iou_thresh: float) -> DetectionResult: ...
    def predict_batch(self, images: List[np.ndarray], ...) -> List[DetectionResult]: ...
```

All segmentation models must implement the `SegmentorBackend` interface:

```python
class SegmentorBackend(Protocol):
    def load(self, weights_path: str, config: dict) -> None: ...
    def segment(self, image: np.ndarray, prompts: List[PromptBox]) -> List[SegmentationResult]: ...
```

Full interface contracts are defined in Phase 7 — Interface Contracts.
