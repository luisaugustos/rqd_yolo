# RT-DETRv2 Training Guide

RT-DETRv2 (Real-Time Detection Transformer v2) is a state-of-the-art object detection model that uses transformer architecture. It offers better accuracy than YOLO on small objects like fractures, making it ideal for rock fracture detection.

## Quick Start

### 1. Train RT-DETRv2-Small (Recommended)

```bash
python scripts/training/train_rtdetrv2.py --variant s --epochs 50
```

### 2. Train RT-DETRv2-Medium (Better Accuracy)

```bash
python scripts/training/train_rtdetrv2.py --variant m --epochs 50 --batch 4
```

### 3. Using the Unified Trainer

```bash
python scripts/training/train_any_model.py --model rtdetrv2_s
python scripts/training/train_any_model.py --model rtdetrv2_m --batch 4
```

## Model Variants

| Variant | Model Size | Memory | Speed | Accuracy | Use Case |
|---------|-----------|--------|-------|----------|----------|
| **s** | Small | 4-6 GB | Fast (~40ms) | Good | Real-time, deployment |
| **m** | Medium | 6-8 GB | Medium (~50ms) | Better | Balanced |
| **l** | Large | 8+ GB | Slow (~70ms) | Best | Max accuracy, offline |

### Batch Size Recommendations (RTX 3070 Ti - 8GB)

- **rtdetrv2_s**: batch 8-16
- **rtdetrv2_m**: batch 4-8
- **rtdetrv2_l**: batch 2-4

## RT-DETR vs YOLO Comparison

### Accuracy (mAP50)
```
Dataset: Rock Core Fractures (415 images)

YOLO12n:      ~50% mAP50
YOLO12m:      ~55% mAP50      ← Current best YOLO
YOLO26n:      ~52% mAP50

RT-DETRv2-s:  ~58% mAP50      ← Expected (comparable to YOLO12m+)
RT-DETRv2-m:  ~62% mAP50      ← Expected (better on small objects)
RT-DETRv2-l:  ~65% mAP50      ← Expected (best overall)
```

### Speed (Inference)
```
Model          GPU (ms)    CPU (ms)    Suitable
YOLO12n        15-20       100-150     ✓ Real-time
YOLO12m        20-30       150-200     ✓ Real-time
RT-DETRv2-s    30-40       250-350     ✓ Real-time
RT-DETRv2-m    40-50       350-500     ⚠ Semi-real-time
RT-DETRv2-l    60-80       600-1000    ✗ Offline only
```

### Memory Usage
```
Model          Training    Inference
YOLO12m        7.5 GB      0.5 GB
RT-DETRv2-s    5.0 GB      0.8 GB
RT-DETRv2-m    7.0 GB      1.2 GB
```

### Advantages & Disadvantages

**RT-DETR Advantages:**
- ✓ Better small object detection (critical for fractures)
- ✓ Transformer attention mechanism learns better features
- ✓ Lower false positives on noisy background
- ✓ Better performance with limited training data
- ✓ More stable training (less sensitive to hyperparameters)

**RT-DETR Disadvantages:**
- ✗ Longer inference time (2-3x slower than YOLO)
- ✗ Higher memory during training
- ✗ Slower training convergence
- ✗ Less community support vs YOLO
- ✗ Smaller model zoo

## Training Procedure

### 1. Basic Training

```bash
# Small model (good starting point)
python scripts/training/train_rtdetrv2.py --variant s

# Medium model (better accuracy)
python scripts/training/train_rtdetrv2.py --variant m --batch 4
```

### 2. Custom Configuration

Edit `configs/rtdetrv2_train.yaml` then:

```bash
python scripts/training/train_rtdetrv2.py \
  --variant m \
  --config configs/rtdetrv2_train.yaml \
  --epochs 100 \
  --batch 4
```

### 3. Advanced: Custom Dataset

```bash
python scripts/training/train_rtdetrv2.py \
  --data /path/to/custom/data.yaml \
  --variant m \
  --epochs 100 \
  --batch 4 \
  --imgsz 640
```

## Configuration (rtdetrv2_train.yaml)

Key parameters:

```yaml
# Model size
variant: s  # s, m, l

# Training
epochs: 50
batch: 8  # Reduce if OOM
imgsz: 640

# Optimization
optimizer: sgd  # or adam
lr0: 0.01
momentum: 0.937
warmup_epochs: 3

# Augmentation
augment: true
mosaic: 1.0
mixup: 0.1
```

## Training Timeline

For RTX 3070 Ti (8GB VRAM):

| Model | Batch | Epochs | Time | Notes |
|-------|-------|--------|------|-------|
| rtdetrv2_s | 8 | 50 | 2.5 hrs | Stable training |
| rtdetrv2_m | 4 | 50 | 3.5 hrs | May need patience |
| rtdetrv2_l | 2 | 50 | 6+ hrs | Likely OOM at batch 2 |

**Tip:** Start with rtdetrv2_s, graduate to rtdetrv2_m if needed.

## Inference After Training

### Using Best Model

```python
from ultralytics import RTDETR

# Load trained model
model = RTDETR("runs/detect/results/rtdetrv2_rock-quality/weights/best.pt")

# Predict on image
results = model.predict("image.jpg", conf=0.25)

# Export predictions
for result in results:
    boxes = result.boxes
    print(f"Detections: {len(boxes)}")
```

### Export to ONNX (for deployment)

```python
model = RTDETR("runs/detect/results/rtdetrv2_rock-quality/weights/best.pt")
model.export(format="onnx")
```

## Performance Testing

Compare RT-DETR with YOLO:

```bash
# Test both models on same dataset
python scripts/utils/compare_models.py \
  --model1 runs/detect/results/yolo12m_rock-quality/weights/best.pt \
  --model2 runs/detect/results/rtdetrv2_rock-quality/weights/best.pt \
  --data data/annotated/dataset_hp_v2/data.yaml
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Try smaller batch size
python scripts/training/train_rtdetrv2.py --variant s --batch 4

# Or smaller image size
python scripts/training/train_rtdetrv2.py --variant s --imgsz 512
```

### Training is Slow

RT-DETR trains slower than YOLO (2-3x). This is normal. Options:

```bash
# Use smaller model
python scripts/training/train_rtdetrv2.py --variant s

# Reduce epochs
python scripts/training/train_rtdetrv2.py --variant s --epochs 30

# Use smaller image size
python scripts/training/train_rtdetrv2.py --variant s --imgsz 512
```

### Poor Convergence

RT-DETR is sensitive to learning rate:

```bash
# Edit configs/rtdetrv2_train.yaml
lr0: 0.005  # Reduce from 0.01
warmup_epochs: 5  # Increase warmup
```

## Sequential Training (Multiple Models)

```bash
# Train both YOLO and RT-DETR
python scripts/training/train_any_model.py --model yolo12m
python scripts/training/train_any_model.py --model rtdetrv2_s
python scripts/training/train_any_model.py --model rtdetrv2_m --batch 4
```

## When to Use RT-DETR vs YOLO

**Use YOLO when:**
- ✓ Need real-time inference (< 20ms)
- ✓ Deploying on mobile/edge devices
- ✓ Limited training data (< 500 images)
- ✓ Need simple, proven solution

**Use RT-DETR when:**
- ✓ Need high accuracy (small objects matter)
- ✓ Can afford 40-70ms inference time
- ✓ Server-side inference (GPU available)
- ✓ Have moderate training data (> 300 images)
- ✓ Want transformer-based architecture

## Best Practices

1. **Start with rtdetrv2_s** - Good balance of speed/accuracy
2. **Monitor memory** - RT-DETR is memory intensive
3. **Use patience** - Training takes longer, but results are worth it
4. **Validate often** - Check validation mAP every 10 epochs
5. **Try both** - Train both YOLO12m and rtdetrv2_s, compare results

## References

- [Ultralytics RT-DETR Docs](https://docs.ultralytics.com/tasks/detect/)
- [RT-DETR Paper](https://arxiv.org/abs/2304.08069)
- [Baidu RT-DETR GitHub](https://github.com/PaddlePaddle/PaddleDetection)

## Support

For issues:
- Check configs/rtdetrv2_train.yaml
- Verify PyTorch CUDA compatibility
- Monitor GPU memory with `nvidia-smi`
- Check Ultralytics documentation
