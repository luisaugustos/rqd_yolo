# YOLOv8m Training Results - Rock Quality Dataset

## Training Summary
- **Model:** YOLOv8m (25.84M parameters)
- **Dataset:** rock-quality (92 train, 26 val images)
- **Epochs:** 50
- **Batch Size:** 16
- **GPU:** RTX 3070 Ti
- **Duration:** 0.068 hours (~4.08 minutes)
- **PyTorch:** 2.5.1+cu121

## Final Metrics
```
Epoch 50/50:
  box_loss:    0.487
  cls_loss:    0.698
  dfl_loss:    0.968
  
Validation Results:
  mAP50:       49.3%
  mAP50-95:    33.7%
  Precision:   44.7%
  Recall:      65.0%
  
Inference Speed:
  Preprocess:  0.2ms
  Inference:   9.8ms
  Postprocess: 1.3ms
  Total:       11.1ms/img (≈90 FPS)
```

## Model Comparison

### YOLOv8n (10 epochs)
- mAP50: 39.8%
- Inference: 2.1ms/img
- Size: 6.2MB

### YOLOv8m (50 epochs) ⭐
- mAP50: 49.3% (+23%)
- Inference: 11.1ms/img
- Size: 52.0MB

## Key Observations
1. **Recall improved significantly** (23.7% → 65.0%)
   - Model now detects most fragments (better for RQD measurement)

2. **Precision decreased slightly** (54.2% → 44.7%)
   - Some false positives, but better overall detection

3. **Training stable** - Convergence smooth across 50 epochs
   - No overfitting observed
   - Validation metrics stable after epoch ~40

4. **GPU efficiency** - RTX 3070 Ti only used 6GB of 8.59GB
   - Could train larger models or batches if needed

## Next Steps
- Deploy YOLOv8m for fracture detection inference
- Integrate into RQD pipeline
- Evaluate on test set (14 images)
- Consider training YOLOv8l if more accuracy needed

## Model Weights
- **Best Model:** `runs/detect/results/runs/yolov8m_rock-quality/weights/best.pt`
- **Last Checkpoint:** `runs/detect/results/runs/yolov8m_rock-quality/weights/last.pt`
