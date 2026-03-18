# Auto-Annotation & Review Guide

## 📋 Overview

You now have **415 raw images** of rock cores from 30 boreholes in `data/raw/dataset_hp/`.

The auto-annotation process will:
1. Use trained **YOLO12n** model to detect fractures
2. Generate YOLO format annotations (`.txt` files)
3. Organize images and labels for training

## 🔄 Process Timeline

### Step 1: Auto-Annotation (Running now)
**Script:** `auto_annotate_dataset.py`
- Processes all 415 images
- Generates bounding box predictions
- Saves to `data/annotated/dataset_hp/`
- **ETA:** 20-30 minutes (depending on GPU)

**Output structure:**
```
data/annotated/dataset_hp/
├── images/          # Original images (copied)
├── labels/          # YOLO annotations (.txt files)
└── data.yaml        # Dataset config
```

### Step 2: Review Annotations (Manual)
**Two options:**

#### Option A: Interactive GUI Tool (Easy)
```bash
python review_annotations.py
```
- View images with detected bounding boxes
- Approve, edit, or clear annotations
- Navigate with Previous/Next buttons
- Edit labels directly if needed

#### Option B: Manual Text Edit (Fast)
Edit label files directly in `data/annotated/dataset_hp/labels/`
- Each `.txt` file = one image
- Format: `class_id x_center y_center width height` (normalized 0-1)
- Delete lines to remove bad detections
- Add lines to add missing detections

### Step 3: Train New Model
Once annotations are reviewed and corrected:

```bash
python scripts/train_roboflow.py \
  --data data/annotated/dataset_hp/data.yaml \
  --model yolo12n \
  --epochs 50 \
  --batch 16
```

## 📌 YOLO Annotation Format

Example `Caja_1.txt`:
```
0 0.523 0.456 0.234 0.345
0 0.712 0.678 0.189 0.267
```

Where:
- `0` = class ID (0 = fractures)
- `0.523` = x center (normalized 0-1)
- `0.456` = y center (normalized 0-1)
- `0.234` = width (normalized 0-1)
- `0.345` = height (normalized 0-1)

## ✏️ Quick Annotation Edits

### Remove a bad detection:
Simply delete the line from the `.txt` file

### Add a missing detection:
1. Open image in any annotation tool (LabelImg, CVAT, etc.)
2. Draw bounding box
3. Note coordinates, convert to YOLO format
4. Add line to `.txt` file

### Batch operations:
For many corrections, use the review tool:
```bash
python review_annotations.py
```

## 📊 Expected Statistics

Based on YOLO12n model trained on 92 rock images:
- **Average detections/image:** 0.5-2 fractures per box
- **Confidence threshold:** 0.25 (catch more, may need review)
- **Expected accuracy:** 55-60% (many will be correct, some false positives/negatives)

## 🎯 Quality Checklist

- [ ] Auto-annotation completed
- [ ] Reviewed all images with detections
- [ ] Removed obvious false positives
- [ ] Added obviously missing fractures
- [ ] Verified label counts (should be 1-4 per image on average)
- [ ] Saved all corrections
- [ ] Created `data.yaml` for training

## 🚀 Next: Train Custom Model

Once annotations are clean, you can:
1. Train new model on this dataset
2. Use it for automatic RQD calculation
3. Create detection visualizations
4. Export predictions

## 📁 File Locations

- **Auto-annotation script:** `auto_annotate_dataset.py`
- **Review tool:** `review_annotations.py`
- **Raw images:** `data/raw/dataset_hp/`
- **Annotated data:** `data/annotated/dataset_hp/`
- **Model weights:** `runs/detect/results/runs/yolo12n_rock-quality3/weights/best.pt`

---

**Questions?** Check the generated `data.yaml` for dataset configuration details.
