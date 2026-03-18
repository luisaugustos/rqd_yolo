#!/usr/bin/env python
"""
Auto-annotate rock core images using trained YOLO12n model.
Generates YOLO format annotations (.txt files) for manual review.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def auto_annotate_dataset(input_dir: str, output_dir: str, model_path: str, conf_threshold: float = 0.25):
    """
    Auto-annotate images using YOLO12n model.

    Args:
        input_dir: Directory with raw images (nested in boreholes)
        output_dir: Directory to save YOLO format annotations
        model_path: Path to trained YOLO model weights
        conf_threshold: Confidence threshold for detections
    """

    print("=" * 70)
    print("YOLO12n AUTO-ANNOTATION SYSTEM")
    print("=" * 70)

    # Load model
    print(f"\n[1/4] Loading YOLO12n model from: {model_path}")
    model = YOLO(model_path)
    print("✓ Model loaded successfully")

    # Find all images
    input_path = Path(input_dir)
    image_files = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png"))
    image_files = [f for f in image_files if f.is_file()]

    print(f"\n[2/4] Found {len(image_files)} images to annotate")

    # Create output structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Process images
    print(f"\n[3/4] Processing images...")

    total_detections = 0
    processed_count = 0

    for idx, image_file in enumerate(image_files):
        # Run inference
        results = model.predict(str(image_file), conf=conf_threshold, verbose=False)

        # Read image to get dimensions
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"  ⚠ Could not read: {image_file.name}")
            continue

        h, w = img.shape[:2]

        # Create safe filename - include parent dir to avoid collisions
        parent_dir = image_file.parent.name
        safe_name = f"{parent_dir}_{image_file.stem}".replace(" ", "_").replace("/", "_")

        # Save image
        output_image = images_dir / f"{safe_name}.jpg"
        cv2.imwrite(str(output_image), img)

        # Generate YOLO annotation
        annotation_lines = []
        detections = 0

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    # Convert to YOLO format (normalized center coordinates)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    box_width = (x2 - x1) / w
                    box_height = (y2 - y1) / h

                    # Clamp to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    box_width = max(0, min(1, box_width))
                    box_height = max(0, min(1, box_height))

                    annotation_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
                    detections += 1

        # Save annotation file
        label_file = labels_dir / f"{safe_name}.txt"
        if annotation_lines:
            with open(label_file, 'w') as f:
                f.write('\n'.join(annotation_lines))
        else:
            # Create empty file for images with no detections
            label_file.touch()

        total_detections += detections
        processed_count += 1

        # Progress
        if (idx + 1) % max(1, len(image_files) // 10) == 0:
            print(f"  Progress: {idx + 1}/{len(image_files)} images processed")

    # Create data.yaml for training
    print(f"\n[4/4] Creating dataset configuration...")

    data_yaml = f"""path: {output_path.resolve()}
train: images
val: images

nc: 1
names:
  0: fractures
"""

    with open(output_path / "data.yaml", 'w') as f:
        f.write(data_yaml)

    # Summary
    print("\n" + "=" * 70)
    print("AUTO-ANNOTATION COMPLETE!")
    print("=" * 70)
    print(f"\nStatistics:")
    print(f"  Images processed: {processed_count}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections/image: {total_detections/processed_count:.2f}")
    print(f"\nOutput saved to: {output_path}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"  Config: {output_path / 'data.yaml'}")

    print(f"\n📌 Next steps:")
    print(f"  1. Review annotations in: {labels_dir}")
    print(f"  2. Correct bounding boxes as needed (open label files with annotation tool)")
    print(f"  3. Train new model: python scripts/train_roboflow.py --data {output_path / 'data.yaml'}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Paths
    input_dir = "C:/Users/luisb/dev/rqd_yolo/data/raw/dataset_hp"
    output_dir = "C:/Users/luisb/dev/rqd_yolo/data/annotated/dataset_hp"

    # Use best YOLO12n model
    model_path = "C:/Users/luisb/dev/rqd_yolo/runs/detect/results/runs/yolo12n_rock-quality3/weights/best.pt"

    # Run annotation
    auto_annotate_dataset(input_dir, output_dir, model_path, conf_threshold=0.25)
