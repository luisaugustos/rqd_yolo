#!/usr/bin/env python
"""Generate synthetic test data for rqd-ai-lab pipeline validation."""

from pathlib import Path
import numpy as np
import cv2
import yaml
from PIL import Image, ImageDraw

def create_test_data():
    """Create minimal synthetic dataset for testing."""

    # Create directory structure
    data_root = Path("data")
    raw_dir = data_root / "raw"
    ann_yolo_dir = data_root / "annotations" / "yolo"
    ann_coco_dir = data_root / "annotations" / "coco"
    splits_dir = data_root / "annotations" / "splits"

    for d in [raw_dir, ann_yolo_dir, ann_coco_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate 5 synthetic test images
    for img_id in range(1, 6):
        img_name = f"test_image_{img_id:03d}.jpg"

        # Create synthetic image (simple geometric shapes)
        img = Image.new("RGB", (1024, 768), color="lightgray")
        draw = ImageDraw.Draw(img)

        # Draw 3 tray rows as horizontal bands
        row_height = 768 // 3
        for row_idx in range(3):
            y_top = row_idx * row_height
            y_bottom = (row_idx + 1) * row_height
            # Row border
            draw.rectangle([0, y_top, 1024, y_bottom], outline="black", width=2)

            # Draw 3-4 synthetic fragments (rectangles) per row
            num_fragments = np.random.randint(3, 5)
            for frag_idx in range(num_fragments):
                x_start = 100 + frag_idx * 200 + np.random.randint(-20, 20)
                y_start = y_top + 50 + np.random.randint(0, 50)
                width = 120 + np.random.randint(-20, 30)
                height = 80 + np.random.randint(-10, 20)

                draw.rectangle(
                    [x_start, y_start, x_start + width, y_bottom - 50],
                    fill="tan",
                    outline="brown",
                    width=1
                )

        # Save image
        img_path = raw_dir / img_name
        img.save(img_path)
        print(f"✓ Created {img_path}")

        # Create corresponding YOLO annotation
        # YOLO format: class_id x_center y_center width height (normalized 0-1)
        yolo_ann = []
        yolo_ann.append("1 0.25 0.25 0.15 0.15")  # class 1: intact_fragment
        yolo_ann.append("1 0.50 0.35 0.12 0.10")
        yolo_ann.append("0 0.35 0.30 0.02 0.08")  # class 0: fracture

        yolo_file = ann_yolo_dir / img_name.replace(".jpg", ".txt")
        yolo_file.write_text("\n".join(yolo_ann))
        print(f"✓ Created {yolo_file}")

    # Create split file
    split_data = {
        "train": ["test_image_001.jpg", "test_image_002.jpg", "test_image_003.jpg"],
        "val": ["test_image_004.jpg"],
        "test": ["test_image_005.jpg"],
    }

    split_file = splits_dir / "split_v1.yaml"
    with open(split_file, "w") as f:
        yaml.dump(split_data, f)
    print(f"✓ Created {split_file}")

    print("\n✓ Test data generation complete!")

if __name__ == "__main__":
    create_test_data()
