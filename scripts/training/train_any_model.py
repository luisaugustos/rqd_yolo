#!/usr/bin/env python
"""
Unified training script for YOLO and RT-DETR models

Supported Models:
- YOLO: yolo8n, yolo8m, yolo11n, yolo11m, yolo12n, yolo12m, yolo26n
- RT-DETR: rtdetrv2_s, rtdetrv2_m, rtdetrv2_l

Usage:
    python scripts/training/train_any_model.py --model yolo12n
    python scripts/training/train_any_model.py --model rtdetrv2_m --batch 4
    python scripts/training/train_any_model.py --model rtdetrv2_s --epochs 50
"""

import sys
import subprocess
from pathlib import Path
import argparse

def train_model(model: str, data: str, epochs: int, batch: int, device: int = 0):
    """Route to appropriate training script based on model type"""

    print(f"\n{'='*70}")
    print(f"Training: {model.upper()}")
    print(f"{'='*70}\n")

    project_root = Path(__file__).parent.parent.parent

    # Determine model type and select training script
    if model.startswith("rtdetrv2"):
        # RT-DETR model
        script = project_root / "scripts" / "training" / "train_rtdetrv2.py"
        variant = model.split("_")[1]  # Extract variant from rtdetrv2_s

        cmd = [
            sys.executable,
            str(script),
            "--data", data,
            "--variant", variant,
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--device", str(device),
        ]

    elif model.startswith("yolo"):
        # YOLO model - use existing train_roboflow.py
        script = project_root / "scripts" / "train_roboflow.py"

        cmd = [
            sys.executable,
            str(script),
            "--data", data,
            "--model", model,
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--device", str(device),
        ]

    else:
        print(f"❌ Unknown model type: {model}")
        print(f"   Supported: yolo*, rtdetrv2_*")
        return False

    # Check script exists
    if not script.exists():
        print(f"❌ Training script not found: {script}")
        return False

    # Run training
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print(f"\n✓ Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO or RT-DETR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YOLO models
  python scripts/training/train_any_model.py --model yolo12n
  python scripts/training/train_any_model.py --model yolo12m --batch 8

  # RT-DETR models
  python scripts/training/train_any_model.py --model rtdetrv2_s
  python scripts/training/train_any_model.py --model rtdetrv2_m --batch 4 --epochs 100

Supported Models:
  YOLO:    yolo8n, yolo8m, yolo11n, yolo11m, yolo12n, yolo12m, yolo26n
  RT-DETR: rtdetrv2_s, rtdetrv2_m, rtdetrv2_l
        """
    )

    parser.add_argument("--model", type=str, required=True,
                       help="Model name (yolo12n, rtdetrv2_s, etc.)")
    parser.add_argument("--data", type=str, default="data/annotated/dataset_hp_v2/data.yaml",
                       help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID (default: 0)")

    args = parser.parse_args()

    success = train_model(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
