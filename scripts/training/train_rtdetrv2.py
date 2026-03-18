#!/usr/bin/env python
"""
RT-DETRv2 Training Script
Real-Time Detection Transformer for rock fracture detection

RT-DETRv2 Advantages:
- Better accuracy than YOLO (especially on small objects)
- Real-time performance on modern GPUs
- Transformer-based architecture (attention mechanism)
- Better with limited training data

Disadvantages:
- Slightly more memory usage
- Longer inference time than YOLO11/12

Available Models:
- rtdetrv2_s: Small (efficient, ~40ms)
- rtdetrv2_m: Medium (balanced, ~50ms)
- rtdetrv2_l: Large (accurate, ~70ms)

Usage:
    python scripts/training/train_rtdetrv2.py --variant s --epochs 50 --batch 8
    python scripts/training/train_rtdetrv2.py --variant m --data data/annotated/dataset_hp_v2/data.yaml
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import RTDETR
import yaml

def train_rtdetrv2(
    data_path: str,
    variant: str = "s",
    epochs: int = 50,
    batch_size: int = 8,
    imgsz: int = 640,
    device: int = 0,
    config_path: str = "configs/rtdetrv2_train.yaml",
):
    """
    Train RT-DETRv2 model for rock fracture detection

    Args:
        data_path: Path to data.yaml file
        variant: Model size (s, m, l)
        epochs: Number of training epochs
        batch_size: Batch size (reduced for transformer memory)
        imgsz: Image size for training
        device: GPU device ID
        config_path: Path to training config (optional)
    """

    print("="*70)
    print("RT-DETRv2 TRAINING - Rock Fracture Detection")
    print("="*70)

    # Load config if provided
    config = {}
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        print(f"\n✓ Loaded config: {config_path}")

    # Model name
    model_name = f"rtdetrv2_{variant}"

    print(f"\n[1/4] Initializing {model_name.upper()}...")
    print(f"  Variant: {variant}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {imgsz}")
    print(f"  Device: cuda:{device}")

    try:
        # Load model - automatically downloads pretrained weights
        model = RTDETR(f"{model_name}.pt")
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Verify dataset
    print(f"\n[2/4] Verifying dataset...")
    data_file = Path(data_path)
    if not data_file.exists():
        print(f"❌ Dataset config not found: {data_path}")
        return False

    print(f"✓ Dataset: {data_file}")

    # Training parameters
    print(f"\n[3/4] Training configuration:")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    train_params = {
        "data": data_path,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": device,
        "patience": config.get('patience', 20),
        "save": True,
        "save_period": -1,
        "close_mosaic": config.get('close_mosaic', 10),
        "project": "runs/detect/results",
        "name": f"{model_name}_rock-quality",
        "exist_ok": True,
        "pretrained": True,
        "warmup_epochs": config.get('warmup_epochs', 3),
        "optimizer": config.get('optimizer', 'sgd'),
    }

    try:
        print(f"\n[4/4] Training {model_name}...")
        results = model.train(**train_params)

        # Training complete
        print(f"\n✓ Training completed successfully!")
        print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Print results
        if hasattr(results, 'results_dict'):
            print(f"\nTraining Metrics:")
            print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
            print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")

        # Model location
        results_dir = Path("runs/detect/results") / f"{model_name}_rock-quality"
        weights_dir = results_dir / "weights"

        if weights_dir.exists():
            print(f"\nModel saved to:")
            print(f"  Best: {weights_dir / 'best.pt'}")
            print(f"  Last: {weights_dir / 'last.pt'}")

        print(f"\nResults directory: {results_dir}")

        return True

    except KeyboardInterrupt:
        print(f"\n⚠ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Train RT-DETRv2 for rock fracture detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Small model (balanced speed/accuracy)
  python scripts/training/train_rtdetrv2.py --variant s

  # Medium model (better accuracy)
  python scripts/training/train_rtdetrv2.py --variant m --epochs 50 --batch 4

  # Large model (best accuracy, slower)
  python scripts/training/train_rtdetrv2.py --variant l --batch 4 --epochs 50

  # Custom dataset and config
  python scripts/training/train_rtdetrv2.py \\
    --data data/annotated/dataset_hp_v2/data.yaml \\
    --config configs/rtdetrv2_train.yaml \\
    --variant m
        """
    )

    parser.add_argument("--data", type=str, default="data/annotated/dataset_hp_v2/data.yaml",
                       help="Path to data.yaml (default: dataset_hp_v2)")
    parser.add_argument("--variant", type=str, default="s", choices=["s", "m", "l"],
                       help="Model variant: s=small, m=medium, l=large (default: s)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size (default: 8, reduce if OOM)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size (default: 640)")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID (default: 0)")
    parser.add_argument("--config", type=str, default="configs/rtdetrv2_train.yaml",
                       help="Path to training config YAML")

    args = parser.parse_args()

    # Train
    success = train_rtdetrv2(
        data_path=args.data,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        config_path=args.config,
    )

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
