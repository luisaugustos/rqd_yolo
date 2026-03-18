#!/usr/bin/env python
"""
Main training orchestration script.
Supports local and remote training with config-based paths.

Usage:
    # Train single model
    python train.py --model yolo12n --epochs 50

    # Sequential training (yolo12n → yolo12m)
    python train.py --sequential

    # Download from Roboflow first, then train
    python train.py --download --sequential

    # Custom config
    python train.py --config config.yaml --model yolo26n
"""

import sys
import argparse
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO models for rock fracture detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model yolo12n --epochs 50
  python train.py --sequential
  python train.py --download --sequential
        """
    )

    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file (default: config.yaml)")
    parser.add_argument("--model", type=str,
                       help="Model to train (yolo12n, yolo12m, yolo26n)")
    parser.add_argument("--epochs", type=int,
                       help="Number of epochs")
    parser.add_argument("--batch", type=int,
                       help="Batch size")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID (default: 0)")
    parser.add_argument("--sequential", action="store_true",
                       help="Train yolo12n, then yolo12m sequentially")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset from Roboflow first")

    args = parser.parse_args()

    project_root = Path(__file__).parent

    print(f"\n{'='*70}")
    print("YOLO Rock Fracture Detection - Training Pipeline")
    print(f"{'='*70}\n")

    # Step 1: Download from Roboflow if requested
    if args.download:
        print("[1/3] Downloading dataset from Roboflow...")
        download_script = project_root / "scripts" / "data" / "download_roboflow.py"
        if download_script.exists():
            try:
                subprocess.run([sys.executable, str(download_script)], cwd=project_root, check=True)
                print("✓ Dataset downloaded\n")
            except subprocess.CalledProcessError:
                print("✗ Download failed\n")
                return 1
        else:
            print(f"⚠ Download script not found: {download_script}\n")

    # Step 2: Train model(s)
    print("[2/3] Starting training...")
    train_script = project_root / "scripts" / "training" / "train_roboflow_config.py"

    cmd = [sys.executable, str(train_script), "--config", args.config]

    if args.sequential:
        cmd.append("--sequential")
    elif args.model:
        cmd.extend(["--model", args.model])
        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.batch:
            cmd.extend(["--batch", str(args.batch)])

    cmd.extend(["--device", str(args.device)])

    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("✓ Training completed\n")
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed with exit code {e.returncode}\n")
        return 1

    # Step 3: Compare results (optional)
    print("[3/3] Training pipeline complete!")
    print(f"\nNext steps:")
    print(f"  - View results: python -m tensorboard --logdir runs/")
    print(f"  - Compare models: python scripts/utils/compare_models.py")
    print(f"  - Deploy model: python rqd_cli.py --model <path>")
    print(f"\n{'='*70}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
