#!/usr/bin/env python
"""
Main training orchestration script.
Supports local and remote training with config-based paths.
Handles both YOLO and RT-DETR models.

Usage:
    # Train single model
    python train.py --model yolo12n --epochs 50
    python train.py --model rtdetrv2_s --epochs 50

    # Sequential training (yolo12n → yolo12m → rtdetrv2_s)
    python train.py --sequential

    # Download from Roboflow first, then train
    python train.py --download --sequential

    # Custom config
    python train.py --config config.yaml --model rtdetrv2_m --batch 4
"""

import sys
import argparse
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO and RT-DETR models for rock fracture detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single model
  python train.py --model yolo12n --epochs 50
  python train.py --model rtdetrv2_s

  # Sequential training (all three models)
  python train.py --sequential

  # Download dataset first, then train
  python train.py --download --sequential

  # Custom batch size for RT-DETR
  python train.py --model rtdetrv2_m --batch 4 --epochs 50

  # Train with custom config
  python train.py --config config.yaml --model rtdetrv2_s
        """
    )

    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file (default: config.yaml)")
    parser.add_argument("--model", type=str,
                       help="Model to train (yolo12n, yolo12m, rtdetrv2_s, rtdetrv2_m, etc.)")
    parser.add_argument("--epochs", type=int,
                       help="Number of epochs")
    parser.add_argument("--batch", type=int,
                       help="Batch size")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID (default: 0)")
    parser.add_argument("--sequential", action="store_true",
                       help="Train models sequentially: yolo12n, yolo12m, rtdetrv2_s")
    parser.add_argument("--yolo-only", action="store_true",
                       help="Sequential training with YOLO models only")
    parser.add_argument("--rtdetr-only", action="store_true",
                       help="Sequential training with RT-DETR models only")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset from Roboflow first")

    args = parser.parse_args()

    project_root = Path(__file__).parent

    print(f"\n{'='*70}")
    print("Rock Fracture Detection - Training Pipeline")
    print(f"{'='*70}\n")

    # Step 1: Download from Roboflow if requested
    if args.download:
        print("[1/4] Downloading dataset from Roboflow...")
        download_script = project_root / "scripts" / "data" / "download_roboflow.py"
        if download_script.exists():
            try:
                subprocess.run([sys.executable, str(download_script)], cwd=project_root, check=True)
                print("✓ Dataset downloaded\n")
            except subprocess.CalledProcessError:
                print("✗ Download failed\n")
                return 1
        else:
            print(f"⚠ Download script not found\n")

    # Step 2: Determine what to train
    print("[2/4] Preparing training...")

    if args.sequential:
        print("  Sequential training mode")
        if args.rtdetr_only:
            models = ["rtdetrv2_s", "rtdetrv2_m"]
            print("  Models: RT-DETR only (s, m)")
        elif args.yolo_only:
            models = ["yolo12n", "yolo12m"]
            print("  Models: YOLO only (12n, 12m)")
        else:
            models = ["yolo12n", "yolo12m", "rtdetrv2_s"]
            print("  Models: All (yolo12n, yolo12m, rtdetrv2_s)")
    elif args.model:
        models = [args.model]
        print(f"  Single model: {args.model}")
    else:
        models = ["yolo12n"]
        print("  Default model: yolo12n")

    print()

    # Step 3: Train each model
    print("[3/4] Starting training...\n")

    train_script = project_root / "scripts" / "training" / "train_any_model.py"

    results = []
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Training {model}...")

        cmd = [
            sys.executable,
            str(train_script),
            "--model", model,
            "--device", str(args.device),
        ]

        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.batch:
            cmd.extend(["--batch", str(args.batch)])

        try:
            result = subprocess.run(cmd, cwd=project_root, check=True)
            results.append((model, True))
            print(f"✓ {model} completed")
        except subprocess.CalledProcessError:
            results.append((model, False))
            print(f"✗ {model} failed")
        except Exception as e:
            results.append((model, False))
            print(f"✗ {model} error: {e}")

    # Step 4: Summary
    print(f"\n[4/4] Training Complete!\n")
    print(f"{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")

    successful = sum(1 for _, s in results if s)
    failed = len(results) - successful

    for model, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {model:20s} {status}")

    print(f"\nResults: {successful} succeeded, {failed} failed")
    print(f"\nNext steps:")
    print(f"  - View results: python -m tensorboard --logdir runs/")
    print(f"  - Compare models: python scripts/utils/compare_models.py")
    print(f"  - Deploy: python rqd_cli.py --model <path>")
    print(f"\n{'='*70}\n")

    # Return success if all completed
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
