#!/usr/bin/env python
"""
Sequential trainer for YOLO12 models.
Automatically trains multiple models one after another.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

MODELS = [
    ("yolo12n", 50, 16),  # (model_name, epochs, batch_size)
    ("yolo12m", 50, 16),
]

def train_model(model_name: str, epochs: int, batch_size: int) -> bool:
    """Train a single model."""
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"{'='*70}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cmd = [
        "python",
        "scripts/train_roboflow.py",
        "--dataset", "rock-quality",
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--device", "0",
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✓ {model_name} training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name} training failed with exit code {e.returncode}")
        return False

def main():
    """Train all models sequentially."""
    print(f"\nYOLO12 Sequential Training Suite")
    print(f"{'='*70}")
    print(f"Dataset: rock-quality (92 train, 26 val, 14 test)")
    print(f"Models to train: {', '.join([m[0] for m in MODELS])}")
    print(f"{'='*70}\n")

    results = {}
    for model_name, epochs, batch_size in MODELS:
        success = train_model(model_name, epochs, batch_size)
        results[model_name] = "✓ Success" if success else "✗ Failed"

    # Summary
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*70}")
    for model_name, status in results.items():
        print(f"{model_name:15} {status}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTo compare results, run:")
    print(f"  python compare_models.py")

if __name__ == "__main__":
    # Note: YOLO12n is already training in background
    print("Note: YOLO12n training is already running in background")
    print("Skipping YOLO12n, proceeding to YOLO12m training...")

    # Train YOLO12m
    train_model("yolo12m", 50, 16)
