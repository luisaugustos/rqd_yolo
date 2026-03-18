#!/usr/bin/env python
"""
Sequential trainer: YOLO12m (50 epochs) → YOLO26n (50 epochs)
Waits for YOLO12m to complete, then automatically starts YOLO26n.
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime

def wait_for_training_completion(model_name: str, max_epochs: int, check_interval: int = 60):
    """Wait for training to complete by monitoring results.csv"""
    print(f"\n⏳ Waiting for {model_name} training to complete...")
    print(f"   Checking every {check_interval} seconds...")

    while True:
        # Find the results.csv for this model
        results_dirs = list(Path("C:/Users/luisb/dev/rqd_yolo/runs/detect/results/runs").glob(f"{model_name}*"))

        if results_dirs:
            results_csv = results_dirs[-1] / "results.csv"  # Get latest run

            if results_csv.exists():
                with open(results_csv) as f:
                    lines = f.readlines()
                    current_epochs = len(lines) - 1  # Subtract header

                    if current_epochs >= max_epochs:
                        print(f"\n✓ {model_name} training COMPLETE! ({current_epochs}/{max_epochs} epochs)")
                        return True
                    else:
                        progress = (current_epochs / max_epochs) * 100
                        print(f"   {model_name}: {current_epochs}/{max_epochs} epochs ({progress:.1f}%)")

        time.sleep(check_interval)

def train_model(model_name: str, epochs: int = 50):
    """Train a YOLO model"""
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} | {epochs} epochs")
    print(f"{'='*70}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cmd = [
        "python",
        "scripts/train_roboflow.py",
        "--dataset", "rock-quality",
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch", "16",
        "--device", "0",
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {model_name} training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name} training failed with exit code {e.returncode}")
        return False

def main():
    """Execute sequential training"""
    print(f"\n{'='*70}")
    print("YOLO Sequential Training: YOLO12m → YOLO26n")
    print(f"{'='*70}")
    print(f"Dataset: rock-quality (92 train, 26 val, 14 test)")
    print(f"Configuration: 50 epochs each, batch size 16, GPU 0")

    # YOLO12m should already be training in background
    print("\n[1/2] YOLO12m is already running in background")
    print("      Waiting for completion...")

    wait_for_training_completion("yolo12m", max_epochs=50, check_interval=120)

    # Now train YOLO26n
    print("\n[2/2] Starting YOLO26n training")
    train_model("yolo26n", epochs=50)

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUITE COMPLETE!")
    print(f"{'='*70}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext steps:")
    print(f"  python compare_models.py")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
