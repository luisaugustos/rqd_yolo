#!/usr/bin/env python
"""
Sequential trainer: YOLO12m (batch 8, 50 epochs) → YOLO26n (50 epochs)
Monitors YOLO12m training and auto-starts YOLO26n when complete.
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime

def monitor_and_train():
    """Monitor YOLO12m batch8 and auto-start YOLO26n"""
    print(f"\n{'='*70}")
    print("Sequential Training: YOLO12m (batch 8) → YOLO26n")
    print(f"{'='*70}\n")

    print("[1/2] Monitoring YOLO12m (batch size 8, 50 epochs)...")

    # Wait for YOLO12m to complete
    max_wait = 18000  # 5 hours max
    check_interval = 120  # Check every 2 minutes
    elapsed = 0

    while elapsed < max_wait:
        # Check for yolo12m results with batch 8
        results_dirs = list(Path("C:/Users/luisb/dev/rqd_yolo/runs/detect/results/runs").glob("yolo12m*"))

        if results_dirs:
            # Get the most recent directory
            latest_dir = max(results_dirs, key=lambda p: p.stat().st_mtime)
            results_csv = latest_dir / "results.csv"

            if results_csv.exists():
                try:
                    with open(results_csv) as f:
                        lines = f.readlines()
                        current_epochs = len(lines) - 1

                        if current_epochs >= 50:
                            print(f"\n✓ YOLO12m training COMPLETE! ({current_epochs}/50 epochs)")
                            break
                        else:
                            progress = (current_epochs / 50) * 100
                            print(f"   YOLO12m: {current_epochs}/50 epochs ({progress:.1f}%) - {latest_dir.name}")
                except Exception as e:
                    print(f"   Error reading results: {e}")

        time.sleep(check_interval)
        elapsed += check_interval

    # Now train YOLO26n
    print(f"\n[2/2] Starting YOLO26n training (50 epochs)...")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cmd = [
        "python",
        "scripts/train_roboflow.py",
        "--dataset", "rock-quality",
        "--model", "yolo26n",
        "--epochs", "50",
        "--batch", "16",
        "--device", "0",
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ YOLO26n training completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ YOLO26n training failed with exit code {e.returncode}")

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUITE COMPLETE!")
    print(f"{'='*70}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext steps:")
    print(f"  python compare_models.py")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    monitor_and_train()
