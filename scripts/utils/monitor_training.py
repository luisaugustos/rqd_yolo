#!/usr/bin/env python
"""Monitor training progress by checking GPU usage and disk space."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def get_gpu_status():
    """Get GPU status from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        lines = result.stdout.strip().split('\n')
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                return {
                    'index': parts[0],
                    'name': parts[1],
                    'temperature': parts[2],
                    'utilization': parts[3],
                    'memory_used': parts[4],
                    'memory_total': parts[5],
                }
    except Exception as e:
        return {'error': str(e)}

def get_training_runs():
    """List active training runs."""
    results_dir = Path("results/runs")
    if not results_dir.exists():
        return []

    runs = []
    for run_dir in sorted(results_dir.glob("*"))[::-1]:
        if run_dir.is_dir():
            results_csv = run_dir / "results.csv"
            if results_csv.exists():
                with open(results_csv) as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        runs.append({
                            'name': run_dir.name,
                            'epochs_completed': len(lines) - 1,
                            'latest_mtime': run_dir.stat().st_mtime,
                        })
    return runs

def main():
    """Show training status."""
    print(f"\n{'='*70}")
    print(f"TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # GPU Status
    print("GPU Status:")
    gpu = get_gpu_status()
    if 'error' in gpu:
        print(f"  Error: {gpu['error']}")
    else:
        print(f"  GPU: {gpu['name']}")
        print(f"  Temperature: {gpu['temperature']}")
        print(f"  Utilization: {gpu['utilization']}")
        print(f"  Memory: {gpu['memory_used']} / {gpu['memory_total']}")

    # Training Runs
    print("\nActive Training Runs:")
    runs = get_training_runs()
    if not runs:
        print("  No training runs found")
    else:
        for run in runs:
            print(f"  • {run['name']}")
            print(f"    Epochs: {run['epochs_completed']}")

    print(f"\n{'='*70}")
    print("Tip: Check logs with:")
    print("  tail -f yolo12n_50epochs.log")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
