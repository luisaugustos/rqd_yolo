"""Compare YOLO model training results.

This script aggregates metrics from multiple trained models and generates
a comparison report showing mAP, recall, F1, precision, and inference speed.
"""

from pathlib import Path
import json
import yaml
from typing import dict, list

RESULTS_DIR = Path("results/runs")


def extract_metrics(run_dir: Path) -> dict:
    """Extract best metrics from a training run."""
    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        return None

    # Read the last line (best results)
    with open(results_csv) as f:
        lines = f.readlines()
        if len(lines) <= 1:
            return None

        # Parse header and last row
        header = lines[0].strip().split(',')
        last_row = lines[-1].strip().split(',')

        metrics = {h: float(v) for h, v in zip(header, last_row) if h and v}
        return metrics


def get_model_info(run_dir: Path) -> dict:
    """Extract model info from args.yaml."""
    args_file = run_dir / "args.yaml"

    if not args_file.exists():
        return {}

    with open(args_file) as f:
        args = yaml.safe_load(f)
        return {
            'model': args.get('model', 'unknown'),
            'epochs': args.get('epochs', 'unknown'),
            'batch_size': args.get('batch_size', 'unknown'),
        }


def main():
    """Generate comparison report."""
    print("="*70)
    print("YOLO Model Training Comparison — Rock-Quality Dataset")
    print("="*70)

    results = []

    # Find all training runs
    if not RESULTS_DIR.exists():
        print("ERROR: No results directory found")
        return

    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics = extract_metrics(run_dir)
        if not metrics:
            continue

        info = get_model_info(run_dir)

        results.append({
            'name': run_dir.name,
            'model': info.get('model', ''),
            'metrics': metrics,
        })

    if not results:
        print("No training results found")
        return

    # Print comparison table
    print(f"\n{'Model':<25} {'mAP50':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'Speed(ms)':>10}")
    print("-" * 75)

    for result in results:
        m = result['metrics']
        name = result['name'][:24]
        map50 = m.get('metrics/mAP50(B)', 'N/A')
        recall = m.get('metrics/recall(B)', 'N/A')
        precision = m.get('metrics/precision(B)', 'N/A')
        f1 = m.get('metrics/F1_curve', 'N/A')
        speed = m.get('inference/GPU_inference(ms)', 'N/A')

        # Format numbers
        if isinstance(map50, float):
            map50 = f"{map50:.4f}"
        if isinstance(recall, float):
            recall = f"{recall:.4f}"
        if isinstance(precision, float):
            precision = f"{precision:.4f}"
        if isinstance(f1, float):
            f1 = f"{f1:.4f}"
        if isinstance(speed, float):
            speed = f"{speed:.2f}"

        print(f"{name:<25} {map50:>10} {recall:>10} {precision:>10} {f1:>10} {speed:>10}")

    print("\n" + "="*70)
    print("Note: Higher mAP50 and Recall are better. Speed is inference time per image.")
    print("="*70)


if __name__ == "__main__":
    main()
