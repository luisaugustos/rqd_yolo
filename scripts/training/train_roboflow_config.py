#!/usr/bin/env python
"""
Training script with portable config-based paths.
Works on local machine and remote server.

Usage:
    python scripts/training/train_roboflow_config.py --model yolo12n --epochs 50
    python scripts/training/train_roboflow_config.py --config config.yaml --model yolo12m
"""

import subprocess
import sys
from pathlib import Path
import yaml
import argparse
from datetime import datetime

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"⚠ Config file not found: {config_file}")
        print("  Using defaults...")
        return {}

    with open(config_file, 'r') as f:
        return yaml.safe_load(f) or {}

def get_project_root():
    """Get project root directory"""
    # Assume this script is in scripts/training/
    return Path(__file__).parent.parent.parent

def resolve_path(path_str, project_root):
    """Convert config path to absolute path"""
    if path_str.startswith('/'):
        return Path(path_str)
    return project_root / path_str

def train_model(model_name: str, epochs: int, batch_size: int, config: dict, device: int = 0) -> bool:
    """Train a single model using scripts/train_roboflow.py"""

    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"{'='*70}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Get paths from config
    project_root = get_project_root()

    dataset_config = config.get('paths', {}).get('dataset_config', 'data/annotated/dataset_hp_v2/data.yaml')
    dataset_path = str(resolve_path(dataset_config, project_root))

    # Build command
    cmd = [
        sys.executable,  # Use python from current environment
        "scripts/train_roboflow.py",
        "--data", dataset_path,
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--device", str(device),
    ]

    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print(f"\n✓ {model_name.upper()} training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name.upper()} training failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Training script not found: scripts/train_roboflow.py")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train YOLO models with config-based paths")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, help="Model to train (yolo12n, yolo12m, yolo26n, etc.)")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--sequential", action="store_true", help="Train yolo12n, then yolo12m")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    training_config = config.get('training', {})

    # Determine what to train
    models_to_train = []

    if args.sequential:
        # Sequential training mode
        models_to_train = [
            ('yolo12n', 50, 16),
            ('yolo12m', 50, 8),
        ]
    elif args.model:
        # Single model from CLI
        epochs = args.epochs or training_config.get('default_epochs', 50)
        batch = args.batch or training_config.get('default_batch', 16)
        models_to_train = [(args.model, epochs, batch)]
    else:
        # Default model from config
        model = training_config.get('default_model', 'yolo12n')
        epochs = training_config.get('default_epochs', 50)
        batch = training_config.get('default_batch', 16)
        models_to_train = [(model, epochs, batch)]

    print(f"\n{'='*70}")
    print(f"YOLO Training with Config-Based Paths")
    print(f"{'='*70}")
    print(f"Project root: {get_project_root()}")
    print(f"Config file: {args.config}")
    print(f"Models to train: {len(models_to_train)}")

    # Train models
    results = []
    for model_name, epochs, batch_size in models_to_train:
        success = train_model(model_name, epochs, batch_size, config, device=args.device)
        results.append((model_name, success))

    # Summary
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*70}")
    for model_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{model_name:15s} {status}")
    print(f"{'='*70}\n")

    # Return exit code based on results
    failed = sum(1 for _, success in results if not success)
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
