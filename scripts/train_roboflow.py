"""Train a YOLO model on a Roboflow-downloaded dataset.

This script trains YOLOv8/v11 on any dataset downloaded with download_roboflow.py.
It uses the data.yaml produced by Roboflow and the hyperparameters from
configs/yolo_train.yaml (overridable via CLI flags).

Usage:
    # Basic training with rock-quality dataset (auto-detects data.yaml)
    python scripts/train_roboflow.py --dataset rock-quality

    # Explicit data.yaml path
    python scripts/train_roboflow.py --data data/roboflow/rock-quality-1/data.yaml

    # Custom model and epochs
    python scripts/train_roboflow.py --dataset rock-quality --model yolov8m --epochs 50

    # Fast smoke-test (3 epochs, small model)
    python scripts/train_roboflow.py --dataset rock-quality --model yolov8n --epochs 3

Outputs are saved under results/runs/<run-name>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "yolo_train.yaml"

# Maps dataset name → expected data.yaml relative path (after download)
DATASET_DATA_YAML: dict[str, str] = {
    "rock-quality": "data/roboflow/rock-quality-1/data.yaml",
    "rock-core-box": "data/roboflow/rock-core-box-1/data.yaml",
}


def load_train_config(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on a Roboflow dataset")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--dataset",
        choices=list(DATASET_DATA_YAML.keys()),
        help="Named dataset (resolves data.yaml automatically)",
    )
    data_group.add_argument(
        "--data",
        type=str,
        help="Explicit path to data.yaml",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLO model variant, e.g. yolov8n, yolov8m, yolov11m (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size (overrides config)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: auto | cpu | 0 | 0,1 (overrides config)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name for output folder (default: auto-generated)",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_TRAIN_CONFIG),
        help=f"Training config YAML (default: {DEFAULT_TRAIN_CONFIG})",
    )
    args = parser.parse_args()

    # --- Resolve data.yaml ---
    if args.data:
        data_yaml = Path(args.data)
    else:
        rel = DATASET_DATA_YAML[args.dataset]
        data_yaml = REPO_ROOT / rel

    if not data_yaml.exists():
        print(
            f"ERROR: data.yaml not found at {data_yaml}\n"
            f"  Download the dataset first:\n"
            f"  python scripts/download_roboflow.py --api-key <KEY> --dataset {args.dataset or 'rock-quality'}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load base config ---
    cfg = load_train_config(Path(args.config))
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    # --- Resolve model weights ---
    if args.model:
        model_weights = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    else:
        model_weights = model_cfg.get("weights", "yolov8m.pt")

    # --- Resolve hyperparams (CLI > config) ---
    epochs = args.epochs or train_cfg.get("epochs", 100)
    imgsz = args.imgsz or train_cfg.get("imgsz", 640)
    batch = args.batch or train_cfg.get("batch_size", 16)
    device = args.device or train_cfg.get("device", "auto")
    project = train_cfg.get("project", "results/runs")
    dataset_tag = args.dataset or data_yaml.parent.name
    name = args.name or f"{Path(model_weights).stem}_{dataset_tag}"
    patience = train_cfg.get("patience", 50)
    workers = train_cfg.get("workers", 8)
    optimizer = train_cfg.get("optimizer", "AdamW")
    lr0 = train_cfg.get("lr0", 0.001)
    lrf = train_cfg.get("lrf", 0.01)
    weight_decay = train_cfg.get("weight_decay", 0.0005)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    amp = train_cfg.get("amp", True)

    # --- Print summary ---
    print("=" * 60)
    print("YOLO Training — Roboflow Dataset")
    print("=" * 60)
    print(f"  data.yaml : {data_yaml}")
    print(f"  model     : {model_weights}")
    print(f"  epochs    : {epochs}")
    print(f"  imgsz     : {imgsz}")
    print(f"  batch     : {batch}")
    print(f"  device    : {device}")
    print(f"  output    : {project}/{name}")
    print("=" * 60)

    # --- Import and train ---
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(model_weights)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        amp=amp,
        exist_ok=False,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    print("\nTraining complete!")
    print(f"  Best weights : {best_weights}")
    print(f"  Results dir  : {Path(project) / name}")

    # --- Quick validation ---
    print("\nRunning validation on best weights …")
    val_model = YOLO(str(best_weights))
    metrics = val_model.val(data=str(data_yaml), imgsz=imgsz, device=device)
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
