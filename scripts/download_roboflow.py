"""Download datasets from Roboflow Universe for RQD training.

Supported datasets:
  - rock-quality      (workspace: tfm-w79wc)
  - rock-core-box     (workspace: maskrcnn-rock-core)

Usage:
    # Download rock-quality dataset (default)
    python scripts/download_roboflow.py --api-key <KEY>

    # Download rock-core-box dataset
    python scripts/download_roboflow.py --api-key <KEY> --dataset rock-core-box

    # Use environment variable instead of --api-key
    export ROBOFLOW_API_KEY=<KEY>
    python scripts/download_roboflow.py --dataset rock-quality

Downloaded files are placed under data/roboflow/<project>-<version>/.
The script prints the data.yaml path to use for YOLO training.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

KNOWN_DATASETS: dict[str, dict] = {
    "rock-quality": {
        "workspace": "tfm-w79wc",
        "project": "rock-quality",
        "default_version": 1,
        "default_format": "yolov8",
        "description": "Rock Quality Designation (RQD) drill core detection",
    },
    "rock-core-box": {
        "workspace": "maskrcnn-rock-core",
        "project": "rock-core-box",
        "default_version": 1,
        "default_format": "coco-segmentation",
        "description": "Rock core box semantic segmentation",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Roboflow datasets for RQD training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k}: {v['description']}" for k, v in KNOWN_DATASETS.items()
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--dataset",
        default="rock-quality",
        choices=list(KNOWN_DATASETS.keys()),
        help="Dataset to download (default: rock-quality)",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override workspace slug",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Override project slug",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Dataset version number (default: latest known version)",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["yolov8", "yolov5", "coco", "coco-segmentation", "png-mask-semantic"],
        help="Export format (default: yolov8 for detection, coco-segmentation for segmentation)",
    )
    parser.add_argument(
        "--output",
        default="data/roboflow",
        help="Output directory (default: data/roboflow)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "ERROR: No API key provided.\n"
            "  Pass --api-key <KEY>  or  export ROBOFLOW_API_KEY=<KEY>\n"
            "  Get your key at: https://app.roboflow.com/settings/api",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: pip install roboflow", file=sys.stderr)
        sys.exit(1)

    # Resolve dataset parameters
    meta = KNOWN_DATASETS[args.dataset]
    workspace = args.workspace or meta["workspace"]
    project_slug = args.project or meta["project"]
    version_num = args.version or meta["default_version"]
    fmt = args.format or meta["default_format"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    location = output_dir / f"{project_slug}-{version_num}"

    print(f"Dataset  : {args.dataset}")
    print(f"Workspace: {workspace}")
    print(f"Project  : {project_slug}  v{version_num}")
    print(f"Format   : {fmt}")
    print(f"Output   : {location}")
    print()

    print(f"Connecting to Roboflow …")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(workspace).project(project_slug)

    print(f"Downloading version {version_num} …")
    dataset = project.version(version_num).download(
        model_format=fmt,
        location=str(location),
        overwrite=False,
    )

    data_yaml = Path(dataset.location) / "data.yaml"
    print(f"\nDataset downloaded to: {dataset.location}")

    if data_yaml.exists():
        import yaml  # type: ignore[import]

        info = yaml.safe_load(data_yaml.read_text())
        nc = info.get("nc", "?")
        names = info.get("names", [])
        print(f"Classes ({nc}): {names}")
        print(f"\ndata.yaml path: {data_yaml.resolve()}")
        print(
            f"\nTo train with this dataset:\n"
            f"  python scripts/train_roboflow.py --data {data_yaml.resolve()}"
        )
    else:
        print(f"WARNING: data.yaml not found at {data_yaml}")

    print("\nDone!")


if __name__ == "__main__":
    main()
