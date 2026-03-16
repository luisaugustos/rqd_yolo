#!/usr/bin/env python3
"""Standalone dataset validation script.

Runs all DQ-001 to DQ-014 checks against the configured dataset and prints
a summary. Exit code 0 = no hard errors; non-zero = errors found.

Usage:
    python scripts/validate_dataset.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def main() -> int:
    """Run dataset validation and return exit code."""
    parser = argparse.ArgumentParser(description="Validate rqd-ai-lab dataset")
    parser.add_argument(
        "--config",
        "-c",
        default="configs/experiment.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(levelname)-8s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    config_path = Path(args.config)
    if not config_path.exists():
        logging.error("Config not found: %s", config_path)
        return 1

    exp_cfg = yaml.safe_load(config_path.read_text()) or {}
    dataset_cfg_path = Path(exp_cfg.get("dataset_config", "configs/dataset.yaml"))
    if not dataset_cfg_path.exists():
        logging.error("Dataset config not found: %s", dataset_cfg_path)
        return 1

    dataset_cfg = yaml.safe_load(dataset_cfg_path.read_text()) or {}

    from src.dataset.loader import DatasetLoader

    loader = DatasetLoader(dataset_cfg, repo_root=config_path.parent.parent)
    report = loader.validate()

    print(f"\n{'='*60}")
    print(f"Dataset Validation Report")
    print(f"{'='*60}")
    print(f"Images checked:  {report.num_images_checked}")
    print(f"Hard errors:     {len(report.errors)}")
    print(f"Warnings:        {len(report.warnings)}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  [{w.rule}] {w.message}")

    if report.errors:
        print("\nErrors:")
        for e in report.errors:
            print(f"  [{e.rule}] {e.message}")
        print("\nValidation FAILED")
        return 1

    print("\nValidation PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
