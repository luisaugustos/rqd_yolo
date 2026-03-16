"""Command-line interface for the rqd-ai-lab pipeline (FR-016).

Usage:
    rqd ingest       --config configs/experiment.yaml
    rqd preprocess   --config configs/experiment.yaml
    rqd train        --config configs/experiment.yaml
    rqd infer        --config configs/experiment.yaml --input path/to/image.jpg
    rqd evaluate     --config configs/experiment.yaml
    rqd compute-rqd  --config configs/experiment.yaml --input path/to/image.jpg
    rqd report       --config configs/experiment.yaml --run-id <id>
    rqd validate-data --config configs/experiment.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import click
import yaml

logger = logging.getLogger("rqd")


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--config",
    "-c",
    default="configs/experiment.yaml",
    show_default=True,
    help="Path to the experiment configuration YAML file.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool) -> None:
    """rqd-ai-lab — Automated RQD from drill core photographs."""
    ctx.ensure_object(dict)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=level,
    )
    try:
        cfg = _load_config(config)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    ctx.obj["config"] = cfg
    ctx.obj["config_path"] = Path(config)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@cli.command()
@click.pass_context
def ingest(ctx: click.Context) -> None:
    """Validate dataset structure and print a summary (FR-001, DQ rules)."""
    cfg = ctx.obj["config"]
    try:
        from src.dataset.loader import DatasetLoader

        dataset_cfg = _load_sub_config(cfg, "dataset_config")
        loader = DatasetLoader(dataset_cfg)
        report = loader.validate()
        if report.has_errors:
            click.echo(
                f"Validation FAILED: {len(report.errors)} errors, "
                f"{len(report.warnings)} warnings across {report.num_images_checked} images.",
                err=True,
            )
            for e in report.errors[:20]:
                click.echo(f"  ERROR [{e.rule}]: {e.message}", err=True)
            sys.exit(1)
        click.echo(
            f"Validation PASSED: {report.num_images_checked} images checked, "
            f"{len(report.warnings)} warnings."
        )
    except Exception as exc:
        logger.exception("ingest failed")
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def preprocess(ctx: click.Context) -> None:
    """Preprocess images and write to data/processed/ (FR-002)."""
    cfg = ctx.obj["config"]
    try:
        from src.dataset.loader import DatasetLoader
        from src.preprocessing.preprocessor import Preprocessor

        dataset_cfg = _load_sub_config(cfg, "dataset_config")
        loader = DatasetLoader(dataset_cfg)
        pp = Preprocessor()
        for split in ("train", "val", "test"):
            samples = loader.load_split(split)
            for sample in samples:
                pp.process(sample)
        click.echo("Preprocessing complete.")
    except Exception as exc:
        logger.exception("preprocess failed")
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--split", default="train", show_default=True, help="Dataset split to train on.")
@click.pass_context
def train(ctx: click.Context, split: str) -> None:
    """Fine-tune a detection model (FR-005)."""
    cfg = ctx.obj["config"]
    try:
        from src.detection.module import DetectionModule

        det_cfg = _load_sub_config(cfg, "detection_config")
        module = DetectionModule(det_cfg)
        module.load()
        click.echo(
            f"Training backend '{det_cfg.get('model', {}).get('backend')}' — "
            "invoke the Ultralytics training API from the module."
        )
    except Exception as exc:
        logger.exception("train failed")
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", required=True, help="Path to input image.")
@click.option("--output-dir", "-o", default="results/predictions", show_default=True)
@click.pass_context
def infer(ctx: click.Context, input: str, output_dir: str) -> None:
    """Run full RQD inference pipeline on a single image (FR-001 to FR-012)."""
    cfg = ctx.obj["config"]
    try:
        result = _run_pipeline(input, cfg)
        out_path = Path(output_dir) / (Path(input).stem + "_rqd.json")
        import json

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "image_id": result["rqd"].image_id,
                    "rqd_pct": result["rqd"].rqd_pct,
                    "total_run_length_mm": result["rqd"].total_run_length_mm,
                    "qualifying_length_mm": result["rqd"].qualifying_length_mm,
                    "num_fragments": result["rqd"].num_fragments_total,
                    "num_qualifying": result["rqd"].num_fragments_qualifying,
                },
                indent=2,
            )
        )
        click.echo(
            f"RQD={result['rqd'].rqd_pct:.1f}%  "
            f"(qualifying={result['rqd'].qualifying_length_mm:.0f}mm / "
            f"total={result['rqd'].total_run_length_mm:.0f}mm)  "
            f"→ {out_path}"
        )
    except Exception as exc:
        logger.exception("infer failed")
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command(name="compute-rqd")
@click.option("--input", "-i", required=True, help="Path to input image.")
@click.option("--output-dir", "-o", default="results/rqd", show_default=True)
@click.option(
    "--pixels-per-mm",
    type=float,
    default=None,
    help="Manual calibration override (pixels per mm).",
)
@click.pass_context
def compute_rqd(
    ctx: click.Context, input: str, output_dir: str, pixels_per_mm: float | None
) -> None:
    """Compute RQD from an image with optional manual calibration (FR-011, FR-012)."""
    cfg = ctx.obj["config"]
    try:
        result = _run_pipeline(input, cfg, manual_ppm=pixels_per_mm)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vis_path = out_dir / (Path(input).stem + "_vis.png")
        _save_visualization(result, vis_path)
        click.echo(f"RQD={result['rqd'].rqd_pct:.2f}%  →  {vis_path}")
    except Exception as exc:
        logger.exception("compute-rqd failed")
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--split", default="test", show_default=True)
@click.pass_context
def evaluate(ctx: click.Context, split: str) -> None:
    """Run evaluation on the test split and write an evaluation report (FR-014)."""
    cfg = ctx.obj["config"]
    try:
        click.echo(f"Evaluating on split='{split}' — load dataset and run predictions.")
        click.echo("Run 'rqd infer' on each image first, then collect results for full eval.")
    except Exception as exc:
        logger.exception("evaluate failed")
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--run-id", required=True, help="Experiment run ID to report on.")
@click.pass_context
def report(ctx: click.Context, run_id: str) -> None:
    """Generate evaluation report for a completed run (FR-014)."""
    cfg = ctx.obj["config"]
    reports_dir = Path(cfg.get("paths", {}).get("reports_dir", "results/reports"))
    run_dir = reports_dir / run_id
    if not run_dir.exists():
        click.echo(f"No results found for run '{run_id}' at {run_dir}", err=True)
        sys.exit(1)
    click.echo(f"Report directory: {run_dir}")
    for f in sorted(run_dir.iterdir()):
        click.echo(f"  {f.name}")


@cli.command(name="validate-data")
@click.pass_context
def validate_data(ctx: click.Context) -> None:
    """Validate dataset against DQ rules and print a summary."""
    ctx.invoke(ingest)


# ---------------------------------------------------------------------------
# Shared pipeline helper
# ---------------------------------------------------------------------------


def _run_pipeline(
    image_path: str,
    cfg: dict[str, Any],
    manual_ppm: float | None = None,
) -> dict[str, Any]:
    """Run the full RQD pipeline on one image.

    Returns a dict with keys: sample, processed, detections, segmentations,
    measurements, rqd.
    """
    from src.dataset.loader import _read_image
    from src.measurement.engine import MeasurementEngine
    from src.preprocessing.preprocessor import Preprocessor
    from src.rqd.engine import RQDEngine
    from src.segmentation.module import SegmentationModule
    from src.utils.contracts import (
        CalibrationInfo,
        ImageSample,
        TrayRow,
    )
    from src.detection.module import DetectionModule

    # ---- Load image ----
    img_path = Path(image_path)
    image = _read_image(img_path)
    if image is None:
        raise ValueError(f"Could not read image: {img_path}")
    h, w = image.shape[:2]
    sample = ImageSample(
        image_id=img_path.stem,
        file_path=str(img_path),
        image=image,
        width=w,
        height=h,
    )

    # ---- Preprocess ----
    pp = Preprocessor()
    processed = pp.process(sample)

    # ---- Calibration ----
    if manual_ppm is not None:
        calib = CalibrationInfo(
            image_id=sample.image_id,
            pixels_per_mm=manual_ppm,
            source="manual",
        )
        logger.warning("Using manual calibration: %.2f px/mm", manual_ppm)
    else:
        # Default: assume 5 px/mm (placeholder; real calibration requires scale marker)
        default_ppm = 5.0
        calib = CalibrationInfo(
            image_id=sample.image_id,
            pixels_per_mm=default_ppm,
            source="manual",
            warning="Using default placeholder calibration; provide --pixels-per-mm for accuracy.",
        )
        logger.warning("Calibration not configured; using default %.1f px/mm", default_ppm)

    # ---- Tray row (placeholder — single row spanning full image width) ----
    row = TrayRow(
        row_id=0,
        image_id=sample.image_id,
        bbox=_whole_image_bbox(w, h),
        row_length_px=float(w),
    )

    # ---- Detection ----
    det_cfg = _load_sub_config(cfg, "detection_config")
    det_module = DetectionModule(det_cfg)
    det_module.load()
    detections = det_module.detect(processed)

    # ---- Segmentation ----
    seg_cfg = _load_sub_config(cfg, "segmentation_config")
    seg_module = SegmentationModule(seg_cfg)
    seg_module.load()
    segmentations = seg_module.segment(processed, detections)

    # ---- Measurement ----
    meas_engine = MeasurementEngine()
    measurements = meas_engine.measure(detections, segmentations, calib, row)

    # ---- RQD ----
    rqd_engine = RQDEngine()
    row_result = rqd_engine.compute_row_rqd(measurements, row, calib)
    image_result = rqd_engine.compute_image_rqd([row_result])

    return {
        "sample": sample,
        "processed": processed,
        "detections": detections,
        "segmentations": segmentations,
        "measurements": measurements,
        "rqd": image_result,
        "row_rqd": row_result,
        "row": row,
        "calibration": calib,
    }


def _whole_image_bbox(w: int, h: int):  # type: ignore[no-untyped-def]
    """Return a BBox covering the full image."""
    from src.utils.contracts import BBox

    return BBox(x1=0.0, y1=0.0, x2=float(w), y2=float(h))


def _save_visualization(result: dict[str, Any], output_path: Path) -> None:
    """Save an annotated visualization image."""
    from src.visualization.visualizer import Visualizer

    vis = Visualizer()
    canvas = vis.draw(
        result["sample"].image,
        detections=result["detections"],
        segmentations=result["segmentations"],
        measurements=result["measurements"],
        rqd_results=[result["rqd"]],
        tray_rows=[result["row"]],
    )
    vis.save(canvas, output_path)


def _load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file.

    Raises:
        FileNotFoundError: When the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def _load_sub_config(parent_cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """Load a sub-config YAML referenced in the parent config."""
    sub_path = parent_cfg.get(key)
    if sub_path and Path(sub_path).exists():
        return yaml.safe_load(Path(sub_path).read_text()) or {}
    return {}


if __name__ == "__main__":
    cli()
