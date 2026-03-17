"""Visualizer — draw detection, segmentation, and RQD overlays on images (FR-013)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.contracts import (
    DetectionResult,
    FragmentMeasurement,
    RQDResult,
    SegmentationResult,
    TrayRow,
)

logger = logging.getLogger(__name__)

# Color scheme (BGR for OpenCV): class_id -> (B, G, R)
_COLORS = {
    "intact_fragment": (86, 180, 233),   # light blue
    "fracture": (230, 159, 0),            # orange
    "tray_row": (0, 114, 178),            # dark blue
    "scale_marker": (0, 158, 115),        # green
    "qualifying": (0, 200, 80),           # bright green
    "non_qualifying": (230, 50, 50),      # red
    "default": (200, 200, 200),           # grey
}


class Visualizer:
    """Draw bounding boxes, masks, lengths, and RQD on images (FR-013).

    Args:
        config: Visualization configuration dict. Relevant keys:
            - draw_boxes (bool): Default True.
            - draw_masks (bool): Default True.
            - draw_lengths (bool): Default True.
            - draw_rqd (bool): Default True.
            - draw_tray_rows (bool): Default True.
            - font_scale (float): Default 0.5.
            - line_thickness (int): Default 2.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._draw_boxes: bool = cfg.get("draw_boxes", True)
        self._draw_masks: bool = cfg.get("draw_masks", True)
        self._draw_lengths: bool = cfg.get("draw_lengths", True)
        self._draw_rqd: bool = cfg.get("draw_rqd", True)
        self._draw_tray_rows: bool = cfg.get("draw_tray_rows", True)
        self._font_scale: float = float(cfg.get("font_scale", 0.5))
        self._thickness: int = int(cfg.get("line_thickness", 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        image: np.ndarray,
        detections: DetectionResult | None = None,
        segmentations: list[SegmentationResult] | None = None,
        measurements: list[FragmentMeasurement] | None = None,
        rqd_results: list[RQDResult] | None = None,
        tray_rows: list[TrayRow] | None = None,
    ) -> np.ndarray:
        """Produce an annotated copy of the input image.

        All overlay elements are configurable via the constructor config.

        Args:
            image: (H, W, 3) uint8 RGB image.
            detections: Detection result to draw boxes from.
            segmentations: Segmentation masks to overlay.
            measurements: Fragment measurements to annotate with lengths.
            rqd_results: Per-row RQD values to overlay.
            tray_rows: Tray row boundaries to draw.

        Returns:
            (H, W, 3) uint8 annotated RGB image.
        """
        import cv2

        canvas = image.copy()

        if self._draw_masks and segmentations:
            canvas = self._overlay_masks(canvas, segmentations)

        if self._draw_tray_rows and tray_rows:
            canvas = self._draw_row_boundaries(canvas, tray_rows)

        if self._draw_boxes and detections:
            canvas = self._draw_detection_boxes(canvas, detections)

        if self._draw_lengths and measurements:
            canvas = self._annotate_lengths(canvas, measurements)

        if self._draw_rqd and rqd_results:
            canvas = self._annotate_rqd(canvas, rqd_results)

        return canvas

    def save(self, image: np.ndarray, output_path: Path) -> None:
        """Save an annotated image to disk.

        Args:
            image: (H, W, 3) uint8 annotated RGB image.
            output_path: Destination file path (.png or .jpg).

        Raises:
            IOError: When the output directory is not writable.
        """
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(output_path), bgr)
        if not success:
            raise IOError(f"Could not write image to {output_path}")
        logger.info("Saved annotated image to %s", output_path)

    # ------------------------------------------------------------------
    # Internal draw helpers
    # ------------------------------------------------------------------

    def _draw_detection_boxes(
        self, canvas: np.ndarray, detections: DetectionResult
    ) -> np.ndarray:
        """Draw bounding boxes with class labels and confidence scores."""
        import cv2

        for box, score, cls_name in zip(detections.boxes, detections.scores, detections.class_names):
            color = _COLORS.get(cls_name, _COLORS["default"])
            pt1 = (int(box.x1), int(box.y1))
            pt2 = (int(box.x2), int(box.y2))
            cv2.rectangle(canvas, pt1, pt2, color, self._thickness)
            label = f"{cls_name} {score:.2f}"
            cv2.putText(
                canvas, label, (pt1[0], max(pt1[1] - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, color, 1, cv2.LINE_AA,
            )
        return canvas

    def _overlay_masks(
        self, canvas: np.ndarray, segmentations: list[SegmentationResult]
    ) -> np.ndarray:
        """Blend segmentation masks as semi-transparent colour overlays."""
        overlay = canvas.copy()
        for seg in segmentations:
            if seg.mask_area_px == 0:
                continue
            color = _COLORS["intact_fragment"]
            mask_bool = seg.mask.astype(bool)
            overlay[mask_bool] = (
                overlay[mask_bool] * 0.5 + np.array(color[::-1]) * 0.5
            ).astype(np.uint8)
        return overlay

    def _draw_row_boundaries(
        self, canvas: np.ndarray, rows: list[TrayRow]
    ) -> np.ndarray:
        """Draw tray row bounding boxes."""
        import cv2

        color = _COLORS["tray_row"]
        for row in rows:
            b = row.bbox
            cv2.rectangle(
                canvas,
                (int(b.x1), int(b.y1)),
                (int(b.x2), int(b.y2)),
                color,
                1,
            )
            cv2.putText(
                canvas,
                f"Row {row.row_id}",
                (int(b.x1) + 4, int(b.y1) + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._font_scale * 0.8,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _annotate_lengths(
        self, canvas: np.ndarray, measurements: list[FragmentMeasurement]
    ) -> np.ndarray:
        """Annotate each fragment with its length in mm.

        Qualifying fragments (≥ threshold) are highlighted in green (AC-013-2).
        """
        import cv2

        for m in measurements:
            color = _COLORS["qualifying"] if m.qualifies_rqd else _COLORS["non_qualifying"]
            b = m.bbox_px
            cx = int((b.x1 + b.x2) / 2)
            cy = int((b.y1 + b.y2) / 2)
            cv2.putText(
                canvas,
                f"{m.length_mm:.0f}mm",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _annotate_rqd(
        self, canvas: np.ndarray, rqd_results: list[RQDResult]
    ) -> np.ndarray:
        """Overlay RQD percentage on each row (AC-013-3) and aggregate at top."""
        import cv2

        color = (255, 255, 255)
        for result in rqd_results:
            if result.scope == "image":
                cv2.putText(
                    canvas,
                    f"RQD={result.rqd_pct:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self._font_scale * 1.4,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return canvas
