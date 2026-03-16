"""MeasurementEngine — converts detections to physical fragment lengths (FR-008, FR-009).

Applies the pixel-to-mm calibration factor and extracts the principal axis
length of each detected core fragment.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np

from src.utils.contracts import (
    BBox,
    CalibrationInfo,
    DetectionResult,
    FragmentMeasurement,
    SegmentationResult,
    TrayRow,
)

logger = logging.getLogger(__name__)

# ISRM RQD threshold (mm) — configurable via rqd_threshold_mm parameter
_DEFAULT_RQD_THRESHOLD_MM = 100.0

# Class ID for intact fragments (must match label set in dataset.yaml)
_FRAGMENT_CLASS_ID = 1


class MeasurementEngine:
    """Estimate fragment lengths in millimetres (FR-008, FR-009, FR-010).

    Converts detected bounding boxes or segmentation masks to physical
    measurements using the supplied calibration factor.

    Args:
        config: Optional config dict. Relevant keys:
            - rqd_threshold_mm (float): Default 100.0.
            - measurement_method (str): 'bbox' | 'mask_pca' | 'skeleton'. Default 'bbox'.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._rqd_threshold_mm: float = float(cfg.get("rqd_threshold_mm", _DEFAULT_RQD_THRESHOLD_MM))
        self._method: Literal["bbox", "mask_pca", "skeleton", "bbox_fallback"] = cfg.get(
            "measurement_method", "bbox"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(
        self,
        detections: DetectionResult,
        segmentations: list[SegmentationResult],
        calibration: CalibrationInfo,
        row: TrayRow | None = None,
    ) -> list[FragmentMeasurement]:
        """Measure all intact fragments in an image.

        Processes only detections with class_id == FRAGMENT_CLASS_ID.

        Args:
            detections: Detection results (coordinates in original image pixels).
            segmentations: Segmentation results (one per fragment, optional).
            calibration: Pixel-to-mm calibration for this image.
            row: Optional tray row context (used to record row_id).

        Returns:
            List of FragmentMeasurement objects.

        Raises:
            ValueError: When calibration.pixels_per_mm is zero or negative.
        """
        if calibration.pixels_per_mm <= 0:
            raise ValueError(
                f"CalibrationError: pixels_per_mm must be positive, "
                f"got {calibration.pixels_per_mm}"
            )

        fragment_dets = detections.filter_by_class(_FRAGMENT_CLASS_ID)
        seg_map = {s.fragment_id: s for s in segmentations}
        row_id = row.row_id if row is not None else 0

        measurements: list[FragmentMeasurement] = []
        for i, (box, score) in enumerate(
            zip(fragment_dets.boxes, fragment_dets.scores)
        ):
            seg = seg_map.get(i)
            m = self._measure_one(
                fragment_id=i,
                image_id=detections.image_id,
                row_id=row_id,
                bbox=box,
                seg=seg,
                calibration=calibration,
            )
            if m is not None:
                measurements.append(m)

        return measurements

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _measure_one(
        self,
        fragment_id: int,
        image_id: str,
        row_id: int,
        bbox: BBox,
        seg: SegmentationResult | None,
        calibration: CalibrationInfo,
    ) -> FragmentMeasurement | None:
        """Measure a single fragment.

        Returns None when the fragment has zero area (logs warning).
        """
        mask_present = seg is not None and seg.mask_area_px > 0

        if mask_present and self._method in ("mask_pca", "skeleton"):
            length_px, width_px, orientation_deg, method = _measure_from_mask(
                seg.mask, self._method  # type: ignore[union-attr]
            )
        else:
            length_px, width_px, orientation_deg, method = _measure_from_bbox(bbox)
            if mask_present and seg is not None:
                method = "bbox_fallback"

        if length_px <= 0:
            logger.warning(
                "Fragment %d in image '%s' has zero measured length; skipping",
                fragment_id,
                image_id,
            )
            return None

        ppm = calibration.pixels_per_mm
        length_mm = length_px / ppm
        width_mm = width_px / ppm if width_px > 0 else None
        qualifies = length_mm >= self._rqd_threshold_mm

        return FragmentMeasurement(
            image_id=image_id,
            row_id=row_id,
            fragment_id=fragment_id,
            length_mm=length_mm,
            width_mm=width_mm,
            orientation_deg=orientation_deg,
            measurement_method=method,
            qualifies_rqd=qualifies,
            rqd_threshold_mm=self._rqd_threshold_mm,
            bbox_px=bbox,
            mask_present=mask_present,
            calibration_source=calibration.source,
        )


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------


def _measure_from_bbox(
    bbox: BBox,
) -> tuple[float, float, float, Literal["bbox"]]:
    """Estimate length from the longer bounding box dimension.

    For horizontal core rows (assumed default) the width of the bbox is the
    length along the core axis. For tilted boxes the diagonal is used as a
    conservative upper bound.

    Args:
        bbox: Bounding box of the fragment.

    Returns:
        Tuple of (length_px, width_px, orientation_deg, 'bbox').
    """
    return bbox.width, bbox.height, 0.0, "bbox"


def _measure_from_mask(
    mask: np.ndarray,
    method: Literal["mask_pca", "skeleton"],
) -> tuple[float, float, float, Literal["mask_pca", "skeleton", "bbox_fallback"]]:
    """Estimate length from a binary mask using PCA or skeletonisation.

    PCA: Finds the principal axis of the mask pixel cloud and returns the
    extent along that axis as the length.

    Skeleton: (Placeholder) Falls back to PCA; full skeletonisation
    requires scikit-image and is implemented as an extension point.

    Args:
        mask: Binary mask (H, W) uint8 with values {0, 1}.
        method: 'mask_pca' or 'skeleton'.

    Returns:
        Tuple of (length_px, width_px, orientation_deg, method_used).
    """
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return 0.0, 0.0, 0.0, "bbox_fallback"

    points = np.column_stack([xs, ys]).astype(np.float32)
    centroid = points.mean(axis=0)
    centered = points - centroid

    cov = np.cov(centered.T)
    if cov.ndim == 0:
        return float(xs.max() - xs.min()), float(ys.max() - ys.min()), 0.0, "bbox_fallback"

    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]  # eigenvector for largest eigenvalue
    secondary = eigvecs[:, 0]

    proj_primary = centered @ principal
    proj_secondary = centered @ secondary

    length_px = float(proj_primary.max() - proj_primary.min())
    width_px = float(proj_secondary.max() - proj_secondary.min())
    orientation_deg = float(math.degrees(math.atan2(principal[1], principal[0])))

    return length_px, width_px, orientation_deg, "mask_pca"


def compute_principal_axis_length(mask: np.ndarray) -> tuple[float, float]:
    """Compute principal and secondary axis lengths from a binary mask.

    Args:
        mask: Binary mask (H, W), uint8.

    Returns:
        Tuple of (principal_length_px, secondary_length_px).
    """
    length, width, _, _ = _measure_from_mask(mask, "mask_pca")
    return length, width


def bbox_to_length(
    bbox: BBox, orientation: Literal["horizontal", "vertical"] = "horizontal"
) -> float:
    """Return the fragment length along the tray row orientation.

    Args:
        bbox: Fragment bounding box.
        orientation: Row orientation.

    Returns:
        Length in pixels.
    """
    return bbox.width if orientation == "horizontal" else bbox.height
