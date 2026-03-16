"""RQDEngine — compute Rock Quality Designation per row and per image (FR-011, FR-012).

Implements the ISRM RQD formula:
    RQD = (sum of intact pieces >= threshold / total core run length) × 100
"""

from __future__ import annotations

import logging

from src.utils.contracts import (
    CalibrationInfo,
    FragmentMeasurement,
    RQDResult,
    TrayRow,
)

logger = logging.getLogger(__name__)

_DEFAULT_RQD_THRESHOLD_MM = 100.0


class RQDEngine:
    """Compute per-row and aggregate RQD values (FR-011, FR-012).

    Args:
        config: Optional config dict. Relevant key:
            - rqd_threshold_mm (float): Default 100.0.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._threshold_mm: float = float(
            cfg.get("rqd_threshold_mm", _DEFAULT_RQD_THRESHOLD_MM)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_row_rqd(
        self,
        measurements: list[FragmentMeasurement],
        row: TrayRow,
        calibration: CalibrationInfo,
    ) -> RQDResult:
        """Compute RQD for a single tray row.

        The run length denominator is taken from row.row_length_mm when
        available; otherwise derived from row.row_length_px and calibration.

        Args:
            measurements: Fragment measurements for this row.
            row: TrayRow providing run length and depth metadata.
            calibration: Pixel-to-mm calibration.

        Returns:
            RQDResult with scope='row'.

        Raises:
            ValueError: When total_run_length_mm is zero or negative.
        """
        total_mm = _row_length_mm(row, calibration)
        if total_mm <= 0:
            raise ValueError(
                f"RQDComputationError: total_run_length_mm={total_mm} must be positive "
                f"for row {row.row_id} in image '{row.image_id}'"
            )

        row_measurements = [m for m in measurements if m.row_id == row.row_id]
        qualifying = [m for m in row_measurements if m.qualifies_rqd]
        qualifying_mm = sum(m.length_mm for m in qualifying)
        rqd_pct = _safe_rqd(qualifying_mm, total_mm)

        if not row_measurements:
            logger.warning(
                "No fragment measurements for row %d in image '%s'; RQD=0",
                row.row_id,
                row.image_id,
            )

        return RQDResult(
            image_id=row.image_id,
            scope="row",
            row_id=row.row_id,
            depth_from_m=None,
            depth_to_m=None,
            total_run_length_mm=total_mm,
            qualifying_length_mm=min(qualifying_mm, total_mm),
            rqd_pct=rqd_pct,
            num_fragments_total=len(row_measurements),
            num_fragments_qualifying=len(qualifying),
            rqd_threshold_mm=self._threshold_mm,
            fragment_measurements=row_measurements,
            calibration_source=calibration.source,
        )

    def compute_image_rqd(
        self,
        row_results: list[RQDResult],
    ) -> RQDResult:
        """Compute aggregate RQD for a full image from per-row results.

        Aggregation sums raw lengths, NOT averages per-row percentages
        (AC-012-2).

        Args:
            row_results: List of per-row RQDResult objects.

        Returns:
            RQDResult with scope='image'.

        Raises:
            ValueError: When there are no row results or total length is zero.
        """
        if not row_results:
            raise ValueError("RQDComputationError: no row results provided for image RQD")

        image_id = row_results[0].image_id
        total_mm = sum(r.total_run_length_mm for r in row_results)
        if total_mm <= 0:
            raise ValueError(
                f"RQDComputationError: aggregate total_run_length_mm={total_mm} must be positive"
            )

        qualifying_mm = sum(r.qualifying_length_mm for r in row_results)
        qualifying_mm = min(qualifying_mm, total_mm)
        rqd_pct = _safe_rqd(qualifying_mm, total_mm)

        all_measurements = [m for r in row_results for m in r.fragment_measurements]
        num_qualifying = sum(r.num_fragments_qualifying for r in row_results)
        calibration_source = row_results[0].calibration_source

        return RQDResult(
            image_id=image_id,
            scope="image",
            row_id=None,
            depth_from_m=None,
            depth_to_m=None,
            total_run_length_mm=total_mm,
            qualifying_length_mm=qualifying_mm,
            rqd_pct=rqd_pct,
            num_fragments_total=len(all_measurements),
            num_fragments_qualifying=num_qualifying,
            rqd_threshold_mm=self._threshold_mm,
            fragment_measurements=all_measurements,
            calibration_source=calibration_source,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _row_length_mm(row: TrayRow, calibration: CalibrationInfo) -> float:
    """Return the run length in mm for a tray row.

    Prefers row.row_length_mm; falls back to pixel conversion.
    """
    if row.row_length_mm is not None and row.row_length_mm > 0:
        return row.row_length_mm
    if calibration.pixels_per_mm > 0:
        return row.row_length_px / calibration.pixels_per_mm
    raise ValueError("Cannot determine row length: neither row_length_mm nor calibration is set")


def _safe_rqd(qualifying_mm: float, total_mm: float) -> float:
    """Compute RQD percentage clamped to [0, 100] (AC-011-2)."""
    if total_mm <= 0:
        return 0.0
    rqd = (qualifying_mm / total_mm) * 100.0
    return max(0.0, min(100.0, rqd))
