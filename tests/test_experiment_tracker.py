"""Tests for ExperimentTracker (FR-015, NFR-004)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.experiment_tracker import ExperimentTracker


class TestExperimentTracker:
    """Tests for local (no MLflow) experiment tracking."""

    def test_start_run_returns_run_id(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(artifact_uri=str(tmp_path), use_mlflow=False)
        run_id = tracker.start_run("test_run")
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_log_params_stored(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(artifact_uri=str(tmp_path), use_mlflow=False)
        tracker.start_run()
        tracker.log_params({"lr": 0.001, "batch": 16})
        assert tracker._params["lr"] == 0.001
        assert tracker._params["batch"] == 16

    def test_log_metric_stored(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(artifact_uri=str(tmp_path), use_mlflow=False)
        tracker.start_run()
        tracker.log_metric("map_50", 0.82)
        assert 0.82 in tracker._metrics["map_50"]

    def test_end_run_writes_json_log(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(artifact_uri=str(tmp_path), use_mlflow=False)
        run_id = tracker.start_run("write_test")
        tracker.log_params({"x": 1})
        tracker.log_metric("loss", 0.5)
        tracker.end_run()
        log_file = tmp_path / run_id / "run_log.json"
        assert log_file.exists()

    def test_log_config_hash_produces_sha256(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("seed: 42\n")
        tracker = ExperimentTracker(artifact_uri=str(tmp_path), use_mlflow=False)
        tracker.start_run()
        digest = tracker.log_config_hash(cfg)
        assert len(digest) == 64  # SHA-256 hex string
        assert tracker._params["config_hash"] == digest

    def test_log_artifact_missing_file_does_not_raise(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(artifact_uri=str(tmp_path), use_mlflow=False)
        tracker.start_run()
        # Should log a warning but not raise
        tracker.log_artifact(tmp_path / "nonexistent_file.pt")

    def test_multiple_metrics_accumulated(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(artifact_uri=str(tmp_path), use_mlflow=False)
        tracker.start_run()
        for i in range(5):
            tracker.log_metric("loss", float(i))
        assert len(tracker._metrics["loss"]) == 5
