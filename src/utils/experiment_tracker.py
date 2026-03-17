"""Experiment tracking wrapper (FR-015, NFR-004).

Provides a unified interface over MLflow for logging parameters, metrics,
and artifacts. Falls back to structured local file logging when MLflow is
unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Logs experiment parameters, metrics, and artifacts (FR-015).

    Wraps MLflow when available; writes a JSON sidecar file otherwise.

    Args:
        tracking_uri: MLflow tracking URI (e.g. 'results/mlruns').
        artifact_uri: Root directory for artifact storage.
        experiment_name: MLflow experiment name.
        use_mlflow: When False, skip MLflow even if installed.
    """

    def __init__(
        self,
        tracking_uri: str = "results/mlruns",
        artifact_uri: str = "results/artifacts",
        experiment_name: str = "rqd_experiment",
        use_mlflow: bool = True,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._artifact_uri = Path(artifact_uri)
        self._experiment_name = experiment_name
        self._run_id: str | None = None
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, list[float]] = {}
        self._mlflow: Any = None
        self._local_log: dict[str, Any] = {}

        if use_mlflow:
            self._mlflow = self._try_init_mlflow()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self, run_name: str | None = None) -> str:
        """Start a new experiment run and return its unique run ID.

        Args:
            run_name: Optional human-readable run name.

        Returns:
            Unique run ID string.
        """
        self._run_id = str(uuid.uuid4())[:8]
        self._params = {}
        self._metrics = {}
        self._local_log = {"run_id": self._run_id, "run_name": run_name, "start_time": time.time()}

        if self._mlflow is not None:
            try:
                self._mlflow.start_run(run_name=run_name or self._run_id)
                self._run_id = self._mlflow.active_run().info.run_id
            except Exception as exc:
                logger.warning("MLflow start_run failed: %s — using local logging", exc)
                self._mlflow = None

        logger.info("Experiment run started: %s (name=%s)", self._run_id, run_name)
        return self._run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of parameters.

        Args:
            params: Key-value parameter pairs.
        """
        self._params.update(params)
        if self._mlflow is not None:
            try:
                self._mlflow.log_params(params)
                return
            except Exception as exc:
                logger.warning("MLflow log_params failed: %s", exc)
        self._local_log.setdefault("params", {}).update(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single scalar metric.

        Args:
            key: Metric name.
            value: Scalar value.
            step: Optional training step / epoch.
        """
        self._metrics.setdefault(key, []).append(value)
        if self._mlflow is not None:
            try:
                self._mlflow.log_metric(key, value, step=step)
                return
            except Exception as exc:
                logger.warning("MLflow log_metric failed: %s", exc)
        self._local_log.setdefault("metrics", {}).setdefault(key, []).append(
            {"value": value, "step": step}
        )

    def log_artifact(self, path: Path) -> None:
        """Log a file as a run artifact.

        Args:
            path: Path to the file to log.
        """
        if not path.exists():
            logger.warning("Artifact not found, skipping: %s", path)
            return
        if self._mlflow is not None:
            try:
                self._mlflow.log_artifact(str(path))
                return
            except Exception as exc:
                logger.warning("MLflow log_artifact failed: %s", exc)
        self._local_log.setdefault("artifacts", []).append(str(path))

    def log_config_hash(self, config_path: Path) -> str:
        """Compute and log SHA-256 hash of a config file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        content = config_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        self.log_params({"config_hash": digest, "config_path": str(config_path)})
        return digest

    def end_run(self) -> None:
        """Finalise and close the current experiment run."""
        self._local_log["end_time"] = time.time()
        if self._run_id:
            out = self._artifact_uri / self._run_id / "run_log.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self._local_log, indent=2))
        if self._mlflow is not None:
            try:
                self._mlflow.end_run()
            except Exception as exc:
                logger.warning("MLflow end_run failed: %s", exc)
        logger.info("Experiment run ended: %s", self._run_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_init_mlflow(self) -> Any:
        """Attempt to import and configure MLflow; return None on failure."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            logger.info("MLflow tracking URI: %s", self._tracking_uri)
            return mlflow
        except ImportError:
            logger.warning("MLflow not installed; using local JSON logging")
            return None
        except Exception as exc:
            logger.warning("MLflow initialisation failed: %s — using local logging", exc)
            return None
