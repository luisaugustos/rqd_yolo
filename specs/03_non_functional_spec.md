# Non-Functional Specification

**Project:** rqd-ai-lab
**Phase:** 3 — Non-Functional Spec
**Version:** 1.0.0
**Date:** 2026-03-16
**Status:** Draft

---

## Overview

This document defines non-functional requirements for the rqd-ai-lab system. Each requirement is assigned a unique identifier in the format `NFR-NNN`. These requirements constrain the system's operational qualities, ensuring it is scientifically rigorous, maintainable, and reproducible.

---

## NFR-001 — Reproducibility

**Title:** Full experiment reproducibility from configuration

**Description:** All training runs and inference experiments must be fully reproducible given the same configuration, random seed, data, and software environment.

**Rationale:** Scientific validity requires that published results can be reproduced by independent researchers. Unreproducible experiments undermine the credibility of benchmarks.

**Constraints:**
- All random seeds must be set from configuration: Python `random`, `numpy`, `torch`, `cuda`.
- CUDA operations must be run in deterministic mode when `deterministic: true` is set in config (may impact speed).
- Dataset shuffling must use a configurable seed.
- Data augmentation must use a configurable seed.
- Model weight initialization must use a configurable seed.

**Verification:**
- Two runs with identical configuration and seed must produce RQD results within ±0.5% of each other.
- Deterministic mode must be verified by a dedicated test that runs two forward passes and checks output equality.

**Traceability:** SC-1, A-10

---

## NFR-002 — Modularity

**Title:** Loosely coupled, independently testable modules

**Description:** The system must be organized into well-defined, independently testable modules with explicit interfaces. No module may directly instantiate another module's internal classes.

**Rationale:** Modularity enables parallel development, independent testing, and model backend substitution without cascading changes.

**Constraints:**
- Each module has a single, well-defined responsibility.
- Modules communicate through documented data contracts (see Phase 7 — Interface Contracts).
- Modules are loadable independently without requiring the full pipeline to be initialized.
- Model backends (YOLOv12, RT-DETRv2, SAM2, etc.) must be interchangeable via a common interface.

**Verification:**
- Each module must have unit tests that exercise it in isolation with mock inputs.
- Swapping a model backend must require only a configuration change, verified by integration test.

**Traceability:** FR-005, FR-007, UG-7

---

## NFR-003 — Maintainability

**Title:** Code clarity and long-term maintainability

**Description:** All source code must be written to a standard that enables future developers to understand, modify, and extend the system without requiring knowledge of the original authors' intent.

**Rationale:** Research codebases frequently outlive the original team. Poor maintainability leads to broken reproducibility over time.

**Constraints:**
- All public functions and classes must have docstrings (Google style).
- Type hints must be applied to all function signatures.
- Cyclomatic complexity of any single function must not exceed 10.
- No magic constants in code; all configurable values come from configuration files.
- No hardcoded file paths anywhere in source or test code.
- Maximum function length: 50 lines (excluding docstrings and blank lines). Functions exceeding this limit should be decomposed.

**Verification:**
- Linting enforced with `ruff` (or equivalent); no lint errors allowed in CI.
- Type checking enforced with `mypy --strict` (or equivalent) on core modules.

**Traceability:** Phase 12 (Code Generation Rules)

---

## NFR-004 — Experiment Traceability

**Title:** Every result traceable to its generating experiment

**Description:** Every evaluation metric, model output, and RQD result must be linked to the experiment run, configuration, and code version that produced it.

**Rationale:** Geotechnical and scientific audits require complete provenance for all computed values. Results that cannot be traced to their source are inadmissible in publication and regulatory contexts.

**Constraints:**
- Every run logs: run ID, UTC timestamp, git commit hash (if available), configuration file hash (SHA-256), model weights hash.
- Experiment logs are append-only and never overwritten.
- Output artifacts are stored in a directory named with the run ID.
- Evaluation reports embed the run ID and configuration hash.

**Verification:**
- Integration test: run the pipeline twice; verify both runs have distinct logged IDs.
- Audit test: given a run ID, all configuration and metrics can be retrieved.

**Traceability:** FR-015, SC-8

---

## NFR-005 — Scientific Auditability

**Title:** All design decisions documented with rationale

**Description:** Every design decision, assumption, and constraint in the system must be documented in the specification files with an explicit rationale. Undocumented behavior is prohibited.

**Rationale:** Research-grade systems must be auditable by third parties. Silent assumptions invalidate published benchmarks.

**Constraints:**
- Ambiguous requirements must be flagged in the spec as `[AMBIGUOUS]` and given an explicit assumption.
- Deviations from standard methodologies must be documented in the relevant spec.
- All evaluation protocols must reference the metric definition (ISRM, COCO, etc.).

**Verification:**
- Every spec document is reviewed for completeness before implementation begins.
- Code-level decisions that are non-obvious must include a comment with the rationale.

**Traceability:** SDD Rules (Rule 6, Rule 7, Rule 8)

---

## NFR-006 — Configuration-Driven Behavior

**Title:** All operational parameters controlled by configuration files

**Description:** All parameters affecting system behavior (model selection, thresholds, paths, hyperparameters, seeds, augmentation policy) must be defined in YAML configuration files and never hardcoded.

**Rationale:** Configuration-driven design makes experiments reproducible, enables parameter sweeps, and prevents silent behavior changes when code is modified.

**Constraints:**
- Configuration files follow a defined schema (validated at startup).
- Missing required configuration keys cause an immediate error with a descriptive message.
- All default values are documented in the configuration schema.
- CLI flags may override individual config values using `key=value` syntax (Hydra-style or equivalent).
- Configuration files are logged as artifacts with every experiment run.

**Verification:**
- Unit test: attempt to start with invalid configuration; verify failure and error message.
- Integration test: changing a configuration value changes system behavior as expected.

**Traceability:** FR-016, NFR-001, Phase 12 (Code Generation Rules)

---

## NFR-007 — Hardware Awareness

**Title:** System adapts to available hardware automatically

**Description:** The system must detect available hardware (CPU, single GPU, multi-GPU) and adapt accordingly without requiring manual configuration changes.

**Rationale:** The system must run for CI and testing on CPU and for production training on GPU, without code changes.

**Constraints:**
- GPU is used automatically if available; CPU is used as fallback.
- Mixed precision (FP16) is enabled by default when GPU is available, with configurable override.
- VRAM usage is reported at model load time.
- A minimum hardware specification is documented: ≥ 8 GB GPU VRAM for full training; CPU-only for inference on single small images.
- Batch size must be configurable to accommodate different VRAM capacities.

**Verification:**
- Smoke test: pipeline runs successfully on CPU with a small test image.
- Hardware detection is tested by mocking `torch.cuda.is_available()`.

**Traceability:** SC-7, A-7

---

## NFR-008 — Deterministic Evaluation Mode

**Title:** A dedicated evaluation mode that eliminates all non-determinism

**Description:** The system must support a `deterministic_eval` mode that disables all stochastic operations during inference and evaluation, ensuring identical results across runs.

**Rationale:** Non-deterministic evaluation results make it impossible to distinguish model improvement from random variation.

**Constraints:**
- In `deterministic_eval` mode: CUDA deterministic mode is enabled, dropout is disabled, test-time augmentation is disabled, and shuffling is disabled.
- `deterministic_eval` is the default mode during evaluation; stochastic options are opt-in.
- Enabling deterministic mode logs a warning if CUDA deterministic mode degrades performance.

**Verification:**
- Test: run evaluation twice in `deterministic_eval` mode; verify all metrics are bit-for-bit identical.

**Traceability:** NFR-001, FR-014

---

## NFR-009 — Logging

**Title:** Structured, leveled logging throughout the system

**Description:** The system must produce structured log output at configurable verbosity levels for all major operations, errors, and warnings.

**Rationale:** Logs are essential for debugging, monitoring, and post-hoc analysis of pipeline runs. Unstructured or missing logs make issue diagnosis impossible.

**Constraints:**
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
- Default log level: INFO.
- Log format includes: timestamp (UTC), log level, module name, message.
- Structured logging (JSON format) is available via configuration for machine parsing.
- All errors are logged before re-raising.
- Calibration fallback to manual mode generates a WARNING log entry.
- Model load events are logged at INFO level with model name and weight file path.

**Verification:**
- Unit test: verify log entries are produced for key events (model load, calibration, RQD result).
- Integration test: run pipeline in DEBUG mode and verify no unexpected silent failures.

**Traceability:** FR-001, FR-004, FR-015

---

## NFR-010 — Test Coverage

**Title:** Minimum test coverage targets for core logic

**Description:** The system must maintain a minimum level of automated test coverage for core functional modules.

**Rationale:** Test coverage provides a quantitative floor for code quality and prevents regressions.

**Constraints:**
- Overall test coverage target: ≥ 80% (line coverage).
- Coverage target for critical modules (measurement engine, RQD engine, calibration): ≥ 90%.
- Tests are organized to mirror the source module structure.
- Coverage is measured and enforced in CI.
- Tests must not require GPU hardware (mocking or CPU fallback required).

**Verification:**
- CI pipeline measures coverage and fails if targets are not met.
- Coverage report is generated as an artifact of every CI run.

**Traceability:** SC-9, Phase 13 (Test Generation Rules)

---

## NFR-011 — Documentation Quality

**Title:** Project and API documentation must be complete and accurate

**Description:** The project must include documentation at three levels: user-facing README, API documentation (docstrings), and specification documents.

**Rationale:** Incomplete documentation prevents reuse and limits the system's value as a research artifact.

**Constraints:**
- README covers: project overview, installation, quick-start example, citation.
- Every public function and class has a docstring with: description, parameters, return values, raises.
- Specification documents (this suite) are the authoritative reference for design decisions.
- Documentation is kept in sync with code; stale documentation is treated as a defect.

**Verification:**
- Automated docstring coverage check (e.g., `interrogate`) enforced in CI with ≥ 80% target.
- README quick-start is tested by running it in a clean environment as part of integration tests.

**Traceability:** Phase 12 (Code Generation Rules), SDD Rules

---

## NFR-012 — Extensibility for Future Models

**Title:** New model backends can be added without modifying existing pipeline logic

**Description:** The detection, segmentation, and foundation model modules must be designed as registries or strategy patterns, allowing new models to be added by implementing a defined interface and registering the new backend.

**Rationale:** Computer vision is a rapidly evolving field. New model families will emerge after initial development. The system must accommodate them with minimal friction.

**Constraints:**
- A new detection model requires implementing a single `DetectorBackend` interface and adding a configuration entry.
- A new segmentation model requires implementing a single `SegmentorBackend` interface.
- No existing tests should require modification when a new backend is added.
- Backend registration uses a configurable string name, not hardcoded imports.

**Verification:**
- Integration test: implement a trivial mock backend; verify it can be selected via configuration and produces correct output format.

**Traceability:** UG-7, NFR-002, FR-005, FR-007

---

## NFR-013 — Inference Latency Measurement

**Title:** Record and report inference latency per pipeline stage

**Description:** The system must measure and report the wall-clock time for each major pipeline stage (preprocessing, detection, segmentation, measurement, RQD computation) for every inference run.

**Rationale:** Performance characterization is essential for comparing model backends and for assessing suitability for operational deployment contexts.

**Constraints:**
- Timing must be collected using high-resolution timers (`time.perf_counter`).
- GPU warm-up runs are excluded from reported latency (at least 3 warm-up iterations before measurement).
- Latency is reported per image and as mean ± std over a batch.
- Timing results are logged as experiment metrics.

**Verification:**
- Integration test: verify timing data is present in experiment log after an evaluation run.

**Traceability:** SC-3, FR-015

---

## NFR-014 — Memory Usage Reporting

**Title:** Report peak GPU and CPU memory usage during inference

**Description:** The system must measure and report peak memory consumption for each major model during inference.

**Rationale:** Memory usage determines the minimum hardware specification and enables fair comparison of model backends.

**Constraints:**
- GPU peak memory is measured using `torch.cuda.max_memory_allocated()` (reset before each measurement).
- CPU peak memory is measured using `tracemalloc` or `psutil`.
- Memory stats are reported per model run and logged as experiment metrics.

**Verification:**
- Integration test: verify memory metrics are present in experiment log.

**Traceability:** SC-7, NFR-007

---

## NFR-015 — Dataset Pipeline Reproducibility

**Title:** Dataset loading and splitting is deterministic and versioned

**Description:** Dataset splits (train/val/test) must be fixed and reproducible. The same split must be used for all experiments to ensure fair comparison.

**Rationale:** Different data splits between experiments invalidate cross-model comparisons.

**Constraints:**
- Dataset splits are defined by a versioned split file (CSV or YAML) committed to the repository.
- Split creation uses a configurable random seed.
- The split file hash is logged with every experiment.
- No in-place data modification; raw data is immutable.

**Verification:**
- Test: verify that loading the dataset twice with the same split file returns identical train/val/test indices.

**Traceability:** FR-001, NFR-001, FR-015

---

## NFR-016 — Experiment Seed Control

**Title:** All stochastic elements are controlled by a global seed hierarchy

**Description:** The system must implement a seed hierarchy where a global seed fans out to per-module seeds, ensuring both global reproducibility and independent module testability.

**Rationale:** A single global seed provides reproducibility; per-module seeds enable isolated module testing without cross-module interference.

**Constraints:**
- Global seed is set in the root configuration and propagated to: Python random, NumPy, PyTorch, CUDA, and dataset shuffling.
- Per-module seed override is supported for testing.
- Seed values are logged at experiment start.

**Verification:**
- Test: set seed; run two identical experiments; verify all output files are identical.

**Traceability:** NFR-001, NFR-008
