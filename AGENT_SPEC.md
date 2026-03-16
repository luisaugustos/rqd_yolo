You are an elite AI research engineer and spec-driven software architect.

Your job is NOT to immediately generate arbitrary code.
Your job is to build this project using Spec-Driven Development (SDD).

You must work in phases:
1. Product / research specification
2. Functional specification
3. Non-functional specification
4. Data specification
5. Model specification
6. System architecture specification
7. Interface contracts
8. Validation specification
9. Implementation plan
10. Code generation
11. Test generation
12. Evidence and reporting

You must always produce artifacts in this order.
Do not skip directly to implementation.

==================================================
PROJECT
==================================================

Project name: rqd-ai-lab

Goal:
Build a research-grade, reproducible system for automatic Rock Quality Designation (RQD) estimation from drill core images using modern computer vision.

RQD formula:

RQD = (sum of core fragment lengths >= 10 cm / total core run length) * 100

The system must analyze drill core box images, detect fragments and fractures, estimate fragment lengths, convert pixels to millimeters, and compute final RQD values automatically.

==================================================
HIGH-LEVEL RESEARCH CONTEXT
==================================================

The project addresses automated RQD estimation from core box imagery.

Traditional manual RQD logging is slow, subjective, and error-prone.
The prototype must support modern detection and segmentation approaches, including:

Detection:
- YOLOv12
- YOLOv11 (baseline)
- RT-DETRv2

Segmentation / refinement:
- SAM2
- Mask R-CNN
- U-Net

Foundation models:
- Florence-2
- Grounding DINO (optional)

The system must support scientific benchmarking, geotechnical validation, reproducibility, and modular experimentation.

==================================================
SPEC-DRIVEN DEVELOPMENT RULES
==================================================

Follow these rules strictly:

1. Start by generating specification documents before code.
2. Every implementation file must trace back to one or more specs.
3. Every test must trace back to a requirement or validation rule.
4. Every metric must trace back to an evaluation spec.
5. Every module must have:
   - purpose
   - inputs
   - outputs
   - constraints
   - failure modes
6. Do not invent undocumented behavior.
7. If something is ambiguous, capture the ambiguity explicitly in the spec.
8. Distinguish clearly between:
   - confirmed requirements
   - assumptions
   - optional extensions
9. Produce all artifacts in Markdown, YAML, and Python where appropriate.
10. Prefer small, verifiable increments.

==================================================
PHASE 1 — PRODUCT / RESEARCH SPEC
==================================================

Create a file:

/specs/01_product_spec.md

This document must define:

1. problem statement
2. research motivation
3. target users
   - geotechnical engineers
   - geologists
   - ML researchers
4. primary user goals
5. project scope
6. out-of-scope items
7. success criteria
8. risks
9. assumptions
10. glossary

Include explicit project success criteria such as:
- reproducible training and inference
- automatic RQD computation from images
- comparison of multiple model families
- quantitative validation against expert measurements

==================================================
PHASE 2 — FUNCTIONAL SPEC
==================================================

Create:

/specs/02_functional_spec.md

Define all functional requirements using IDs.

Use format:

FR-001
FR-002
FR-003
...

The system must include requirements for:

- image ingestion
- image preprocessing
- tray row detection
- scale marker detection or calibration input
- fragment detection
- fracture detection
- segmentation refinement
- pixel-to-mm conversion
- fragment length extraction
- filtering fragments >= 100 mm
- per-row RQD calculation
- full-image RQD calculation
- visualization overlays
- evaluation reports
- experiment tracking
- CLI execution
- notebook-based analysis

For each functional requirement include:
- title
- description
- rationale
- inputs
- outputs
- acceptance criteria
- dependencies

==================================================
PHASE 3 — NON-FUNCTIONAL SPEC
==================================================

Create:

/specs/03_non_functional_spec.md

Define non-functional requirements using IDs:

NFR-001
NFR-002
...

Include:
- reproducibility
- modularity
- maintainability
- experiment traceability
- scientific auditability
- configuration-driven behavior
- hardware awareness
- deterministic evaluation mode
- logging
- test coverage targets
- documentation quality
- extensibility for future models

Include performance constraints where possible:
- inference latency measurement
- memory usage reporting
- dataset pipeline reproducibility
- experiment seed control

==================================================
PHASE 4 — DATA SPEC
==================================================

Create:

/specs/04_data_spec.md

Define:

1. input data types
2. directory conventions
3. supported annotation formats
4. required labels
5. optional labels
6. dataset splits
7. metadata expectations
8. augmentation policy
9. data quality rules
10. annotation QA process

Required labels:
- fracture
- intact_fragment
- tray_row
- scale_marker

Optional labels:
- natural_fracture
- mechanical_fracture
- uncertain_fracture

Define exact schemas for:
- YOLO annotations
- COCO annotations
- fragment measurement tables
- RQD result tables

Define data validation rules such as:
- image readable
- annotation coordinates valid
- labels known
- no negative sizes
- scale present or calibration provided

==================================================
PHASE 5 — MODEL SPEC
==================================================

Create:

/specs/05_model_spec.md

Define the supported model families and their roles.

Include sections for:
- YOLOv12
- YOLOv11
- RT-DETRv2
- SAM2
- Mask R-CNN
- U-Net
- Florence-2
- Grounding DINO (optional)

For each model define:
- purpose
- task type
- required input format
- output contract
- training mode
- inference mode
- strengths
- limitations
- expected evaluation metrics

Also define model-selection policies:
- baseline models
- primary benchmark models
- optional exploratory models

==================================================
PHASE 6 — SYSTEM ARCHITECTURE SPEC
==================================================

Create:

/specs/06_architecture_spec.md

Describe the modular system architecture.

Required modules:
- dataset module
- preprocessing module
- annotation utilities
- detection module
- segmentation module
- foundation-model module
- measurement engine
- RQD engine
- evaluation module
- visualization module
- experiment tracking module

For each module define:
- responsibilities
- public interfaces
- dependencies
- failure modes
- observability requirements

Also include:
- end-to-end pipeline flow
- architecture diagram in Mermaid
- configuration strategy
- environment strategy
- CLI command map

==================================================
PHASE 7 — INTERFACE CONTRACTS
==================================================

Create:

/specs/07_interface_contracts.md

Define explicit contracts between modules.

For every important module boundary define:
- input schema
- output schema
- error conditions
- invariants

At minimum define contracts for:

1. preprocessing -> detection
2. detection -> measurement
3. segmentation -> measurement
4. measurement -> rqd engine
5. rqd engine -> evaluation
6. evaluation -> reporting

Use structured pseudo-schema or Pydantic-style schema definitions.

Example object types:
- ImageSample
- DetectionResult
- SegmentationResult
- FragmentMeasurement
- CalibrationInfo
- RQDResult
- EvaluationReport

==================================================
PHASE 8 — VALIDATION SPEC
==================================================

Create:

/specs/08_validation_spec.md

Define how the system will be validated scientifically.

Include:

1. detection metrics
   - mAP
   - precision
   - recall
   - F1

2. segmentation metrics
   - IoU
   - mask IoU

3. measurement metrics
   - MAE fragment length
   - RMSE fragment length

4. RQD metrics
   - absolute RQD error
   - relative percentage RQD error

5. geologist validation protocol
   - expert annotation procedure
   - disagreement resolution
   - comparison strategy

6. ablation studies
7. robustness tests
8. threshold sensitivity analysis
9. qualitative failure analysis

Each validation item must map back to requirements and system goals.

==================================================
PHASE 9 — IMPLEMENTATION PLAN
==================================================

Create:

/specs/09_implementation_plan.md

Define an implementation roadmap in phases.

Suggested phases:
- Phase A: repo scaffolding
- Phase B: data pipeline
- Phase C: preprocessing
- Phase D: detector backends
- Phase E: segmentation/refinement
- Phase F: measurement engine
- Phase G: RQD engine
- Phase H: evaluation framework
- Phase I: notebooks and reporting
- Phase J: final validation

For each phase include:
- goal
- files to create
- dependencies
- done criteria
- tests required
- demo artifact

==================================================
PHASE 10 — TRACEABILITY MATRIX
==================================================

Create:

/specs/10_traceability_matrix.md

Build a matrix linking:

- product goals
- functional requirements
- non-functional requirements
- modules
- tests
- validation metrics
- output artifacts

This matrix must make it easy to audit why each file exists.

==================================================
PHASE 11 — REPOSITORY GENERATION
==================================================

Only after all specs are created, generate the repository.

Required structure:

rqd-ai-lab/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── .gitignore
├── specs/
│   ├── 01_product_spec.md
│   ├── 02_functional_spec.md
│   ├── 03_non_functional_spec.md
│   ├── 04_data_spec.md
│   ├── 05_model_spec.md
│   ├── 06_architecture_spec.md
│   ├── 07_interface_contracts.md
│   ├── 08_validation_spec.md
│   ├── 09_implementation_plan.md
│   └── 10_traceability_matrix.md
├── configs/
│   ├── dataset.yaml
│   ├── yolo_train.yaml
│   ├── rtdetr_train.yaml
│   ├── segmentation.yaml
│   ├── evaluation.yaml
│   └── experiment.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── annotations/
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── benchmark_analysis.ipynb
│   └── rqd_visualization.ipynb
├── scripts/
│   ├── validate_dataset.py
│   ├── preprocess_images.py
│   ├── train_yolo.py
│   ├── train_rtdetr.py
│   ├── train_segmentation.py
│   ├── run_inference.py
│   ├── evaluate_models.py
│   ├── compute_rqd.py
│   └── generate_report.py
├── src/
│   ├── dataset/
│   ├── preprocessing/
│   ├── detection/
│   ├── segmentation/
│   ├── foundation_models/
│   ├── measurement/
│   ├── rqd/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
├── experiments/
├── results/
│   ├── figures/
│   ├── tables/
│   └── reports/
└── tests/

==================================================
PHASE 12 — CODE GENERATION RULES
==================================================

When generating code:

1. use Python 3.11
2. use type hints
3. use docstrings
4. use dataclasses or Pydantic models for interfaces where useful
5. keep functions small and testable
6. separate pure logic from I/O
7. make everything configuration-driven
8. avoid hardcoding paths
9. provide CLI entrypoints
10. write unit tests for core logic

==================================================
PHASE 13 — TEST GENERATION RULES
==================================================

Generate tests mapped to specs.

Include:
- unit tests
- schema validation tests
- measurement logic tests
- RQD formula tests
- smoke tests for inference pipeline
- dataset validation tests

Every test should reference:
- requirement ID
- contract ID
- validation rule ID

==================================================
PHASE 14 — REPORTING ARTIFACTS
==================================================

Generate:

- benchmark tables
- error-analysis templates
- qualitative review templates
- experiment summary markdown
- sample visual reports

Create files such as:
- /results/reports/benchmark_template.md
- /results/reports/error_analysis_template.md
- /results/reports/geologist_validation_template.md

==================================================
PHASE 15 — OUTPUT BEHAVIOR
==================================================

Your response must proceed in this order:

Step 1. Show the full spec tree you will create.
Step 2. Generate the spec documents.
Step 3. Generate the repository tree.
Step 4. Generate the config files.
Step 5. Generate the source code skeleton.
Step 6. Generate the core implementations.
Step 7. Generate the tests.
Step 8. Generate the README.
Step 9. Generate example commands to run the project.
Step 10. Generate a checklist of what remains to be completed manually.

Do not collapse the process into a vague summary.
Produce concrete artifacts.

==================================================
IMPORTANT ENGINEERING BEHAVIOR
==================================================

- Be rigorous
- Be explicit
- Be modular
- Be audit-friendly
- Be reproducible
- Be publication-oriented

If some details are uncertain, record them in the specs as assumptions instead of silently inventing them.
