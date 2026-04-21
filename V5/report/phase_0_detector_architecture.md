# Phase 0 Detector Architecture Report

## 1) Repo Alignment
- Existing detector baseline already lives in `src/detectors` with M10 entrypoints at:
  - `src/pipelines/m10_train_detector.py`
  - `src/pipelines/m10_eval_detector.py`
  - `src/pipelines/m10_infer_detector.py`
- M9.2 split + scenario contracts are available and usable now (`*_scenarios_v2.csv`, `split_manifest_v2.json`, `split_balance_report_v2.json`).
- Phase 0 decision: evolve current detector architecture in place, avoid parallel subsystem.

## 2) Proposed Detector File Tree (Phase 1 Target)
- Keep current root:
  - `src/detectors`
- Add domain contracts and DTO modules:
  - `src/detectors/domain/interfaces/*`
  - `src/detectors/domain/models/*`
  - `src/detectors/domain/enums/*`
- Add shared preprocessing/features/utils:
  - `src/detectors/shared/preprocessing/*`
  - `src/detectors/shared/features/*`
  - `src/detectors/shared/utils/*`
- Add branch service layers:
  - `src/detectors/cyber/{heuristics,features,ml,services}/*`
  - `src/detectors/physical/{features,ml,services}/*`
- Add fusion/postprocessing specializations:
  - `src/detectors/fusion/{strategies,services}/*`
  - `src/detectors/postprocessing/*`
- Add training/inference modules:
  - `src/detectors/training/{datasets,trainers,evaluators,reports}/*`
  - `src/detectors/inference/*`

Compatibility policy:
- Keep existing modules (`models.py`, `configs.py`, `runtime.py`) as wrappers during migration.
- Keep public m10 pipeline file names unchanged.

## 3) Contract Summary

### Core interfaces
- `Detector`: lifecycle (`fit`, `predict`, `save`, `load`), top-level hybrid orchestration.
- `BranchDetector`: cyber/physical branch common API (`fit`, `score`).
- `FeatureExtractor`: deterministic feature generation for branch inputs.
- `Preprocessor`: PMU alignment, NaN policy, normalization, window materialization.
- `FusionStrategy`: combine branch outputs into `{p_abnormal, p_cyber, p_physical}`.
- `Postprocessor`: hysteresis, state machine, chunk extraction.

### Core DTOs
- `DetectionInput`: windows, feature names, timestamps, metadata.
- `DetectionOutput`: branch probabilities, fused probability, frame/stable predictions, chunks.
- `BranchScore`: probability vector + backend details.
- `WindowSample`: per-window training/eval sample with labels and metadata.
- `DetectorConfig`: typed config across preprocessing, branches, fusion, postprocessing, artifact paths.
- `DetectorMetrics`: binary metrics + calibration + delay/FP governance metrics.

Serialization policy:
- Model binaries in artifact path; companion JSON for config/schema/calibration.
- Frame/chunk outputs as CSV/Parquet; run summaries as JSON/Markdown.

## 4) Data Flow Summary
1. Read split CSV (or manifest-derived list) and scenario directories.
2. Load PMU CSVs by scenario (`scenario/pmu/*.csv`), merge on `TIMESTAMP`.
3. Reconcile labels from PMU `Event` and `labels/event_frame_labels.csv`.
4. Build frame-level binary label (`EVENT != 0`) and family tags:
   - normal: `{0}`
   - physical-heavy: `{1,2,3,4}`
   - cyber-heavy: `{5,7}`
   - concurrent/ambiguous: `{6,8}`
5. Preprocess (numeric coercion, NaN strategy, DATA_PRESENT handling, scaling, angle-safe transforms, optional positive-sequence/estimator features).
6. Build sliding windows with metadata.
7. Window label rule:
   - `window_positive_ratio = mean(frame_binary_abnormal)`
   - `window_binary = 1 if ratio >= threshold else 0`
8. Score cyber and physical branches.
9. Fuse scores into `p_abnormal` plus branch explainability.
10. Postprocess to stable decisions and chunk proposals.
11. Export frame-level and chunk-level outputs for downstream classifier/localizer.

## 5) Pipeline Plan

### `m10_train_detector.py`
- Build train/val datasets.
- Train cyber + physical branches.
- Tune fusion/postprocessing thresholds on validation.
- Save model/config/calibration artifacts.
- Emit training report and threshold sweep artifacts.

### `m10_eval_detector.py`
- Load trained detector and split.
- Emit full metrics:
  - confusion
  - ROC/PR
  - calibration (Brier, ECE)
  - FP/min
  - detection delay
  - per-scenario metrics
  - family-group metrics (cyber-heavy, physical-heavy, concurrent/ambiguous)

### `m10_infer_detector.py`
- Load trained detector.
- Run on one PMU scenario.
- Emit stable binary alerts + chunk proposals for downstream tasks.

## 6) Test Plan Summary

### Unit
- DTO/interface contracts
- label mapping and family mapping
- window builder
- NaN/DATA_PRESENT policy
- angle transforms
- cyber rules
- fusion
- hysteresis/state machine/chunker
- evaluator metrics and report schema

### Integration
- train/eval/infer smoke tests
- one-scenario end-to-end
- small split end-to-end

### E2E
- Full run: scenario generation -> splits -> train -> eval -> infer -> report/chunk contract checks.

## 7) Acceptance Criteria (Phase 3 Ready Targets)
- F1 abnormal >= 0.78
- Precision abnormal >= 0.75
- Recall abnormal >= 0.82
- Balanced accuracy >= 0.80
- False positives per minute <= 2.0
- Detection delay p95 <= 2.0 s
- ROC AUC >= 0.88
- PR AUC >= 0.85
- Brier <= 0.18
- ECE <= 0.08
- Family minimum recall:
  - physical-heavy >= 0.80
  - cyber-heavy >= 0.75
  - concurrent/ambiguous >= 0.70

## 8) Planning Docs Created
- `AGENTS.md`
- `PLANS.md`
- `report/phase_0_detector_architecture.json`
- `report/phase_0_detector_architecture.md`

## 9) Risks and Next Phase
Main risks:
- sparse val/test coverage for some cyber/concurrent classes,
- optional LightGBM dependency variability,
- schema drift between scenario labels and detector ingestion.

Next phase:
- `phase_1_foundations` (contracts + scaffolding + contract-focused tests).

## Verdict
- `phase_0_complete: true`
- Ready to start Phase 1 without implementing full detector retraining logic yet.
