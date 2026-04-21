# Phase 2: Branches And Fusion

## What Works End-to-End
Phase 2 now provides a working hybrid detector stack on top of Phase 1 foundations:
- cyber branch (heuristics + tabular ML)
- physical branch (heuristics + temporal TCN-style model)
- rule-gated fusion
- smoothing + explicit event state machine + chunk extraction
- train/eval/infer pipelines under `src/detectors/pipelines/`

The system can now:
1. train a hybrid detector,
2. evaluate on scenario split CSVs,
3. run inference on a scenario directory,
4. emit stable binary predictions and candidate chunks.

## Implemented Branch Algorithms

### Cyber branch
- Rules:
  - missing/data-present anomalies,
  - stuck-value signatures,
  - spike/outlier signatures,
  - timestamp irregularity checks.
- Features:
  - rule evidence scores + temporal tabular summaries.
- ML:
  - LightGBM preferred when available,
  - deterministic numpy logistic fallback otherwise.
- Output:
  - `BranchScore` with probabilities, backend, evidence, and feature importance.

### Physical branch
- Rules:
  - voltage sag,
  - current surge,
  - frequency/ROCOF deviation,
  - estimator innovation signal.
- Features:
  - temporal windows + auxiliary summary features.
- Temporal model:
  - `TCNPhysicalDetector` (TCN-style dilated temporal features with deterministic training/inference).
- Fallback:
  - auxiliary logistic baseline (`XGBPhysicalDetector` file path kept for architecture consistency).

## Fusion And Postprocessing
- Fusion:
  - `RuleGatedFusion` with transparent gating/weight boosts for strong branch evidence.
- Postprocessing:
  - exponential smoother,
  - explicit state machine (`NORMAL`, `ABNORMAL_ACTIVE`, `RECOVERY`),
  - chunk builder with configurable min length and padding.

## Metrics Available In Phase 2
Evaluation now exports at minimum:
- confusion matrix counts,
- F1/precision/recall (abnormal class),
- balanced accuracy,
- false positives per minute,
- detection delay,
- ROC AUC,
- PR AUC,
- per-scenario metrics,
- threshold sweep output from training pipeline.

## Tests
- Unit + integration + e2e suites added for Phase 2 components.
- Targeted Phase-2 execution result: `16 passed`.

## Known Gaps For Phase 3
- Hyperparameter/threshold hardening and calibration expansion.
- Broader split-based robustness validation by family and scenario difficulty.
- Stronger production-grade temporal modeling path if deep-learning stack is enabled.

## Verdict
- `phase_2_complete: true`
- `ready_for_phase_3: true`
