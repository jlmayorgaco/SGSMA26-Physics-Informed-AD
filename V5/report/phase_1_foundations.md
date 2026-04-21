# Phase 1 Foundations

## What Now Exists
Phase 1 implemented the detector foundations without building final detector branches.

### Domain contracts
- Interfaces added under `src/detectors/domain/interfaces/`:
  - `Detector`
  - `BranchDetector`
  - `FeatureExtractor`
  - `Preprocessor`
  - `FusionStrategy`
  - `Postprocessor`
- DTO/model dataclasses added under `src/detectors/domain/models/`:
  - `DetectionInput`, `DetectionOutput`, `BranchScore`, `WindowSample`, `DetectorConfigV2`, `DetectorMetricsV2`
- Enums added under `src/detectors/domain/enums/`:
  - `EventFamily`, `DetectorMode`

### Shared preprocessing foundations
Implemented in `src/detectors/shared/preprocessing/`:
- `pmu_alignment.py`: robust PMU frame alignment by timestamp.
- `nan_masking.py`: deterministic NaN masking and fill behavior.
- `normalization.py`: fit/apply normalization state.
- `angle_features.py`: wrap-safe angle transforms and sin/cos derivatives.
- `positive_sequence_features.py`: optional positive-sequence feature hook.
- `estimator_feature_adapter.py`: optional estimator metadata injection.
- `pmu_window_builder.py`: window generation + metadata-rich `WindowSample` creation.

### Shared features and utilities
Implemented in:
- `src/detectors/shared/features/` (`temporal_stats`, `derivative_features`)
- `src/detectors/shared/utils/` (`label_mapping`, `thresholding`, `hysteresis`, `calibration`)

## Data Flow Enabled By Phase 1
`split csv -> split records -> scenario PMU load -> aligned frame -> preprocessing -> window builder -> binary label mapping + metadata -> WindowSample / DetectionInput`

Concretely:
1. Split loader reads train/val/test manifests with typed records.
2. Dataset builder loads PMU CSV + frame labels per scenario.
3. Preprocessing applies NaN policy, normalization, angle-safe transforms, optional estimator hooks.
4. Window builder creates deterministic windows (size/stride/label threshold).
5. Label mapping centralizes binary target:
   - Event 0 => `y=0`
   - Event 1..8 => `y=1`
6. Metadata preservation includes:
   - scenario id/split/timestamps/window indices,
   - event coarse/family,
   - cyber/physical/concurrent flags,
   - subtype/origin/difficulty/data-present ratio.

## Intentionally Unimplemented In Phase 1
- Final cyber branch detector logic.
- Final physical branch detector logic.
- Final fusion logic.
- End-to-end detector training/optimization.

Only light architecture stubs were added for cyber/physical/fusion/state-machine to keep Phase 2 wiring clean.

## Tests Added And Status
### Unit
- `test_detection_models.py`
- `test_interfaces_compile.py`
- `test_label_mapping.py`
- `test_pmu_alignment.py`
- `test_nan_masking.py`
- `test_angle_features.py`
- `test_window_builder.py`
- `test_split_loader.py`
- `test_dataset_builder.py`
- `test_estimator_feature_adapter_stub.py`

### Integration
- `test_phase1_dataset_flow_smoke.py`
- `test_phase1_binary_label_pipeline.py`
- `test_phase1_report_generation.py`

Execution result:
- `17 passed`

## Phase 2 Build Targets
Phase 2 should build directly on these foundations:
1. Implement cyber branch models/services on top of `BranchDetector` contracts.
2. Implement physical branch models/services on top of `BranchDetector` contracts.
3. Replace fusion stub with calibrated strategy implementations.
4. Connect postprocessing/state-machine/chunker to final branch outputs.
5. Wire train/eval/infer orchestrators to new branch implementations while keeping M10 entrypoints stable.

## Verdict
- `phase_1_complete: true`
- `ready_for_phase_2: true`
