# PLANS

## Scope
Plan for detector module delivery after M9.2 readiness.
Target: hybrid binary detector (normal vs abnormal) that feeds downstream classifier/localizer.

## Phase 0 (Current): Architecture and Contracts
### Deliverables
- Repository alignment and target file tree for `src/detectors`.
- Explicit interfaces/DTO contracts for detector components.
- End-to-end data flow spec from scenario artifacts to chunk proposals.
- Pipeline specs for train/eval/infer.
- Test matrix and acceptance criteria.
- Reports:
  - `report/phase_0_detector_architecture.json`
  - `report/phase_0_detector_architecture.md`

### Exit Criteria
- Phase 0 report marked complete.
- Contract ownership and serialization rules are explicit.
- No full detector reimplementation yet.

## Phase 1: Foundations (Contracts + Scaffolding)
### Milestones
1. Add domain interface files and DTO modules in `src/detectors/domain`.
2. Add shared preprocessing adapters (alignment, NaN masking, angle-safe transforms, optional estimator features).
3. Add training dataset builders and split loaders under `src/detectors/training/datasets`.
4. Add minimal runnable stubs for branch services/fusion/postprocessing APIs.
5. Add unit tests for contracts and preprocessing semantics.

### Dependencies
- M9.2 split manifests and scenario label schema.
- Existing M10 baseline modules for compatibility.

### Exit Criteria
- New contracts imported without breaking current pipelines.
- Unit test subset for contracts/preprocessing passes.

## Phase 2: Detector Implementation (Hybrid)
### Milestones
1. Implement cyber branch rules and tabular model training path.
2. Implement physical temporal branch (TCN-first) with baseline fallback.
3. Implement fusion strategies and threshold tuning/calibration workflow.
4. Implement postprocessing state machine + chunk extraction.
5. Integrate estimator-assisted feature adapter behind config flag.
6. Expand integration tests for train/eval/infer.

### Dependencies
- Phase 1 contracts and dataset builders.
- Existing m10 pipelines as orchestrators.

### Exit Criteria
- Train/eval/infer run on split v2 without manual patching.
- Metrics and report artifacts generated consistently.

## Phase 3: Hardening and Readiness
### Milestones
1. Tune thresholds and calibration with validation-first policy.
2. Add per-scenario and family-group diagnostics (cyber-heavy, physical-heavy, concurrent).
3. Add false-positive/minute and delay governance checks.
4. Add downstream chunk-contract validation for classifier/localizer handoff.
5. Final readiness report and acceptance gate.

### Dependencies
- Phase 2 implementation complete.
- Adequate scenario coverage in val/test for event families.

### Exit Criteria (Target)
- F1 abnormal >= 0.78
- Recall abnormal >= 0.82
- Precision abnormal >= 0.75
- Balanced accuracy >= 0.80
- FP/min <= 2.0 on normal-heavy data
- Detection delay p95 <= 2.0 s
- Calibration: ECE <= 0.08
- Family coverage:
  - physical-heavy recall >= 0.80
  - cyber-heavy recall >= 0.75
  - concurrent/ambiguous recall >= 0.70

## Risks and Mitigations
- Sparse labels in val/test for some event types:
  - mitigate with family-level and pooled fallback reporting.
- Optional dependencies (e.g., LightGBM) may be missing:
  - maintain deterministic fallback model.
- Leakage risks from split changes:
  - enforce manifest-based loading and no ad-hoc random split in detector pipelines.

## Ownership Notes
- Keep public pipeline names stable.
- Keep outputs stable for CI/integration tests.
- Defer major refactors to bounded phases; avoid mixing architecture and model tuning in one pass.
