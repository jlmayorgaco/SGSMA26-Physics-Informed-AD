# AGENTS

## Purpose
This repository contains the PMU / IEEE-39 / ANDES pipeline through M9.2 and the M10 detector baseline.
This document defines working rules for detector evolution (M10+) so classifier/localizer integration stays stable.

## Architecture Rules
- Preserve current module ownership: `src/simulation` generates scenarios, `src/detectors` detects, `src/evaluation` reports.
- Keep `src/pipelines/m10_train_detector.py`, `m10_eval_detector.py`, and `m10_infer_detector.py` as the public M10 entrypoints.
- Use small composable modules; avoid monolithic files and hidden side effects.
- Keep data contracts explicit in typed DTOs under `src/detectors/domain/models`.
- Preserve backward compatibility with existing `src/detectors/models.py` and `src/detectors/configs.py` during migration.
- Prefer deterministic behavior: seeded training, deterministic split loading, explicit threshold files.

## Detector Conventions
- Detector target is binary (`event=0` vs `event in 1..8`) while preserving per-event metadata for downstream stages.
- Hybrid structure is mandatory:
  - cyber branch (rules + tabular model)
  - physical branch (temporal model, TCN-first)
  - fusion
  - postprocessing (hysteresis/state machine/chunker)
- Event families for reporting:
  - normal: `{0}`
  - physical-heavy: `{1,2,3,4}`
  - cyber-heavy: `{5,7}`
  - concurrent/ambiguous: `{6,8}`
- Keep estimator-assisted signals optional and feature-gated by config.

## Coding Style
- Python 3.10+ with full type hints on public functions/classes.
- Use dataclasses for internal DTOs and clear Protocol/ABC interfaces for components.
- Keep function-level responsibilities narrow; no giant orchestration in branch modules.
- Fail fast on malformed inputs and missing required columns.
- Prefer explicit column names/constants over magic strings spread across files.

## Testing Expectations
- Unit tests for contracts, label mapping, windowing, NaN handling, branch rules, fusion, and postprocessing.
- Integration tests for train/eval/infer pipelines on tiny generated splits.
- E2E test from scenario generation to detector report artifacts.
- New tests must be deterministic and runnable without ANDES unless explicitly marked.

## Artifact and Report Conventions
- Training/eval/infer artifacts go under `output/detector_m10/` (or overridden output root).
- Planning/finalization reports go under `report/` as paired JSON + Markdown.
- JSON reports should use stable top-level sections and machine-readable numeric fields.
- Markdown reports should summarize key verdicts, risks, and next actions.

## Guardrails
- No giant god files.
- No silent schema changes to split CSV or scenario label files.
- No coupling detector internals directly to downstream classifier/localizer internals.
- Reuse existing repository patterns before introducing new abstractions.
