# Architecture (Phase 1)

## Goal
Phase 1 introduces a clean architecture that coexists with legacy scripts while preserving current behavior.

## Folder responsibilities
- `src/domain`: core entities/constants and shared domain vocabulary.
- `src/application/use_cases`: orchestration-level use cases (currently scaffolded).
- `src/infrastructure/legacy`: thin adapters that call legacy scripts directly.
- `src/infrastructure/io`: repository interfaces for CSV/JSON/artifacts (scaffolded).
- `src/signals`, `src/physics`, `src/estimation`, `src/evaluation`: small pure utilities migrated safely first.
- `src/cli`: future command entrypoints; currently minimal wiring.

## Legacy coexistence strategy
- Legacy scripts (`m0`..`m4`) stay untouched and remain source-of-truth.
- Adapters expose stable function names so tests and future modules can depend on new APIs.
- No algorithmic changes are made in this phase.

## Why adapters first
Adapters decouple new architecture from legacy file/module layouts.
This allows future phase-2 migrations to happen incrementally with parity tests guarding behavior.
