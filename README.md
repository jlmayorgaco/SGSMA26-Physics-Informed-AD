# Proyecto PMU (Phase 1)

This directory contains the Phase-1 refactor scaffold for the SGSMA 2026 synchrophasor anomaly-detection project.

## Phase-1 status
- Legacy scripts (`m0_chunks.py` .. `m4_andes_faults_type1.py`) remain source-of-truth.
- New `src/` currently provides scaffold modules, pure utility helpers, and legacy adapters.
- Tests are focused on parity safety and migration readiness.

## Running tests
From `proyecto_pmu/`:

```bash
pytest -q
pytest tests/unit -q
pytest tests/integration -q
pytest -m "not andes" -q
```

## Future migration
Phase 2 will incrementally move business logic from legacy scripts into the new architecture behind parity/regression tests.
