# Testing Strategy (Phase 1)

## Test layers
- `unit/`: pure deterministic math/signal helper tests.
- `integration/`: adapter-to-legacy wiring tests on tiny fixtures.
- `regression/`: golden/snapshot contracts for key outputs.
- `smoke/`: importability/wiring checks for CLI modules.

## ANDES dependency handling
- Tests requiring ANDES use marker `@pytest.mark.andes`.
- Those tests skip gracefully when `andes` is unavailable.
- Pure unit tests do not require ANDES and should always run.

## Parity-first migration
The objective is not output improvement yet.
The objective is safety: establish behavioral contracts now so future migration preserves expected outputs.
