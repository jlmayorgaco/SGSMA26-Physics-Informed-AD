# Parity Contracts (Phase 1)

## Legacy m0 observable outputs
Expected output folder:
- `output/SCENARIO_RAW0001/chunks/`

Key artifacts:
- `output/SCENARIO_RAW0001/chunks_metadata.json`
- `output/SCENARIO_RAW0001/chunk0_index.json`
- `output/SCENARIO_RAW0001/raw_signal_sanity_summary.csv`

## Legacy m1 observable outputs
Expected output folder:
- `output/SCENARIO_RAW0001_NORMALIZED/chunks/`

Key artifacts:
- `output/SCENARIO_RAW0001_NORMALIZED/normalization_baselines.csv`
- `output/SCENARIO_RAW0001_NORMALIZED/normalized_chunk_index.json`

## Legacy m4 observable outputs
Expected folder structure:
- `output/SIM.../simulation/`
- `output/SIM.../estimated/`

Key artifacts:
- `estimation_metrics_long.csv`
- `estimation_summary_by_bus.csv`
- `estimation_summary_by_signal.csv`
- `estimation_report.json`

## Metric behavior notes
- Angle errors must use wrapped angular differences, not naive subtraction.
- PMU buses in `estimated/` may be passthrough.
- Non-PMU buses may be YBUS-estimated.

## Split policy notes
- No random row shuffling.
- Contiguous temporal splits only (`70/15/15`).
