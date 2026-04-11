# Engineering Review Artifacts

These artifacts audit the visible SGSMA 2026 run from an electrical-engineering
perspective. They are not hidden-test claims; they document why each visible
alarm fired and what the Ybus state proxy estimated from the eight PMUs.

Key result:

- Full detector visible audit: precision 1.000,
  recall 1.000, F1 1.000,
  TP/FP/FN 12/0/0.
- Main chi2/DATA_PRESENT detector only: precision 1.000,
  recall 0.750, F1 0.857,
  TP/FP/FN 9/0/3.
- The improvement comes from adding the deterministic phase-unbalance detector
  for label 7 bad-data bursts and from matching stacked labels as separate
  transitions.
- Per-bus sample audit: macro-F1 0--8
  0.861, visible-label macro-F1 0--7
  0.968, weighted-F1
  0.999.

Files:

- `detection_and_scenario_audit.json`: alarm-by-alarm source, thresholds,
  labels, locations, top residual buses, and figure links.
- `units.json`: channel units and derived-score units.
- `per_bus_sample_metrics.json`: each PMU-specific Event column compared with
  the matching per-bus prediction stream.
- `ieee39_topology.json`: buses, PMU flags, branch list, and diagram metadata.
- `scenario_table.csv`: flat table for spreadsheet review.
- `figures/`: IEEE-39 topology diagram and one plot per alarm.
