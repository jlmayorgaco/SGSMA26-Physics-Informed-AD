# Plot Checklist Against guidelines.pdf

Explicit or strongly implied plots/tables requested by the guide:

- Confusion matrix: covered by `fig01_detection_confusion_matrix` and `fig02_event_confusion_matrix`.
- Per-class precision/recall/F1 table: covered by `fig03_event_per_class_metrics` and `models_bus_agnostic/raw_event_per_class_metrics.csv`.
- Detection metrics including false alarms: covered by `fig04_raw001_task_summary`, `fig06_efficiency_summary`, and `models_bus_agnostic/guidelines_metrics.json`.
- Localization metrics: covered by `fig05_localization_diagnostics` and `fig11_localizer_promotion_curve`.
- Model complexity/runtime: covered by `fig06_efficiency_summary` and `guidelines_metrics.json`.
- Training/validation curves: no neural epoch training curve exists for the final reviewer-facing physics model; the closest validation curve is `fig11_localizer_promotion_curve`, and RAW-vs-SIM validation comparison is `fig08_raw_sim_metric_lines`.
- RAW and simulation comparison: covered by `fig08_raw_sim_metric_lines`, `fig09_raw_sim_event_distribution_lines`, and `fig10_raw_sim_waveform_overlay`.
