# Plot Checklist Against guidelines.pdf

Explicit or strongly implied plots/tables requested by the guide:

- Detection metrics including false alarms: covered by `fig04_input_data_task_summary`, `fig06_efficiency_summary`, and `models/final_metrics.json`.
- Localization metrics: covered by `fig11_localizer_promotion_curve`, `fig12_localizer_training_validation_curve`, and `models/final_metrics.json`.
- Model complexity/runtime: covered by `fig06_efficiency_summary` and `models/final_metrics.json`.
- Training/validation curves: covered by `fig12_localizer_training_validation_curve` and `fig13_training_validation_curves`. The final reviewer-facing model is a windowed ExtraTrees/hybrid ML pipeline, so the report uses feature-ablation validation curves and loss proxies rather than fabricated neural epoch-loss curves.
- Input data and simulation comparison: covered by `fig08_raw_sim_metric_lines`, `fig09_raw_sim_event_distribution_lines`, and `fig10_raw_sim_waveform_overlay`.
- Guidelines coverage summary: covered by `fig14_guidelines_plot_coverage`.
