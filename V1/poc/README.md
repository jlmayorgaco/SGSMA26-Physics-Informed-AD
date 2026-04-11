# SGSMA 2026 Comparative POC

This directory is a post-contest research bench for journal-paper ablations. It
does not replace or modify the validated competition pipeline under `src/`.

## What It Runs

The POC compares four estimators against four classifiers:

- Estimators: `E1_LSE`, `E2_EKF`, `E3_UKF`, `E4_GNN`
- Classifiers: `C1_RULES`, `C2_LGBM`, `C3_TCN`, `C4_TRANSFORMER`

All estimators emit a common innovation object. The detector and localizer are
shared across every row so the ablation isolates estimator/classifier behavior.

## Reproduce

Generate and validate one synthetic 3LG fault scenario:

```bash
python -m poc.main --mode generate-one-fault
```

Run the full 4x4 ablation:

```bash
python -m poc.main --mode full-ablation
```

Run the POC test suite:

```bash
python -m pytest poc/tests -q
```

## Outputs

Generated synthetic scenarios are written under `poc/data_synth/` and are
gitignored. Ablation artifacts are written under `poc/results/`:

- `ablation_table.csv`: all 16 combinations
- `ablation_heatmap.png`: real-event Macro-F1 heatmap
- `per_class_f1.csv`: per-class F1 for the best compactness-adjusted row
- `report.md`: generated summary with gap and failure-case notes
- `run_<timestamp>.log`: environment, seed, and timing log

## Current Caveats

The local synthesis wrapper uses the repo's existing physics surrogate when a
direct ANDES runtime is not available. The CSV schema is still validated through
`src/io/load_csv.py` without modification. The report always shows both
synthetic held-out performance and external validation on the nine real
competition events; gaps above 0.15 are flagged as distribution shift.

