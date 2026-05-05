# Bus-Agnostic SGSMA Model Bundle

This is the default reviewer-facing model bundle used by `main.py`.

It accepts any IEEE-39 PMU folder containing `Bus*.csv` files. The bus IDs are
read from filenames and are used as metadata, not as fixed feature-column names.

Run:

```powershell
python main.py --input-dir data\RAW0001
```

Default output:

```text
<input-dir>\predictions.csv
<input-dir>\prediction_diagnostics.json
```

The legacy fixed-PMU model remains under `models/` for reproducibility, but it is
not the default reviewer entrypoint.
