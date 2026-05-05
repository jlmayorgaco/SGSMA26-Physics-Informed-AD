# Reproducibility

## Runtime Environment

- Python: 3.10 or newer.
- Install runtime dependencies with `pip install -r requirements.txt`.
- The reviewer entrypoint is `main.py`.

## Inference Command

Run inference on any folder containing `Bus*.csv` PMU files:

```powershell
python main.py --input-dir <RAW_FOLDER> --output <RAW_FOLDER>\predictions.csv
```

When running `main.py` from the extracted submission ZIP and the RAW folder is outside that extracted folder, pass an absolute input path:

```powershell
$raw = (Resolve-Path .\data\RAW0001).Path
python .\submission_check\main.py --input-dir $raw --output .\submission_check\predictions.csv
```

The output CSV schema is:

```text
TIMESTAMP, Bus, Predicted_Event, Predicted_Location
```

The command also writes `prediction_diagnostics.json` next to the output CSV.

The final entrypoint applies the same ML runtime to all inputs:

- Inputs are split into 30 s windows.
- Each window is scored with the validated ExtraTrees/hybrid ML bundle in `models/`.
- The bus-agnostic physics runtime is used only if the ML bundle is unavailable.

## Local Validation

From the repository root, the included local validation data can be checked with:

```powershell
python main.py --input-dir data\RAW0001 --output data\RAW0001\predictions.csv
python -m pytest tests\test_bus_agnostic_features.py -q
```

The full historical suite can be run with:

```powershell
python -m pytest
```

Some migration/regression tests are skipped unless the optional ANDES runtime or removed legacy scripts are available.
