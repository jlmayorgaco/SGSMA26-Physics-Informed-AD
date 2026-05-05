# SGSMA 2026 Physics-Informed PMU Anomaly Detection

Bus-agnostic anomaly detection, event classification, and event localization for the SGSMA 2026 synchrophasor competition on the IEEE 39-bus system.

The reviewer-facing entrypoint is `main.py`. It accepts any folder containing `Bus*.csv` PMU files, infers the available PMU buses dynamically, and writes the required submission schema:

```text
TIMESTAMP, Bus, Predicted_Event, Predicted_Location
```

## Repository Layout

```text
main.py                         Reviewer-facing inference entrypoint
src/models/hybrid_submission.py Final router: ML for SIM/chunks, physics fallback for long RAW
src/models/bus_agnostic.py      Runtime bus-agnostic physics model
src/features/pmu_discovery.py   Dynamic PMU bus discovery utilities
models/                         Validated ExtraTrees detector/classifier/localizer bundle
models_bus_agnostic/            Final metrics and runtime config
figures/                        IEEE-ready report figures
Final.md                        Final method and artifact summary
report.md                       Full technical report notes
sgsma_2026_final_submission.zip Minimal reviewer package
```

Large local training artifacts and exploratory outputs are intentionally not part of the committed reviewer path.
The compact submission archive includes the `models/` runtime bundle even though that directory is ignored in normal development.

## Quick Start

### 1. Clone and enter the repository

```powershell
git clone https://github.com/jlmayorgaco/SGSMA26-Physics-Informed-AD.git
cd SGSMA26-Physics-Informed-AD
```

### 2. Create a Python environment

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

For development tests:

```powershell
pip install -e ".[dev]"
```

## Run Inference

Use the provided RAW folder or any future RAW folder with PMU files named like `Bus2_*.csv`.
In this repository the checked-in local validation folder is `data\RAW0001` with four zeros.

```powershell
python main.py --input-dir data\RAW0001 --output data\RAW0001\predictions.csv
```

If you are unsure where the files are, run:

```powershell
Get-ChildItem -Recurse data -Filter "Bus*.csv" | Select-Object -First 10 FullName
```

For another machine or hidden test set:

```powershell
python main.py --input-dir C:\path\to\RAW0002 --output C:\path\to\RAW0002\predictions.csv
```

The command also writes `prediction_diagnostics.json` next to the output CSV.

Runtime routing:

- short SIM/chunk-style folders use the validated ExtraTrees/hybrid ML bundle in `models/`;
- long RAW streams use the bus-agnostic physics runtime to preserve timestamp-aligned predictions.

## Validate the Reviewer Package

The compact submission archive can be tested directly:

```powershell
Expand-Archive .\sgsma_2026_final_submission.zip -DestinationPath .\submission_check -Force
python .\submission_check\main.py --input-dir data\RAW0001 --output .\submission_check\predictions.csv
```

Expected output columns:

```text
TIMESTAMP, Bus, Predicted_Event, Predicted_Location
```

## Tests

Run the submission-focused tests:

```powershell
python -m pytest tests\test_bus_agnostic_features.py -q
```

The full historical test suite includes legacy migration/regression tests that may require additional local legacy modules and ANDES runtime setup. The reviewer-facing inference path is validated by the command above and by running `main.py` from the zip archive.

## Report Metrics and Figures

Final metrics required by `guidelines.pdf` are stored in:

```text
models_bus_agnostic/guidelines_metrics.json
models_bus_agnostic/raw_detection_confusion_matrix.csv
models_bus_agnostic/raw_event_confusion_matrix.csv
models_bus_agnostic/raw_event_per_class_metrics.csv
models_bus_agnostic/raw_localization_errors.csv
```

IEEE-ready figures are in `figures/` as both `.pdf` and `.png`.

Current RAW0001 local validation summary:

```text
Detection abnormal F1: 1.0000
Event classification weighted F1: 1.0000
Event classification macro F1 over observed RAW0001 classes: 1.0000
Localization Top-1: 0.8333
```

## Bus-Agnostic Design

The system may use bus IDs as coordinates in the current network instance, but it does not treat a fixed bus such as BUS29 or BUS39 as intrinsically special. In the final router, ML models are used for the validated SIM/chunk regime and physics/topology rules are used for long RAW streams. PMU buses are inferred from input filenames/columns and used only to compute topology-relative quantities such as:

- candidate-to-PMU electrical distance,
- Zbus diffusion compatibility,
- candidate degree and line endpoints,
- severity-weighted distance to observed PMUs,
- data quality and robust signal deviations.

This keeps the inference path compatible with a future RAW folder that has a different PMU placement.
