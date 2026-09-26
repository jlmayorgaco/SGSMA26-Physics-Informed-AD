# SGSMA 2026 Physics-Informed PMU Anomaly Detection

Bus-agnostic anomaly detection, event classification, and event localization for the SGSMA 2026 synchrophasor competition on the IEEE 39-bus system.

The reviewer-facing entrypoint is `main.py`. It accepts any folder containing `Bus*.csv` PMU files, infers the available PMU buses dynamically, and writes the required submission schema:

```text
TIMESTAMP, Bus, Predicted_Event, Predicted_Location
```

## Repository Layout

```text
main.py                         Reviewer-facing inference entrypoint
src/                            Competition package and physics-informed runtime
tests/                          Unit, smoke, integration, and regression tests
pipelines/                      Reproducible competition data pipelines
research/pmu_hybrid_dae_bayes/  Standalone sparse-PMU hybrid-DAE campaign
paper/                          Conference manuscript, evidence, figures, and review notes
presentation/ and slides/       Tutorials and presentation sources
data/ and configs/              Small checked-in inputs and configuration
MODEL.md                        Final model contract and reported metrics
sgsma_2026_final_submission.zip Minimal reviewer package
```

Large local training artifacts, simulator outputs, caches, and scratch runs are intentionally ignored.
The compact submission archive includes the `models/` runtime bundle even though that directory is ignored in normal development.

## Research and paper

The post-competition hidden-state reconstruction study is documented in:

- `research/pmu_hybrid_dae_bayes/README.md` — scope, leakage contract, and commands;
- `research/pmu_hybrid_dae_bayes/src/` — physics, measurement, simulation, and estimator code;
- `research/pmu_hybrid_dae_bayes/tests/` — campaign-specific validation.

The current IEEE manuscript and its reproducibility evidence live in `paper/`. Build and
benchmark commands are documented in `paper/README.md`.

`RAW0001` is treated as an organizer-supplied IEEE-39 reference simulation, not field PMU data or independent external validation.

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

Runtime model:

- the validated ExtraTrees/hybrid ML bundle in `models/` is applied in fixed 30 s windows for all inputs;
- the physics runtime is only a fallback if the ML bundle is unavailable.

## Competition Day XLSX Packaging

When the two live test workbooks arrive, generate the organizer-ready submission folder with:

```powershell
python scripts\prepare_competition_day_submission.py `
  --test1 C:\path\to\Test1.xlsx `
  --test2 C:\path\to\Test2.xlsx
```

This creates `output\competition_day\J_Mayorga_SGSMA2026\` with:

```text
J_Mayorga_Results_Test1.xlsx
J_Mayorga_Results_Test2.xlsx
```

The workbook adapter preserves each bus sheet, fills the `label` column, and adds the `Metrics` sheet required by the policy PDF. See `COMPETITION_DAY.md` for the short runbook.
Diagnostics are written outside the upload folder under `output\competition_day\_diagnostics\`.

## Validate the Reviewer Package

The compact submission archive can be tested directly:

```powershell
Expand-Archive .\sgsma_2026_final_submission.zip -DestinationPath .\submission_check -Force
$raw = (Resolve-Path .\data\RAW0001).Path
python .\submission_check\main.py --input-dir $raw --output .\submission_check\predictions.csv
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

The final selected-model metrics and runtime contract are maintained in `MODEL.md`.
The manuscript figures and quantitative evidence are under `paper/figures/` and
`paper/evidence/`.

Current selected-model validation summary:

```text
SIM detector accuracy:        0.9693
SIM classifier macro-F1:      0.9598
SIM localization Top-1:       0.8524
RAW0001 detector accuracy:    1.0000
RAW0001 classifier macro-F1:  1.0000
RAW0001 localization Top-1:   0.6667
```

## Bus-Agnostic Design

The system may use bus IDs as coordinates in the current network instance, but it does not treat a fixed bus such as BUS29 or BUS39 as intrinsically special. The final runtime uses the same validated ML windowing path for all inputs. PMU buses are inferred from input filenames/columns and used only to compute topology-relative quantities such as:

- candidate-to-PMU electrical distance,
- Zbus diffusion compatibility,
- candidate degree and line endpoints,
- severity-weighted distance to observed PMUs,
- data quality and robust signal deviations.

This keeps the inference path compatible with a future RAW folder that has a different PMU placement.
