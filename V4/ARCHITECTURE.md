# Architecture Overview — V4 Pipeline

This document describes the software architecture of the SGSMA 2026 PMU Anomaly Detection pipeline (version 4). The system is designed as a modular, milestone‑driven collection of Python modules that transform raw PMU CSV files into detection, classification, and localization predictions.

## 1. High‑Level Data Flow

```
Raw CSV files (8 buses)
        │
        ▼
   ┌─────────────┐
   │   Loader    │  – canonicalize headers, merge on TIMESTAMP,
   │ (src/analysis/loader.py) │  infer bus IDs, compute sampling rates
   └─────────────┘
        │
        ▼
   ┌─────────────┐
   │   Analyzer  │  – per‑bus statistics, event‑span extraction,
   │ (m0_analize_raw_data.py) │  normal‑operation baselines, cross‑bus correlations
   └─────────────┘
        │
        ▼
   ┌─────────────────────────────────────────────────────────┐
   │              Synthetic Data Generation (planned)         │
   │  (m1_generate_synth_data.py, src/simulator/*)           │
   │  – ANDES‑based dynamic simulations of faults, outages,  │
   │    generation/load steps, cyber‑physical events         │
   └─────────────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────┐
   │   Feature   │  – extract ~30–50 features per detected event:
   │  Extractor  │    residual statistics, spatial pattern,
   │ (planned)   │    spectral energy, cyber indicators,
   │             │    Jacobian cosine similarities
   └─────────────┘
        │
        ▼
   ┌─────────────┐
   │   Detector  │  – lightweight change detector (CUSUM/Page‑Hinkley)
   │  Training   │    on ROCOF and |V| residuals; calibrated on first 60 s
   │ (planned)   │
   └─────────────┘
        │
        ▼
   ┌─────────────┐
   │ Classifier  │  – LightGBM multi‑class model (≤200k params)
   │  Training   │    trained on real + synthetic events
   │ (planned)   │
   └─────────────┘
        │
        ▼
   ┌─────────────┐
   │  Localizer  │  – cosine match between observed perturbation pattern
   │ (planned)   │    and power‑flow Jacobian columns (zero learned params)
   └─────────────┘
        │
        ▼
   ┌─────────────┐
   │  Inference  │  – end‑to‑end pipeline that reads raw CSVs and writes
   │   Pipeline  │    Predicted_Event and Predicted_Location columns
   │ (planned)   │
   └─────────────┘
```

## 2. Core Components

### 2.1 Configuration (`src/config/`)

Centralized definitions of constants, enums, dataclasses, and runtime configuration.

- **`constants.py`** – canonical column names, measurement groups, small numerical guards.
- **`enums.py`** – `EventId` (0–8) and `ReportScope` enumerations.
- **`models.py`** – dataclasses: `BusData`, `EventSpan`, `TimeWindow`, `OutputPaths`, etc.
- **`config.py`** – `AnalysisConfig` dataclass with validation and output‑folder properties.
- **`cli.py`** – shared CLI‑parsing utilities (planned).

### 2.2 Data Loading & Canonicalization (`src/analysis/loader.py`)

- Reads CSV files with flexible separators (`sep=None`).
- Detects and strips bus prefixes (`BUS2_VA_ANG` → `VA_ang`).
- Validates required columns are present.
- Converts numeric columns, infers sampling rate from timestamps.
- Returns a `BusData` object (bus ID, DataFrame, sampling rate).

### 2.3 Analysis Modules (`src/analysis/`)

Each module performs a specific analytical task and returns structured dictionaries or DataFrames.

| Module | Purpose |
|--------|---------|
| `integrity.py` | Timestamp alignment, missing‑data detection, dataset‑level integrity checks. |
| `baseline.py` | Filter normal‑operation samples (`Event == 0`, `DATA_PRESENT == 1`), compute summary statistics. |
| `stats.py` | Basic statistics (mean, std, percentiles) with optional circular handling for angles. |
| `noise.py` | Trend removal, robust residual estimation, noise modeling. |
| `spectral.py` | Spectral energy in low‑frequency bands (0.1–2 Hz). |
| `hilbert_analysis.py` | Hilbert‑transform envelope and instantaneous phase. |
| `events.py` | Extract contiguous event spans, summarize event durations, compute per‑channel event profiles. |
| `missing.py` | Profile missing‑data patterns (NaN runs, `DATA_PRESENT` flags). |
| `power.py` | Compute three‑phase instantaneous power (P, Q) from voltage and current phasors. |
| `cross_bus.py` | Cross‑bus correlations and rankings (which PMU reacts most/least to each event). |
| `catalog.py` | (placeholder) Event‑library management. |
| `electrical_analyizer.py` | (placeholder) Electrical‑distance and sensitivity analysis. |
| `faults_analizer.py` | (placeholder) Fault‑specific feature extraction. |
| `noise_analizer.py` | (placeholder) Advanced noise‑and‑artifact analysis. |

### 2.4 Visualization (`src/plotter/`)

Matplotlib‑based plotting utilities organized by scope.

- `common.py` – shared style settings, figure‑saving helpers.
- `bus_plots.py` – per‑bus signal panels, distribution histograms, Hilbert transforms.
- `event_plots.py` – event‑overlay plots (voltage, frequency across buses).
- `dataset_plots.py` – cross‑bus correlation heatmaps, missing‑data raster.
- `general_plots.py` – normal‑operation signal traces and histograms.
- `noises_plots.py`, `stats_plots.py`, `diagram_plots.py`, `electrical_plots.py` – placeholders for future visualizations.

### 2.5 Utilities (`src/utils/`)

General‑purpose helper functions.

| Module | Purpose |
|--------|---------|
| `filesystem.py` | Directory creation, path validation, safe file writing. |
| `naming.py` | Bus‑ID inference from file names, column‑name normalization. |
| `numeric.py` | Safe division, circular mean, angle unwrapping. |
| `signals.py` | Signal‑processing wrappers (filtering, detrending). |
| `windows.py` | Contiguous‑run detection, window slicing. |
| `serialization.py` | JSON/CSV saving with type‑conversion to built‑in Python types. |

### 2.6 Simulator (`src/simulator/`)

Planned ANDES‑based dynamic simulator for synthetic event generation.

- `ieee39_andes_simulator.py` – will programmatically run ANDES simulations of the IEEE 39‑bus system, inject faults/outages/steps, and output synthetic PMU CSV files in the same format as the real data.

### 2.7 Milestone Scripts (`m0_*` … `m9_*`)

Orchestrate the end‑to‑end pipeline. Each script is a standalone entry point that uses the modules above.

- `m0_analize_raw_data.py` – the only fully implemented milestone; produces comprehensive dataset/bus/event reports.
- `m1_generate_synth_data.py` – (planned) generate synthetic events.
- `m2_validate_synth_data.py` – (planned) validate synthetic‑data distribution against real events.
- `m3_multiple_synth_data.py` – (planned) expand synthetic dataset with multiple scenarios.
- `m4_feature_analysis.py` – (planned) extract features from detected event windows.
- `m5_ieee39_hidden_nodes_estimation.py` – (planned) estimate voltages at non‑PMU buses using Ybus‑weighted harmonic extension.
- `m6_faults_detection_training.py` – (planned) train the Tier‑1 detector (CUSUM/Page‑Hinkley).
- `m7_faults_detection_testing.py` – (planned) evaluate detector performance.
- `m8_faults_classification_training.py` – (planned) train the LightGBM classifier.
- `m9_faults_classification_testing.py` – (planned) evaluate classifier and localizer metrics.

## 3. Key Design Decisions

### 3.1 Modularity & Separation of Concerns

Each analysis module does one thing and returns structured data. Plotting is isolated in `plotter/`. This makes testing, debugging, and extending the pipeline straightforward.

### 3.2 Configuration‑Driven Execution

The `AnalysisConfig` dataclass holds all runtime parameters (time windows, thresholds, export toggles). This ensures consistency across modules and simplifies command‑line interface design.

### 3.3 Output Organization

The analyzer creates a self‑contained output folder with subdirectories:

- `dataset/` – scenario‑wide integrity, event catalog, cross‑bus profiles.
- `general/` – normal‑operation‑only analysis.
- `buses/` – per‑bus full‑timeline analysis.
- `events/` – per‑event and per‑event‑per‑bus detailed EDA.

Each subdirectory contains `csv/`, `plots/`, `reports/`, and `json/` folders, making it easy to locate any generated artifact.

### 3.4 Physics‑First Approach

- The pipeline uses the grid topology (Ybus/Zbus) to estimate hidden‑bus voltages and compute Jacobian sensitivity columns.
- Synthetic events are generated with a dynamic simulator (ANDES) rather than static power‑flow deltas, ensuring realistic transients.
- Localization is based on cosine matches with theoretical sensitivity columns, requiring zero learned parameters.

### 3.5 Parameter‑Efficient Modeling

To minimize the scoring penalty, the classifier is LightGBM with a small number of trees and shallow depth. The detector is a simple change‑detection rule. No deep‑learning components are used.

## 4. Dependencies

The project relies on the following key Python packages (pinned in the root `pyproject.toml`):

- **Data manipulation**: `numpy`, `scipy`, `pandas`
- **Grid analysis**: `pandapower` (Ybus, Zbus, power‑flow Jacobians)
- **Dynamic simulation**: `andes` (for synthetic event generation)
- **Machine learning**: `lightgbm`, `scikit‑learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Excel reading**: `openpyxl`
- **Optional acceleration**: `jax` (for fast Jacobian computations; fallback to NumPy if not available)

## 5. Extension Points

The architecture is designed to be extended in the following ways:

1. **New analysis modules** – add a `.py` file in `src/analysis/` that follows the existing pattern (pure functions returning dictionaries/DataFrames).
2. **New plot types** – add functions in `src/plotter/` that accept the relevant data structures and write plots to a provided `Path`.
3. **New milestone scripts** – create a new `mX_*.py` script that imports the needed modules and orchestrates a new pipeline step.
4. **Alternative simulators** – if ANDES proves problematic, implement a fallback swing‑equation integrator in `src/simulator/fallback.py` with the same interface.
5. **Additional detectors/classifiers** – plug in new detection or classification algorithms as long as they respect the parameter budget and output formats.

## 6. Testing Strategy

Unit tests are placed in the root `tests/` directory (outside V4). Each milestone should be accompanied by tests that verify:

- Correct header canonicalization and loading.
- Accurate event‑span extraction.
- Proper handling of missing data (`NaN`, `DATA_PRESENT = 0`).
- Consistency of cross‑bus correlations and rankings.
- That synthetic data matches the expected format and labeling.

Run the test suite with `make test` (root) or `pytest tests/` after each milestone.

## 7. Future Work

- Implement the missing milestone scripts (M1–M9).
- Integrate the ANDES simulator and generate a diverse synthetic dataset.
- Develop the feature extractor, detector, classifier, and localizer modules.
- Create an end‑to‑end inference pipeline (`run_inference.py`) that reads raw CSVs and writes competition‑format predictions.
- Add comprehensive unit and integration tests.
- Optimize performance for the full 90‑minute dataset (target <5 minutes on a laptop CPU).

---

*This document reflects the architecture as of the completion of M0. It will be updated as new milestones are implemented.*