# SGSMA 2026 — Physics-Informed PMU Anomaly Detection (Version 4)

End-to-end pipeline for the SGSMA 2026 Synchrophasor Anomaly Detection Competition.
Detects, classifies (9 event types), and localizes power-system events in real-time
PMU data from the IEEE 39-bus New England system.

This version (V4) focuses on a modular, extensible architecture with comprehensive
data analysis, synthetic event generation, feature extraction, and machine learning
model training.

## Quick Start

### Prerequisites

- Python 3.11+ (tested with 3.13)
- Dependencies installed via `pip` or `uv` (see root `pyproject.toml`)

### Installation

1. Clone the repository and navigate to the V4 directory:

```bash
git clone <repo-url>
cd SGSMA26-Physics-Informed-AD/V4
```

2. Install dependencies from the root project (shared environment):

```bash
cd ..
pip install -e .  # or uv sync
cd V4
```

3. Place raw competition data in `data/RAW0001/`:
   - `Bus2_Competition_Data_nanmask.csv`
   - `Bus5_Competition_Data_nanmask.csv`
   - `Bus6_Competition_Data_nanmask.csv`
   - `Bus10_Competition_Data_nanmask.csv`
   - `Bus19_Competition_Data_nanmask.csv`
   - `Bus22_Competition_Data_nanmask.csv`
   - `Bus29_Competition_Data_nanmask.csv`
   - `Bus39_Competition_Data_nanmask.csv`

4. Place metadata files in `data/metadata/`:
   - `IEEE 39 Bus Power System.raw` (PSS/E RAW format)
   - `Event Timeline & Location.xlsx`
   - `PMUbus_ Location.txt`

### Running the Analysis Pipeline

The pipeline is organized as a series of milestone scripts (`m0_*` to `m9_*`). Start with raw data analysis:

```bash
python m0_analize_raw_data.py --input-dir data/RAW0001 --output-dir outputs/analysis
```

This will generate a detailed report of the dataset, per‑bus statistics, event windows,
cross‑bus correlations, and baseline normal‑operation profiles.

### Synthetic Data Generation (Planned)

Future milestones will generate physics‑based synthetic events using ANDES simulations
(`m1_generate_synth_data.py`) and validate them against the real data distribution
(`m2_validate_synth_data.py`).

### Feature Extraction & Model Training (Planned)

- `m4_feature_analysis.py`: extract discriminative features from event windows
- `m5_ieee39_hidden_nodes_estimation.py`: estimate voltages at non‑PMU buses using grid topology
- `m6_faults_detection_training.py`: train a lightweight detector (e.g., χ² innovation test)
- `m8_faults_classification_training.py`: train a multi‑class LightGBM classifier
- `m7_faults_detection_testing.py` / `m9_faults_classification_testing.py`: evaluate on held‑out data

## Project Structure

```
V4/
├── m0_analize_raw_data.py          # Main RAW PMU scenario analyzer
├── m1_generate_synth_data.py       # ANDES‑based synthetic event generator (planned)
├── m2_validate_synth_data.py       # Synthetic‑data validation (planned)
├── m3_multiple_synth_data.py       # Multi‑scenario synthetic dataset (planned)
├── m4_feature_analysis.py          # Feature extraction & selection (planned)
├── m5_ieee39_hidden_nodes_estimation.py  # Topology‑aware state estimation (planned)
├── m6_faults_detection_training.py       # Detector training (planned)
├── m7_faults_detection_testing.py        # Detector evaluation (planned)
├── m8_faults_classification_training.py  # Classifier training (planned)
├── m9_faults_classification_testing.py   # Classifier evaluation (planned)
├── src/
│   ├── config/                     # Configuration dataclasses, constants, enums
│   ├── analysis/                   # Data analysis modules (loader, stats, events, spectral, …)
│   ├── plotter/                    # Visualization utilities for buses, events, datasets
│   ├── utils/                      # General‑purpose helpers (filesystem, signals, windows, …)
│   └── simulator/                  # ANDES‑based IEEE‑39 dynamic simulator (planned)
├── data/
│   ├── RAW0001/                    # Raw competition CSV files (git‑ignored)
│   └── metadata/                   # Grid topology and event timeline files
└── outputs/                        # Generated reports, plots, and processed data (git‑ignored)
```

## Key Design Decisions

1. **Modular, milestone‑driven development** – Each `m*` script tackles a well‑defined subtask,
   making the pipeline easy to debug and extend.
2. **Physics‑informed synthetic data** – Planned ANDES simulations ensure synthetic events
   follow real electromechanical transients, not just steady‑state deltas.
3. **Topology‑aware feature extraction** – The grid’s Ybus/Zbus and power‑flow Jacobians are used
   to estimate hidden‑bus voltages and compute sensitivity‑based localization features.
4. **Lightweight, interpretable models** – Priority on LightGBM and simple detectors to keep
   parameter counts low (scoring penalty λ·log₁₀N_params).
5. **Strict train/val/test separation** – Contiguous time‑block splits prevent leakage; synthetic
   data is added only to the training set.

## Development Notes

- The codebase uses type hints and dataclasses extensively.
- All random seeds are controlled via a central configuration.
- The analyzer (`m0_analize_raw_data.py`) produces a self‑contained output folder with CSV,
  JSON, PNG, and Markdown reports for easy inspection.
- The project follows the guidelines in the root `CLAUDE.md`; refer to that document for
  competition‑specific constraints and validation rules.

## License & Attribution

Developed for the SGSMA 2026 Synchrophasor Anomaly Detection Competition.
See the root `README.md` for license details.