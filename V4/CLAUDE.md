# CLAUDE.md — SGSMA 2026 PMU Anomaly Detection (V4)

Authoritative project context for V4 of the SGSMA 2026 Synchrophasor Anomaly Detection pipeline.
This document describes the architecture, design decisions, and development workflow for the
modular, milestone‑driven codebase in the `V4/` directory.

## 1. Project Goal

Compete in the SGSMA 2026 Synchrophasor Anomaly Detection Competition (deadline **April 15 2026**).
Three tasks on PMU data from the IEEE 39‑bus system:

1. **Detection** – binary normal vs. abnormal  
2. **Classification** – 9 event labels (0 normal, 1 fault, 2 line outage, 3 gen change, 4 load change,
   5 missing data, 6 missing+physical, 7 bad data, 8 unknown)  
3. **Localization** – bus and/or line of origin (39 candidates, only 8 PMU‑observed)

**Scoring** penalizes model size:  
`Score = Macro‑F1 − λ·log₁₀(N_params)`, λ ∈ [0.02, 0.05].  
Compact, physics‑informed models win. Do not propose end‑to‑end deep learning.

**Test set is hidden.** The `Event` column is provided in the supplied CSVs (training data),
but the organizers will run our code on a withheld dataset where `Event` is blank.
Generalization to unseen events, timings, and locations is the actual evaluation criterion —
memorizing the 9 known events is worthless.

## 2. Critical Corrections to the Official Spec

Discovered by inspecting the actual CSV files. Trust this section over the PDF guide.

### 2.1 Column order is `ANG` before `MAG`, not the other way around

The spec §5.1 lists columns as `VA_mag, VA_ang`. **The real CSVs have them reversed.**
Actual header for `Bus2_Competition_Data_nanmask.csv`:

```
TIMESTAMP, BUS2_VA_ANG, BUS2_VA_MAG, BUS2_VB_ANG, BUS2_VB_MAG,
BUS2_VC_ANG, BUS2_VC_MAG, BUS2_IA_ANG, BUS2_IA_MAG, BUS2_IB_ANG,
BUS2_IB_MAG, BUS2_IC_ANG, BUS2_IC_MAG, BUS2_Freq, BUS2_ROCOF,
DATA_PRESENT, Event
```

Always inspect the actual header before parsing. The loader in `src/analysis/loader.py`
canonicalizes column names regardless of prefix order.

### 2.2 Column names are bus‑prefixed

Each CSV uses prefixes like `BUS2_`, `BUS5_`, `BUS10_`, etc. After merging the 8 files on
`TIMESTAMP`, you get 8 × 14 = 112 measurement columns plus 8 `DATA_PRESENT` flags plus one
`Event` column. Do not collapse the prefixes — they disambiguate which PMU each channel belongs to.

### 2.3 `Event` labels are mostly global, with bus‑specific cyber/bad‑data exceptions

Physical labels 1–4 are shared across the PMU CSVs, but cyber and bad‑data labels are
bus‑specific in the provided files. For example, Bus 29 carries the missing‑data labels 5/6
while other PMUs continue reporting the overlapping physical event, and label 7 appears only
on the corrupted PMU channels. This means:

- A physical load change at Bus 7 still propagates through all observed PMUs, even though no PMU sits on Bus 7.
- Missing‑data and bad‑data labels must be audited with bus‑local `DATA_PRESENT` and three‑phase consistency,
  not by assuming all eight `Event` columns are identical.
- Localization is a separate task: which bus *originated* the event. Ground truth comes from
  `Event_Timeline_and_Location.xlsx`, joined to the CSV by approximate timestamp.
- We estimate non‑PMU bus behavior with the Ybus full‑state proxy and use Jacobian sensitivities for origin inference.

### 2.4 Units verified from real data

- Voltage magnitude in **volts** (~208 000 V at 345 kV nominal — line‑to‑neutral RMS)
- Voltage angle in **degrees**
- Current magnitude in **amperes** (~585 A typical)
- Current angle in **degrees**
- Frequency in **Hz** (nominal 60)
- ROCOF in **Hz/s**

### 2.5 Timestamps are not exactly periodic

`TIMESTAMP` values like `0.1330000000016298` show that the sampling clock is nominally 30 fps
but includes floating‑point drift. **Do not assume `Δt = 1/30` exactly.** Compute Δt per step
from successive timestamps for integrators and windowing logic. Round to 3 decimals only when
writing the submission CSV (per spec §11).

### 2.6 Event timeline (from `Event_Timeline_and_Location.xlsx`)

| #  | Approx time | Type             | Location       | Label |
|----|-------------|------------------|----------------|-------|
| 1  | 10 min      | Cyber data drop  | Bus 29         | 5     |
| 3  | 20 min      | 3LG fault        | Bus 39         | 1     |
| 4  | 40 min      | Line outage      | Bus 24–23      | 2     |
| 5  | 45 min      | Cyber data drop  | Bus 29         | 5     |
| 7  | 50 min      | Cyber + physical | Bus 29 & Bus 2 | 6     |
| 8  | 50 min      | Gen change       | Bus 2          | 3     |
| 9  | 55 min      | Gen change       | Bus 2          | 3     |
| 10 | 65 min      | Load change      | Bus 7          | 4     |
| 11 | 70 min      | Load change      | Bus 7          | 4     |

Event numbers 2 and 6 are absent from the organizer’s numbering — this is intentional,
not a missing event in the data.

PMU buses: **2, 5, 6, 10, 19, 22, 29, 39**. Events at non‑PMU buses (Bus 7, Bus 24) must be
inferred topologically.

## 3. Tiered Strategy — Build T1 First, T2 Only If Time Allows

Do not start with the UKF. The path of least risk is:

### Tier 1 — Lean physics‑informed baseline (target: working end‑to‑end in 2 days)

- **Detector**: CUSUM or Page‑Hinkley change detector on ROCOF and `|V|` deviations from
  rolling robust statistics. Per‑PMU detection, fused by OR with a debouncing window.
- **Feature extractor**: hand‑crafted features over a 3‑second window centered on each detector
  firing — innovation‑style residuals computed against a rolling‑mean baseline rather than a Kalman filter.
  ~30–50 features.
- **Classifier**: LightGBM, ~100–200 trees, depth 4–6. ~50–100k effective parameters.
- **Localizer**: cosine match between observed perturbation pattern $\bar\nu \in \mathbb{R}^8$
  (per‑PMU energy of the residual) and theoretical sensitivity columns $J_k = \partial h / \partial P_k$
  from the power‑flow Jacobian. Pure physics, zero learned parameters.
- **Augmentation**: ~900–1200 synthetic events from ANDES (or fallback: our own swing integrator).

This is already competitive. Decision point at end of Day 3: if validation Macro‑F1 > 0.7 and
there is time remaining, escalate to T2. Otherwise polish T1 and ship.

### Tier 2 — UKF upgrade (only if T1 ships and time remains)

Add a 30‑state UKF (10 generators × $\delta, \omega, P_m$) over the Kron‑reduced swing model.
Replace the rolling‑mean residuals with normalized innovations $\eta_t = \nu_t^\top S_t^{-1} \nu_t$
for the detector, and replace the residual features with Kalman innovation features for the classifier.
The localizer is unchanged. The augmented $\hat P_m$ states give a direct generation‑change indicator
that improves label‑3 classification.

The UKF wins on calibrated false‑alarm rate (χ² test gives a guaranteed FP/min) and on cleaner
features under heavy load. It does not change the architecture of the classifier or the localizer.

### Why not deep learning at all

- 9 labeled events in 90 minutes is far too few for any neural sequence model to learn from real data alone.
- The parameter penalty in scoring (`λ·log₁₀ N_params`) makes a 10M‑param model pay ~0.21,
  vs ~0.15 for LightGBM at 100k. The DL model would need to beat LightGBM by **+0.06 Macro‑F1**
  just to break even.
- Synthetic data closes some of the gap but never fully — and with augmentation, LightGBM benefits
  at least as much as a CNN.
- Interpretability matters for the report: every prediction in the physics‑informed pipeline traces
  back to a measurable quantity.

## 4. Repository Layout (V4‑specific)

```
V4/
├── m0_analize_raw_data.py          # RAW PMU scenario analyzer (complete)
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
│   ├── analysis/                   # Data analysis modules
│   │   ├── loader.py              # CSV loading, header inspection, canonicalization
│   │   ├── integrity.py           # Timestamp alignment, missing‑data detection
│   │   ├── baseline.py            # Normal‑operation filtering & summary
│   │   ├── stats.py               # Basic statistics (mean, std, percentiles)
│   │   ├── noise.py               # Noise modeling (trend‑removal, residuals)
│   │   ├── spectral.py            # Spectral energy, low‑frequency bands
│   │   ├── hilbert_analysis.py    # Hilbert‑transform envelope & instantaneous phase
│   │   ├── events.py              # Event‑span extraction & summarization
│   │   ├── missing.py             # Missing‑data profiling
│   │   ├── power.py               # Three‑phase power computation
│   │   ├── cross_bus.py           # Cross‑bus correlations & rankings
│   │   ├── catalog.py             # (placeholder)
│   │   ├── electrical_analyizer.py # (placeholder)
│   │   ├── faults_analizer.py     # (placeholder)
│   │   └── noise_analizer.py      # (placeholder)
│   ├── plotter/                   # Visualization utilities
│   │   ├── common.py              # Shared plotting helpers
│   │   ├── bus_plots.py           # Per‑bus signal & distribution panels
│   │   ├── event_plots.py         # Event‑overlay plots (voltage, frequency)
│   │   ├── dataset_plots.py       # Dataset‑level heatmaps, missing‑data raster
│   │   ├── general_plots.py       # Normal‑operation signal histograms
│   │   ├── noises_plots.py        # (placeholder)
│   │   ├── stats_plots.py         # (placeholder)
│   │   ├── diagram_plots.py       # (placeholder)
│   │   └── electrical_plots.py    # (placeholder)
│   ├── utils/                     # General‑purpose helpers
│   │   ├── filesystem.py          # Directory creation, path validation
│   │   ├── naming.py              # Bus‑ID inference, column normalization
│   │   ├── numeric.py             # Safe division, circular mean, unwrap
│   │   ├── signals.py             # Signal‑processing wrappers
│   │   ├── windows.py             # Contiguous‑run detection, window slicing
│   │   └── serialization.py       # JSON/CSV saving with type conversion
│   └── simulator/                 # Dynamic simulation (planned)
│       └── ieee39_andes_simulator.py
├── data/                          # Input data (git‑ignored)
│   ├── RAW0001/                   # Raw competition CSVs
│   └── metadata/                  # .raw, .xlsx, .txt files
└── outputs/                       # Generated reports & plots (git‑ignored)
```

## 5. Python Environment

**Python 3.11** (not 3.12 — pandapower and ANDES have rough edges on 3.12 as of early 2026).
Shared dependencies are defined in the root `pyproject.toml`. Key packages:

- numpy, scipy, pandas
- pandapower (for Ybus/Zbus, power‑flow Jacobians)
- andes (for dynamic simulation of synthetic events)
- jax or numba (optional, for fast Jacobian computations)
- lightgbm (classifier)
- scikit‑learn (metrics, splits)
- matplotlib, seaborn (plotting)
- openpyxl (reading event‑timeline .xlsx)

## 6. Conventions — Non‑negotiable

- **Inspect headers before parsing.** Never hardcode column order from the spec.
  `src/analysis/loader.py` reads the actual header and canonicalizes it.
- **No row shuffling, ever.** Splits are contiguous time blocks per spec §9.
  Sliding windows must not straddle split boundaries — leave a 1‑window buffer.
- **Synthetic data goes in train only.** Real competition data fills validation and test contiguously.
  Mixing synthetic into validation defeats the purpose.
- **`DATA_PRESENT` is always a feature**, even after imputation. The model must know when values are synthetic.
- **Calibrate detector thresholds and (T2) `R` matrix from the first 60 seconds** of the dataset,
  which is ground‑truth normal. Do not inflate “for safety” — same bias trap as the PI‑GRU work in OpenFreqBench.
- **Use ROCOF and `|I|` as early‑warning channels**, not Frequency. At 30 fps a 3LG fault lasts ~3 cycles;
  Frequency averages too slowly to catch it before the window closes.
- **Random seeds fixed and logged.** `numpy`, `random`, `lightgbm`, `jax` all seeded from a single config value (default 42).
- **Parameter counting includes everything.** LightGBM ≈ `n_trees × avg_leaves × 2` (split + leaf value).
  Report honestly in the efficiency table.
- **Compute Δt per step** from `TIMESTAMP` differences, not from a hardcoded `1/30` constant.

## 7. Milestones — Execute in Order

### M0. Scaffold + IO + grid model + sanity checks (completed)

- `m0_analize_raw_data.py` loads the 8 CSVs, canonicalizes columns, computes per‑bus statistics,
  extracts event spans, and outputs a comprehensive report (JSON, CSV, PNG, Markdown).
- **Tests**: header layout, base‑case voltages, Ybus shape, Jacobian shape, electrical‑distance symmetry.

### M1. Synthetic data generation (planned)

- `m1_generate_synth_data.py`: programmatic ANDES runs producing labeled synthetic events
  (faults, line outages, generation/load steps, cyber‑physical combinations).
- Output as the same CSV format as the real data so downstream code is unchanged.

### M2. Synthetic‑data validation (planned)

- Compare distribution of synthetic features to real‑event features; adjust simulation parameters
  to close the gap.

### M3. Multiple synthetic scenarios (planned)

- Expand the synthetic dataset with variations in location, severity, timing, and noise.

### M4. Feature extraction (planned)

- Extract ~30–50 features per detected event window: residual statistics, spatial pattern,
  spectral energy, cyber indicators, Jacobian cosine similarities.

### M5. Hidden‑bus state estimation (planned)

- Use Ybus‑weighted harmonic extension to estimate voltages at all 39 buses from the 8 PMU observations.

### M6. Detector training (planned)

- Train a lightweight detector (CUSUM/Page‑Hinkley) on ROCOF and `|V|` residuals.

### M7. Detector evaluation (planned)

- Measure detection precision, recall, FP/min, and detection delay on the validation split.

### M8. Classifier training (planned)

- Train a LightGBM multi‑class classifier on the feature set (augmented with synthetic events).

### M9. Classifier evaluation (planned)

- Compute Macro‑F1, per‑class P/R/F1, confusion matrix, and localization Top‑1/Top‑3 scores.

## 8. Working Rules for Development

- **Run the test suite after every milestone.** Do not move on with red tests.
- **Commit after each milestone** with a clear message: `M0: RAW analyzer outputs dataset/bus/event reports`.
- **No shuffling, no leakage, no shortcuts on the split.** Re‑read §6 if tempted.
- **If a milestone reveals the architecture is wrong**, stop and surface the issue rather than papering over it.
- **Parameter budget**: total trainable parameters under 200k. If LightGBM grows past that,
  reduce `n_estimators` first, then `num_leaves`.
- **Time budget**: if any milestone takes more than 1.5× expected effort, stop and report what is blocking.
- **T1 first, always.** Do not start M8 (UKF) until M0–M7 ship and tests are green.

## 9. Things to Ask Before Changing

- Anything that touches the train/val/test split definition
- Adding a deep‑learning component
- Anything that increases parameter count by more than 2×
- Skipping the M0 sanity checks
- Replacing ANDES augmentation with pandapower‑only steady‑state deltas

## 10. Things to Do Without Asking

- Bug fixes in `src/`
- Adding tests
- Improving feature extractors within the budget
- Tuning LightGBM hyperparameters via grid or Bayesian search on the validation split
- Adding more synthetic events to augmentation, as long as they go to train only and respect the scenario grid

## 11. Definition of Done

Running `python m0_analize_raw_data.py --input-dir data/RAW0001 --output-dir outputs/analysis`
produces a self‑contained report folder with:

- Dataset‑level integrity JSON
- Per‑bus normal‑operation baselines
- Event‑span catalog
- Cross‑bus correlation heatmaps
- Missing‑data raster plot
- README_outputs.txt explaining the layout

All planned milestones (M1–M9) are implemented, tested, and produce the required outputs.
The final pipeline (`run_inference.py`) reads raw CSVs, writes predictions with correct timestamps,
and logs all competition metrics.

---  

*This document is maintained in the `V4/` directory. Refer to the root `CLAUDE.md` for
higher‑level competition context and historical decisions.*