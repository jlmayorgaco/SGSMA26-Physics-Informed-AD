# CLAUDE.md — SGSMA 2026 Synchrophasor Anomaly Detection

Authoritative project context for Claude Code. Read this fully before touching any file. This document overrides assumptions from the official competition guide where the two disagree (corrections from real-data inspection are listed in §2).

---

## 1. Project goal

Compete in the SGSMA 2026 Synchrophasor Anomaly Detection Competition. Deadline **April 15, 2026**. Three tasks on PMU data from the IEEE 39-bus system:

1. **Detection** — binary normal vs. abnormal
2. **Classification** — 9 event labels (0 normal, 1 fault, 2 line outage, 3 gen change, 4 load change, 5 missing data, 6 missing+physical, 7 bad data, 8 unknown)
3. **Localization** — bus and/or line of origin (39 candidates, only 8 PMU-observed)

**Scoring** penalizes model size: `Score = Macro-F1 − λ · log10(N_params)`, λ ∈ [0.02, 0.05]. Compact, physics-informed models win. Do not propose end-to-end deep learning.

**Test set is hidden.** The `Event` column is provided in the supplied CSVs (training data), but the organizers will run our code on a withheld dataset where `Event` is blank. Generalization to unseen events, timings, and locations is the actual evaluation criterion — memorizing the 9 known events is worthless.

---

## 2. Critical corrections to the official spec

These were discovered by inspecting the actual CSV files. Trust this section over the PDF guide.

### 2.1 Column order is `ANG` before `MAG`, not the other way around

The spec §5.1 lists columns as `VA_mag, VA_ang`. **The real CSVs have them reversed.** Actual header for `Bus2_Competition_Data_nanmask.csv`:

```
TIMESTAMP, BUS2_VA_ANG, BUS2_VA_MAG, BUS2_VB_ANG, BUS2_VB_MAG,
BUS2_VC_ANG, BUS2_VC_MAG, BUS2_IA_ANG, BUS2_IA_MAG, BUS2_IB_ANG,
BUS2_IB_MAG, BUS2_IC_ANG, BUS2_IC_MAG, BUS2_Freq, BUS2_ROCOF,
DATA_PRESENT, Event
```

Always inspect the actual header in M1 before parsing. Do not hardcode the spec order.

### 2.2 Column names are bus-prefixed

Each CSV uses prefixes like `BUS2_`, `BUS5_`, `BUS10_`, etc. After merging the 8 files on `TIMESTAMP`, you get 8 × 14 = 112 measurement columns plus 8 `DATA_PRESENT` flags plus one `Event` column. Do not collapse the prefixes — they disambiguate which PMU each channel belongs to.

### 2.3 `Event` labels are global per timestep, identical across all 8 CSVs

The label for a given `TIMESTAMP` is the same in every bus CSV. This means:

- A load change at Bus 7 (minute 65) shows `Event = 4` in **all 8 CSVs simultaneously**, even though no PMU sits on Bus 7. The classifier learns to recognize the propagated signature across the 8 observed buses.
- Localization is a separate task: which bus *originated* the event. Ground truth comes from `Event_Timeline_and_Location.xlsx`, joined to the CSV by approximate timestamp. The `Event` column alone does not contain location.
- We do **not** need to estimate Bus 7 measurements. The localizer infers origin topologically via Jacobian sensitivities (see §6).

### 2.4 Units verified from real data

- Voltage magnitude in **volts** (~208 000 V at 345 kV nominal — line-to-neutral RMS, so 345 kV / √3 ≈ 199 kV, with some operating offset)
- Voltage angle in **degrees**
- Current magnitude in **amperes** (~585 A typical)
- Current angle in **degrees**
- Frequency in **Hz** (nominal 60)
- ROCOF in **Hz/s**

### 2.5 Timestamps are not exactly periodic

`TIMESTAMP` values like `0.1330000000016298` show that the sampling clock is nominally 30 fps but includes floating-point drift. **Do not assume `Δt = 1/30` exactly.** Compute Δt per step from successive timestamps for the integrator and the windowing logic. Round to 3 decimals only when writing the submission CSV (per spec §11).

### 2.6 Event timeline (from `Event_Timeline_and_Location.xlsx`)

| #  | Approx time | Type             | Location       | Label |
|----|-------------|------------------|----------------|-------|
| 1  | 10 min      | Cyber data drop  | Bus 29         | 5     |
| 3  | 20 min      | 3LG fault        | Bus 39         | 1     |
| 4  | 40 min      | Line outage      | Bus 24–23      | 2     |
| 5  | 45 min      | Cyber data drop  | Bus 29         | 5     |
| 7  | 50 min      | Cyber + physical | Bus 29 & Bus 2 | 6     |
| 8  | 50 min      | Gen change       | Bus 2          | 3     |
| 9  | 55 min      | Gen change       | Bus 2          | 3     |
| 10 | 65 min      | Load change      | Bus 7          | 4     |
| 11 | 70 min      | Load change      | Bus 7          | 4     |

Event numbers 2 and 6 are absent from the organizer's numbering — this is intentional, not a missing event in the data.

PMU buses: **2, 5, 6, 10, 19, 22, 29, 39**. Events at non-PMU buses (Bus 7, Bus 24) must be inferred topologically.

---

## 3. Tiered strategy — build T1 first, T2 only if time allows

Do not start with the UKF. The path of least risk is:

### Tier 1 — Lean physics-informed baseline (target: working end-to-end in 2 days)

- **Detector**: CUSUM or Page-Hinkley change detector on ROCOF and `|V|` deviations from rolling robust statistics. Per-PMU detection, fused by OR with a debouncing window.
- **Feature extractor**: hand-crafted features over a 3-second window centered on each detector firing — innovation-style residuals computed against a rolling-mean baseline rather than a Kalman filter. ~30–50 features.
- **Classifier**: LightGBM, ~100–200 trees, depth 4–6. ~50–100k effective parameters.
- **Localizer**: cosine match between observed perturbation pattern $\bar\nu \in \mathbb{R}^8$ (per-PMU energy of the residual) and theoretical sensitivity columns $J_k = \partial h / \partial P_k$ from the power-flow Jacobian. Pure physics, zero learned parameters.
- **Augmentation**: ~900–1200 synthetic events from ANDES (or fallback: our own swing integrator).

This is already competitive. Decision point at end of Day 3: if validation Macro-F1 > 0.7 and there is time remaining, escalate to T2. Otherwise polish T1 and ship.

### Tier 2 — UKF upgrade (only if T1 ships and time remains)

Add a 30-state UKF (10 generators × $\delta, \omega, P_m$) over the Kron-reduced swing model. Replace the rolling-mean residuals with normalized innovations $\eta_t = \nu_t^\top S_t^{-1} \nu_t$ for the detector, and replace the residual features with Kalman innovation features for the classifier. The localizer is unchanged. The augmented $\hat P_m$ states give a direct generation-change indicator that improves label-3 classification.

The UKF wins on calibrated false-alarm rate (χ² test gives a guaranteed FP/min) and on cleaner features under heavy load. It does not change the architecture of the classifier or the localizer.

### Why not deep learning at all

- 9 labeled events in 90 minutes is far too few for any neural sequence model to learn from real data alone.
- The parameter penalty in scoring (`λ · log10 N_params`) makes a 10M-param model pay ~0.21, vs ~0.15 for LightGBM at 100k. The DL model would need to beat LightGBM by **+0.06 Macro-F1** just to break even.
- Synthetic data closes some of the gap but never fully — and with augmentation, LightGBM benefits at least as much as a CNN.
- Interpretability matters for the report: every prediction in the physics-informed pipeline traces back to a measurable quantity.

---

## 4. Repository layout

```
.
├── CLAUDE.md                          ← this file
├── README.md                          ← reproduction instructions, written in M10
├── pyproject.toml                     ← pinned dependencies
├── Makefile                           ← setup, test, train, infer, report, submission
├── .gitignore
├── data/
│   ├── raw/                           ← 8 Bus*_Competition_Data_nanmask.csv (read-only)
│   ├── metadata/                      ← .raw, .xlsx, .txt
│   ├── synthetic/                     ← ANDES augmentation output (gitignored)
│   ├── splits/                        ← train/val/test contiguous time-block indices
│   └── processed/                     ← cached merged dataframes (gitignored)
├── src/
│   ├── __init__.py
│   ├── io/
│   │   ├── load_csv.py                ← header inspection + 8-CSV merge on TIMESTAMP
│   │   ├── load_events.py             ← parse Event_Timeline_and_Location.xlsx
│   │   └── label_utils.py             ← extract event transition timestamps from CSVs
│   ├── grid/
│   │   ├── load_case.py               ← parse .raw via pandapower → Ybus, Zbus, branches
│   │   ├── kron_reduce.py             ← Kron reduction to 10 generator internal nodes
│   │   ├── electrical_distance.py     ← d_ij = |Z_ii + Z_jj − 2 Z_ij|
│   │   └── jacobians.py               ← ∂h/∂P_k and ∂h/∂Y_ij sensitivity columns (JAX)
│   ├── dynamics/                      ← T2 only
│   │   ├── swing.py                   ← classical 2nd-order model + RK4
│   │   ├── parameters.py              ← H_i, D_i, x'_d from andes.cases.ieee39
│   │   └── measurement.py             ← h(x) → 14 channels × 8 PMU buses (JAX)
│   ├── estimator/                     ← T2 only
│   │   ├── ukf.py                     ← square-root UKF, Merwe sigma points
│   │   ├── calibration.py             ← Q, R from first 60 s of normal operation
│   │   └── nan_handling.py            ← inflate R on missing PMU channels
│   ├── detector/
│   │   ├── baseline.py                ← T1: CUSUM / Page-Hinkley on ROCOF + |V|
│   │   ├── chi2.py                    ← T2: χ² test on normalized innovation
│   │   └── debounce.py                ← shared k-frame debouncing logic
│   ├── classifier/
│   │   ├── features.py                ← residual feature extractor (~30–50 features)
│   │   ├── train_lgbm.py              ← LightGBM training with augmentation
│   │   └── predict.py
│   ├── localizer/
│   │   ├── cosine_match.py            ← argmax_k cos(J_k, ν̄) for buses
│   │   └── line_match.py              ← argmax_(i,j) cos(J_ij, ν̄) for line outages
│   ├── augmentation/
│   │   ├── andes_sim.py               ← primary: ANDES event simulation
│   │   ├── fallback_sim.py            ← backup: our own swing integrator
│   │   └── scenario_grid.py           ← (type × location × severity × timing) generator
│   ├── pipeline/
│   │   ├── run_inference.py           ← CLI: data/raw/ → predictions
│   │   └── make_submission.py         ← write Predicted_Event, Predicted_Location columns
│   └── eval/
│       ├── metrics.py                 ← all metrics from spec §10
│       └── splits.py                  ← contiguous 70/15/15 with buffer
├── notebooks/
│   ├── 01_eda.ipynb                   ← inspect headers, transitions, units
│   ├── 02_label_alignment.ipynb       ← join xlsx with CSV transitions for localization GT
│   ├── 03_baseline_detector.ipynb     ← T1 detector tuning
│   ├── 04_classifier_ablation.ipynb
│   └── 05_ukf_sanity.ipynb            ← T2 only: χ² calibration check
├── report/
│   ├── sgsma2026_report.tex
│   ├── figures/
│   └── Makefile
└── tests/
    ├── test_io.py
    ├── test_grid.py
    ├── test_jacobians.py
    ├── test_detector.py
    ├── test_classifier.py
    ├── test_localizer.py
    └── test_pipeline.py
```

---

## 5. Python environment

**Python 3.11** (not 3.12 — pandapower and ANDES have rough edges on 3.12 as of early 2026).

`pyproject.toml` uses `uv` or `pip` for installation. Pinned dependencies:

```toml
[project]
name = "sgsma2026-pmu-ad"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "numpy>=1.26,<2.1",
    "scipy>=1.11,<1.14",
    "pandas>=2.1,<2.3",
    "pandapower>=2.14,<3.0",
    "andes>=1.9,<2.0",
    "jax>=0.4.28,<0.5",
    "jaxlib>=0.4.28,<0.5",
    "lightgbm>=4.3,<5.0",
    "scikit-learn>=1.4,<1.6",
    "matplotlib>=3.8,<4.0",
    "seaborn>=0.13,<0.14",
    "openpyxl>=3.1,<4.0",
    "tqdm>=4.66,<5.0",
    "pyyaml>=6.0,<7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<9.0",
    "pytest-cov>=5.0,<6.0",
    "ruff>=0.4,<0.6",
    "ipykernel>=6.29,<7.0",
    "jupyter>=1.0,<2.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"
```

CPU-only JAX is fine for this project (UKF runs in seconds even at 162k steps). If JAX install gives trouble on Windows, fall back to NumPy + `scipy.optimize.approx_fprime` for Jacobians.

---

## 6. Conventions — non-negotiable

- **Inspect headers before parsing.** Never hardcode column order from the spec. M1 must read the actual header of one CSV and assert the layout matches `(BUSk_VA_ANG, BUSk_VA_MAG, …)`. If the layout differs, surface it.
- **No row shuffling, ever.** Splits are contiguous time blocks per spec §9. Sliding windows must not straddle split boundaries — leave a 1-window buffer.
- **Synthetic data goes in train only.** Real competition data fills validation and test contiguously. Mixing synthetic into validation defeats the purpose.
- **`DATA_PRESENT` is always a feature**, even after imputation. The model must know when values are synthetic.
- **Calibrate detector thresholds and (T2) `R` matrix from the first 60 seconds** of the dataset, which is ground-truth normal. Do not inflate "for safety" — same bias trap as the PI-GRU work in OpenFreqBench.
- **Use ROCOF and `|I|` as early-warning channels**, not Frequency. At 30 fps a 3LG fault lasts ~3 cycles; Frequency averages too slowly to catch it before the window closes.
- **Random seeds fixed and logged.** `numpy`, `random`, `lightgbm`, `jax` all seeded from a single config value (default 42).
- **Parameter counting includes everything.** LightGBM ≈ `n_trees × avg_leaves × 2` (split + leaf value). Report honestly in the efficiency table.
- **Compute Δt per step** from `TIMESTAMP` differences, not from a hardcoded `1/30` constant.

---

## 7. Tooling — chosen, do not re-litigate

| Use | Tool | Why |
|---|---|---|
| Parse `.raw`, Ybus, Zbus, electrical distance, PF Jacobian | **pandapower** | Reads PSS/E RAW directly, mature, Python-native |
| Dynamic parameters $H_i, D_i, x'_d$ | **`andes.cases.ieee39`** | Pre-calibrated, matches the case the organizers used |
| Synthetic event generation (T2) | **ANDES** primary, our own swing integrator as fallback | ANDES integrates the full DAE; pandapower cannot |
| UKF + Jacobians (T2) | **JAX + NumPy** | Autodiff for measurement Jacobian, no manual derivatives |
| Classifier | **LightGBM** | Smaller and faster than XGBoost, native categorical support |
| Metrics + splits utilities | **scikit-learn** | Standard, no opinion |

**Do NOT use**: PSS/E (commercial), MATPOWER (Octave dependency), PyTorch, TensorFlow, Transformers, or any deep learning framework. They are explicitly out of scope for this project.

### Why not pandapower for synthetic events

pandapower is a power-flow / quasi-static solver. It does not integrate swing equations, has no concept of inertia, and produces no electromechanical transients. Synthetic events generated from pandapower would be steady-state deltas — completely distinct in distribution from the dynamic-simulation data the organizers provide. Training on those would create a gap that destroys validation performance. ANDES (or our own swing integrator) is mandatory for dynamic augmentation.

---

## 8. M1 sanity checks — run before any modeling

These four checks must pass in M1 before proceeding. They take 10 minutes and prevent silent disasters in later milestones.

1. **Header inspection.** Load `Bus2_Competition_Data_nanmask.csv`, print its columns, and assert the layout is `(TIMESTAMP, BUS2_VA_ANG, BUS2_VA_MAG, ..., DATA_PRESENT, Event)`. If the order differs, log it and proceed with the actual order — never the spec order.

2. **Event transitions.** For each of the 8 CSVs, extract the rows where `Event` changes value:
   ```python
   transitions = df[df['Event'].diff().fillna(0) != 0][['TIMESTAMP', 'Event']]
   ```
   Confirm there are ~9 distinct events, that the timestamps cluster near minutes 10, 20, 40, 45, 50, 55, 65, 70, and that the labels are consistent across all 8 buses at each transition.

3. **NaN behavior during cyber events.** During the intervals where `Event ∈ {5, 6}` in `Bus29_Competition_Data_nanmask.csv`, confirm that all 14 measurement columns are NaN and `DATA_PRESENT == 0`. Confirm the *other* 7 CSVs continue to report valid data during those same intervals.

4. **Power-flow base case.** Parse the `.raw` with pandapower, run a base-case power flow, and confirm voltages at the 8 PMU buses match Table 1 of the spec to 3 decimals (e.g., bus 39 → |V| = 1.0300 p.u., θ = −10.05°). This validates that pandapower is reading the case correctly.

If any check fails, stop and surface the issue. Do not paper over it.

---

## 9. Milestones — execute in order

### M1. Scaffold + IO + grid model + sanity checks

- Create the directory tree, `pyproject.toml`, `Makefile`, `.gitignore`, `README.md` skeleton.
- `src/io/load_csv.py`: header inspection, 8-CSV merge on `TIMESTAMP`, returns a single dataframe with 112 measurement columns + 8 `DATA_PRESENT` flags + `Event`.
- `src/io/load_events.py`: parse the xlsx into a list of `(approx_time_sec, label, location_bus)` tuples.
- `src/io/label_utils.py`: extract event transitions from the merged CSV; produce a clean `(start_ts, end_ts, label, location)` table by joining transitions with the xlsx.
- `src/grid/load_case.py`: pandapower → Ybus, Zbus, branch list, generator list. Verify base-case voltages match spec Table 1.
- `src/grid/electrical_distance.py`: $d_{ij} = |Z_{ii} + Z_{jj} - 2 Z_{ij}|$ for the 39 × 39 distance matrix.
- `src/grid/jacobians.py`: power-flow Jacobian inversion → $J_k$ columns for all 39 buses, projected onto the 8 PMU measurement vector.
- Run the four M1 sanity checks.
- **Tests**: header layout, base-case voltages, Ybus shape, Jacobian shape, electrical-distance symmetry.

### M2. Splits and ground truth tables

- `src/eval/splits.py`: contiguous 70/15/15 time-block split with 1-window buffer. Returns `(train_idx, val_idx, test_idx)` over the merged dataframe.
- `notebooks/02_label_alignment.ipynb`: visualize each event window across all 8 PMUs, confirm the join with the xlsx is correct.
- **Tests**: split coverage = 100%, splits are disjoint, no window straddles a boundary.

### M3. Tier 1 detector

- `src/detector/baseline.py`: rolling-window robust statistics (median + MAD) over a 30 s baseline, compute z-scores of ROCOF and `|V|` per PMU, fuse with `OR`. Page-Hinkley change detector on top of the fused score.
- `src/detector/debounce.py`: declare abnormal after `k` consecutive frames above threshold; clear after `k` consecutive frames below.
- Tune threshold and `k` on the train split to hit FP/min < 0.1 on the first 10 minutes (which are normal).
- **Tests**: detector fires within 1 s of every known event in the train split; FP rate < 0.1/min on the first 10 minutes.

### M4. Tier 1 feature extractor + classifier

- `src/classifier/features.py`: for each detected event window (3 s centered on detector firing), extract:
  - Per-channel residual statistics (mean, max, std, p95) split by voltage-mag, voltage-ang, current-mag, current-ang, frequency, ROCOF
  - Spatial pattern: $\bar\nu \in \mathbb{R}^8$ (per-PMU residual energy), its entropy, its argmax
  - Spectral energy in 0.1–2 Hz band of the frequency residual (inter-area oscillation indicator)
  - Cyber indicators: NaN count, number of PMUs with `DATA_PRESENT = 0`, longest NaN run
  - Topology-projected features: cosine similarities of $\bar\nu$ to each $J_k$ (top-3 values + argmax)
- `src/classifier/train_lgbm.py`: LightGBM with `num_leaves=31`, `n_estimators=200`, `learning_rate=0.05`, `class_weight='balanced'`, early stopping on validation. Log feature importances.
- **Tests**: trained model achieves Macro-F1 > 0.6 on validation real events alone (without augmentation yet).

### M5. Tier 1 localizer

- `src/localizer/cosine_match.py`: given $\bar\nu$ at event detection, compute cosine similarity to all 39 columns $J_k$, return Top-1 and Top-3.
- `src/localizer/line_match.py`: same logic for line outages over the 34 branches with $J_{ij}$.
- The pipeline runs both bus-mode and line-mode, picks the one with higher confidence based on the predicted event type from the classifier (line outage → line mode, everything else → bus mode).
- **Tests**: Top-1 ≥ 5/9 on real events, Top-3 ≥ 8/9.

### M6. Augmentation

- `src/augmentation/andes_sim.py`: programmatic ANDES runs producing labeled synthetic events:
  - 200 three-phase faults, varied bus, fault impedance, clearing time
  - 100 line outages, varied branch and pre-event load level
  - 200 generation step changes, ±5 to ±50 MW, varied generator
  - 200 load step changes, ±5 to ±50 MW, varied load bus
  - 100 PMU dropout events at varied PMU buses and durations
  - 100 cyber+physical concurrent events
- `src/augmentation/fallback_sim.py`: same interface but using our own swing integrator (only used if ANDES fails to install). Document which one was used in the final report.
- Output as the same CSV format as the real data so downstream code is unchanged. Save to `data/synthetic/`.
- Re-run M4 training with augmentation (training set only). Confirm validation Macro-F1 improves to > 0.85.
- **Tests**: synthetic events have correct shape, valid `Event` labels, valid `Location` metadata.

### M7. End-to-end pipeline + submission

- `src/pipeline/run_inference.py`: CLI taking `--data data/raw/` and producing predictions. CPU only, < 5 min for the full 90 minutes.
- `src/pipeline/make_submission.py`: append `Predicted_Event` and `Predicted_Location` columns to each bus CSV with timestamps preserved exactly. Also write a single combined `submission.csv`.
- `src/eval/metrics.py`: compute every metric from spec §10 — detection (precision, recall, F1, FP/min, detection delay), classification (Macro-F1, Weighted-F1, per-class P/R/F1, full 9×9 confusion matrix), localization (Top-1, Top-3, mean electrical-distance error using `Zbus`).
- **Tests**: submission file matches input timestamps exactly (3-decimal rounding), all metrics produce numeric output.

### M8. Tier 2 — UKF (only if T1 ships and time remains)

Decision point: end of Day 3. If T1 validation Macro-F1 > 0.7 and ≥ 2 days remain, proceed. Otherwise skip and go to M9.

- `src/dynamics/parameters.py`: pull $H_i, D_i, x'_d$ from `andes.cases.ieee39`.
- `src/dynamics/swing.py`: 30-state model ($\delta_i, \omega_i, P_{m,i}$ for $i = 1..10$), RK4 integrator using per-step Δt from timestamps.
- `src/dynamics/measurement.py`: JAX-traced $h(x)$ producing 14 channels × 8 PMU buses.
- `src/estimator/ukf.py`: square-root UKF with Merwe sigma points ($\alpha = 10^{-3}, \beta = 2, \kappa = 0$).
- `src/estimator/calibration.py`: empirical $R$ from first 60 s; physical $Q$.
- `src/estimator/nan_handling.py`: set $R$ rows/cols to $10^{12}$ when `DATA_PRESENT = 0`.
- `src/detector/chi2.py`: $\eta_t = \nu_t^\top S_t^{-1} \nu_t$, threshold from $\chi^2_{n_z, 1-\alpha}$.
- Replace M3 detector input with χ² statistic; replace M4 features with innovation features. Re-train LightGBM.
- **Tests**: KS test of $\eta_t$ vs $\chi^2_{n_z}$ on first 60 s with $p > 0.01$; UKF runs the full 90 min without divergence; detector fires within 1 s of every known event with FP < 0.1/min.

### M9. Report

- `report/sgsma2026_report.tex`: 3–6 pages, IEEE conference style.
- Sections: method overview with architecture diagram, preprocessing (NaN handling, normalization, features), training (seeds, hyperparameters, hardware, augmentation), results (all four metric blocks + confusion matrix figure + training curves), efficiency table (parameter count, model size MB, inference time per minute of data, hardware spec), discussion (failure modes, ablations, what changed between T1 and T2 if T2 was built).
- Build via `latexmk -pdf` from `report/Makefile`.

### M10. Reproducibility + packaging

- Top-level `Makefile` targets: `setup`, `test`, `train`, `infer`, `report`, `submission` (zips code + predictions + report into `submission_sgsma2026.zip`).
- `README.md` with one-command reproduction: `make setup && make submission`.
- `requirements.txt` frozen, Python version recorded.
- All random seeds threaded from a single config (default 42).

---

## 10. Working rules for Claude Code

- **Run the test suite after every milestone.** Do not move on with red tests.
- **Commit after each milestone** with a clear message: `M3: Page-Hinkley detector hits 9/9 events with FP/min=0.04`.
- **No shuffling, no leakage, no shortcuts on the split.** Re-read §6 if tempted.
- **If a milestone reveals the architecture is wrong**, stop and surface the issue rather than papering over it.
- **Parameter budget**: total trainable parameters under 200k. If LightGBM grows past that, reduce `n_estimators` first, then `num_leaves`.
- **Time budget**: if any milestone takes more than 1.5× expected effort, stop and report what is blocking.
- **T1 first, always.** Do not start M8 (UKF) until M1–M7 ship and tests are green.

---

## 11. Things to ask before changing

- Anything that touches the train/val/test split definition
- Adding a deep learning component
- Anything that increases parameter count by more than 2×
- Skipping the M1 sanity checks
- Replacing ANDES augmentation with pandapower-only steady-state deltas

## 12. Things to do without asking

- Bug fixes in `src/`
- Adding tests
- Improving feature extractors within the budget
- Tuning LightGBM hyperparameters via grid or Bayesian search on the validation split
- Adding more synthetic events to augmentation, as long as they go to train only and respect the scenario grid

---

## 13. Definition of done

Running `make submission` from a clean clone, with `data/raw/` and `data/metadata/` populated, produces `submission_sgsma2026.zip` containing:

- All source code under `src/`
- All 8 prediction CSVs with timestamps matching the input exactly
- Combined `submission.csv`
- Compiled `report.pdf`
- `README.md` with reproduction steps
- `requirements.txt` and frozen environment

All tests pass. Every metric from spec §10 is computed and logged. Random seed and library versions are documented. No deep learning frameworks installed.

---

## 14. Open questions to resolve as we go

- Does the UKF in T2 actually beat T1 enough to justify its complexity? Run an ablation in M8 and report both numbers.
- Should label 8 (unknown) be a fallback when classifier max-probability < threshold, or never predicted? Spec §14 Q6 warns against overuse — default to never predicting it unless ablation shows otherwise.
- For line outage localization, does $J_{ij}$ cosine match beat a learned multiclass head? Compare in M5/M8.
- Top-1 vs Top-3 weighting in the final aggregate score is unclear from the spec — report both prominently.
