# SGSMA 2026 — Physics-Informed PMU Anomaly Detection

End-to-end pipeline for the SGSMA 2026 Synchrophasor Anomaly Detection Competition.
Detects, classifies (9 event types), and localizes power-system events in real-time
PMU data from the IEEE 39-bus New England system.

## One-command reproduction

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd SGSMA26-Physics-Informed-AD

# 2. Install dependencies
make setup

# 3. Place raw data (read-only)
#    data/raw/Bus{2,5,6,10,19,22,29,39}_Competition_Data_nanmask.csv
#    data/metadata/IEEE 39 Bus Power System.raw
#    data/metadata/Event Timeline & Location.xlsx

# 4. Generate synthetic training data (≈30 s)
make augment

# 5. Run inference (trains model, writes predictions, prints metrics)
make infer

# 6. Compile report (requires latexmk + LaTeX)
make report

# 7. Package everything
make submission
# → submission_sgsma2026.zip
```

## Architecture

```
Raw PMU CSVs (8 buses x 30 fps x 90 min)
         |
         v
   load_csv.py --- header inspection, positional merge
         |
         v
  Chi2Detector --- eta_t = (z - h0 - delta)^T diag(R)^-1 (z - h0 - delta)
   (chi2.py)       + DATA_PRESENT dropout flag
         |         OR-combined, debounced (k_on=3, k_off=15),
         |         then refined into stacked sub-events when
         |         dropout transitions and sustained eta spikes co-occur
         |
         v alarm onset indices
  FeatureExtractor --- 44 features / onset
   (features.py)       post-onset residuals x {mean, max, std}, spatial nu_bar,
                       spectral energy, cyber indicators, Jacobian cosines,
                       Ybus harmonic full-state residuals
         |
         v
   LightGBM clf --- 200 trees, 31 leaves, balanced classes
  (train_lgbm.py)   + physics synthetic events from swing-equation simulator
                    + targeted Bus7, line 24-23, and cyber-physical cases
         |
         v predicted label (0-8)
  CosineLocalizer --- argmax_k |cos(nu_bar, J_k)| (bus mode)
 (cosine_match.py)    argmax_ij |cos(nu_bar, J_i-J_j)| (line mode)
                      DATA_PRESENT direct (cyber mode)
                      Ybus-state residuals promote non-PMU hard cases
         |
         v
  make_submission.py --- per-bus CSVs + combined submission.csv
```

## Key design decisions

| Component | Choice | Reason |
|---|---|---|
| Detector | χ² innovation | Calibrated FP/min; no UKF needed for T1 |
| Classifier | LightGBM | <12,400 params; scoring-efficient |
| State proxy | Ybus-weighted harmonic extension | Estimates all 39 bus voltage magnitudes/angles from 8 PMUs |
| Localizer | Jacobian cosine match + state residual priors | Zero learned parameters; improves Bus 7 and line 24-23 |
| Augmentation | Swing-equation RK4 + topology attenuation | Dynamic transients, not static deltas |
| Splits | Contiguous 70/15/15 | No shuffling per competition rules |
| Real-label training | Event transitions | Keeps stacked `5 -> 6 -> 3` events separate |
| Targeted scenarios | Bus 7, line 24-23, post-cyber physical | Directly trains/evaluates weak competition cases |

## Random seed

All randomness is seeded from `SEED=42` (configurable via `--seed`).
Logged at startup: `Random seed: 42`.

## Parameter count

LightGBM fitted model: ≤ 12,400 effective parameters (200 trees × 31 leaves × 2).
Early stopping typically reduces to ~60–80 trees (~3,700–4,960 params).
Scoring penalty: λ × log₁₀(12400) ≈ 0.05 × 4.09 ≈ 0.20.

## Repository layout

```
src/
  io/         — CSV loading, event parsing
  grid/       — Ybus, Zbus, Jacobian sensitivity columns
  estimator/  — topology-constrained all-39-bus voltage-state proxy
  detector/   — Chi2Detector, debounce
  classifier/ — features, LightGBM training
  localizer/  — cosine-match bus and line localization
  augmentation/ — physics-based synthetic event generator
  eval/       — splits, metrics
  pipeline/   — run_inference CLI, submission writer
tests/        — pytest suite (198 passed, 1 expected xfail)
data/
  raw/        — competition CSVs (read-only)
  metadata/   — .raw grid file, event timeline xlsx
  synthetic/  — generated synthetic events (gitignored)
report/       — LaTeX source and compiled PDF
```

## Test suite

```bash
make test          # runs all tests
# expected: 198 passed, 1 xfailed, 0 pytest errors
```

## ANDES full-state reconstruction experiment

Run one non-PMU Bus 7 three-phase fault at the midpoint of an ANDES IEEE-39
simulation, observe only the 8 competition PMU buses, reconstruct the full
39-bus voltage state with the Ybus estimator, and write metrics/plots/report:

```bash
python -m src.pipeline.andes_fault_reconstruction \
  --fault-bus 7 \
  --t-final 10 \
  --fps 30 \
  --out experiments/andes_fault_bus7
```

Outputs include `truth_all_buses.csv`, `pmu_observed.csv`,
`non_pmu_truth.csv`, `full_state_estimate.csv`, `metrics.csv`, PNG figures,
and `report.md`.

## Python version

```
Python 3.13.12  (project targets 3.11+; tested on 3.13)
```

## Frozen dependencies

See `requirements.txt` for exact pinned versions.
Key packages: numpy, scipy, pandas, lightgbm, scikit-learn, matplotlib, openpyxl.
