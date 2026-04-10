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
Raw PMU CSVs (8 buses × 30 fps × 90 min)
         │
         ▼
   load_csv.py  ─── header inspection, positional merge
         │
         ▼
  Chi2Detector  ─── η_t = (z − h₀ − δ)ᵀ diag(R)⁻¹ (z − h₀ − δ)
   (chi2.py)        + DATA_PRESENT dropout flag
         │           OR-combined, debounced (k_on=3, k_off=15)
         │
         ▼ alarm onset indices
  FeatureExtractor  ─── 37 features / onset
   (features.py)        residuals × {mean, max, std}, spatial ν̄,
                         spectral energy, cyber indicators, Jacobian cosines
         │
         ▼
   LightGBM clf  ─── 200 trees, 31 leaves, balanced classes
  (train_lgbm.py)    + 180 synthetic events from swing-eq simulator
         │
         ▼ predicted label (0–8)
  CosineLocalizer  ─── argmax_k |cos(ν̄, J_k)| (bus mode)
 (cosine_match.py)     argmax_ij |cos(ν̄, J_i−J_j)| (line mode)
                        DATA_PRESENT direct (cyber mode)
         │
         ▼
  make_submission.py  ─── per-bus CSVs + combined submission.csv
```

## Key design decisions

| Component | Choice | Reason |
|---|---|---|
| Detector | χ² innovation | Calibrated FP/min; no UKF needed for T1 |
| Classifier | LightGBM | <12,400 params; scoring-efficient |
| Localizer | Jacobian cosine match | Zero learned parameters |
| Augmentation | Swing-equation RK4 | Dynamic transients, not static deltas |
| Splits | Contiguous 70/15/15 | No shuffling per competition rules |

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
  detector/   — Chi2Detector, debounce
  classifier/ — features, LightGBM training
  localizer/  — cosine-match bus and line localization
  augmentation/ — physics-based synthetic event generator
  eval/       — splits, metrics
  pipeline/   — run_inference CLI, submission writer
tests/        — pytest suite (56+ tests, all green)
data/
  raw/        — competition CSVs (read-only)
  metadata/   — .raw grid file, event timeline xlsx
  synthetic/  — generated synthetic events (gitignored)
report/       — LaTeX source and compiled PDF
```

## Test suite

```bash
make test          # runs all tests
# expected: 56+ passed, 1 xfailed, 0 errors
```

## Python version

```
Python 3.13.12  (project targets 3.11+; tested on 3.13)
```

## Frozen dependencies

See `requirements.txt` for exact pinned versions.
Key packages: numpy, scipy, pandas, lightgbm, scikit-learn, matplotlib, openpyxl.
