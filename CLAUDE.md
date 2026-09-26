# CLAUDE.md

Guidance for working in this repository. Read this before editing or running anything.

## What this is

Physics-informed multi-PMU **anomaly detection, event classification, and localization** for the
**SGSMA 2026** competition on the **IEEE 39-bus** system. Given 8 PMU CSV streams, the pipeline
detects abnormal behavior, classifies the event type, and localizes it to a bus.

The competition provides **8 PMU CSVs** (buses **2, 5, 6, 10, 19, 22, 29, 39**) plus IEEE-39
metadata (`data/metadata/`) and precomputed topology (`data/topology/ieee39/`). The real provided
dataset is `data/RAW0001`. The model is **bus-agnostic**: PMU buses are discovered from filenames
(`Bus2_*.csv` → bus 2 via `src/features/pmu_discovery.discover_bus_csvs`), not hardcoded, so it
generalizes to a hidden RAW folder with different PMU placement.

### Event label convention (0–8)

```
0 = normal              5 = missing data
1 = fault               6 = missing data + physical event
2 = line outage         7 = bad data
3 = generation change   8 = unknown/mixed proxy
4 = load change/drop
```

## Runtime contract (the deliverable)

Single entrypoint:

```powershell
python main.py --input-dir <RAW_OR_SIM_FOLDER> --output <OUTPUT_CSV>
```

- Defaults: reads `data/RAW0001`, writes `<input-dir>/predictions.csv`.
- `main.py` → `src/models/hybrid_submission.run_hybrid_submission_prediction(...)`.
- **Route selection (in `hybrid_submission.py`):** if `models/final_model_config.json` exists →
  **`ml_windowed`** route (model `sgms_extra_trees_windowed_v2`, fixed **30 s** windows via
  `SGSMAFinalModel`). Otherwise → **physics fallback** (`src/models/bus_agnostic.run_bus_agnostic_prediction`,
  configured by `models_bus_agnostic/`).
- **Output CSV columns (exact):** `TIMESTAMP, Bus, Predicted_Event, Predicted_Location`.
- Also writes `prediction_diagnostics.json` next to the CSV (route, model, timing, per-window
  predictions, Top-3 location candidates). Only Top-1 goes in the official CSV.

The ML bundle the `ml_windowed` route loads from `models/`:

```
final_model_config.json, feature_columns.json
classifiers/physical_event_classifier.joblib
detector/bad_data_detector.joblib, detector/missing_composition_detector.joblib
detector/hierarchical_model_config.json
localizer/final_dynamic_localizers.joblib   # hybrid localizer reconstructed from this
```

Detector and classifier are `SimpleImputer + ExtraTreesClassifier` with hierarchical
physics-informed gates; localizers are typed ExtraTrees + a reconstructed hybrid localizer.
Features are `base_v2 + rolling + rls_kalman + graph_temporal` (~45k), built per window by
`src/data_factory/feature_extractor_v2.extract_window_features_v2` and
`src/data_factory/dynamic_feature_extractor_v3.extract_dynamic_features_from_frames`.

See **`MODEL.md`** for the authoritative runtime contract and reported metrics
(SIM: detector 0.969, classifier macro-F1 0.960, localizer Top-1 0.852; RAW0001 detector/classifier 1.0).

## ⚠️ Repo-state gotchas

1. **`models/` is gitignored** (large `.joblib` artifacts), as are `models_submission_v2/`,
   `deepseek/`, `report/`, `workbench/`, `output/`, `plots/`, `release/`. A fresh `git clone` has
   **no `models/`**, so `main.py` falls back to the physics route until you either train the bundle
   or restore it from the tracked **`sgsma_2026_final_submission.zip`** (which bundles `models/`).
   The bundle is present in this working copy — just not tracked.
2. **`src/` IS tracked and is a full Clean-Architecture package** (not stubs). `models_bus_agnostic/`
   is tracked only for its README/JSON/CSV metric tables; its `.joblib` files are untracked.
3. Before claiming the runtime works end to end, **actually run it** — RAW0001 takes ~14 min
   (~9 s per PMU-minute, CPU-only). Use a small SIM folder for a fast check.

## Layout

`src/` follows a layered (Clean Architecture) design — keep new code in the matching layer:

```
main.py                  # single reviewer entrypoint (tracked)
MODEL.md / README.md     # runtime contract + overview (authoritative)
sgsma_2026_final_submission.zip   # compact reviewer package (bundles models/)
configs/                 # data.yaml, simulation.yaml, training_task{1,2,3}.yaml
models/                  # GITIGNORED validated ML bundle loaded by ml_windowed route
models_bus_agnostic/     # physics-fallback config + guideline metric tables (partly tracked)
data/RAW0001/            # 8 provided PMU CSVs
data/metadata/           # IEEE_39_Bus_Power_System.raw, event timeline, PMU locations
data/topology/ieee39/    # precomputed Ybus/Zbus, branches_physical.csv, manifest
tests/                   # pytest: unit/ integration/ regression/ smoke/ (+ fixtures/)
pipelines/               # numbered scripts: m0..m4 (methodology), p10..p17 (per-event rankers)
m1..m4_*.py (root)       # standalone preprocessing/ANDES scripts
scripts/                 # report/figure generation
figures/ report/ presentation/   # paper/deck artifacts (report/ gitignored)
workbench/ deepseek/     # GITIGNORED scratch: simulated SIM data, exploratory ML

src/
  domain/                # constants, events, models, topology (pure domain types)
  application/use_cases/  # orchestration per stage (chunk, normalize, calibrate, train_task{1,2,3}, export…)
  infrastructure/        # andes/, io/ (csv/json/artifact repos), legacy/ adapters
  cli/                   # train / evaluate / generate_data / prepare_dataset / export_submission
  data_engineering/      # raw_loader, normalization, windowing, imputation, feature_factory
  data_factory/          # feature_extractor_v2/v4, dynamic_feature_extractor_v3, final_model (SGSMAFinalModel)
  features/              # pmu_discovery, bus_agnostic feature helpers
  models/                # hybrid_submission (runtime), bus_agnostic (physics), localizer/
  physics/              # ybus, zbus, electrical_distance, positive_sequence, state_estimation
  calibration/          # noise profiling + ANDES event-0 calibration
  estimation/           # Ybus / temporal-regularized state estimators
  evaluation/           # detection / classification / localization / efficiency metrics
  ml/                   # datasets, model_registry, tasks (detector/classifier/localizer), trainer
  signals/ simulation/ submission/ helpers/ utils/
```

Methodology stages: **M0–M4** = chunking, normalization, noise profiling, ANDES event-0
calibration, fault simulation. **P10–P17** = bus-agnostic + per-event-type localizer/ranker training.

## Commands

```powershell
python -m pytest                                   # full suite (last known: 246 passed, 10 skipped)
python -m pytest tests\test_bus_agnostic_features.py -q   # focused submission test
python main.py --input-dir data\RAW0001 --output data\RAW0001\predictions.csv
python main.py --input-dir workbench\simulated\sgsma_generated\SIM00642\pmu --output workbench\_tmp\SIM00642_predictions.csv
```

- Install: `pip install -r requirements.txt` (Python 3.10+); dev extras via `pip install -e ".[dev]"`.
- ANDES tests are skipped unless `RUN_ANDES_TESTS=1` (see `tests/conftest.py`, `andes` marker).
- Environment: Windows, CPU-only is the validated target. Shell is PowerShell (use PS syntax).

## Data-handling facts (verified against the real CSVs)

These correct naive assumptions; honor them in any data-loading code:

- **`Event` column is NOT globally consistent across the 8 CSVs.** Cyber events (5, 6) appear only
  in Bus29's CSV; Bus2/Bus39 show Event=7 (bad data) where others show 0; events 1–4 are consistent
  across all buses. Derive a global event as the max non-zero across per-bus columns.
- **Timestamps drift ~1e-11 s across CSVs.** All 8 have exactly 161,379 rows in the same order. An
  outer merge on `TIMESTAMP` inflates to ~172k rows — **align by row index**, not by merge.
  (The runtime sorts each frame by `TIMESTAMP` independently and processes per-bus.)
- **pandapower 3.x dropped PSS/E RAW import** (`from_pss2_raw` gone). Topology is parsed directly
  from the `.raw`; precomputed Ybus/Zbus already live in `data/topology/ieee39/`. Physics code is in
  `src/physics/` (`ybus.py`, `zbus.py`, `electrical_distance.py`). Note: some `src/domain/`
  modules (e.g. `topology.py`) are Phase-1 scaffolds with TODOs — verify before relying on them.
- **PSS/E bus numbers ≠ competition bus numbers.** The `.raw` uses PSS/E numbering (e.g. PSS/E bus
  12 = 'BUS2'); competition filenames use names (BUS2, BUS5, …). Map accordingly.
- **Kron-reduced Ybus is not symmetric** (phase-shifting transformers) — physically correct.

## Conventions

- Match surrounding style: `from __future__ import annotations`, type hints, `pathlib`, modern
  union syntax (`X | None`). Place code in the correct `src/` layer (domain vs application vs infra).
- Keep the official 4-column output contract exact — graders parse it.
- Don't add runtime dependencies on `deepseek/` or `workbench/`; the deliverable depends only on
  `src/`, `models/` (or the zip bundle), `models_bus_agnostic/`, and `data/`.
- Branch: work happens on `MVP3`; `main` is the PR base.
