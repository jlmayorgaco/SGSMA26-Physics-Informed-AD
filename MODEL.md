# Final Model

## Reviewer Entrypoint

The final submission entrypoint is:

```powershell
python main.py --input-dir <RAW_OR_SIM_FOLDER> --output <OUTPUT_CSV>
```

The active runtime route is:

```text
model = sgms_extra_trees_windowed_v2
route = ml_windowed
window_seconds = 30
```

`main.py` calls `src/models/hybrid_submission.py`, which applies the validated ML bundle in fixed 30 s windows for all inputs. The bus-agnostic physics model is retained only as a fallback if the ML bundle is unavailable.

## Output Contract

The generated prediction CSV contains exactly the required columns:

```text
TIMESTAMP, Bus, Predicted_Event, Predicted_Location
```

The runtime also writes `prediction_diagnostics.json` next to the output CSV. Diagnostics include the route, model name, inference time, window-level predictions, and Top-3 location candidates per window.

## Model Family

The final runtime uses a validated ML pipeline:

- Detector: `SimpleImputer + ExtraTreesClassifier`, with hierarchical physics-informed gates.
- Event classifier: `SimpleImputer + ExtraTreesClassifier`, with hierarchical event logic.
- Localizer: typed ExtraTrees-based localizers and a hybrid localizer reconstructed from `final_dynamic_localizers.joblib`.

The model bundle loaded at runtime is:

```text
models/final_model_config.json
models/feature_columns.json
models/classifiers/physical_event_classifier.joblib
models/detector/bad_data_detector.joblib
models/detector/missing_composition_detector.joblib
models/detector/hierarchical_model_config.json
models/localizer/final_dynamic_localizers.joblib
```

The large redundant `models/localizer/final_hybrid_localizer.joblib` artifact is excluded from the compact zip; the localizer is reconstructed from the dynamic localizer bundle.

## Features

The selected feature variant is:

```text
base_v2 + rolling + rls_kalman + graph_temporal
```

Total selected feature count:

```text
n_features = 45162
```

### Base V2 Features

Extracted by:

```text
src/data_factory/feature_extractor_v2.py
```

Feature families:

- sample count, duration, and time-step statistics;
- timestamp gap and missing-run features;
- `DATA_PRESENT` and NaN/data-quality features;
- voltage magnitude and angle features;
- current magnitude and angle features;
- frequency and ROCOF features;
- robust z-score features;
- pre/early/mid/late window statistics;
- derivatives and rolling robust deviations;
- phase-spread and phase-consistency features;
- P/Q proxy features;
- global PMU aggregations: mean, max, min, std.

### Dynamic V3 Features

Extracted by:

```text
src/data_factory/dynamic_feature_extractor_v3.py
```

Feature families:

- rolling multiscale statistics;
- RLS/Kalman level innovation features;
- graph-temporal PMU features;
- pairwise PMU differences;
- pairwise PMU correlations;
- time-to-peak differences;
- weighted graph differences;
- global data-quality features.

## Event Labels

The event labels follow the SGSMA convention:

```text
0 = normal
1 = fault
2 = line outage
3 = generation change/outage
4 = load change/drop
5 = missing data
6 = missing data + physical event
7 = bad data
8 = unknown/mixed proxy
```

## Reported Performance

The current final ML bundle performance is stored in:

```text
models/final_metrics.json
```

### SIM Performance

```text
Detector accuracy:        0.9693
Classifier accuracy:      0.9673
Classifier macro-F1:      0.9598
Localizer Top-1 exact:    0.8524
```

### RAW0001 Performance

```text
Detector accuracy:        1.0000
Classifier accuracy:      1.0000
Classifier macro-F1:      1.0000
Localizer Top-1 exact:    0.6667
```

## Guideline Metrics

The SGSMA guide asks for detection, classification, localization, and efficiency/model-complexity reporting. The currently available metrics are below.

### Task 1: Detection, Normal vs Abnormal

SIM:

```text
Detector accuracy = 0.9693
```

RAW0001:

```text
Detector accuracy = 1.0000
```

RAW0001 detection confusion matrix from `models_bus_agnostic/raw_detection_confusion_matrix.csv`:

```text
                 pred_normal  pred_abnormal
true_normal                9              0
true_abnormal              0             12
```

RAW0001 abnormal precision/recall/F1:

```text
Abnormal precision = 1.0000
Abnormal recall    = 1.0000
Abnormal F1        = 1.0000
False alarm rate   = 0.0 FP/min
Detection delay    = not computed for the chunk-level RAW0001 summary
```

### Task 2: Event Classification

SIM:

```text
Classifier accuracy = 0.9673
Classifier macro-F1 = 0.9598
```

RAW0001:

```text
Classifier accuracy = 1.0000
Classifier macro-F1 = 1.0000
Weighted-F1         = 1.0000
```

RAW0001 per-class metrics:

```text
event,precision,recall,f1_score,support
0,1.0,1.0,1.0,9
1,1.0,1.0,1.0,1
2,1.0,1.0,1.0,1
3,1.0,1.0,1.0,1
4,1.0,1.0,1.0,1
5,1.0,1.0,1.0,4
6,1.0,1.0,1.0,1
7,1.0,1.0,1.0,3
8,0.0,0.0,0.0,0
```

RAW0001 event confusion matrix, rows=true and columns=predicted:

```text
[[9,0,0,0,0,0,0,0,0],
 [0,1,0,0,0,0,0,0,0],
 [0,0,1,0,0,0,0,0,0],
 [0,0,0,1,0,0,0,0,0],
 [0,0,0,0,1,0,0,0,0],
 [0,0,0,0,0,4,0,0,0],
 [0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,3,0],
 [0,0,0,0,0,0,0,0,0]]
```

### Task 3: Localization

SIM:

```text
Localizer Top-1 exact = 0.8524
```

RAW0001:

```text
Localizer Top-1 exact = 0.6667
```

Top-3 localization is generated in `prediction_diagnostics.json` as `top3_location` per 30 s window. The compact official CSV keeps only the required Top-1 `Predicted_Location` column.

### Task 4: Efficiency and Model Complexity

Runtime command:

```powershell
python main.py --input-dir <RAW_OR_SIM_FOLDER> --output <OUTPUT_CSV>
```

Runtime behavior:

```text
Window length: 30 s
Output: timestamp-aligned predictions for every TIMESTAMP and Bus row
Active model: ExtraTrees/hybrid ML bundle
Fallback: bus-agnostic physics runtime only if ML bundle is missing
```

Model/package size:

```text
Compact submission zip: about 33 MB
The largest redundant localizer artifact is excluded from the zip.
```

Hardware used for local validation:

```text
Windows CPU-only local environment
```

Observed local runtime checks:

```text
SIM00642 extracted zip test: route=ml_windowed, model=sgms_extra_trees_windowed_v2
RAW0001 extracted zip test: generated timestamp-aligned output successfully
RAW0001 timing: 823.67 s inference for 89.65 min of PMU data, about 9.19 s compute per PMU-data minute
```

## Validation Commands

Run a SIM/chunk-style check:

```powershell
python main.py --input-dir workbench\simulated\sgsma_generated\SIM00642\pmu --output workbench\_tmp\SIM00642_predictions.csv
```

Run RAW0001:

```powershell
python main.py --input-dir data\RAW0001 --output data\RAW0001\predictions.csv
```

Run tests:

```powershell
python -m pytest
```

Last full local test result:

```text
246 passed, 10 skipped
```
