# SGSMA 2026 Final Technical Report

## 1. Executive Summary

This project implements a bus-agnostic, physics-informed anomaly detection, event classification, and localization system for the SGSMA 2026 IEEE 39-bus PMU competition.

The final reviewer entrypoint is:

```powershell
python main.py --input-dir <RAW_FOLDER> --output <RAW_FOLDER>\predictions.csv
```

The generated prediction file follows the `guidelines.pdf` schema:

```text
TIMESTAMP, Bus, Predicted_Event, Predicted_Location
```

The final runtime path applies the validated ExtraTrees/hybrid ML bundle in fixed 30 s windows for all inputs. The bus-agnostic physics runtime is retained only as a fallback if the ML bundle is unavailable. The code infers PMU buses from filenames and columns, then uses them as topology coordinates for feature computation.

## 2. Bus-Agnostic Contract

The system is allowed to know that a measurement comes from a bus in the current case. It is not allowed to learn that a specific bus ID is always important.

Allowed use of bus IDs:

- Read `Bus*.csv` files.
- Infer which buses have PMU data.
- Query topology, Zbus/electrical distance, branch endpoints, and node degree.
- Emit final labels such as `BUS7`, `LINE23-24`, or `PMU29`.

Forbidden use of bus IDs:

- Fixed model features like `is_BUS29`.
- Rules like `if strongest_pmu == BUS39`.
- Assuming the observed PMUs are always `(2, 5, 6, 10, 19, 22, 29, 39)`.
- RAW0001 chunk-specific corrections.

Implemented support:

- `src/features/pmu_discovery.py`: PMU bus discovery from filenames and feature columns.
- `src/models/bus_agnostic.py`: reviewer-facing runtime prediction.
- `src/data_factory/dynamic_feature_extractor_v3.py`: dynamic feature extraction from observed frames, not fixed RAW0001 PMUs.
- `src/models/localizer/hybrid.py`: topology residual ranker infers observed PMUs from each row.

## 3. Data and Training Assets

Retained final data/workbench structure:

```text
workbench/scenarios/sgsma_generated
workbench/simulated/sgsma_generated
workbench/features/sgsma_generated
workbench/raw_current_eval
workbench/event3_generation_ranker_dynamic_pmus_check
workbench/final_submission_bus_agnostic
```

The 5000 generated scenarios are preserved under the `sgsma_generated` folders. Smoke runs, failed experiments, and temporary workbench outputs were removed.

Validation strategy:

- Simulated scenario training uses scenario/group-aware splits to avoid leakage.
- RAW0001 is used as local validation and reporting only.
- Candidate localizer improvements are promoted only if they are bus-agnostic and improve or preserve RAW/SIM guardrails.

## 4. Preprocessing

### 4.1 Timestamp Alignment

Each PMU CSV is read and sorted by `TIMESTAMP`. Runtime output preserves the original timestamp and bus alignment.

### 4.2 NaN and Missing Data Handling

Missing data is not blindly imputed away. It is represented explicitly through:

- `DATA_PRESENT` fraction.
- Per-signal NaN fraction.
- Maximum NaN fraction across channels.
- Timestamp gap ratio.
- Missing-run indicators.

Signal arrays are interpolated only where needed for robust signal-processing operations such as angle unwrapping or derivatives. Missingness remains available as a model feature.

### 4.3 Robust Normalization

For each PMU signal, a robust pre-event baseline is computed:

```text
center = median(pre-event samples)
scale = 1.4826 * MAD(pre-event samples)
robust_z(t) = |x(t) - center| / max(scale, eps)
```

This reduces sensitivity to outliers and makes features comparable across PMUs and RAW placements.

### 4.4 Angle Processing

Voltage/current angle channels are unwrapped in degrees before derivative and spread calculations:

```text
theta_unwrapped = unwrap(theta * pi/180) * 180/pi
```

This prevents artificial jumps at the `-180/180` boundary from becoming false events.

## 5. Feature Engineering

The feature set is organized by physical meaning. The strongest features are the ones that map PMU time-series behavior into electrical and topology-relative quantities.

### 5.1 Data Quality Features

Features:

- `DATA_PRESENT` fraction.
- NaN fraction by PMU and signal.
- Maximum timestamp gap ratio.
- Missing-run severity.
- Global min/mean data-present fraction.
- Global max NaN fraction.

Why relevant:

- Missing PMU frames and bad measurements are competition event types.
- These features separate communication/instrumentation anomalies from physical grid disturbances.
- They are usually among the most important detector features for `event5` and `event7`.

### 5.2 Robust Voltage and Current Features

Features:

- Robust z-score max for `VA/VB/VC_MAG`.
- Robust z-score max for `IA/IB/IC_MAG`.
- Full-window span.
- Early/mid/late pre-event deltas.
- Maximum absolute derivative.
- Rolling energy and rolling slope.
- Time-to-peak.

Why relevant:

- Faults create voltage depression and current spikes.
- Load changes create sustained voltage/current magnitude shifts.
- Line outages redistribute current and voltage stress across electrically nearby PMUs.

Most relevant for:

- Detector: `voltage_dev_max`, `current_dev_max`, `severity_max`.
- Classifier: separating fault/outage/load/generation changes.
- Localizer: severity pattern over observed PMUs.

### 5.3 Frequency and ROCOF Features

Features:

- Robust frequency deviation.
- ROCOF absolute magnitude.
- Frequency/ROCOF rolling energy.
- Frequency/ROCOF late-pre delta.
- Hilbert instantaneous frequency statistics where available.

Why relevant:

- Generation trips and load imbalance appear strongly in frequency and ROCOF.
- Faults may be local and fast; generation events are system-wide but topology-weighted.

Most relevant for:

- `event3`/`event6` generation-related classification/localization.

### 5.4 Three-Phase Symmetrical Components

Computed from phase phasors:

```text
V0 = (Va + Vb + Vc) / 3
V1 = (Va + a Vb + a^2 Vc) / 3
V2 = (Va + a^2 Vb + a Vc) / 3
```

Features:

- Positive-sequence magnitude.
- Negative-sequence magnitude.
- Zero-sequence magnitude.
- Sequence imbalance ratios.
- Sequence robust deviations.

Why relevant:

- Balanced normal operation is positive-sequence dominated.
- Faults and phase imbalance increase negative/zero sequence.
- Helps avoid confusing true physical faults with single-channel bad data.

### 5.5 P/Q Proxy Features

Approximate complex power per phase:

```text
S_proxy = V * conj(I)
P_proxy = Re(S_proxy)
Q_proxy = Im(S_proxy)
```

Features:

- P/Q early/mid/late deltas.
- P/Q robust z-score.
- P/Q span.
- P/Q derivative.

Why relevant:

- Load events change P and Q demand.
- Generator trips change active-power balance and induce frequency response.
- Line outages redistribute apparent power.

Most relevant for:

- `event4` load-change localization and classification.

### 5.6 Graph Signal Processing Features

The observed PMU severities are treated as a graph signal on the measured buses.

Features:

- PMU spread by signal.
- Weighted graph total variation.
- Pairwise PMU difference weighted by inverse Zbus distance.
- Time-to-peak range across PMUs.
- Diffusion cosine score.
- Diffusion residual norm.
- Severity entropy.
- Top1-top2 severity margin.

Electrical basis:

Disturbances propagate according to electrical coupling, not numeric bus order. Zbus effective distance approximates how strongly a disturbance at one location should be observed at each PMU.

Candidate diffusion model:

```text
expected_pmu_response(candidate, pmu, tau)
    = exp(-z_eff(candidate, pmu) / tau)

diffusion_cosine
    = cosine(observed_severity_vector, expected_response_vector)
```

Most relevant for:

- Localizer.
- RAW0002 robustness because features are relative to the observed PMU set.

### 5.7 Candidate-Relative Localization Features

For each candidate bus or line:

- Candidate type: bus, line, load bus, generator bus, transformer/generator terminal.
- Candidate degree.
- Minimum/mean/max electrical distance to observed PMUs.
- Severity-weighted candidate distance.
- Inverse distance score.
- Diffusion cosine and residual over multiple `tau`.
- Endpoint-observed count for line candidates.
- Line metadata: impedance, X/R proxy, tap flag, endpoint degree.

Important: candidate IDs are used to query topology and write labels, but candidate ID is not used as a direct model feature.

Most relevant localization features:

1. Diffusion cosine/residual.
2. Severity-weighted distance to observed PMUs.
3. Inverse distance score.
4. Candidate type and endpoint class.
5. Candidate degree and line impedance metadata.
6. Severity entropy and top1-top2 margin.

## 6. Models Used

### 6.1 Runtime Windowed ML Submission Model

Files:

- `src/models/hybrid_submission.py`
- `src/data_factory/final_model.py`
- `src/models/bus_agnostic.py`

Type:

- ExtraTrees/hybrid ML applied in fixed 30 s windows.
- Rule-calibrated, physics-informed fallback only when the ML bundle is unavailable.
- Uses robust PMU and global features.
- Reads arbitrary PMU bus placement from input files.

Purpose:

- Reviewer-facing entrypoint.
- Produces valid SGSMA prediction CSV.
- Uses the validated ML route consistently while preserving timestamp-aligned CSV output.

### 6.2 Legacy ExtraTrees Detector and Classifier

Model family:

- Scikit-learn ExtraTrees ensembles.
- Hierarchical detector/classifier from the frozen validated model bundle.

Purpose:

- Reproducible training and validation baseline.
- Strong RAW0001 detector/classifier behavior.

Performance:

- RAW0001 detector: 100%.
- RAW0001 classifier: 100% over observed classes.
- SIM detector: 96.93%.
- SIM classifier accuracy: 96.73%.
- SIM classifier macro-F1: 95.98%.

### 6.3 Typed Localizer and Hybrid Topology Ranker

Files:

- `src/models/localizer/hybrid.py`
- `pipelines/p15_agnostic_location_guardrails.py`
- `pipelines/p16_train_event3_generation_ranker.py`

Model family:

- Typed localizer by location class.
- Topology residual ranker.
- Event-specific candidate ranker for generation-related bus localization.

Purpose:

- Improve localization while preserving bus-agnostic behavior.
- Use candidate-relative topology and signal features.

Promotion decision:

- P15 topology canonicalization promoted.
- P16 event3/event6 ranker promoted for RAW reporting.
- P14 event2 line ranker and P17 event4 load ranker were not promoted because they did not improve final RAW Top-1 in a general way.

## 7. Final RAW0001 Metrics Required by Guidelines

Metrics are saved in:

```text
models_bus_agnostic/guidelines_metrics.json
models_bus_agnostic/raw_detection_confusion_matrix.csv
models_bus_agnostic/raw_event_confusion_matrix.csv
models_bus_agnostic/raw_event_per_class_metrics.csv
models_bus_agnostic/raw_localization_errors.csv
```

### 7.1 Task 1: Detection, Normal vs Abnormal

Normal = label `0`; abnormal = labels `1-8`.

```text
TP = 12
TN = 9
FP = 0
FN = 0
Accuracy = 1.0000
Abnormal precision = 1.0000
Abnormal recall = 1.0000
Abnormal F1 = 1.0000
False alarm rate = 0.0 FP/min
Detection delay = not computed from promoted chunk-level summary
```

Detection confusion matrix:

```text
                 pred_normal  pred_abnormal
true_normal                9              0
true_abnormal              0             12
```

### 7.2 Task 2: Event Classification

Labels: `0..8`.

```text
Accuracy = 1.0000
Weighted-F1 = 1.0000
Macro-F1 over observed RAW0001 classes = 1.0000
Macro-F1 over forced labels 0..8 with absent event8 zero_division=0 = 0.8889
```

Per-class metrics:

```text
event,precision,recall,f1,support
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

Event confusion matrix, rows=true, columns=pred, labels `0..8`:

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

### 7.3 Task 3: Localization

```text
Localized event support = 12
Top-1 exact accuracy = 0.8333 = 10/12
Mean electrical-distance error = 0.003214
Median electrical-distance error = 0.000000
```

Remaining localization errors:

```text
chunk14_event2: true LINE23-24, predicted LINE22-35
chunk22_event4: true BUS7, predicted BUS12
```

Top-3 note:

- Top-3 was evaluated in candidate-ranker experiments.
- The promoted final RAW0001 prediction artifact is Top-1 only, so the final report stores Top-3 as not applicable for the promoted artifact.

### 7.4 Efficiency and Model Complexity

Reviewer runtime model:

```text
Entrypoint: python main.py --input-dir <RAW_FOLDER> --output <RAW_FOLDER>\predictions.csv
Trainable neural parameters: 0
Runtime model bundle: models_bus_agnostic
models_bus_agnostic metrics/config size: < 1 MB
Hardware used for validation: Windows 11 CPU-only, Intel64/AMD64, 22 logical CPUs
Runtime smoke test: RAW0001 prediction completed successfully
```

Legacy reproducibility bundle:

```text
models_submission_v2 + models_bus_agnostic size: about 414 MB
```

## 8. SIM Metrics

From the frozen selected model:

```text
SIM detector accuracy = 0.9693
SIM classifier accuracy = 0.9673
SIM classifier macro-F1 = 0.9598
SIM localizer Top-1 exact = 0.8524
```

RAW frozen baseline before agnostic localizer improvements:

```text
RAW detector accuracy = 1.0000
RAW classifier accuracy = 1.0000
RAW classifier macro-F1 = 1.0000
RAW localizer Top-1 exact = 0.6667
```

RAW after promoted agnostic localization improvements:

```text
RAW localizer Top-1 exact = 0.8333
```

## 9. Deliverables Checklist Against guidelines.pdf

Required executable code:

- Present: `main.py`.

Required output format:

- Present: `TIMESTAMP, Bus, Predicted_Event, Predicted_Location`.

Required detection metrics:

- Present: precision, recall, F1, FP/FN/TP/TN, false alarm rate.

Required classification metrics:

- Present: accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, full 9x9 confusion matrix.

Required localization metrics:

- Present: Top-1 exact accuracy, mean/median electrical distance, localization error table.

Required efficiency metrics:

- Present: trainable parameter count, model size, runtime entrypoint, hardware, window description.

Required report content:

- Present: architecture, preprocessing, feature engineering, training/validation approach, results, efficiency, and deliverable schema.

## 10. Final Package Contents

Final package:

```text
sgsma_2026_final_submission.zip
```

Important included files:

```text
main.py
Final.md
report.md
models_bus_agnostic/
models_bus_agnostic/guidelines_metrics.json
models_bus_agnostic/raw_detection_confusion_matrix.csv
models_bus_agnostic/raw_event_confusion_matrix.csv
models_bus_agnostic/raw_event_per_class_metrics.csv
models_bus_agnostic/raw_localization_errors.csv
src/features/pmu_discovery.py
src/models/bus_agnostic.py
src/data_factory/dynamic_feature_extractor_v3.py
src/models/localizer/hybrid.py
```

## 11. Validation Commands

Tests:

```powershell
python -m pytest -q
```

Last result:

```text
12 passed
```

Runtime smoke test:

```powershell
python main.py --input-dir data\RAW0001 --output workbench\final_runtime_check\predictions.csv
```

Result:

- Completed successfully.
- Detected PMU buses dynamically.
- Produced schema-compatible predictions.

