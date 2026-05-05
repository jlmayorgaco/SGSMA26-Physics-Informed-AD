# Final SGSMA 2026 Method Notes

## Submission Status

This repository is organized around a bus-agnostic SGSMA 2026 inference path. The reviewer entrypoint is:

```powershell
python main.py --input-dir data\RAW0001
```

For any future `RAW0002` folder, the intended call is the same:

```powershell
python main.py --input-dir data\RAW0002 --output data\RAW0002\predictions.csv
```

The required output schema from `guidelines.pdf` is:

```text
TIMESTAMP, Bus, Predicted_Event, Predicted_Location
```

The reviewer entrypoint now applies the validated ExtraTrees/hybrid ML bundle in `models/` over fixed 30 s windows for all inputs. The bus-agnostic physics runtime in `models_bus_agnostic/` is retained only as a fallback if the ML bundle is unavailable. PMU buses are inferred from `Bus*.csv` filenames and feature columns at runtime.

## Design Principle

The model may know which bus a measurement comes from in the current network instance, but it must not learn that a specific bus ID is always special. Bus IDs are used only as topology coordinates for the current case:

- Allowed: candidate-to-PMU electrical distance, candidate degree, line endpoints, load/generator/terminal type, observed PMU set, diffusion mismatch.
- Not allowed: fixed features such as `is_BUS29`, `strongest_is_BUS39`, or rules assuming that RAW0002 contains BUS29/BUS39.

This is implemented in `src/features/pmu_discovery.py`, `src/models/bus_agnostic.py`, `src/data_factory/dynamic_feature_extractor_v3.py`, and `src/models/localizer/hybrid.py`.

## Architecture

The final architecture has three layers:

1. Runtime ML inference
   - Applies the validated ExtraTrees detector/classifier/localizer bundle in fixed 30 s windows.
   - Broadcasts each window prediction to the timestamp-aligned output rows in that window.
   - Reads arbitrary `Bus*.csv` files.
   - Infers observed PMU buses dynamically.
   - Extracts robust per-PMU and global features.
   - Emits row-aligned predictions with `TIMESTAMP` and `Bus`.

2. Physics-informed feature/ranking layer
   - Uses IEEE-39 topology, branch data, and Zbus effective distances.
   - Converts bus identities into relative candidate features.
   - Scores candidates by compatibility between observed PMU severity and expected electrical diffusion.

3. Validation and promotion layer
   - RAW0001 is used only as a local validation benchmark.
   - Candidate improvements are promoted only if they are general, bus-agnostic, and do not rely on chunk IDs or fixed bus IDs.
   - P15/P16 were promoted for RAW analysis; P14/P17 were kept as experiments because they did not improve final Top-1.

## Feature Families

### Data Quality Features

Computed per PMU and globally:

- `DATA_PRESENT` fraction.
- NaN fraction across voltage/current/frequency/ROCOF channels.
- Maximum timestamp gap ratio.
- Missing-run severity.

Electrical/signal rationale: communication loss and corrupted PMU streams are not physical network events. They should be detected as data anomalies (`event5`/`event7`) without forcing a bus or line disturbance.

### Robust PMU Deviation Features

For each PMU signal, the extractor computes a robust baseline from the early/pre-event segment:

```text
z(t) = |x(t) - median(pre)| / robust_scale(pre)
robust_scale = 1.4826 * MAD
```

Features include max absolute robust z-score, span, derivative max, early/mid/late deltas, rolling energy, rolling slope, and peak timing.

Electrical rationale:

- Voltage magnitude drops and current magnitude spikes indicate faults.
- Sustained voltage/current shifts indicate load or topology changes.
- Frequency and ROCOF deviations indicate generation/load imbalance.
- Angle discontinuities and inter-PMU phase changes indicate topology or power-flow redistribution.

### Symmetrical Components

Three-phase phasors are transformed into zero, positive, and negative sequence components:

```text
V0 = (Va + Vb + Vc) / 3
V1 = (Va + a Vb + a^2 Vc) / 3
V2 = (Va + a^2 Vb + a Vc) / 3
```

Electrical rationale:

- Balanced operation has dominant positive sequence.
- Faults and phase imbalance increase zero/negative sequence energy.
- This helps separate physical faults from simple data spikes.

### P/Q Proxy Features

Per phase:

```text
S_proxy = V * conj(I)
P_proxy = Re(S_proxy)
Q_proxy = Im(S_proxy)
```

Features are deltas, spans, derivatives, and robust z-scores of P/Q proxies.

Electrical rationale:

- Load changes produce sustained active/reactive power shifts.
- Generation trips produce active-power imbalance and frequency response.
- Line outages redistribute apparent power across nearby PMUs.

### Graph Signal Processing Features

The PMU severity vector is treated as a graph signal over the observed PMU set.

Features:

- PMU spread and weighted graph total variation.
- Pairwise differences weighted by inverse Zbus distance.
- Time-to-peak spread across PMUs.
- Diffusion cosine between observed severity and expected candidate influence.
- Diffusion residual norm.
- Candidate weighted distance to severity.

Electrical rationale: disturbances propagate through electrical coupling, not numeric bus order. Zbus distance gives a topology-aware distance, so a candidate is plausible if its expected influence pattern matches the observed PMU severity pattern.

### Candidate-Relative Localization Features

For each candidate bus/line:

- Candidate type: bus, line, load-class bus, generator bus, transformer/generator terminal.
- Candidate degree.
- Minimum/mean/max Zbus distance to observed PMUs.
- Severity-weighted candidate-to-PMU distance.
- Inverse distance score.
- Diffusion cosine/residual over multiple decay constants.
- Severity entropy and top1-top2 severity margin.
- Line metadata for line candidates: impedance, X/R proxy, tap/transformer-like flags, endpoint degrees.

No candidate feature column encodes a specific bus or line ID.

## Models and Promotion Decisions

### Detector

Task: normal vs abnormal.

Inputs: order-invariant global PMU features, robust signal deviations, missing-data features, and physical thresholds.

Current RAW0001 result:

```text
Abnormal precision: 1.0000
Abnormal recall:    1.0000
Abnormal F1:        1.0000
Accuracy:           1.0000
```

### Event Classifier

Task: classify labels 0-8.

Inputs: global feature table and event-specific signal families.

Current RAW0001 result:

```text
Accuracy:    1.0000
Macro-F1:    1.0000
Weighted-F1: 1.0000
```

RAW0001 confusion matrix, labels 0-8:

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

### Localizer

Task: predict bus, line, or PMU location.

Final RAW0001 result after agnostic guardrail/ranker validation:

```text
Top-1 exact:              0.8333 = 10/12
Mean electrical distance: 0.003214
```

RAW0001 localization progression:

```text
Frozen baseline:      66.67% = 8/12
P15 topology guard:   75.00% = 9/12
P15 + P16 ranker:     83.33% = 10/12
```

Remaining RAW0001 localization misses:

```text
chunk14_event2: true LINE23-24, predicted LINE22-35
chunk22_event4: true BUS7, predicted BUS12
```

P14 event2 and P17 event4 were evaluated but not promoted because they did not improve RAW Top-1 in a defensible, general way.

## Training Data and Validation

The repository retains the 5000-scenario simulation set:

```text
workbench/scenarios/sgsma_generated
workbench/simulated/sgsma_generated
workbench/features/sgsma_generated
```

The training strategy used group/scenario-aware splits to avoid leakage across windows from the same simulated event. Placement-augmented experiments randomly sampled observed PMU subsets from available bus streams to test RAW0002-style PMU changes.

RAW0001 is used only for local reporting and promotion checks. The final entrypoint does not hardcode RAW0001 PMU placement.

## Efficiency

Reviewer-facing runtime path:

- Active ML bundle: ExtraTrees detector/classifier/localizer in `models/`.
- Physics fallback: `models_bus_agnostic`, used only if the ML bundle is unavailable.
- Required runtime dependencies: pandas, numpy, scipy, scikit-learn, joblib, matplotlib.
- Windowing: per-timestamp robust features plus short rolling/dynamic features for training/evaluation pipelines; runtime prediction preserves original timestamp alignment.
- Hardware used for validation: Windows 11, Intel64/AMD64 CPU, 22 logical CPUs, CPU-only.

Legacy sklearn model bundle kept for reproducibility:

```text
models_submission_v2 + models_bus_agnostic size: about 414 MB
```

## Guideline Checklist

Required by `guidelines.pdf`:

- Executable code: `main.py`.
- Separate prediction CSV schema: `TIMESTAMP, Bus, Predicted_Event, Predicted_Location`.
- Detection metrics: reported above.
- Classification macro-F1, weighted-F1, per-class/confusion matrix: reported above.
- Localization Top-1 and electrical distance: reported above.
- Efficiency/model size: reported above.
- Leakage prevention: scenario/group-aware split and RAW0001 kept as validation only.
- Technical report source: `report/ieee_method_report`.

## Current Final Artifacts

Keep:

```text
main.py
models_bus_agnostic/
models_submission_v2/
src/
pipelines/
tests/
report/
data/topology/
data/metadata/
workbench/scenarios/sgsma_generated/
workbench/simulated/sgsma_generated/
workbench/features/sgsma_generated/
workbench/raw_current_eval/
workbench/event3_generation_ranker_dynamic_pmus_check/
sgsma_2026_final_submission.zip
```

Generated smoke/failed-experiment directories are not required for the final package and can be removed after their reports have been summarized here.
