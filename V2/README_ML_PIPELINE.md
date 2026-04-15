# V2 PMU Grid ML Pipeline

## Contract

The model uses only the eight PMU CSV streams defined by the guidelines:

`2, 5, 6, 10, 19, 22, 29, 39`

Synthetic scenarios may also write CSVs for the other 31 IEEE-39 buses. Those
files are hidden truth for training targets and metrics only. They are never
used as model input.

## Outputs

The trained pipeline predicts:

- event label: `0` through `8`
- event interval: `start_sec`, `end_sec`
- location: bus, line, or multi-bus target
- 39-bus state map: one predicted state label for every IEEE-39 bus
- plot artifacts: training metrics, label counts, SIM split, prediction
  timeline, and 39-bus state heatmap

## Commands

Generate raw-like synthetic scenarios:

```powershell
python V2\pipelines\run_pmu_pipeline.py generate --config V2\pipelines\make_synth_data_rawlike.json --n-scenarios 5000 --synthetic-dir data\synthetic_v2_rawlike --plot-first-n 25 --plot-buses all
```

The raw-like config keeps the competition contract: only buses `2, 5, 6, 10,
19, 22, 29, 39` are model inputs. The remaining 31 bus CSVs are hidden truth
for 39-bus state targets, localization review, and plots only. The generator
does not sample raw PMU waveforms.

Generate the next physical-focus dataset when we want more positive event
coverage without reusing raw waveforms:

```powershell
python V2\pipelines\run_pmu_pipeline.py generate --config V2\pipelines\make_synth_data_physical_focus.json --n-scenarios 5000 --synthetic-dir data\synthetic_v2_physical_focus --plot-first-n 25 --plot-buses all
```

This config inherits the raw-like calibration and topology, then increases
fault, line-outage, generation/load-change, dropout, bad-data, cyber-physical,
and multi-event coverage. The first scenarios force known hard cases such as
bus 7 faults, line 23-24, generator bus 2, bus 29 cyber cases, every PMU bus
dropout/bad-data, and all major phase-fault subtypes.

Backfill all-node plots for an already generated dataset without regenerating
CSVs:

```powershell
python V2\pipelines\generate_missing_plots.py --dataset-dir data\synthetic_v2_rawlike --plot-buses all
```

Train/test with a 70/30 SIM split:

```powershell
python V2\pipelines\run_pmu_pipeline.py train --synthetic-dir data\synthetic_v2_rawlike --output-dir models_rawlike --runs 1 --train-fraction 0.70
```

Train the staged physics-aware pipeline:

```powershell
python V2\pipelines\run_pmu_pipeline.py train-staged --synthetic-dir data\synthetic_v2_rawlike --output-dir models_staged_rawlike --detector-model lightgbm --event-model lightgbm --location-model lightgbm --state-model extratrees --runs 1 --train-fraction 0.70
```

The staged pipeline is the refactor target for the hackathon model: detector,
event classifier, location head, 39-bus state estimator, Ybus/KCL/Zbus/swing
feature layer, physics-aware localization table, and transparent temporal
post-processing. It still uses only the eight PMU CSV streams as features.

Benchmark several compact candidates from the same existing CSVs and copy the
winner to `models_benchmark_rawlike/pmu_grid_model.pkl`:

```powershell
python V2\pipelines\run_pmu_pipeline.py benchmark --synthetic-dir data\synthetic_v2_rawlike --output-dir models_benchmark_rawlike --models lightgbm,histgb,extratrees,randomforest,linear_sgd,mlp,graph_topology,physics_ybus --runs 1 --train-fraction 0.70
```

The graph and physics candidates still use only the 8 PMU CSV streams as
inputs. The graph candidate diffuses PMU disturbance features over IEEE-39
topology. The physics candidate reconstructs hidden-bus voltage phasors with a
Ybus/KCL harmonic layer, then feeds transparent residual features to the
classifier. The final benchmark winner is selected by model quality with a
small serialized-size penalty.

Generate and train in one command:

```powershell
python V2\pipelines\run_pmu_pipeline.py run-all --config V2\pipelines\make_synth_data_rawlike.json --n-scenarios 5000 --synthetic-dir data\synthetic_v2_rawlike --output-dir models_rawlike --runs 1 --train-fraction 0.70 --plot-first-n 25 --plot-buses all
```

For a full generation that writes review artifacts for every scenario during
generation, use `--plot-all --plot-buses all`. That creates all-node plots for
all 5,000 scenarios and can produce hundreds of thousands of PNGs, so the
recommended overnight path is the script below.

Infer from raw or synthetic PMU CSVs:

```powershell
python V2\pipelines\run_pmu_pipeline.py infer --model-path models_benchmark_rawlike\pmu_grid_model.pkl --input-dir data\raw --out models_benchmark_rawlike\raw_predictions.json
```

Infer with the staged physics-aware model:

```powershell
python V2\pipelines\run_pmu_pipeline.py infer-staged --model-path models_staged_rawlike\pmu_grid_model.pkl --input-dir data\raw --out models_staged_rawlike\raw_predictions.json
```

Evaluate raw predictions against visible raw `Event` labels:

```powershell
python V2\pipelines\evaluate_raw_predictions.py --predictions models_benchmark_rawlike\raw_predictions.json --raw-dir data\raw --out models_benchmark_rawlike\raw_predictions_eval.json
```

Run the whole overnight flow:

```powershell
powershell -ExecutionPolicy Bypass -File V2\pipelines\run_rawlike_overnight.ps1
```

Run the physical-focus overnight flow:

```powershell
powershell -ExecutionPolicy Bypass -File V2\pipelines\run_rawlike_overnight.ps1 -ConfigPath V2\pipelines\make_synth_data_physical_focus.json -SyntheticDir data\synthetic_v2_physical_focus -OutputDir models_benchmark_physical_focus
```

Pipeline data/model paths are resolved from the `V2` project directory. The
overnight script `-ConfigPath` accepts an absolute path or a repo-relative path.

To force plots for every generated scenario, add `-PlotAll` to the overnight
script command.

If the 5,000 CSV scenarios already exist and you only want to benchmark,
infer, and evaluate raw performance:

```powershell
powershell -ExecutionPolicy Bypass -File V2\pipelines\run_rawlike_overnight.ps1 -SkipGenerate -SkipCompleted
```

If a benchmark is interrupted after some model folders finish, resume it without
retraining completed models:

```powershell
python V2\pipelines\run_pmu_pipeline.py benchmark --synthetic-dir data\synthetic_v2_rawlike --output-dir models_benchmark_rawlike --models lightgbm,histgb,extratrees,randomforest,linear_sgd,mlp,graph_topology,physics_ybus --runs 1 --train-fraction 0.70 --skip-completed
```

## Plots

Training writes figures under:

```text
models/figures/
```

Model benchmarking writes:

```text
models_benchmark/pmu_model_benchmark.csv
models_benchmark/pmu_model_benchmark.json
models_benchmark/pmu_grid_model.pkl
```

Scenario review backfill writes:

```text
data/synthetic_v2/SIM_####/plots/*.png
data/synthetic_v2/SIM_####/engineering_review.md
data/synthetic_v2/plot_generation_summary.json
```

Inference writes figures under the prediction output folder:

```text
models/figures/<prediction_name>_timeline.png
models/figures/<prediction_name>_bus_state_heatmap.png
```
