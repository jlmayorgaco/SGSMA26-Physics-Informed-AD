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

Generate synthetic scenarios:

```powershell
python V2\pipelines\run_pmu_pipeline.py generate --n-scenarios 5000 --synthetic-dir data\synthetic_v2 --no-plots
```

Train/test with a 70/30 SIM split:

```powershell
python V2\pipelines\run_pmu_pipeline.py train --synthetic-dir data\synthetic_v2 --output-dir models --runs 5 --train-fraction 0.70
```

Generate and train in one command:

```powershell
python V2\pipelines\run_pmu_pipeline.py run-all --n-scenarios 5000 --synthetic-dir data\synthetic_v2 --output-dir models --runs 5 --train-fraction 0.70 --no-plots
```

Infer from raw or synthetic PMU CSVs:

```powershell
python V2\pipelines\run_pmu_pipeline.py infer --model-path models\pmu_grid_model.pkl --input-dir data\raw --out models\raw_predictions.json
```

Relative paths are resolved from the `V2` project directory.

## Plots

Training writes figures under:

```text
models/figures/
```

Inference writes figures under the prediction output folder:

```text
models/figures/<prediction_name>_timeline.png
models/figures/<prediction_name>_bus_state_heatmap.png
```
