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

Backfill plots for an already generated dataset without regenerating CSVs:

```powershell
python V2\pipelines\generate_missing_plots.py --dataset-dir data\synthetic_v2 --plot-buses all
```

Train/test with a 70/30 SIM split:

```powershell
python V2\pipelines\run_pmu_pipeline.py train --synthetic-dir data\synthetic_v2 --output-dir models --runs 5 --train-fraction 0.70
```

Benchmark several compact candidates from the same existing CSVs and copy the
winner to `models_benchmark/pmu_grid_model.pkl`:

```powershell
python V2\pipelines\run_pmu_pipeline.py benchmark --synthetic-dir data\synthetic_v2 --output-dir models_benchmark --models lightgbm,histgb,extratrees --runs 3 --train-fraction 0.70
```

Generate and train in one command:

```powershell
python V2\pipelines\run_pmu_pipeline.py run-all --n-scenarios 5000 --synthetic-dir data\synthetic_v2 --output-dir models --runs 5 --train-fraction 0.70 --no-plots
```

For a future full generation that writes review artifacts during generation,
use `--plot-all --plot-buses all` instead of `--no-plots`.

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
