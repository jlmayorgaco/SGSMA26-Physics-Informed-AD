param(
    [int]$NScenarios = 5000,
    [string]$SyntheticDir = "data\synthetic_v2_rawlike",
    [string]$OutputDir = "models_benchmark_rawlike",
    [string]$RawDir = "data\raw",
    [string]$ConfigPath = "",
    [int]$Runs = 1,
    [double]$TrainFraction = 0.70,
    [int]$TrainSeed = 20260412,
    [int]$PlotFirstN = 25,
    [switch]$PlotAll,
    [switch]$SkipGenerate,
    [switch]$SkipBenchmark,
    [switch]$SkipRawEval,
    [switch]$SkipCompleted
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $ProjectRoot
$LogDir = Join-Path $ProjectRoot $OutputDir
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Models = "lightgbm,histgb,extratrees,randomforest,linear_sgd,mlp,graph_topology,physics_ybus"
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $Config = Join-Path $ProjectRoot "pipelines\make_synth_data_rawlike.json"
}
elseif ([System.IO.Path]::IsPathRooted($ConfigPath)) {
    $Config = $ConfigPath
}
else {
    $Config = Join-Path $RepoRoot $ConfigPath
}
$Pipeline = Join-Path $ProjectRoot "pipelines\run_pmu_pipeline.py"
$Evaluator = Join-Path $ProjectRoot "pipelines\evaluate_raw_predictions.py"
$ModelPath = Join-Path $OutputDir "pmu_grid_model.pkl"
$RawPredictions = Join-Path $OutputDir "raw_predictions.json"
$RawEval = Join-Path $OutputDir "raw_predictions_eval.json"

Push-Location $RepoRoot
try {
    if (-not $SkipGenerate) {
        $plotArgs = @("--plot-first-n", "$PlotFirstN", "--plot-buses", "all")
        if ($PlotAll) {
            $plotArgs = @("--plot-all", "--plot-buses", "all")
        }

        python $Pipeline generate `
            --config $Config `
            --n-scenarios $NScenarios `
            --synthetic-dir $SyntheticDir `
            @plotArgs `
            2>&1 | Tee-Object -FilePath (Join-Path $LogDir "01_generate_rawlike.log")
    }

    if (-not $SkipBenchmark) {
        $benchmarkArgs = @()
        if ($SkipCompleted) {
            $benchmarkArgs += "--skip-completed"
        }
        python $Pipeline benchmark `
            --synthetic-dir $SyntheticDir `
            --output-dir $OutputDir `
            --models $Models `
            --runs $Runs `
            --train-fraction $TrainFraction `
            --train-seed $TrainSeed `
            @benchmarkArgs `
            2>&1 | Tee-Object -FilePath (Join-Path $LogDir "02_benchmark_models.log")
    }

    if (-not $SkipRawEval) {
        python $Pipeline infer `
            --model-path $ModelPath `
            --input-dir $RawDir `
            --out $RawPredictions `
            2>&1 | Tee-Object -FilePath (Join-Path $LogDir "03_infer_raw.log")

        python $Evaluator `
            --predictions $RawPredictions `
            --raw-dir $RawDir `
            --out $RawEval `
            2>&1 | Tee-Object -FilePath (Join-Path $LogDir "04_evaluate_raw.log")
    }
}
finally {
    Pop-Location
}
