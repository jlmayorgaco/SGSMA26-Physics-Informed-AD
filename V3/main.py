from src.physics.estimator_analysis import run_and_analyze_estimators
from src.physics.global_wls_angle_residual_ekf_estimator import GlobalWLSAngleResidualEKFConfig

SCENARIO_ID = "SIM_0001"
REPRESENTATION = "positive_sequence"

# Two hypotheses for the measurement-base experiment.
# Each maps to its own output directory so results are never overwritten.
EXPERIMENTS: dict[str, GlobalWLSAngleResidualEKFConfig] = {
    "nominal_base": GlobalWLSAngleResidualEKFConfig(
        representation=REPRESENTATION,
        n_modes=8,
        measurement_base_mode="nominal",
    ),
    "measurement_override_345": GlobalWLSAngleResidualEKFConfig(
        representation=REPRESENTATION,
        n_modes=8,
        measurement_base_mode="override_345",
    ),
}

all_experiment_artifacts: dict[str, dict] = {}

for experiment_tag, dynamic_config in EXPERIMENTS.items():
    output_root = f"artifacts/estimator_analysis/{experiment_tag}"

    print("=" * 80)
    print(f"EXPERIMENT: {experiment_tag}")
    print(f"  measurement_base_mode = {dynamic_config.measurement_base_mode!r}")
    print(f"  output_root           = {output_root}")
    print("=" * 80)

    artifacts = run_and_analyze_estimators(
        scenario_id=SCENARIO_ID,
        representation=REPRESENTATION,
        dynamic_config=dynamic_config,
        output_root=output_root,
    )

    all_experiment_artifacts[experiment_tag] = artifacts

    print(f"\nARTIFACTS GENERATED — {experiment_tag}")
    for subset_name, subset_artifacts in artifacts.items():
        print(f"\n  SUBSET: {subset_name}")
        print(f"  {subset_artifacts}")

print("\n" + "=" * 80)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 80)
for experiment_tag, artifacts in all_experiment_artifacts.items():
    print(f"\n  {experiment_tag}:")
    for subset_name in artifacts:
        comp_json = artifacts[subset_name]["comparison"]["comparison_json"]
        print(f"    {subset_name}: {comp_json}")
