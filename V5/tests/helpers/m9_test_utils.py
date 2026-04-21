from __future__ import annotations

from pathlib import Path

from src.simulation.m9.generator import generate_scenario, generate_template_matrix
from src.simulation.m9.validation import run_m9_validation_suite

RAW_PATH = Path("data/metadata/IEEE_39_Bus_Power_System.raw")
PMU_LOCATION_PATH = Path("data/metadata/PMUbus_ Location.txt")
REFERENCE_PMU_DIR = Path("data/RAW0001")


def generate_single(tmp_path: Path, template: str, scenario_id: str = "SIMTEST", seed: int = 7):
    scenarios_root = tmp_path / "scenarios"
    manifest = generate_scenario(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=scenarios_root,
        scenario_template=template,
        scenario_id=scenario_id,
        seed=seed,
        fps=30.0,
        use_andes=False,
        save_plots=False,
    )
    return manifest, scenarios_root / scenario_id


def generate_templates(tmp_path: Path):
    scenarios_root = tmp_path / "scenarios"
    manifests = generate_template_matrix(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=scenarios_root,
        seed=11,
        fps=30.0,
        use_andes=False,
        save_plots=False,
    )
    return manifests, scenarios_root


def generate_validation_suite(tmp_path: Path):
    return run_m9_validation_suite(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        seed=13,
        fps=30.0,
        use_andes=False,
        save_plots=False,
    )
