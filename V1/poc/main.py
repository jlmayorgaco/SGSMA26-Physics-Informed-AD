"""Orchestrator for the SGSMA 2026 post-contest comparative POC.

This file intentionally contains orchestration only.  Estimators, classifiers,
the shared chi-squared detector, the cosine localizer, and ablation/reporting
live in sibling packages under ``poc/`` so each component can be swapped without
touching the competition code in ``src/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from poc.classifiers import CLASSIFIERS
from poc.estimators import ESTIMATORS
from poc.evaluation.ablation import run_full_ablation
from poc.schema import Scenario, set_global_seed
from poc.synthesis.andes_runner import AndesRunner

LOG = logging.getLogger("poc.main")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "poc" / "config.yaml"


class NanHandler:
    """NaN handling facade.

    Method: per-channel median fill with DATA_PRESENT left intact.  This is the
    physics-first baseline used before estimator-specific covariance inflation.
    Parameter count: zero.
    """

    def transform(self, x):
        import numpy as np

        med = np.nanmedian(x, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        return np.where(np.isnan(x), med[None, :], x)


class FeatureScaler:
    """Robust z-score scaler using median/MAD from normal baseline.

    Equation: ``z = (x - median) / (1.4826*MAD + eps)``.  Parameter count equals
    two fixed calibration vectors but is reported in the estimator that owns it.
    """

    def fit(self, x):
        import numpy as np

        self.center_ = np.nanmedian(x, axis=0)
        mad = np.nanmedian(np.abs(x - self.center_[None, :]), axis=0)
        self.scale_ = np.where(mad > 1e-9, 1.4826 * mad, 1.0)
        return self

    def transform(self, x):
        return (x - self.center_[None, :]) / self.scale_[None, :]


class MultiPMUFusion:
    """PMU fusion facade using the fixed competition bus order.

    Method: concatenate the eight 14-channel PMUs into a 112-dimensional vector.
    Parameter count: zero.
    """


class Detector:
    """Shared detector facade backed by ``poc.detection.chi2_detector``."""


class Classificator:
    """Classifier facade backed by the registry in ``poc.classifiers``."""


class Localizer:
    """Topology localizer facade backed by ``poc.localization.cosine_localizer``."""


class Metrics:
    """Metrics facade backed by ``poc.evaluation.metrics``."""


class SubmissionWriter:
    """POC placeholder for writing competition-schema predictions.

    The journal ablation does not overwrite the validated ``src/`` submission
    path, so this class remains a reporting adapter rather than a producer of the
    April 15 contest artifact.
    """


class Plotter:
    """Plotting facade backed by ``poc.evaluation.plots``."""


def load_config(path: Path) -> dict:
    """Load YAML config and normalize path strings relative to repo root."""

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def generate_one_fault(cfg: dict) -> list[Path]:
    """Generate one declarative 3LG fault scenario and verify CSV schema."""

    paths = cfg["paths"]
    runner = AndesRunner(
        raw_data_dir=ROOT / paths["raw_data"],
        output_dir=ROOT / paths["poc_synth"],
        seed=int(cfg["seed"]),
    )
    scenario = Scenario(
        scenario_id="fault_smoke_0000",
        label=1,
        event_bus=39,
        t_event=5.0,
        duration=5.0 / 60.0,
        category="faults",
        params={"fault_impedance": 0.01, "clearing_cycles": 5},
    )
    out_dir = runner.generate_scenario(scenario)
    runner.verify_with_src_loader(out_dir)
    return sorted(out_dir.glob("Bus*_Competition_Data_nanmask.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="SGSMA 2026 comparative POC")
    parser.add_argument(
        "--mode",
        choices=["generate-one-fault", "smoke-ablation", "full-ablation"],
        default="smoke-ablation",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    )

    cfg = load_config(args.config)
    set_global_seed(int(cfg["seed"]))
    LOG.info("Loaded %d estimators and %d classifiers", len(ESTIMATORS), len(CLASSIFIERS))

    if args.mode == "generate-one-fault":
        paths = generate_one_fault(cfg)
        LOG.info("Generated %d bus CSVs under %s", len(paths), paths[0].parent if paths else "n/a")
        return

    run_full_ablation(
        cfg=cfg,
        repo_root=ROOT,
        smoke=(args.mode == "smoke-ablation"),
    )


if __name__ == "__main__":
    main()

