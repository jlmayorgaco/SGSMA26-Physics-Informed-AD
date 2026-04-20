from __future__ import annotations

import argparse
from pathlib import Path
import json
from dataclasses import asdict

from src.detectors.configs import DetectorConfig
from src.detectors.data.preprocessing import SharedPreprocessor
from src.detectors.pipeline_utils import build_batches_for_records, load_records_from_split, save_detection_artifacts
from src.detectors.runtime import HybridDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M10 evaluate hybrid detector")
    p.add_argument("--split-csv", type=Path, default=Path("test_scenarios_v2.csv"))
    p.add_argument("--workspace-root", type=Path, default=Path("."))
    p.add_argument("--model-path", type=Path, default=Path("output/detector_m10/detector_model.pkl"))
    p.add_argument("--output-root", type=Path, default=Path("output/detector_m10"))
    p.add_argument("--run-name", type=str, default="test")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    records = load_records_from_split(a.split_csv, workspace_root=a.workspace_root.resolve())
    detector = HybridDetector.load(a.model_path)
    preprocessor = SharedPreprocessor(detector.config.preprocessing)
    # Rehydrate stored feature schema by doing a tiny internal fit from first record.
    batch = build_batches_for_records(records, config=detector.config, preprocessor=preprocessor, fit=True)
    artifacts = detector.evaluate(batch)
    save_detection_artifacts(a.output_root, a.run_name, artifacts)
    summary = {
        "split_csv": str(a.split_csv),
        "windows": int(len(batch.y)),
        "metrics": asdict(artifacts.metrics),
    }
    (a.output_root / f"{a.run_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
