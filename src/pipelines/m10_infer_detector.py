from __future__ import annotations

import argparse
from pathlib import Path
import json
from dataclasses import asdict

from src.detectors.configs import DetectorConfig
from src.detectors.data.ingestion import ScenarioRecord
from src.detectors.data.preprocessing import SharedPreprocessor
from src.detectors.pipeline_utils import build_batches_for_records, save_detection_artifacts
from src.detectors.runtime import HybridDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M10 inference pipeline for one scenario")
    p.add_argument("--scenario-dir", type=Path, required=True)
    p.add_argument("--scenario-id", type=str, default="INFER")
    p.add_argument("--model-path", type=Path, default=Path("output/detector_m10/detector_model.pkl"))
    p.add_argument("--output-root", type=Path, default=Path("output/detector_m10"))
    return p.parse_args()


def main() -> int:
    a = parse_args()
    detector = HybridDetector.load(a.model_path)
    rec = ScenarioRecord(scenario_id=a.scenario_id, scenario_dir=a.scenario_dir.resolve(), split="infer")
    preprocessor = SharedPreprocessor(detector.config.preprocessing)
    batch = build_batches_for_records([rec], config=detector.config, preprocessor=preprocessor, fit=True)
    artifacts = detector.evaluate(batch)
    save_detection_artifacts(a.output_root, f"infer_{a.scenario_id}", artifacts)
    summary = {
        "scenario_id": a.scenario_id,
        "scenario_dir": str(a.scenario_dir),
        "windows": int(len(batch.y)),
        "metrics_if_labels_exist": asdict(artifacts.metrics),
        "chunks": [asdict(c) for c in artifacts.event_chunks],
    }
    (a.output_root / f"infer_{a.scenario_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
