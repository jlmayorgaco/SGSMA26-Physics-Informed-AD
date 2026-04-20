from __future__ import annotations

import argparse
from pathlib import Path
import json
from dataclasses import asdict

from src.detectors.configs import DetectorConfig
from src.detectors.data.preprocessing import SharedPreprocessor
from src.detectors.metrics import metrics_to_frame
from src.detectors.pipeline_utils import build_batches_for_records, load_records_from_split, save_detection_artifacts
from src.detectors.runtime import HybridDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M10 train hybrid detector (binary: Event 0 vs 1..8)")
    p.add_argument("--train-split", type=Path, default=Path("train_scenarios_v2.csv"))
    p.add_argument("--val-split", type=Path, default=Path("val_scenarios_v2.csv"))
    p.add_argument("--workspace-root", type=Path, default=Path("."))
    p.add_argument("--output-root", type=Path, default=Path("output/detector_m10"))
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--window-stride", type=int, default=16)
    p.add_argument("--decision-threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    config = DetectorConfig()
    config.window.size = int(a.window_size)
    config.window.stride = int(a.window_stride)
    config.fusion.decision_threshold = float(a.decision_threshold)
    config.model_path = a.output_root / "detector_model.pkl"
    config.report_dir = a.output_root

    train_records = load_records_from_split(a.train_split, workspace_root=a.workspace_root.resolve())
    val_records = load_records_from_split(a.val_split, workspace_root=a.workspace_root.resolve())
    preprocessor = SharedPreprocessor(config.preprocessing)
    train_batch = build_batches_for_records(train_records, config=config, preprocessor=preprocessor, fit=True)
    val_batch = build_batches_for_records(val_records, config=config, preprocessor=preprocessor, fit=False)

    detector = HybridDetector(config=config)
    detector.fit(train_batch)
    detector.save(config.model_path)

    train_artifacts = detector.evaluate(train_batch)
    val_artifacts = detector.evaluate(val_batch)
    save_detection_artifacts(a.output_root, "train", train_artifacts)
    save_detection_artifacts(a.output_root, "val", val_artifacts)
    metrics = {
        "train": asdict(train_artifacts.metrics),
        "val": asdict(val_artifacts.metrics),
        "train_windows": int(len(train_batch.y)),
        "val_windows": int(len(val_batch.y)),
        "model_path": str(config.model_path),
    }
    (a.output_root / "training_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
