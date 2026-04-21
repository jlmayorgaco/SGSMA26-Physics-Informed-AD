from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.domain.services.hybrid_event_detector import HybridEventDetector
from src.detectors.pipelines.common import build_frame_output, write_json
from src.detectors.training.datasets.detector_dataset_builder import DetectorDatasetBuilder, merge_detection_inputs
from src.detectors.training.datasets.detector_split_loader import SplitRecord


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M10 inference pipeline")
    p.add_argument("--scenario-dir", type=Path, required=True)
    p.add_argument("--scenario-id", type=str, default="INFER")
    p.add_argument("--model-path", type=Path, default=Path("output/detector_m10_2_ready/models/detector_model.pkl"))
    p.add_argument("--output-root", type=Path, default=Path("output/detector_m10_2_ready"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    detector = HybridEventDetector.load(args.model_path)
    builder = DetectorDatasetBuilder(detector.config if isinstance(detector.config, DetectorConfigV2) else DetectorConfigV2())
    if detector.preprocessing_state is not None:
        builder.normalization_state = detector.preprocessing_state  # type: ignore[assignment]
        builder.feature_columns = list(detector.feature_names)

    record = SplitRecord(
        scenario_id=args.scenario_id,
        scenario_dir=args.scenario_dir.resolve(),
        split="infer",
        template_name="",
        event_coarse=None,
        difficulty_level="",
        scenario_family="",
        seed_family="",
    )
    dataset = builder.build_for_records([record], fit=False)
    inputs = merge_detection_inputs(dataset.by_scenario.values(), scenario_id=args.scenario_id, split="infer")
    output = detector.predict(inputs)

    if "y_binary" in inputs.metadata.columns:
        y_true = inputs.metadata["y_binary"].to_numpy(dtype=int)
    else:
        y_true = np.zeros((inputs.x_windows.shape[0],), dtype=int)

    frame = build_frame_output(output, y_true, inputs.timestamps, inputs.metadata)
    frame_path = metrics_dir / f"infer_{args.scenario_id}_frame_predictions_v2.csv"
    chunk_path = metrics_dir / f"infer_{args.scenario_id}_chunks_v2.csv"
    frame.to_csv(frame_path, index=False)
    frame.to_csv(metrics_dir / f"infer_{args.scenario_id}_frame_predictions_v4.csv", index=False)
    pd.DataFrame(output.chunks).to_csv(chunk_path, index=False)
    pd.DataFrame(output.chunks).to_csv(metrics_dir / f"infer_{args.scenario_id}_chunks_v4.csv", index=False)

    # backward-compatible artifacts
    frame.to_csv(output_root / f"infer_{args.scenario_id}_frame_predictions.csv", index=False)
    pd.DataFrame(output.chunks).to_csv(output_root / f"infer_{args.scenario_id}_chunks.csv", index=False)

    write_json(
        metrics_dir / f"infer_{args.scenario_id}_summary_v2.json",
        {
            "scenario_id": args.scenario_id,
            "windows": int(inputs.x_windows.shape[0]),
            "stable_abnormal_windows": int(frame["y_pred_stable"].sum()) if not frame.empty else 0,
            "frame_predictions": str(frame_path),
            "chunks": str(chunk_path),
        },
    )
    write_json(
        metrics_dir / f"infer_{args.scenario_id}_summary_v4.json",
        {
            "scenario_id": args.scenario_id,
            "windows": int(inputs.x_windows.shape[0]),
            "stable_abnormal_windows": int(frame["y_pred_stable"].sum()) if not frame.empty else 0,
            "frame_predictions": str(metrics_dir / f"infer_{args.scenario_id}_frame_predictions_v4.csv"),
            "chunks": str(metrics_dir / f"infer_{args.scenario_id}_chunks_v4.csv"),
        },
    )
    write_json(
        output_root / f"infer_{args.scenario_id}_summary.json",
        {
            "scenario_id": args.scenario_id,
            "windows": int(inputs.x_windows.shape[0]),
            "frame_predictions": str(frame_path),
            "chunks": str(chunk_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
