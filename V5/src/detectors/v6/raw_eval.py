from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.detectors.v6.cyber_simple import CyberChunkDecision, CyberV6Detector


_EVENT_PATTERN = re.compile(r"event_(\d+)")


@dataclass(slots=True)
class ChunkEvalRow:
    chunk_dir: str
    true_label: int
    pred_label: int
    true_event5: int
    pred_event5: int
    true_event7: int
    pred_event7: int
    true_cyber: int
    pred_cyber: int
    missing_score: float
    missing_buses: str
    top_event7_bus: str
    top_event7_score: float
    top_event7_feature_hits: int


def _parse_event_label_from_name(name: str) -> int:
    match = _EVENT_PATTERN.search(name)
    if not match:
        return 0
    return int(match.group(1))


def _load_chunk_labels(m0_root: Path) -> dict[str, int]:
    metadata_path = m0_root / "chunks_metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels: dict[str, int] = {}
        for chunk in payload.get("chunks", []):
            chunk_dir = str(chunk.get("chunk_dir", "")).strip()
            if not chunk_dir:
                continue
            labels[chunk_dir] = int(chunk.get("label", _parse_event_label_from_name(chunk_dir)))
        return labels
    labels = {}
    chunks_root = m0_root / "chunks"
    for chunk_dir in sorted(path for path in chunks_root.iterdir() if path.is_dir()):
        labels[chunk_dir.name] = _parse_event_label_from_name(chunk_dir.name)
    return labels


def _binary_metrics(true_values: list[int], pred_values: list[int]) -> dict[str, float | int]:
    tp = sum(1 for t, p in zip(true_values, pred_values) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_values, pred_values) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_values, pred_values) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_values, pred_values) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _top_bus_feature_hits(decision: CyberChunkDecision) -> int:
    if not decision.event7.per_bus:
        return 0
    row = decision.event7.per_bus[0]
    return int(row.feature_hits)


def run_v6_cyber_raw_eval(m0_root: Path, output_root: Path) -> dict[str, Any]:
    chunks_root = m0_root / "chunks"
    if not chunks_root.exists() or not chunks_root.is_dir():
        raise FileNotFoundError(f"Missing chunks directory: {chunks_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    labels = _load_chunk_labels(m0_root)

    detector = CyberV6Detector()
    chunk_paths = sorted(path for path in chunks_root.iterdir() if path.is_dir())

    normal_chunks = []
    for chunk_path in chunk_paths:
        true_label = labels.get(chunk_path.name, _parse_event_label_from_name(chunk_path.name))
        if true_label != 0:
            continue
        normal_chunks.append(detector.load_chunk_frames(chunk_path))
    detector.fit_event7_thresholds(normal_chunks)

    eval_rows: list[ChunkEvalRow] = []
    label_confusion: dict[str, int] = {}
    for chunk_path in chunk_paths:
        true_label = labels.get(chunk_path.name, _parse_event_label_from_name(chunk_path.name))
        chunk_frames = detector.load_chunk_frames(chunk_path)
        decision = detector.classify_chunk(chunk_frames)
        true_event5 = int(true_label == 5)
        true_event7 = int(true_label == 7)
        pred_event5 = int(decision.predicted_label == 5)
        pred_event7 = int(decision.predicted_label == 7)
        true_cyber = int(true_label in {5, 7})
        pred_cyber = int(decision.predicted_label in {5, 7})
        row = ChunkEvalRow(
            chunk_dir=chunk_path.name,
            true_label=true_label,
            pred_label=decision.predicted_label,
            true_event5=true_event5,
            pred_event5=pred_event5,
            true_event7=true_event7,
            pred_event7=pred_event7,
            true_cyber=true_cyber,
            pred_cyber=pred_cyber,
            missing_score=float(decision.event5.chunk_score),
            missing_buses=";".join(decision.event5.missing_buses),
            top_event7_bus=decision.event7.top_bus or "",
            top_event7_score=float(decision.event7.top_score),
            top_event7_feature_hits=_top_bus_feature_hits(decision),
        )
        eval_rows.append(row)
        key = f"true_{true_label}_pred_{decision.predicted_label}"
        label_confusion[key] = label_confusion.get(key, 0) + 1

    table = pd.DataFrame([asdict(row) for row in eval_rows])
    table.to_csv(output_root / "v6_cyber_raw_predictions.csv", index=False)

    event5_metrics = _binary_metrics(table["true_event5"].tolist(), table["pred_event5"].tolist())
    event7_metrics = _binary_metrics(table["true_event7"].tolist(), table["pred_event7"].tolist())
    cyber_metrics = _binary_metrics(table["true_cyber"].tolist(), table["pred_cyber"].tolist())

    report = {
        "input": {"m0_root": str(m0_root.resolve()), "chunks_dir": str(chunks_root.resolve())},
        "thresholds": asdict(detector.thresholds),
        "summary": {
            "n_chunks": int(len(table)),
            "n_true_event5": int(table["true_event5"].sum()),
            "n_true_event7": int(table["true_event7"].sum()),
            "n_pred_event5": int(table["pred_event5"].sum()),
            "n_pred_event7": int(table["pred_event7"].sum()),
            "n_pred_cyber": int(table["pred_cyber"].sum()),
        },
        "metrics": {
            "event5_binary": event5_metrics,
            "event7_binary": event7_metrics,
            "cyber_binary_5_or_7": cyber_metrics,
        },
        "label_confusion_counts": label_confusion,
    }
    (output_root / "v6_cyber_raw_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# V6 Cyber RAW Evaluation",
        "",
        f"- Chunks evaluados: {report['summary']['n_chunks']}",
        f"- Threshold outlier_rate: {report['thresholds']['outlier_rate']:.6f}",
        f"- Threshold jump_rate: {report['thresholds']['jump_rate']:.6f}",
        f"- Threshold stuck_rate: {report['thresholds']['stuck_rate']:.6f}",
        f"- Threshold replay_rate: {report['thresholds']['replay_rate']:.6f}",
        "",
        "## Metricas",
        "",
        f"- Event5 precision/recall/f1: {event5_metrics['precision']:.3f} / {event5_metrics['recall']:.3f} / {event5_metrics['f1']:.3f}",
        f"- Event7 precision/recall/f1: {event7_metrics['precision']:.3f} / {event7_metrics['recall']:.3f} / {event7_metrics['f1']:.3f}",
        f"- Cyber(5|7) precision/recall/f1: {cyber_metrics['precision']:.3f} / {cyber_metrics['recall']:.3f} / {cyber_metrics['f1']:.3f}",
        "",
        "## Conteos True/Pred",
        "",
    ]
    for key in sorted(label_confusion):
        md_lines.append(f"- {key}: {label_confusion[key]}")
    (output_root / "v6_cyber_raw_eval.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V6 cyber-only evaluation over M0 chunked RAW data.")
    parser.add_argument(
        "--m0-root",
        type=Path,
        default=Path("output/M0_RAW0001_NEWARCH"),
        help="M0 output root containing chunks_metadata.json and chunks/ directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/detector_m10/v6_cyber_raw_eval"),
        help="Directory where reports and predictions will be written.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = run_v6_cyber_raw_eval(m0_root=args.m0_root, output_root=args.output_root)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
