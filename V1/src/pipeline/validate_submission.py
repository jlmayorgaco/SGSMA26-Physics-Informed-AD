"""Validate SGSMA prediction CSVs before packaging."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.load_csv import PMU_BUSES
from src.pipeline.make_submission import verify_timestamps


def _parse_labels(text: str | None) -> set[int]:
    if not text:
        return set()
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def validate_predictions(
    pred_dir: Path | str,
    data_dir: Path | str,
    *,
    required_labels: set[int] | None = None,
) -> dict:
    """Validate prediction files and return a JSON-serializable report."""
    pred_dir = Path(pred_dir)
    data_dir = Path(data_dir)
    required_labels = required_labels or set()
    errors: list[str] = []
    bus_reports: dict[int, dict] = {}

    timestamp_ok = verify_timestamps(pred_dir, data_dir)
    all_labels: set[int] = set()
    expected_rows = None

    for bus in PMU_BUSES:
        src_path = data_dir / f"Bus{bus}_Competition_Data_nanmask.csv"
        pred_path = pred_dir / f"Bus{bus}_Predictions.csv"
        if not src_path.exists():
            errors.append(f"Missing source CSV for Bus{bus}: {src_path}")
            continue
        if not pred_path.exists():
            errors.append(f"Missing prediction CSV for Bus{bus}: {pred_path}")
            continue

        src_rows = len(pd.read_csv(src_path, usecols=["TIMESTAMP"]))
        pred = pd.read_csv(pred_path)
        expected_rows = src_rows if expected_rows is None else expected_rows
        required_cols = ["TIMESTAMP", "Predicted_Event", "Predicted_Location"]
        if list(pred.columns) != required_cols:
            errors.append(f"Bus{bus}: columns must be exactly {required_cols}, got {list(pred.columns)}")
        if len(pred) != src_rows:
            errors.append(f"Bus{bus}: row count {len(pred)} != source {src_rows}")

        if pred.isna().any().any():
            errors.append(f"Bus{bus}: prediction file contains NaN")

        labels = pd.to_numeric(pred["Predicted_Event"], errors="coerce").to_numpy()
        locs = pd.to_numeric(pred["Predicted_Location"], errors="coerce").to_numpy()
        if not np.all(np.isfinite(labels)):
            errors.append(f"Bus{bus}: non-numeric Predicted_Event values")
        if not np.all(np.isfinite(locs)):
            errors.append(f"Bus{bus}: non-numeric Predicted_Location values")

        labels_i = labels.astype(int)
        locs_i = locs.astype(int)
        if not np.all(labels_i == labels):
            errors.append(f"Bus{bus}: Predicted_Event must be integer-valued")
        if not np.all(locs_i == locs):
            errors.append(f"Bus{bus}: Predicted_Location must be integer-valued")
        bad_labels = sorted(set(labels_i.tolist()) - set(range(9)))
        if bad_labels:
            errors.append(f"Bus{bus}: invalid labels {bad_labels}")
        bad_locs = sorted({int(x) for x in locs_i if x != -1 and not (1 <= int(x) <= 39)})
        if bad_locs:
            errors.append(f"Bus{bus}: invalid locations {bad_locs}")

        counts = {int(k): int(v) for k, v in pred["Predicted_Event"].value_counts().sort_index().items()}
        all_labels.update(counts)
        bus_reports[bus] = {
            "rows": int(len(pred)),
            "timestamp_ok": bool(timestamp_ok.get(bus, False)),
            "label_counts": counts,
            "abnormal_rows": int((labels_i != 0).sum()),
        }
        if not timestamp_ok.get(bus, False):
            errors.append(f"Bus{bus}: timestamps do not match source CSV")

    combined_path = pred_dir / "submission.csv"
    if not combined_path.exists():
        errors.append(f"Missing combined submission CSV: {combined_path}")
    else:
        combined = pd.read_csv(combined_path)
        expected_combined_rows = len(PMU_BUSES) * int(expected_rows or 0)
        if len(combined) != expected_combined_rows:
            errors.append(
                f"submission.csv rows {len(combined)} != expected {expected_combined_rows}"
            )
        required_combined_cols = ["TIMESTAMP", "Bus", "Predicted_Event", "Predicted_Location"]
        if list(combined.columns) != required_combined_cols:
            errors.append(
                f"submission.csv columns must be exactly {required_combined_cols}, got {list(combined.columns)}"
            )

    missing_required = sorted(required_labels - all_labels)
    if missing_required:
        errors.append(f"Required labels missing from predictions: {missing_required}")

    return {
        "ok": not errors,
        "errors": errors,
        "prediction_dir": str(pred_dir),
        "data_dir": str(data_dir),
        "required_labels": sorted(required_labels),
        "observed_labels": sorted(all_labels),
        "bus_reports": bus_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, default=Path("predictions"))
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--required-labels", default="1,2,3,4,5,6,7")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = validate_predictions(
        args.pred,
        args.data,
        required_labels=_parse_labels(args.required_labels),
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
