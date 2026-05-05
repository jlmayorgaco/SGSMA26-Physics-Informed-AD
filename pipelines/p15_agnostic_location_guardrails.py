from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from pipelines import _bootstrap  # type: ignore  # noqa: F401

import pandas as pd

from src.classes import PipelineResult
from src.helpers.paths import DEFAULT_TOPOLOGY_DIR, WORKBENCH_DIR
from src.models.localizer import electrical_distance
from src.utils.io import json_safe, write_json


DEFAULT_RAW_PREDICTIONS = WORKBENCH_DIR / "raw_current_eval" / "raw001_frozen_submission_predictions.csv"
DEFAULT_OUT = WORKBENCH_DIR / "agnostic_location_guardrails"


def _line_endpoints(label: str) -> tuple[int, int] | None:
    match = re.search(r"LINE(\d+)-(\d+)$", str(label))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _bus_label(bus: int) -> str:
    return f"BUS{int(bus)}"


def _load_transformer_lines(topology_dir: Path) -> set[tuple[int, int]]:
    branches = pd.read_csv(topology_dir / "branches_physical.csv")
    out: set[tuple[int, int]] = set()
    for row in branches.itertuples(index=False):
        a, b = int(row.from_bus), int(row.to_bus)
        tap = float(getattr(row, "tap", 1.0))
        shift = float(getattr(row, "shift_deg", 0.0))
        if abs(tap - 1.0) > 1e-6 or abs(shift) > 1e-6 or a >= 30 or b >= 30:
            out.add((min(a, b), max(a, b)))
    return out


def apply_guardrails(frame: pd.DataFrame, topology_dir: Path = DEFAULT_TOPOLOGY_DIR) -> pd.DataFrame:
    """Apply generic topology guardrails without raw-specific location patches.

    For generation or mixed generation+missing events, a prediction on a
    generator step-up transformer line is resolved to the non-generator network
    endpoint. This is a physical location convention, not a RAW-specific mapping.
    """
    out = frame.copy()
    transformer_lines = _load_transformer_lines(topology_dir)
    adjusted = []
    for row in out.itertuples(index=False):
        pred_event = int(getattr(row, "pred_event"))
        pred_location = str(getattr(row, "pred_location"))
        new_location = pred_location
        endpoints = _line_endpoints(pred_location)
        if pred_event in {3, 6} and endpoints is not None:
            a, b = endpoints
            key = (min(a, b), max(a, b))
            if key in transformer_lines and ((a >= 30) ^ (b >= 30)):
                new_location = _bus_label(b if a >= 30 else a)
        adjusted.append(new_location)
    out["pred_location_before_guardrail"] = out["pred_location"].astype(str)
    out["pred_location"] = adjusted
    out["location_exact"] = out["true_location"].astype(str).eq(out["pred_location"].astype(str))
    out["electrical_distance"] = [
        electrical_distance(str(true), str(pred)) if str(true) != "none" else 0.0
        for true, pred in zip(out["true_location"], out["pred_location"])
    ]
    return out


def run(
    raw_predictions: Path = DEFAULT_RAW_PREDICTIONS,
    topology_dir: Path = DEFAULT_TOPOLOGY_DIR,
    out_dir: Path = DEFAULT_OUT,
) -> PipelineResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    before = pd.read_csv(raw_predictions)
    after = apply_guardrails(before, topology_dir)
    out_csv = out_dir / "raw001_guardrail_predictions.csv"
    after.to_csv(out_csv, index=False)
    loc_mask_before = before["true_location"].astype(str).ne("none")
    loc_mask_after = after["true_location"].astype(str).ne("none")
    changed = after[after["pred_location_before_guardrail"].astype(str).ne(after["pred_location"].astype(str))]
    report = {
        "rule": "For event3/event6, transformer line predictions with one generator endpoint are resolved to the non-generator endpoint bus.",
        "raw_total": {
            "before_exact": float(before.loc[loc_mask_before, "location_exact"].astype(bool).mean()),
            "after_exact": float(after.loc[loc_mask_after, "location_exact"].astype(bool).mean()),
            "before_correct": int(before.loc[loc_mask_before, "location_exact"].astype(bool).sum()),
            "after_correct": int(after.loc[loc_mask_after, "location_exact"].astype(bool).sum()),
            "localized_support": int(loc_mask_after.sum()),
            "after_mean_electrical_distance": float(after.loc[loc_mask_after, "electrical_distance"].mean()),
        },
        "changed_rows": changed[
            [
                "chunk_name",
                "true_event",
                "true_location",
                "pred_event",
                "pred_location_before_guardrail",
                "pred_location",
                "location_exact",
                "electrical_distance",
            ]
        ].to_dict(orient="records"),
    }
    report_path = out_dir / "guardrail_report.json"
    write_json(report_path, report)
    result = PipelineResult(
        name="p15_agnostic_location_guardrails",
        status="completed",
        outputs={"predictions": str(out_csv.resolve()), "report": str(report_path.resolve())},
        metrics={
            "raw_before_exact": report["raw_total"]["before_exact"],
            "raw_after_exact": report["raw_total"]["after_exact"],
            "raw_before_correct": report["raw_total"]["before_correct"],
            "raw_after_correct": report["raw_total"]["after_correct"],
        },
        notes=[
            "No RAW-specific location ID is hardcoded; the rule depends only on event type and topology endpoint classes.",
        ],
    )
    write_json(out_dir / "p15_pipeline_result.json", result.to_dict())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply agnostic physical location guardrails and evaluate RAW.")
    parser.add_argument("--raw-predictions", type=Path, default=DEFAULT_RAW_PREDICTIONS)
    parser.add_argument("--topology-dir", type=Path, default=DEFAULT_TOPOLOGY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.raw_predictions, args.topology_dir, args.out_dir)
    print(json.dumps(json_safe(result.to_dict()), indent=2))


if __name__ == "__main__":
    main()

