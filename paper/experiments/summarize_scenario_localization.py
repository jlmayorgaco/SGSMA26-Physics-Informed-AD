"""Summarize scenario-level localization from saved row predictions.

Each abnormal scenario receives one physical and/or integrity decision.  The
decision is the modal row-level Top-1 candidate over rows for which the target
is defined.  Empty outputs participate in the vote as ``NONE`` and therefore
remain errors.  A tie is assigned to the candidate that first reached Top-1,
which preserves the temporal order of each scenario.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "paper"
    / "evidence"
    / "revised"
    / "causal_benchmark"
    / "target_complete"
    / "target_complete"
    / "primary"
    / "test_predictions.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "evidence" / "revised"


def _top1(value: object) -> str:
    if pd.isna(value) or str(value) == "":
        return "NONE"
    return str(value).split("|", 1)[0]


def _deterministic_mode(values: pd.Series) -> str:
    counts = values.value_counts(dropna=False)
    best = int(counts.max())
    tied = {str(value) for value, count in counts.items() if int(count) == best}
    return next(str(value) for value in values if str(value) in tied)


def scenario_decisions(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    truth_col = f"true_{target}"
    pred_col = f"{target}_top3"
    required = {"model", "model_seed", "scenario_id", "true_event", truth_col, pred_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for (model, seed, scenario), group in frame.groupby(
        ["model", "model_seed", "scenario_id"], sort=False
    ):
        valid = group[truth_col].notna() & group[truth_col].astype(str).ne("")
        if not bool(valid.any()):
            continue
        truths = sorted(group.loc[valid, truth_col].astype(str).unique())
        if len(truths) != 1:
            raise ValueError(f"{scenario} has multiple {target} truths: {truths}")
        votes = group.loc[valid, pred_col].map(_top1)
        prediction = _deterministic_mode(votes)
        events = sorted(group.loc[valid, "true_event"].astype(int).unique())
        if len(events) != 1:
            raise ValueError(f"{scenario} has multiple active event labels: {events}")
        rows.append(
            {
                "model": str(model),
                "model_seed": int(seed),
                "scenario_id": str(scenario),
                "event": int(events[0]),
                "target_type": target,
                "truth": truths[0],
                "prediction": prediction,
                "correct": int(prediction == truths[0]),
                "prediction_present": int(prediction != "NONE"),
            }
        )
    return pd.DataFrame(rows)


def summarize(predictions: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(predictions)
    decisions = pd.concat(
        [scenario_decisions(frame, "physical"), scenario_decisions(frame, "integrity")],
        ignore_index=True,
    )
    by_seed = (
        decisions.groupby(["model", "model_seed", "target_type"], as_index=False)
        .agg(
            scenario_n=("scenario_id", "nunique"),
            scenario_top1=("correct", "mean"),
            scenario_prediction_coverage=("prediction_present", "mean"),
        )
        .sort_values(["target_type", "model", "model_seed"], kind="stable")
    )
    summary = (
        by_seed.groupby(["model", "target_type"], as_index=False)
        .agg(
            scenario_n=("scenario_n", "mean"),
            scenario_top1_mean=("scenario_top1", "mean"),
            scenario_top1_std=("scenario_top1", lambda x: float(np.std(x, ddof=1))),
            scenario_prediction_coverage_mean=("scenario_prediction_coverage", "mean"),
            scenario_prediction_coverage_std=(
                "scenario_prediction_coverage", lambda x: float(np.std(x, ddof=1))
            ),
        )
        .sort_values(["target_type", "model"], kind="stable")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_dir / "scenario_localization_decisions.csv", index=False)
    by_seed.to_csv(output_dir / "scenario_localization_by_seed.csv", index=False)
    summary.to_csv(output_dir / "scenario_localization_summary.csv", index=False)
    return by_seed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    by_seed, summary = summarize(args.predictions.resolve(), args.output_dir.resolve())
    print(by_seed.to_string(index=False))
    print("\nAcross-seed summary")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
