"""Derive per-event recall and routed localization from retained test predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = (
    ROOT
    / "paper"
    / "evidence"
    / "revised"
    / "causal_benchmark"
    / "target_complete"
    / "target_complete"
    / "primary"
)
OUTPUT = ROOT / "paper" / "evidence" / "revised"


def _ranked_first(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split("|").str[0]


def _ranked_contains(truth: pd.Series, ranked: pd.Series) -> np.ndarray:
    expected = truth.fillna("").astype(str).to_numpy()
    candidates = ranked.fillna("").astype(str).str.split("|").to_list()
    return np.asarray([target in row[:3] for target, row in zip(expected, candidates)], dtype=bool)


def summarize(predictions: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(predictions, keep_default_na=False)
    rows: list[dict[str, object]] = []
    for (model, seed), group in frame.groupby(["model", "model_seed"], sort=True):
        for event in range(9):
            selected = group.loc[group["true_event"] == event]
            row: dict[str, object] = {
                "model": model,
                "model_seed": int(seed),
                "event": event,
                "support": int(len(selected)),
                "event_recall": float((selected["pred_event"] == event).mean()),
            }
            event_ok = selected["pred_event"].to_numpy() == event
            physical_first = _ranked_first(selected["physical_top3"]).to_numpy()
            integrity_first = _ranked_first(selected["integrity_top3"]).to_numpy()
            physical_truth = selected["true_physical"].astype(str).to_numpy()
            integrity_truth = selected["true_integrity"].astype(str).to_numpy()
            physical_ok = (physical_truth == "") | (physical_first == physical_truth)
            integrity_ok = (integrity_truth == "") | (integrity_first == integrity_truth)
            row["joint_exact"] = float(np.mean(event_ok & physical_ok & integrity_ok))
            for role in ("physical", "integrity"):
                eligible = selected.loc[selected[f"true_{role}"] != ""]
                row[f"{role}_support"] = int(len(eligible))
                if len(eligible):
                    first = _ranked_first(eligible[f"{role}_top3"])
                    row[f"{role}_top1"] = float(
                        (first.to_numpy() == eligible[f"true_{role}"].astype(str).to_numpy()).mean()
                    )
                    row[f"{role}_top3"] = float(
                        _ranked_contains(eligible[f"true_{role}"], eligible[f"{role}_top3"]).mean()
                    )
                else:
                    row[f"{role}_top1"] = float("nan")
                    row[f"{role}_top3"] = float("nan")
            rows.append(row)
    by_seed = pd.DataFrame(rows)
    value_columns = [
        "event_recall",
        "joint_exact",
        "physical_top1",
        "physical_top3",
        "integrity_top1",
        "integrity_top3",
    ]
    support_columns = ["support", "physical_support", "integrity_support"]
    aggregate_rows: list[dict[str, object]] = []
    for (model, event), group in by_seed.groupby(["model", "event"], sort=True):
        aggregate: dict[str, object] = {"model": model, "event": int(event)}
        for column in support_columns:
            aggregate[column] = int(group[column].iloc[0])
        for column in value_columns:
            aggregate[f"{column}_mean"] = float(group[column].mean())
            aggregate[f"{column}_std"] = float(group[column].std(ddof=1))
        aggregate_rows.append(aggregate)
    return by_seed, pd.DataFrame(aggregate_rows)


def main() -> None:
    by_seed, summary = summarize(BENCHMARK / "test_predictions.csv")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(OUTPUT / "per_event_metrics_by_seed.csv", index=False)
    summary.to_csv(OUTPUT / "per_event_metrics_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
