"""Paired scenario-block bootstrap for the causal benchmark.

The forest-seed standard deviations in the main metrics describe estimator
randomness on one fixed split.  This script adds uncertainty over test
trajectories by resampling complete scenarios, stratified by event type, while
preserving the paired comparison between models.  Forest seeds are also
resampled as paired blocks.
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT = (
    REPO_ROOT / "paper" / "evidence" / "revised" / "paired_scenario_bootstrap.csv"
)


def _top1(series: pd.Series) -> np.ndarray:
    return series.fillna("").astype(str).str.split("|", regex=False).str[0].to_numpy()


def _scenario_stats(frame: pd.DataFrame) -> dict[str, float]:
    frame = frame.sort_values("time_s", kind="stable")
    truth = frame["true_event"].to_numpy(dtype=int)
    pred = frame["pred_event"].to_numpy(dtype=int)
    abnormal = truth != 0
    pred_abnormal = pred != 0
    false = (~abnormal) & pred_abnormal
    episodes = int(np.sum(false & ~np.r_[False, false[:-1]]))

    time_s = frame["time_s"].to_numpy(dtype=float)
    dt_s = np.diff(time_s, prepend=time_s[0])
    if len(dt_s) > 1:
        dt_s[0] = float(np.median(dt_s[1:]))

    physical_truth = frame["true_physical"].fillna("").astype(str).to_numpy()
    physical_mask = physical_truth != ""
    physical_pred = _top1(frame["physical_top3"])
    active_physical_pred = np.where(physical_pred[physical_mask] == "", "NONE", physical_pred[physical_mask])
    if np.any(physical_mask):
        values, counts = np.unique(active_physical_pred, return_counts=True)
        max_count = int(np.max(counts))
        tied = set(values[counts == max_count].astype(str))
        scenario_prediction = next(
            str(value) for value in active_physical_pred if str(value) in tied
        )
        scenario_truth_values = np.unique(physical_truth[physical_mask])
        if len(scenario_truth_values) != 1:
            raise ValueError(f"scenario has multiple physical truths: {scenario_truth_values.tolist()}")
        scenario_correct = int(scenario_prediction == str(scenario_truth_values[0]))
        scenario_n = 1
    else:
        scenario_correct = 0
        scenario_n = 0
    return {
        "tp": int(np.sum(abnormal & pred_abnormal)),
        "fp": int(np.sum((~abnormal) & pred_abnormal)),
        "fn": int(np.sum(abnormal & (~pred_abnormal))),
        "normal_minutes": float(np.sum(dt_s[~abnormal]) / 60.0),
        "false_alarm_episodes": episodes,
        "physical_n": int(np.sum(physical_mask)),
        "physical_correct": int(np.sum(physical_mask & (physical_pred == physical_truth))),
        "physical_scenario_n": scenario_n,
        "physical_scenario_correct": scenario_correct,
    }


def _metric(total: dict[str, float], name: str) -> float:
    if name == "detection_f1":
        denominator = 2.0 * total["tp"] + total["fp"] + total["fn"]
        return 2.0 * total["tp"] / denominator if denominator else np.nan
    if name == "false_alarm_episodes_per_normal_min":
        return total["false_alarm_episodes"] / total["normal_minutes"]
    if name == "physical_top1":
        return total["physical_correct"] / total["physical_n"]
    if name == "physical_scenario_top1":
        return total["physical_scenario_correct"] / total["physical_scenario_n"]
    raise KeyError(name)


def _sum_stats(
    lookup: dict[tuple[str, int, str], dict[str, float]],
    model: str,
    seeds: np.ndarray,
    scenarios: list[str],
) -> dict[str, float]:
    keys = (
        "tp", "fp", "fn", "normal_minutes", "false_alarm_episodes",
        "physical_n", "physical_correct", "physical_scenario_n", "physical_scenario_correct",
    )
    total = {key: 0.0 for key in keys}
    for seed in seeds:
        for scenario in scenarios:
            stats = lookup[(model, int(seed), scenario)]
            for key in keys:
                total[key] += stats[key]
    return total


def run_bootstrap(
    predictions: Path,
    output: Path,
    *,
    replicates: int = 5000,
    bootstrap_seed: int = 20260807,
) -> pd.DataFrame:
    frame = pd.read_csv(predictions, keep_default_na=False)
    frame["model_seed"] = frame["model_seed"].astype(int)
    frame["true_event"] = frame["true_event"].astype(int)
    frame["pred_event"] = frame["pred_event"].astype(int)

    lookup: dict[tuple[str, int, str], dict[str, float]] = {}
    event_by_scenario: dict[str, int] = {}
    for (model, seed, scenario), group in frame.groupby(
        ["model", "model_seed", "scenario_id"], sort=False
    ):
        lookup[(str(model), int(seed), str(scenario))] = _scenario_stats(group)
        event_by_scenario[str(scenario)] = int(group["true_event"].max())

    strata: dict[int, list[str]] = {}
    for scenario, event in event_by_scenario.items():
        strata.setdefault(event, []).append(scenario)
    for values in strata.values():
        values.sort()

    seeds = np.array(sorted(frame["model_seed"].unique()), dtype=int)
    comparisons = [
        ("typed-flat", "typed", "flat", "detection_f1"),
        ("typed-flat", "typed", "flat", "false_alarm_episodes_per_normal_min"),
        ("typed-flat", "typed", "flat", "physical_top1"),
        ("typed-flat", "typed", "flat", "physical_scenario_top1"),
        ("typed_topology-typed", "typed_topology", "typed", "physical_top1"),
        ("typed_topology-typed", "typed_topology", "typed", "physical_scenario_top1"),
    ]

    rng = np.random.default_rng(bootstrap_seed)
    samples: dict[tuple[str, str], list[float]] = {
        (comparison, metric): [] for comparison, _, _, metric in comparisons
    }
    for _ in range(replicates):
        sampled_scenarios: list[str] = []
        for event in sorted(strata):
            source = np.asarray(strata[event], dtype=object)
            sampled_scenarios.extend(
                rng.choice(source, size=len(source), replace=True).astype(str).tolist()
            )
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        totals: dict[str, dict[str, float]] = {}
        for model in sorted(frame["model"].unique()):
            totals[str(model)] = _sum_stats(
                lookup, str(model), sampled_seeds, sampled_scenarios
            )
        for comparison, left, right, metric in comparisons:
            samples[(comparison, metric)].append(
                _metric(totals[left], metric) - _metric(totals[right], metric)
            )

    full_scenarios = sorted(event_by_scenario)
    full_totals = {
        model: _sum_stats(lookup, model, seeds, full_scenarios)
        for model in sorted(frame["model"].unique())
    }
    rows: list[dict[str, float | int | str]] = []
    for comparison, left, right, metric in comparisons:
        values = np.asarray(samples[(comparison, metric)], dtype=float)
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "estimate": _metric(full_totals[left], metric)
                - _metric(full_totals[right], metric),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": bootstrap_seed,
                "scenario_blocks": len(full_scenarios),
                "model_seed_blocks": len(seeds),
            }
        )
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "status": "complete",
                "method": "paired event-stratified scenario-block and paired model-seed bootstrap",
                "predictions": str(predictions),
                "results": result.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    result = run_bootstrap(
        args.predictions.resolve(),
        args.output.resolve(),
        replicates=args.replicates,
        bootstrap_seed=args.seed,
    )
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
