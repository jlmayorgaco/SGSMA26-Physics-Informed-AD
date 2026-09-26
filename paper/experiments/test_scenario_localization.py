from __future__ import annotations

import pandas as pd

from paper.experiments.summarize_scenario_localization import scenario_decisions


def test_scenario_vote_counts_empty_outputs_as_errors() -> None:
    frame = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "model_seed": [11] * 4,
            "scenario_id": ["s"] * 4,
            "true_event": [1] * 4,
            "true_physical": ["BUS2"] * 4,
            "physical_top3": ["BUS2|BUS3", None, "BUS3|BUS2", None],
        }
    )
    result = scenario_decisions(frame, "physical")
    assert result.loc[0, "prediction"] == "NONE"
    assert result.loc[0, "correct"] == 0
    assert result.loc[0, "prediction_present"] == 0


def test_scenario_vote_breaks_tie_by_first_top1_occurrence() -> None:
    frame = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "model_seed": [11] * 4,
            "scenario_id": ["s"] * 4,
            "true_event": [1] * 4,
            "true_physical": ["BUS3"] * 4,
            "physical_top3": ["BUS3", "BUS2", "BUS2", "BUS3"],
        }
    )
    result = scenario_decisions(frame, "physical")
    assert result.loc[0, "prediction"] == "BUS3"
    assert result.loc[0, "correct"] == 1


def test_scenario_vote_uses_only_rows_with_defined_truth() -> None:
    frame = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "model_seed": [11] * 4,
            "scenario_id": ["s"] * 4,
            "true_event": [0, 1, 1, 1],
            "true_physical": [None, "BUS2", "BUS2", "BUS2"],
            "physical_top3": ["BUS9", "BUS2|BUS3", "BUS2", "BUS3"],
        }
    )
    result = scenario_decisions(frame, "physical")
    assert result.loc[0, "prediction"] == "BUS2"
    assert result.loc[0, "correct"] == 1
