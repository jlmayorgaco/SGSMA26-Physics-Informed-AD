from __future__ import annotations

import numpy as np

from paper.experiments.causal_grouped_benchmark import (
    _flat_event_scores,
    _select_validation_threshold,
    _synthetic_dataset,
    assign_primary_splits,
    audit_primary_splits,
    build_scenario_records,
    extract_causal_features,
    load_candidate_universes,
)


class _ProbabilityModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.probabilities[: len(X)]


def test_line_universe_excludes_transformers() -> None:
    universes = load_candidate_universes()
    assert len(universes.lines) == 34
    assert "LINE10-32" not in universes.lines
    assert "LINE2-3" in universes.lines
    assert universes.generator_buses == (
        "BUS2",
        "BUS6",
        "BUS10",
        "BUS19",
        "BUS20",
        "BUS22",
        "BUS23",
        "BUS25",
        "BUS29",
        "BUS39",
    )
    assert len(universes.load_buses) == 18
    assert "BUS31" not in universes.load_buses
    assert "BUS39" in universes.load_buses


def test_primary_split_is_scenario_disjoint_and_target_complete() -> None:
    universes = load_candidate_universes()
    records = build_scenario_records(universes, replicas_per_target=5, smoke=True)
    assignments = assign_primary_splits(records, split_seed=91)
    audit_primary_splits(records, assignments)
    assert len(assignments) == len(records)
    assert set(assignments.values()) == {"train", "validation", "test"}


def test_features_do_not_read_future_samples() -> None:
    original = _synthetic_dataset()
    changed = _synthetic_dataset()
    cutoff = 40
    changed.observed_frequency_hz[cutoff:, 2] += 5.0
    original_features, _, _ = extract_causal_features(original, window_s=2.0)
    changed_features, _, _ = extract_causal_features(changed, window_s=2.0)
    np.testing.assert_array_equal(original_features[:cutoff], changed_features[:cutoff])
    assert not np.array_equal(original_features[cutoff:], changed_features[cutoff:])


def test_flat_event_threshold_uses_abnormal_probability_and_preserves_class() -> None:
    model = _ProbabilityModel(
        np.array(
            [
                [0.70, 0.20, 0.10],
                [0.35, 0.15, 0.50],
                [0.15, 0.70, 0.15],
            ]
        )
    )
    score, label = _flat_event_scores(model, np.zeros((3, 1)))
    np.testing.assert_allclose(score, [0.30, 0.65, 0.85])
    np.testing.assert_array_equal(label, [1, 2, 1])
    threshold = _select_validation_threshold(score, np.array([False, True, True]))
    assert threshold > score[0]
    np.testing.assert_array_equal(np.where(score >= threshold, label, 0), [0, 2, 1])
