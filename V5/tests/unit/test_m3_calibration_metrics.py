from __future__ import annotations

import numpy as np

from src.calibration.calibration_metrics import chunk_stability_penalty, composite_score


def test_composite_score_current_mag_finite() -> None:
    s = composite_score(0.1, 0.1, 0.1, 0.1, "current_mag")
    assert np.isfinite(s)


def test_composite_score_frequency_finite() -> None:
    s = composite_score(0.1, 0.1, 0.1, 0.1, "frequency")
    assert np.isfinite(s)


def test_chunk_stability_penalty_zero_for_single_chunk() -> None:
    chunks = [{"df": {"BUS10_VA_MAG": np.array([1.0, 2.0, 3.0])}}]
    spec = {"support_status": "supported_direct"}
    sim = np.array([1.0, 2.0, 3.0])
    # using dict-like df for this simple case would fail; use pandas DataFrame
    import pandas as pd

    chunks = [{"df": pd.DataFrame({"BUS10_VA_MAG": [1.0, 2.0, 3.0]})}]
    assert chunk_stability_penalty(chunks, "10", "VA_MAG", sim, spec) == 0.0


def test_chunk_stability_penalty_finite_for_multiple_chunks() -> None:
    import pandas as pd

    chunks = [
        {"df": pd.DataFrame({"BUS10_VA_MAG": [1.0, 2.0, 3.0]})},
        {"df": pd.DataFrame({"BUS10_VA_MAG": [1.1, 2.1, 3.1]})},
    ]
    spec = {"support_status": "supported_direct"}
    sim = np.array([1.0, 2.0, 3.0])
    assert np.isfinite(chunk_stability_penalty(chunks, "10", "VA_MAG", sim, spec))
