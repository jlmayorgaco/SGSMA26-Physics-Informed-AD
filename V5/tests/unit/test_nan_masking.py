from __future__ import annotations

import pandas as pd

from src.detectors.shared.preprocessing.nan_masking import apply_nan_masking


def test_nan_masking_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "A": [1.0, None, 3.0],
            "B": [None, None, 4.0],
            "EVENT": [0, 0, 1],
            "DATA_PRESENT": [1, 0, 1],
        }
    )
    clean, mask = apply_nan_masking(frame, feature_columns=["A", "B"])
    assert clean["A"].isna().sum() == 0
    assert clean["B"].isna().sum() == 0
    assert list(mask.columns) == ["A__is_nan", "B__is_nan"]
    assert int(mask["A__is_nan"].sum()) == 1
    assert int(mask["B__is_nan"].sum()) == 2

