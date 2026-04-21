from __future__ import annotations

import pandas as pd

from src.detectors.shared.preprocessing.nan_masking import apply_nan_masking


def test_preprocessing_ablation_fill_modes_supported() -> None:
    frame = pd.DataFrame({"A": [1.0, None, 3.0], "B": [None, 2.0, None], "EVENT": [0, 0, 1], "DATA_PRESENT": [1.0, 0.0, 1.0]})
    for mode in ["ffill_bfill", "bfill_only", "ffill_limit1", "interpolate"]:
        out, mask = apply_nan_masking(frame, feature_columns=["A", "B"], fill_method=mode)
        assert out[["A", "B"]].isna().sum().sum() == 0
        assert {"A__is_nan", "B__is_nan"}.issubset(set(mask.columns))
