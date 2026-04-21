from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.shared.preprocessing.angle_features import add_angle_derived_features, wrap_degrees


def test_angle_wrap_safe_transform() -> None:
    arr = np.array([181.0, -181.0, 360.0, -360.0])
    wrapped = wrap_degrees(arr)
    assert np.allclose(wrapped, np.array([-179.0, 179.0, 0.0, 0.0]))


def test_angle_feature_columns_added() -> None:
    frame = pd.DataFrame({"BUS10_VA_ANG": [0.0, 90.0, 180.0]})
    out = add_angle_derived_features(frame)
    assert "BUS10_VA_ANG__sin" in out.columns
    assert "BUS10_VA_ANG__cos" in out.columns
    assert "BUS10_VA_ANG__wrapped" in out.columns

