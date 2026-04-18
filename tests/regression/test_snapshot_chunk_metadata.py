from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.infrastructure.legacy.m0_adapter import get_chunk_metadata


@pytest.mark.regression
def test_snapshot_chunk_metadata(snapshot_dir: Path) -> None:
    event_slice = pd.DataFrame({"Bus10": [0, 1], "Bus19": [0, 0]})
    actual = get_chunk_metadata(event_slice, start_t=0.0, end_t=1.0, order_idx=1, dominant_label=1)
    expected = json.loads((snapshot_dir / "snapshot_chunk_metadata.json").read_text(encoding="utf-8"))

    assert actual == expected
