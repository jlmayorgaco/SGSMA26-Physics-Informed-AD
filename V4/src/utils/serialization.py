from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .filesystem import ensure_dir


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]

    return obj


def save_json(path: Path, payload: dict[str, Any], indent: int = 2) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=indent, ensure_ascii=False)


def save_dataframe_csv(path: Path, df: pd.DataFrame, index: bool = False) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=index)