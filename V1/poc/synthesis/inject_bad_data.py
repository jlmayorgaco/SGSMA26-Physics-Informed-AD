"""Artificial label-7 corruption injection for PMU measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd

from poc.schema import CHANNEL_SUFFIXES


def inject_bad_data(
    df: pd.DataFrame,
    *,
    bus: int,
    start_frame: int,
    duration_frames: int,
    mode: str = "spike",
    magnitude: float = 8.0,
) -> pd.DataFrame:
    """Inject spike or drift corruption while keeping DATA_PRESENT equal to one."""

    out = df.copy()
    stop = min(len(out), start_frame + duration_frames)
    frame = np.arange(stop - start_frame, dtype=float)
    for suffix in CHANNEL_SUFFIXES:
        col = f"BUS{bus}_{suffix}"
        if col not in out:
            continue
        vals = out.loc[start_frame : stop - 1, col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        scale = float(np.nanstd(finite)) if len(finite) > 1 else 1.0
        if scale < 1e-9:
            scale = max(1.0, abs(float(np.nanmean(finite))) * 0.01) if len(finite) else 1.0
        if mode == "drift":
            vals = vals + magnitude * scale * (frame / max(len(frame) - 1, 1))
        else:
            vals = vals.copy()
            vals[:: max(1, len(vals) // 5)] += magnitude * scale
        out.loc[start_frame : stop - 1, col] = vals
    out.loc[start_frame : stop - 1, "Event"] = 7
    return out

