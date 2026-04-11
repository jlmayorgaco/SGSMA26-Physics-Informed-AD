"""NaN/missing-data handling for the UKF measurement update.

Per CLAUDE.md §3: when DATA_PRESENT == 0 for a PMU bus, inflate the R
rows and columns corresponding to that bus's channels to 1e12 so the
update step effectively ignores those measurements.

The function is called once per timestep and is intentionally cheap:
it copies R (or masks it in-place on a pre-allocated buffer) and returns
a view/copy suitable for passing to ukf.update().
"""
from __future__ import annotations

import numpy as np

from src.io.load_csv import PMU_BUSES  # [2, 5, 6, 10, 19, 22, 29, 39]

# Per-bus channel slice start index in the flat measurement vector.
# E.g., if N_CHANNELS_PER_BUS = 4, bus index 0 → channels 0:4, bus index 1 → 4:8, …
R_INFLATE_VALUE = 1e12


def inflate_R(
    R_base: np.ndarray,
    data_present: dict[int, bool],
    n_channels_per_bus: int,
) -> np.ndarray:
    """Return a modified R matrix with inflated rows/cols for missing PMU buses.

    Args:
        R_base: (n_z, n_z) base measurement noise covariance (diagonal usually).
        data_present: mapping {bus_number: True/False}, e.g. {2: True, 29: False, …}
        n_channels_per_bus: number of channels per bus in the flat measurement vector.

    Returns:
        R_eff: (n_z, n_z) modified R; missing bus rows/cols set to R_INFLATE_VALUE.
               Diagonal-off-diagonal structure is preserved — only diagonal inflation
               is needed when R_base is diagonal.
    """
    R_eff = R_base.copy()
    for bus_idx, bus in enumerate(PMU_BUSES):
        if not data_present.get(bus, True):
            start = bus_idx * n_channels_per_bus
            end = start + n_channels_per_bus
            # Inflate row and column (for full covariance matrix)
            R_eff[start:end, :] = R_INFLATE_VALUE
            R_eff[:, start:end] = R_INFLATE_VALUE
            # Restore diagonal to R_INFLATE_VALUE (avoid double-inflation on diagonal)
            R_eff[start:end, start:end] = np.eye(n_channels_per_bus) * R_INFLATE_VALUE
    return R_eff


def data_present_flags(row: "pd.Series") -> dict[int, bool]:
    """Extract per-bus DATA_PRESENT flags from a merged-DataFrame row.

    Args:
        row: one row of the merged DataFrame (or a dict-like).

    Returns:
        {bus_number: bool} — True if DATA_PRESENT == 1 for that bus.
    """
    flags: dict[int, bool] = {}
    for bus in PMU_BUSES:
        col = f"BUS{bus}_DATA_PRESENT"
        val = row.get(col, 1) if hasattr(row, "get") else getattr(row, col, 1)
        flags[bus] = bool(int(val)) if val is not None else True
    return flags


def extract_z(
    row: "pd.Series",
    channel_cols: list[str],
    fill_nan: float = 0.0,
) -> np.ndarray:
    """Extract the measurement vector z from a merged-DataFrame row.

    NaN values (missing channels) are filled with fill_nan; the corresponding
    R entries should be inflated so the filter ignores them.

    Args:
        row: one row of the merged DataFrame.
        channel_cols: ordered list of column names to extract.
        fill_nan: replacement value for NaN entries (irrelevant after R inflation).

    Returns:
        z: (n_z,) float64 array.
    """
    z = np.array([
        float(row[c]) if (c in row and not _is_nan(row[c])) else fill_nan
        for c in channel_cols
    ], dtype=float)
    return z


def _is_nan(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True
