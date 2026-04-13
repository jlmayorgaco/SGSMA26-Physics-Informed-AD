"""Physics evidence used by the event localizer.

The public input remains only the eight PMU streams.  This module turns those
eight phasors into transparent candidate evidence:

* complex Ybus/KCL harmonic reconstruction over all 39 buses,
* Zbus electrical-distance priors from the PMUs with the strongest residuals,
* PMU-direct evidence for events at observed buses,
* phase-balance hints for label-1 fault subtype explanations.
"""
from __future__ import annotations

import numpy as np

from src.grid.electrical_distance import electrical_distance
from src.io.load_csv import PMU_BUSES


def _angle_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def _normalize(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    arr = np.asarray(list(values.values()), dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return {int(k): 0.0 for k in values}
    return {int(k): float((v - lo) / (hi - lo)) for k, v in values.items()}


def _window(df, start: int, end: int):
    start = max(0, int(start))
    end = max(start + 1, min(len(df), int(end)))
    return df.iloc[start:end]


def observed_complex_voltage(df, grid, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Return observed PMU Ybus indices and complex positive-sequence voltages."""
    win = _window(df, start, end)
    obs_idx: list[int] = []
    obs_v: list[complex] = []
    for bus, idx in zip(PMU_BUSES, grid.pmu_bus_indices):
        mag_cols = [f"BUS{bus}_{ph}_MAG" for ph in ("VA", "VB", "VC") if f"BUS{bus}_{ph}_MAG" in win]
        ang_col = f"BUS{bus}_VA_ANG"
        if not mag_cols or ang_col not in win:
            continue
        mags = win[mag_cols].to_numpy(dtype=float)
        angles = win[ang_col].to_numpy(dtype=float)
        if not np.isfinite(mags).any() or not np.isfinite(angles).any():
            continue
        base_v_ln = grid.buses[int(idx)].base_kv * 1000.0 / np.sqrt(3.0)
        vm_pu = float(np.nanmean(mags) / max(base_v_ln, 1.0))
        angle_rad = np.deg2rad(float(np.rad2deg(np.angle(np.nanmean(np.exp(1j * np.deg2rad(angles)))))))
        if np.isfinite(vm_pu) and np.isfinite(angle_rad):
            obs_idx.append(int(idx))
            obs_v.append(vm_pu * np.exp(1j * angle_rad))
    return np.asarray(obs_idx, dtype=int), np.asarray(obs_v, dtype=complex)


def reconstruct_complex_voltage(
    df,
    grid,
    start: int,
    end: int,
    *,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Reconstruct all-bus complex voltages from PMU phasors only."""
    base = np.asarray([b.vm_pu * np.exp(1j * np.deg2rad(b.va_deg)) for b in grid.buses], dtype=complex)
    obs_idx, obs_v = observed_complex_voltage(df, grid, start, end)
    if len(obs_idx) == 0:
        return base

    full = base.copy()
    full[obs_idx] = obs_v
    n = len(base)
    obs_set = set(obs_idx.tolist())
    hidden = np.asarray([i for i in range(n) if i not in obs_set], dtype=int)
    if len(hidden) == 0:
        return full

    Y = np.asarray(grid.Ybus, dtype=complex)
    A = Y[np.ix_(hidden, hidden)] + regularization * np.eye(len(hidden), dtype=complex)
    rhs = -Y[np.ix_(hidden, obs_idx)] @ obs_v
    try:
        full[hidden] = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        full[hidden] = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return full


def ybus_residual_by_bus(
    df,
    grid,
    onset_frame: int,
    *,
    fps: float = 30.0,
    window_sec: float = 3.0,
    baseline_sec: float = 30.0,
) -> dict[int, float]:
    """Return normalized all-bus Ybus/KCL residual energy around an onset."""
    lookahead = max(1, int(round(fps * window_sec)))
    baseline = max(1, int(round(fps * baseline_sec)))
    t0 = int(onset_frame)
    b0 = max(0, t0 - baseline)
    b1 = max(b0 + 1, t0)
    e1 = min(len(df), t0 + lookahead)
    try:
        v0 = reconstruct_complex_voltage(df, grid, b0, b1)
        v1 = reconstruct_complex_voltage(df, grid, t0, e1)
    except Exception:
        return {int(bus): 0.0 for bus in grid.ext_bus_order}

    dvm = np.abs(v1) - np.abs(v0)
    dva = np.deg2rad(_angle_diff_deg(np.rad2deg(np.angle(v1)), np.rad2deg(np.angle(v0))))
    dvm = dvm - np.nanmedian(dvm)
    dva = dva - np.nanmedian(dva)
    energy = np.sqrt((dvm / 0.002) ** 2 + (dva / np.deg2rad(0.1)) ** 2)
    energy = np.where(np.isfinite(energy), energy, 0.0)
    return {int(bus): float(energy[i]) for i, bus in enumerate(grid.ext_bus_order)}


def zbus_distance_prior(nu: np.ndarray, grid) -> dict[int, float]:
    """Score candidate buses by electrical closeness to PMUs with high residuals."""
    if grid.Zbus is None or len(grid.pmu_bus_indices) == 0:
        return {int(bus): 0.0 for bus in grid.ext_bus_order}
    nu = np.asarray(nu, dtype=float)
    if np.nanmax(nu) <= 1e-12:
        return {int(bus): 0.0 for bus in grid.ext_bus_order}
    weights = np.maximum(nu, 0.0)
    weights = weights / max(float(weights.sum()), 1e-12)
    D = electrical_distance(grid.Zbus)
    raw: dict[int, float] = {}
    for row, bus in enumerate(grid.ext_bus_order):
        dist = D[row, grid.pmu_bus_indices]
        raw[int(bus)] = float(np.sum(weights / (dist + 1e-6)))
    return _normalize(raw)


def swing_residual_by_bus(
    df,
    onset_frame: int,
    grid=None,
    *,
    fps: float = 30.0,
    window_sec: float = 3.0,
    baseline_sec: float = 30.0,
) -> dict[int, float]:
    """Return a safe swing-equation-inspired frequency/ROCOF residual prior.

    This deliberately avoids importing the optional UKF/ANDES/SymPy stack.  It
    measures local electromechanical stress from PMU frequency and ROCOF
    residuals, then spreads the evidence to all IEEE-39 buses through Zbus
    electrical distance when the grid is available.
    """
    lookahead = max(1, int(round(fps * window_sec)))
    baseline = max(1, int(round(fps * baseline_sec)))
    t0 = int(onset_frame)
    b0 = max(0, t0 - baseline)
    b1 = max(b0 + 1, t0)
    e1 = min(len(df), t0 + lookahead)
    base = df.iloc[b0:b1]
    win = df.iloc[t0:e1]

    pmu_scores: dict[int, float] = {}
    for bus in PMU_BUSES:
        fcol = f"BUS{bus}_Freq"
        rcol = f"BUS{bus}_ROCOF"
        score = 0.0
        if fcol in df and len(win):
            f0 = float(base[fcol].dropna().mean()) if len(base[fcol].dropna()) else np.nan
            f1 = win[fcol].to_numpy(dtype=float)
            if np.isfinite(f0) and np.isfinite(f1).any():
                score += float(np.nanmax(np.abs(f1 - f0)) / 0.02)
        if rcol in df and len(win):
            r0 = float(base[rcol].dropna().median()) if len(base[rcol].dropna()) else 0.0
            r1 = win[rcol].to_numpy(dtype=float)
            if np.isfinite(r1).any():
                score += float(np.nanmax(np.abs(r1 - r0)) / 0.05)
        pmu_scores[int(bus)] = max(0.0, score)

    if grid is None or grid.Zbus is None:
        return _normalize(pmu_scores)

    pmu_vec = np.asarray([pmu_scores.get(bus, 0.0) for bus in PMU_BUSES], dtype=float)
    if np.nanmax(pmu_vec) <= 1e-12:
        return {int(bus): 0.0 for bus in grid.ext_bus_order}
    weights = pmu_vec / max(float(pmu_vec.sum()), 1e-12)
    D = electrical_distance(grid.Zbus)
    raw: dict[int, float] = {}
    for row, bus in enumerate(grid.ext_bus_order):
        dist = D[row, grid.pmu_bus_indices]
        raw[int(bus)] = float(np.sum(weights / (dist + 1e-6)))
    return _normalize(raw)


def pmu_direct_prior(nu: np.ndarray) -> dict[int, float]:
    """Map the 8-dimensional PMU residual vector directly to PMU bus scores."""
    nu = np.asarray(nu, dtype=float)
    if np.nanmax(nu) <= 1e-12:
        return {int(bus): 0.0 for bus in PMU_BUSES}
    vals = np.maximum(nu, 0.0) / max(float(np.nanmax(nu)), 1e-12)
    return {int(bus): float(vals[i]) for i, bus in enumerate(PMU_BUSES)}


def jacobian_prior(nu: np.ndarray, J_cols: dict[int, np.ndarray]) -> dict[int, float]:
    """Return normalized absolute cosine scores against Jacobian columns."""
    nu = np.asarray(nu, dtype=float)
    nnu = np.linalg.norm(nu)
    if nnu < 1e-30:
        return {int(bus): 0.0 for bus in J_cols}
    raw: dict[int, float] = {}
    for bus, col in J_cols.items():
        col = np.asarray(col, dtype=float)
        ncol = np.linalg.norm(col)
        raw[int(bus)] = 0.0 if ncol < 1e-30 else float(abs(np.dot(nu, col) / (nnu * ncol)))
    return _normalize(raw)


def combined_bus_scores(
    *,
    nu: np.ndarray,
    J_cols: dict[int, np.ndarray],
    grid,
    ybus_energy: dict[int, float] | None = None,
    state_energy: dict[int, float] | None = None,
    swing_energy: dict[int, float] | None = None,
) -> list[dict]:
    """Return an auditable bus candidate score table."""
    buses = sorted(set(J_cols) | set(getattr(grid, "ext_bus_order", [])))
    jac = jacobian_prior(nu, J_cols)
    zbus = zbus_distance_prior(nu, grid) if grid is not None else {}
    pmu = pmu_direct_prior(nu)
    ybus = _normalize(ybus_energy or {})
    state = _normalize(state_energy or {})
    swing = _normalize(swing_energy or {})

    rows: list[dict] = []
    for bus in buses:
        j = jac.get(bus, 0.0)
        y = ybus.get(bus, 0.0)
        z = zbus.get(bus, 0.0)
        p = pmu.get(bus, 0.0)
        s = state.get(bus, 0.0)
        w = swing.get(bus, 0.0)
        combined = 0.34 * j + 0.26 * max(y, s) + 0.14 * z + 0.16 * p + 0.10 * w
        rows.append(
            {
                "bus": int(bus),
                "score": float(combined),
                "jacobian": float(j),
                "ybus": float(y),
                "state": float(s),
                "zbus": float(z),
                "pmu_direct": float(p),
                "swing": float(w),
            }
        )
    rows.sort(key=lambda row: (-row["score"], row["bus"]))
    return rows


def top_buses_from_scores(score_table: list[dict], k: int = 3) -> list[tuple[int, float]]:
    return [(int(row["bus"]), float(row["score"])) for row in score_table[:k]]


def fault_subtype_hint(df, onset_frame: int, *, fps: float = 30.0, window_sec: float = 0.5) -> dict:
    """Explain label-1 balance as a fault subtype hint.

    The official classifier still emits label 1.  This explanation separates
    balanced three-phase faults from unbalanced phase disturbances.
    """
    half = max(1, int(round(fps * window_sec)))
    t0 = max(0, int(onset_frame) - half)
    t1 = min(len(df), int(onset_frame) + half)
    base0 = max(0, t0 - int(round(fps * 2.0)))
    base = df.iloc[base0:t0] if base0 < t0 else df.iloc[:max(1, t0)]
    win = df.iloc[t0:t1]
    best: dict | None = None
    for bus in PMU_BUSES:
        mags = []
        drops = []
        for phase in ("A", "B", "C"):
            col = f"BUS{bus}_V{phase}_MAG"
            if col not in df:
                continue
            before = float(base[col].dropna().mean()) if len(base[col].dropna()) else np.nan
            during = float(win[col].dropna().mean()) if len(win[col].dropna()) else np.nan
            if np.isfinite(before) and np.isfinite(during) and before > 1e-12:
                drop = max(0.0, (before - during) / before)
                mags.append(during)
                drops.append(drop)
        if len(drops) != 3:
            continue
        severity = float(max(drops))
        unbalance = float(np.std(drops) / max(np.mean(drops), 1e-12))
        cand = {"bus": int(bus), "severity": severity, "unbalance": unbalance, "drops": drops}
        if best is None or cand["severity"] > best["severity"]:
            best = cand
    if best is None or best["severity"] < 1e-4:
        return {"fault_subtype": "weak_or_high_impedance", "confidence": 0.0, "bus": None}
    drops = np.asarray(best["drops"], dtype=float)
    active = np.where(drops > max(0.05, 0.35 * float(drops.max())))[0]
    if best["unbalance"] < 0.15 and len(active) == 3:
        subtype = "three_phase"
    elif len(active) == 1:
        subtype = f"single_line_ground_{'ABC'[int(active[0])]}"
    elif len(active) == 2:
        subtype = f"line_line_{''.join('ABC'[int(i)] for i in active)}"
    else:
        subtype = "unbalanced_or_high_impedance"
    confidence = float(min(1.0, best["severity"] * 2.0) * min(1.0, 0.5 + best["unbalance"]))
    return {
        "fault_subtype": subtype,
        "confidence": confidence,
        "bus": int(best["bus"]),
        "phase_voltage_drop_fraction": [float(v) for v in drops],
    }
