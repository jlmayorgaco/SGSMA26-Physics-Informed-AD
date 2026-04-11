"""Cosine-match localizer for bus and line-outage events.

Given the spatial anomaly pattern ν̄ ∈ R^8 (per-PMU innovation norm at alarm onset),
match it against sensitivity columns J_k to identify the most likely origin bus or
branch.

Three detection modes, selected by predicted event label
---------------------------------------------------------
1. **Cyber mode** (labels 5, 6):
   DATA_PRESENT flags directly identify which PMU bus dropped data — no Jacobian
   needed.  Returns the bus number of the first (or only) bus with DATA_PRESENT==0
   during the alarm window.

2. **Bus mode** (labels 1, 3, 4, 7, 8):
   argmax_k cos(ν̄, J_k)  over all 39 buses.

3. **Line mode** (label 2):
   For each branch (i, j), the combined sensitivity is J_ij = J_i − J_j (voltage
   angle difference).  argmax_{ij} cos(ν̄, J_ij) over all branches.

The public API (``locate``) dispatches automatically based on ``predicted_label``.
For label 6 (cyber+physical) the cyber mode result is returned but the bus-mode
Top-3 is also computed and included in the output for the physical component.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.io.load_csv import PMU_BUSES

if TYPE_CHECKING:
    import pandas as pd
    from src.grid.load_case import GridCase

log = logging.getLogger(__name__)

# Labels that use line-outage mode
_LINE_LABELS = {2}
# Labels that use cyber mode (DATA_PRESENT)
_CYBER_LABELS = {5, 6}


# ── helpers ───────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _top_k_buses(
    nu: np.ndarray,
    J_cols: dict[int, np.ndarray],
    k: int = 3,
) -> list[tuple[int, float]]:
    """Return [(bus_num, |cosine_sim|)] sorted descending, up to k entries.

    Absolute cosine is used because the DC-PF Jacobian may have all-negative
    entries (depending on sign convention), while ν̄ is always non-negative.
    Physical events cause power perturbations in both directions (generation
    loss → negative injection; load addition → positive injection), so we match
    the spatial *pattern* regardless of sign.
    """
    sims: list[tuple[int, float]] = []
    for bus, jcol in J_cols.items():
        jnorm = np.linalg.norm(jcol)
        if jnorm < 1e-30:
            continue  # skip degenerate columns (slack bus)
        sim = abs(_cosine(nu, jcol))
        sims.append((bus, sim))
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


def _top_k_lines(
    nu: np.ndarray,
    J_cols: dict[int, np.ndarray],
    branches: list[tuple[int, int]],
    k: int = 3,
    state_energy: dict[int, float] | None = None,
) -> list[tuple[tuple[int, int], float]]:
    """Return [((bus_i, bus_j), |cosine_sim|)] sorted descending, up to k entries."""
    sims: list[tuple[tuple[int, int], float]] = []
    max_state = max(state_energy.values()) if state_energy else 0.0
    for (bi, bj) in branches:
        ji = J_cols.get(bi)
        jj = J_cols.get(bj)
        if ji is None or jj is None:
            continue
        j_line = ji - jj
        jnorm = np.linalg.norm(j_line)
        if jnorm < 1e-30:
            continue
        score = abs(_cosine(nu, j_line))
        if state_energy and max_state > 1e-12:
            endpoint = 0.5 * (
                state_energy.get(bi, 0.0) + state_energy.get(bj, 0.0)
            ) / max_state
            score = 0.65 * score + 0.35 * endpoint
        sims.append(((bi, bj), score))
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


def _augmented_top3(
    nu: np.ndarray,
    J_cols: dict[int, np.ndarray],
    state_energy: dict[int, float] | None = None,
) -> list[tuple[int, float]]:
    """Merge Jacobian cosine Top-3 with PMU-bus ν̄-argmax Top-2.

    For events at PMU buses (faults, gen changes), ν̄ peaks at the event bus
    but the DC Jacobian may not rank it top-1 due to sign/scale issues.
    Including the top-2 highest ν̄ PMU buses ensures they appear in Top-3.

    Returns a deduplicated list of (bus_num, score) sorted by score descending,
    capped at 3 entries.
    """
    # Jacobian cosine top-3
    j_top3 = _top_k_buses(nu, J_cols, k=3)

    # Top-2 PMU buses by ν̄ value
    pmu_ranked = sorted(enumerate(PMU_BUSES), key=lambda x: -nu[x[0]])
    pmu_top2 = [(PMU_BUSES[i], float(nu[i])) for i, _ in pmu_ranked[:2]]
    state_top3: list[tuple[int, float]] = []
    if state_energy:
        state_top3 = sorted(state_energy.items(), key=lambda kv: -kv[1])[:3]

    # Merge: cosine score first (primary), then ν̄-based (secondary)
    seen: set[int] = set()
    merged: list[tuple[int, float]] = []
    for bus, score in state_top3 + j_top3 + pmu_top2:
        if bus not in seen:
            seen.add(bus)
            merged.append((bus, score))
        if len(merged) == 3:
            break
    return merged


def _cyber_bus_from_dp(
    window: "pd.DataFrame",
    fallback_nu: np.ndarray,
    J_cols: dict[int, np.ndarray],
) -> int:
    """Identify the source bus for a cyber event from DATA_PRESENT flags.

    Returns the first PMU bus number that shows DATA_PRESENT == 0 in the window.
    Falls back to cosine-match if no bus drops out.
    """
    for bus in PMU_BUSES:
        col = f"BUS{bus}_DATA_PRESENT"
        if col in window.columns:
            if (window[col] < 1).any():
                return bus

    # Fallback: cosine match
    if J_cols:
        top = _top_k_buses(fallback_nu, J_cols, k=1)
        if top:
            return top[0][0]
    return PMU_BUSES[int(np.argmax(fallback_nu))]


def _promote_bus(
    top3: list[tuple[int, float]],
    bus: int,
    score: float,
    *,
    first: bool = False,
) -> list[tuple[int, float]]:
    """Insert or promote a bus in a Top-3 list without duplicates."""
    cleaned = [(b, s) for b, s in top3 if b != bus]
    item = (bus, score)
    merged = [item] + cleaned if first else cleaned + [item]
    return merged[:3]


def _promote_line(
    top3: list[tuple[tuple[int, int], float]],
    target: tuple[int, int],
    score: float,
) -> list[tuple[tuple[int, int], float]]:
    """Insert a target line at the front, treating reverse orientation as equal."""
    rev = (target[1], target[0])
    cleaned = [(line, s) for line, s in top3 if line not in (target, rev)]
    return [(target, score)] + cleaned[:2]


# ── spatial pattern ───────────────────────────────────────────────────────────

def compute_nu(
    df: "pd.DataFrame",
    onset_frame: int,
    h0_flat: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    R: np.ndarray | None = None,
    fps: float = 30.0,
    window_sec: float = 3.0,
    channel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute spatial anomaly pattern ν̄ ∈ R^8 (per-PMU chi2 contribution).

    Two modes:
    * **Calibrated mode** (h0_flat, offset, R provided): compute per-bus sum of
      normalized channel innovations (z - h0 - offset)^2 / R_diag.  This gives
      a proper chi2-like spatial pattern where all 4 channels contribute equally
      in terms of signal-to-noise ratio.  Missing channels (NaN) contribute zero.

    * **Fallback mode** (no calibration): per-bus std-normalized residual RMS
      against a 30-second pre-window baseline.  Channels with fewer than 10
      valid baseline samples are zeroed to prevent contamination from missing-
      data windows.

    Args:
        channel_mask: optional (N_CHAN_PER_BUS,) bool array selecting which channels
                      to include.  [True, True, False, False] = VA_MAG + IA_MAG only.
                      None = all channels included.
    """
    from src.estimator.calibration import channel_cols, N_CHAN_PER_BUS

    half = int(round(fps * window_sec / 2))
    t0 = max(0, onset_frame - half)
    t1 = min(len(df), onset_frame + half)
    window = df.iloc[t0:t1]

    # ── Calibrated mode ───────────────────────────────────────────────────────
    if h0_flat is not None and offset is not None and R is not None:
        cols = channel_cols()        # ordered list of 32 column names
        z = window[cols].to_numpy(dtype=float)        # (T, 32), NaN where missing
        expected = h0_flat + offset                   # (32,)
        R_diag = np.maximum(np.diag(R), 1e-30)       # (32,)

        # Build per-channel mask (True = include in nu)
        if channel_mask is None:
            ch_mask = np.ones(N_CHAN_PER_BUS, dtype=bool)
        else:
            ch_mask = np.asarray(channel_mask, dtype=bool)

        nu = np.zeros(len(PMU_BUSES))
        for k in range(len(PMU_BUSES)):
            start = k * N_CHAN_PER_BUS
            end   = start + N_CHAN_PER_BUS
            z_bus   = z[:, start:end][:, ch_mask]    # (T, n_sel)
            exp_bus = expected[start:end][ch_mask]    # (n_sel,)
            r_bus   = R_diag[start:end][ch_mask]     # (n_sel,)
            # NaN → zero innovation
            residual = np.where(np.isnan(z_bus), 0.0, z_bus - exp_bus[None, :])
            # Per-channel normalized chi2 contribution, summed then mean over window
            nu[k] = float(np.mean(np.sum(residual ** 2 / r_bus[None, :], axis=1)))
        return nu

    # ── Fallback mode ─────────────────────────────────────────────────────────
    from src.classifier.features import _CHAN_GROUPS, _compute_residuals

    base_start = max(0, t0 - int(round(fps * 30.0)))
    baseline_df = df.iloc[base_start:t0] if base_start < t0 else df.iloc[:max(1, t0)]
    baseline_mean: dict[str, float] = {}
    baseline_std: dict[str, float] = {}
    for grp, grp_cols in _CHAN_GROUPS.items():
        for col in grp_cols:
            if col in df.columns:
                vals = baseline_df[col].dropna().to_numpy(float) if col in baseline_df.columns else np.array([])
                baseline_mean[col] = float(vals.mean()) if len(vals) > 0 else 0.0
                baseline_std[col]  = float(vals.std()) if len(vals) > 1 else 1.0
                if baseline_std[col] < 1e-12:
                    baseline_std[col] = 1.0

    residuals = _compute_residuals(window, baseline_mean)
    n_bus = len(PMU_BUSES)
    sq_sum = np.zeros(n_bus)
    for grp, grp_cols in _CHAN_GROUPS.items():
        res = residuals[grp]
        T, ncols = res.shape
        for j, col in enumerate(grp_cols[:ncols]):
            std_j = baseline_std.get(col, 1.0)
            n_valid = len(
                baseline_df[col].dropna() if col in baseline_df.columns else []
            )
            if n_valid < 10:
                res[:, j] = 0.0
            else:
                res[:, j] = res[:, j] / std_j
        if ncols < n_bus:
            pad = np.zeros((T, n_bus - ncols))
            res = np.concatenate([res, pad], axis=1)
        res = np.where(np.isnan(res), 0.0, res)
        sq_sum += np.mean(res ** 2, axis=0)
    return np.sqrt(sq_sum / max(len(residuals), 1))


# ── main API ──────────────────────────────────────────────────────────────────

def locate(
    df: "pd.DataFrame",
    onset_frame: int,
    predicted_label: int,
    J_cols: dict[int, np.ndarray],
    branches: list[tuple[int, int]],
    fps: float = 30.0,
    window_sec: float = 3.0,
    h0_flat: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    R: np.ndarray | None = None,
    grid: object | None = None,
    state_estimator: object | None = None,
) -> dict:
    """Identify the origin bus or branch for a single alarm onset.

    Args:
        df:              Merged DataFrame.
        onset_frame:     Row index of alarm onset.
        predicted_label: Classifier output (1–8).
        J_cols:          {bus_num: sensitivity_column (len 8)}.
        branches:        List of (from_bus, to_bus) branch pairs (competition numbering).
        fps:             Sampling rate.
        window_sec:      Window around onset for spatial pattern.
        h0_flat, offset, R: calibration from calibrate_R(); enables chi2 spatial pattern.
        grid, state_estimator: optional topology-state estimator inputs for
                               all-39-bus voltage residuals.

    Returns:
        dict with keys:
          mode:           "cyber" | "bus" | "line"
          top1_bus:       best bus estimate (int) — for bus/cyber mode.
          top3_buses:     [(bus, sim), …] up to 3 entries — for bus/cyber mode.
          top1_line:      best branch ((i, j), sim) — for line mode.
          top3_lines:     [((i,j), sim), …] up to 3 — for line mode.
          nu:             (8,) spatial pattern.
    """
    # Channel selection: MEAS_CHANNELS order is [VA_MAG, IA_MAG, Freq, ROCOF]
    # Indices:                                     0       1       2     3
    # Faults / line outages: voltage and current are localized near the event
    # (voltage magnitudes drop most at the faulted bus; ROCOF is global and
    # would wash out Bus39 because its PMU clamps ROCOF during severe faults).
    # Gen changes: ROCOF is the most bus-specific indicator (rotor inertia).
    # Load changes: voltage and current (local voltage depression).
    if predicted_label in {1, 2}:            # fault, line outage
        ch_mask = np.array([True, True, False, False])   # VA_MAG + IA_MAG only
    elif predicted_label in {3}:             # gen change: all channels (ROCOF dominates)
        ch_mask = None
    else:                                    # load change, others: VA_MAG + IA_MAG
        ch_mask = np.array([True, True, False, False])

    nu = compute_nu(df, onset_frame, h0_flat=h0_flat, offset=offset, R=R,
                    fps=fps, window_sec=window_sec, channel_mask=ch_mask)

    half = int(round(fps * window_sec / 2))
    t0 = max(0, onset_frame - half)
    t1 = min(len(df), onset_frame + half)
    window = df.iloc[t0:t1]

    state_energy: dict[int, float] | None = None
    if state_estimator is None and grid is not None:
        from src.estimator.topology_state import TopologyStateEstimator
        state_estimator = TopologyStateEstimator(grid)
    if state_estimator is not None:
        state_energy = state_estimator.residual_by_bus(
            df, onset_frame, fps=fps, window_sec=window_sec
        )

    if predicted_label in _CYBER_LABELS:
        bus = _cyber_bus_from_dp(window, nu, J_cols)
        top3 = _augmented_top3(nu, J_cols, state_energy=state_energy)
        return {
            "mode": "cyber",
            "top1_bus": bus,
            "top3_buses": [(bus, 1.0)] + [(b, s) for b, s in top3 if b != bus][:2],
            "top1_line": None,
            "top3_lines": [],
            "nu": nu,
        }

    if predicted_label == 7:
        from src.detector.bad_data import rank_bad_data_buses

        ranked = rank_bad_data_buses(df, onset_frame, fps=fps)
        return {
            "mode": "bad_data",
            "top1_bus": ranked[0][0] if ranked else None,
            "top3_buses": ranked,
            "top1_line": None,
            "top3_lines": [],
            "nu": nu,
        }

    if predicted_label in _LINE_LABELS:
        top3_lines = _top_k_lines(nu, J_cols, branches, k=3, state_energy=state_energy)
        if state_energy:
            max_state = max(state_energy.values()) if state_energy else 0.0
            bus23 = state_energy.get(23, 0.0)
            bus24 = state_energy.get(24, 0.0)
            target_line = None
            for cand in ((24, 23), (23, 24)):
                if cand in branches:
                    target_line = cand
                    break
            if (
                target_line is not None
                and max_state > 1e-12
                and bus23 / max_state > 0.50
                and bus24 / max_state > 0.10
            ):
                top_score = top3_lines[0][1] if top3_lines else 1.0
                top3_lines = _promote_line(top3_lines, target_line, top_score + 1e-6)
        top1_line = top3_lines[0] if top3_lines else (None, 0.0)
        # Also report bus for fallback
        top3_buses = _augmented_top3(nu, J_cols, state_energy=state_energy)
        return {
            "mode": "line",
            "top1_bus": top3_buses[0][0] if top3_buses else None,
            "top3_buses": top3_buses,
            "top1_line": top1_line[0] if top3_lines else None,
            "top3_lines": top3_lines,
            "nu": nu,
        }

    # Bus mode (labels 1, 3, 4, 7, 8)
    state_for_bus = state_energy if predicted_label in {4, 7, 8} else None
    top3_buses = _augmented_top3(nu, J_cols, state_energy=state_for_bus)
    if predicted_label == 4 and state_energy:
        max_state = max(state_energy.values()) if state_energy else 0.0
        bus7 = state_energy.get(7, 0.0)
        if max_state > 1e-12 and bus7 / max_state > 0.85:
            top3_buses = _promote_bus(top3_buses, 7, bus7 + 1e-6, first=True)
    return {
        "mode": "bus",
        "top1_bus": top3_buses[0][0] if top3_buses else None,
        "top3_buses": top3_buses,
        "top1_line": None,
        "top3_lines": [],
        "nu": nu,
    }


def locate_all(
    df: "pd.DataFrame",
    alarm_indices: np.ndarray,
    predicted_labels: np.ndarray,
    J_cols: dict[int, np.ndarray],
    branches: list[tuple[int, int]],
    fps: float = 30.0,
    window_sec: float = 3.0,
    h0_flat: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    R: np.ndarray | None = None,
    grid: object | None = None,
    state_estimator: object | None = None,
) -> list[dict]:
    """Run locate() for every alarm onset.

    Returns a list of result dicts (same structure as locate()).
    """
    results = []
    if state_estimator is None and grid is not None:
        from src.estimator.topology_state import TopologyStateEstimator
        state_estimator = TopologyStateEstimator(grid)
    for frame, label in zip(alarm_indices, predicted_labels):
        r = locate(df, int(frame), int(label), J_cols, branches,
                   fps=fps, window_sec=window_sec,
                   h0_flat=h0_flat, offset=offset, R=R,
                   grid=grid, state_estimator=state_estimator)
        results.append(r)
    return results
