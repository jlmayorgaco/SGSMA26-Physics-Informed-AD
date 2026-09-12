"""Covariance and physically constrained process-discrepancy primitives (E04-A5).

The PowerDynamics export contains a reduced physical state matrix but no explicit
input Jacobian.  ``build_physical_dictionary`` therefore constructs a transparent,
column-sparse dictionary from the exported differential state inventory.  Columns
are state-space input directions (governor/mechanical, generator set-point and
load-equivalent network forcing) rather than an unconstrained dense learned map.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def marginal_hidden_covariance(L_hidden: np.ndarray, P_aug: np.ndarray, physical_dim: int = 114) -> np.ndarray:
    """Return the hidden-output covariance from the *marginal* physical block."""
    L = np.asarray(L_hidden, dtype=float)
    P = np.asarray(P_aug, dtype=float)
    Pxx = P[:physical_dim, :physical_dim]
    return L @ Pxx @ L.T


def conditional_physical_covariance(Pxx: np.ndarray, Pxc: np.ndarray, Pcc: np.ndarray) -> np.ndarray:
    """Conditional covariance, exposed only for the audit toy (never for scoring)."""
    Pxx, Pxc, Pcc = map(lambda a: np.asarray(a, dtype=float), (Pxx, Pxc, Pcc))
    return (Pxx - Pxc @ np.linalg.solve(Pcc, Pxc.T) + (Pxx - Pxc @ np.linalg.solve(Pcc, Pxc.T)).T) / 2


def hidden_component_nees(errors: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Per-component NEES-like statistic using marginal variance."""
    e = np.asarray(errors, dtype=float)
    c = np.asarray(covariances, dtype=float)
    return e * e / np.maximum(np.diagonal(c, axis1=-2, axis2=-1), 1e-18)


def build_physical_dictionary(descriptor_csv: str | Path, physical_dim: int = 114) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a sparse, interpretable ``B_phys`` from PowerDynamics state names.

    The reduced A export has one row per differential state, in the same order as
    differential entries in ``pd_descriptor_inventory.csv``.  We select governor
    states, machine speed/angle and AVR set-point states; load P/Q directions are
    represented by paired network-equivalent speed/angle forcing directions.  This
    is deliberately a fixed dictionary: only the low-rank factor ``G`` is fitted.
    """
    d = pd.read_csv(descriptor_csv)
    diff = d[d["kind"].astype(str).eq("differential")].reset_index(drop=True)
    if len(diff) < physical_dim:
        raise ValueError(f"descriptor has {len(diff)} differential states, expected {physical_dim}")
    diff = diff.iloc[:physical_dim].copy()
    cols: list[np.ndarray] = []
    rows: list[dict] = []

    def add(direction_id: str, typ: str, bus: int, state_name: str, description: str, idx: int) -> None:
        v = np.zeros(physical_dim)
        v[idx] = 1.0
        cols.append(v)
        rows.append({"direction_id": direction_id, "type": typ, "bus": int(bus), "state_index": int(idx),
                     "state_name": state_name, "description": description, "norm": 1.0})

    # Keep the dictionary compact and physically legible.  All ten generator buses
    # are represented; missing fields simply do not create a column.
    for _, row in diff.iterrows():
        bus, name = int(row["bus"]), str(row["state_name"])
        idx = int(row.name)
        if name == "xg1":
            add(f"gov_mechanical_ref_bus{bus}", "governor_mechanical_reference", bus, name,
                "TGOV1 mechanical-power reference perturbation", idx)
        elif name == "xg2":
            add(f"gov_turbine_state_bus{bus}", "governor_turbine", bus, name,
                "TGOV1 turbine/governor internal forcing", idx)
        elif name == "vfout":
            add(f"generator_setpoint_bus{bus}", "generator_setpoint", bus, name,
                "AVR field/set-point disturbance", idx)
        elif name == "ω":
            add(f"load_P_equiv_bus{bus}", "load_active_power_equivalent", bus, name,
                "network-equivalent active-load forcing on electromechanical speed", idx)
        elif name == "δ":
            add(f"load_Q_equiv_bus{bus}", "load_reactive_power_equivalent", bus, name,
                "network-equivalent reactive-load forcing on rotor angle", idx)
    if not cols:
        raise ValueError("no physical differential directions found")
    return np.column_stack(cols), pd.DataFrame(rows)


def fit_physical_low_rank(process_residuals: np.ndarray, B_phys: np.ndarray, rank: int) -> dict:
    """Fit only a low-rank forcing factor from observed-state process residuals."""
    r = np.asarray(process_residuals, dtype=float)
    B = np.asarray(B_phys, dtype=float)
    if r.ndim != 2 or r.shape[1] != B.shape[0]:
        raise ValueError("residual and B_phys dimensions do not agree")
    rank = int(rank)
    projected = r @ B  # B columns are orthonormal sparse directions.
    mean = projected.mean(axis=0)
    xc = projected - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    q = min(rank, vt.shape[0])
    basis = vt[:q].T
    scale = np.maximum(np.std(xc @ basis, axis=0, ddof=1), 1e-12)
    scores = (xc @ basis) / scale
    a = []; qvar = []
    for j in range(q):
        u, v = scores[:-1, j], scores[1:, j]
        f = float(np.clip((u @ v) / max(u @ u, 1e-30), -0.995, 0.995))
        a.append(f); qvar.append(float(np.mean((v - f * u) ** 2)))
    G = basis * scale[None, :]
    Fc = np.diag(a); Qc = np.diag(np.maximum(qvar, 1e-12))
    Pc = np.diag(np.diag(Qc) / np.maximum(1 - np.asarray(a) ** 2, 1e-12))
    return {"rank": q, "mean": mean, "basis": basis, "scale": scale, "G": G,
            "Bc": B @ G, "Fc": Fc, "Qc": Qc, "Pc": Pc,
            "explained": (s[:q] ** 2).cumsum() / max((s ** 2).sum(), 1e-30)}


def propagate_hidden_process_covariance(Pxx: np.ndarray, Pxc: np.ndarray, Pcc: np.ndarray,
                                        Bc: np.ndarray, Qc: np.ndarray, L_hidden: np.ndarray) -> np.ndarray:
    """One-step marginal covariance propagation for an augmented process model."""
    Pxx, Pxc, Pcc, Bc, Qc, L = map(np.asarray, (Pxx, Pxc, Pcc, Bc, Qc, L_hidden))
    # Latent process noise enters x through the same physical coupling Bc.
    Pn = Pxx + Bc @ (Pcc + Qc) @ Bc.T + Pxc @ Bc.T + Bc @ Pxc.T
    return L @ ((Pn + Pn.T) / 2) @ L.T


def augmented_process_prediction(A: np.ndarray, Bc: np.ndarray, Fc: np.ndarray,
                                 Qx: np.ndarray, Qc: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Reference one-step covariance prediction for the augmented process."""
    A, Bc, Fc = map(lambda a: np.asarray(a, dtype=float), (A, Bc, Fc))
    n, r = A.shape[0], Fc.shape[0]
    Aa = np.zeros((n + r, n + r)); Aa[:n, :n] = A; Aa[:n, n:] = Bc; Aa[n:, n:] = Fc
    Qa = np.zeros_like(Aa); Qa[:n, :n] = np.asarray(Qx, dtype=float); Qa[n:, n:] = np.asarray(Qc, dtype=float)
    Pn = Aa @ np.asarray(P, dtype=float) @ Aa.T + Qa
    return (Pn + Pn.T) / 2
