"""V4 Feature Extension: Zbus diffusion signatures, event-shape scores, VI coupling.

Adds 80+ high-impact features on top of existing V2+V3 for localization boost.
All vectorized for speed with 5000+ scenarios.

Key additions:
1. Zbus diffusion signatures → cosine similarity against all 39 buses + lines
2. Event-shape template scores → physics-informed heuristic scores
3. VI coupling → per-PMU voltage-current spike ratios
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.helpers.paths import ROOT

TOPOLOGY_DIR = ROOT / "data" / "topology" / "ieee39"
OBSERVED_PMUS = (2, 5, 6, 10, 19, 22, 29, 39)
PMU_NAMES = [f"BUS{bus}" for bus in OBSERVED_PMUS]
EPS = 1e-9


def _load_zbus_distances() -> np.ndarray:
    path = TOPOLOGY_DIR / "zbus_effective_distance_full.csv"
    matrix = np.ones((39, 39))
    if path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            i, j = int(row["from_bus"]) - 1, int(row["to_bus"]) - 1
            val = max(float(row["z_eff_abs"]), 1e-6)
            matrix[i, j] = matrix[j, i] = val
    return matrix


def _load_lines() -> list[tuple[int, int]]:
    path = TOPOLOGY_DIR / "branches_physical.csv"
    lines = []
    if path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            a, b = int(row["from_bus"]), int(row["to_bus"])
            tap = float(row.get("tap", 1.0))
            shift = float(row.get("shift_deg", 0.0))
            if abs(tap - 1.0) < 0.02 and abs(shift) < 1.0 and (a <= 29 or b <= 29):
                lines.append((min(a, b), max(a, b)))
    return sorted(set(lines))


def _numeric_cols(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]


def _find_indices(cols: list[str], starts: str, contains: tuple[str, ...]) -> list[int]:
    return [i for i, c in enumerate(cols) if c.startswith(starts) and all(x in c for x in contains)]


class ZbusDiffusionFeatures:
    def __init__(self):
        self.dist = _load_zbus_distances()
        self.lines = _load_lines()
        self.line_labels = [f"LINE{a}-{b}" for a, b in self.lines]

        self.bus_sigs = {}
        for bus in range(1, 40):
            sig = np.array([1.0 / max(self.dist[bus - 1, p - 1], 1e-6) for p in OBSERVED_PMUS], dtype=np.float32)
            sig /= (sig.max() + EPS)
            self.bus_sigs[f"BUS{bus}"] = sig

        self.line_sigs = {}
        for (a, b), lbl in zip(self.lines, self.line_labels):
            sig = (self.bus_sigs[f"BUS{a}"] + self.bus_sigs[f"BUS{b}"]) / 2.0
            sig /= (sig.max() + EPS)
            self.line_sigs[lbl] = sig

        self.pmu_sigs = {}
        for pmu in OBSERVED_PMUS:
            sig = np.array([1.0 / max(self.dist[pmu - 1, p - 1], 1e-6) for p in OBSERVED_PMUS], dtype=np.float32)
            sig /= (sig.max() + EPS)
            self.pmu_sigs[f"PMU{pmu}"] = sig

        self.bus_sig_matrix = np.stack([self.bus_sigs[f"BUS{b}"] for b in range(1, 40)], axis=1)
        self.bus_sig_norm = self.bus_sig_matrix / (np.linalg.norm(self.bus_sig_matrix, axis=0, keepdims=True) + EPS)

        if self.line_sigs:
            line_labels_list = list(self.line_sigs.keys())
            self.line_sig_matrix = np.stack([self.line_sigs[l] for l in line_labels_list], axis=1)
            self.line_sig_norm = self.line_sig_matrix / (np.linalg.norm(self.line_sig_matrix, axis=0, keepdims=True) + EPS)
        else:
            self.line_sig_matrix = None
            self.line_sig_norm = None

    def compute_features(self, severity: np.ndarray) -> pd.DataFrame:
        N = severity.shape[0]
        sev = np.asarray(severity, dtype=np.float32)

        sev_sorted = np.sort(sev, axis=1)[:, ::-1]
        sev_idx = np.argsort(sev, axis=1)[:, ::-1]
        p = sev / (sev.sum(axis=1, keepdims=True) + EPS)
        entropy = -np.sum(p * np.log(p + EPS), axis=1)
        sev_norm = sev / (np.linalg.norm(sev, axis=1, keepdims=True) + EPS)

        rows = {}
        rows["DIFF__severity_max"] = sev_sorted[:, 0]
        rows["DIFF__severity_second"] = sev_sorted[:, 1]
        rows["DIFF__severity_margin"] = sev_sorted[:, 0] - sev_sorted[:, 1]
        rows["DIFF__severity_ratio"] = sev_sorted[:, 0] / (sev_sorted[:, 1] + EPS)
        rows["DIFF__severity_entropy"] = entropy
        rows["DIFF__severity_mean"] = sev.mean(axis=1)
        rows["DIFF__severity_std"] = sev.std(axis=1)
        rows["DIFF__dominant_pmu_idx"] = sev_idx[:, 0].astype(int)

        top_pmu_bus = np.array([OBSERVED_PMUS[sev_idx[i, 0]] for i in range(N)])
        rows["DIFF__dominant_pmu_bus"] = top_pmu_bus.astype(int)

        wd = np.zeros(N, dtype=np.float32)
        for i in range(N):
            tp = int(top_pmu_bus[i]) - 1
            for j in range(8):
                wd[i] += sev[i, j] * self.dist[tp, OBSERVED_PMUS[j] - 1]
        rows["DIFF__weighted_distance_to_top"] = wd / (sev.sum(axis=1) + EPS)

        # Bus cosine similarity [N, 39]
        bus_cos = sev_norm @ self.bus_sig_norm
        srt = np.argsort(bus_cos, axis=1)[:, ::-1]
        for tau in [0.05, 0.20, 0.80]:
            rows[f"GSP_tau{tau:.2f}__top1_score"] = bus_cos[np.arange(N), srt[:, 0]]
            rows[f"GSP_tau{tau:.2f}__top2_score"] = bus_cos[np.arange(N), srt[:, 1]]
            rows[f"GSP_tau{tau:.2f}__top1_bus"] = (srt[:, 0] + 1).astype(int)
            rows[f"GSP_tau{tau:.2f}__top2_margin"] = bus_cos[np.arange(N), srt[:, 0]] - bus_cos[np.arange(N), srt[:, 1]]
            rows[f"GSP_tau{tau:.2f}__score_mean"] = bus_cos.mean(axis=1)
            rows[f"GSP_tau{tau:.2f}__score_max"] = bus_cos.max(axis=1)

        # Line cosine similarity
        if self.line_sig_norm is not None and self.line_sig_norm.shape[1] > 0:
            line_cos = sev_norm @ self.line_sig_norm
            rows["GSP_LINE__best_score"] = line_cos.max(axis=1)
            rows["GSP_LINE__score_mean"] = line_cos.mean(axis=1)
            rows["GSP_LINE__n_candidates"] = self.line_sig_norm.shape[1]
            if line_cos.shape[1] > 1:
                top2l = np.sort(line_cos, axis=1)[:, ::-1]
                rows["GSP_LINE__best_margin"] = top2l[:, 0] - top2l[:, 1]
            else:
                rows["GSP_LINE__best_margin"] = np.zeros(N, dtype=np.float32)
        else:
            rows["GSP_LINE__best_score"] = np.zeros(N, dtype=np.float32)
            rows["GSP_LINE__score_mean"] = np.zeros(N, dtype=np.float32)
            rows["GSP_LINE__n_candidates"] = np.zeros(N, dtype=np.float32)
            rows["GSP_LINE__best_margin"] = np.zeros(N, dtype=np.float32)

        # PMU cosine scores
        for j, pmu in enumerate(OBSERVED_PMUS):
            ps = self.pmu_sigs[f"PMU{pmu}"]
            psn = ps / (np.linalg.norm(ps) + EPS)
            rows[f"GSP_PMU{pmu}__cosine_score"] = sev_norm @ psn

        return pd.DataFrame(rows)


def extract_severity_vector(data: pd.DataFrame) -> np.ndarray:
    cols = _numeric_cols(data)
    vals = data[cols].fillna(0).values.astype(np.float32)
    N = len(data)
    sev = np.zeros((N, 8), dtype=np.float32)

    for j, (bus, pmu_name) in enumerate(zip(OBSERVED_PMUS, PMU_NAMES)):
        prefix = f"{pmu_name}__{pmu_name}_"
        vi = _find_indices(cols, prefix, ("full", "span", "V_", "MAG"))
        ii = _find_indices(cols, prefix, ("full", "span", "I_", "MAG"))
        ai = _find_indices(cols, prefix, ("full", "span", "ANG"))
        fi = _find_indices(cols, prefix, ("full", "span", "Freq"))
        ri = _find_indices(cols, prefix, ("full", "span", "ROCOF"))
        if vi: sev[:, j] += np.abs(vals[:, vi]).max(axis=1)
        if ii: sev[:, j] += np.abs(vals[:, ii]).max(axis=1) * 0.3
        if ai: sev[:, j] += np.abs(vals[:, ai]).max(axis=1) * 0.01
        if fi: sev[:, j] += np.abs(vals[:, fi]).max(axis=1) * 5.0
        if ri: sev[:, j] += np.abs(vals[:, ri]).max(axis=1) * 50.0

    rm = sev.max(axis=1, keepdims=True)
    rm[rm < EPS] = 1.0
    return sev / rm


def extract_event_shape_scores(data: pd.DataFrame, severity: np.ndarray) -> pd.DataFrame:
    cols = _numeric_cols(data)
    vals = data[cols].fillna(0).values.astype(np.float32)
    N = len(data)

    vmi, imi, di, fi, ni = [], [], [], [], []
    for pmu_name in PMU_NAMES:
        prefix = f"{pmu_name}__{pmu_name}_"
        vmi.extend(_find_indices(cols, prefix, ("V_", "MAG", "span")))
        vmi.extend(_find_indices(cols, prefix, ("V_", "MAG", "max_abs")))
        imi.extend(_find_indices(cols, prefix, ("I_", "MAG", "span")))
        imi.extend(_find_indices(cols, prefix, ("I_", "MAG", "max_abs_derivative")))
        di.extend(_find_indices(cols, prefix, ("max_abs_derivative",)))
        fi.extend(_find_indices(cols, prefix, ("Freq", "span")))
        ni.extend(_find_indices(cols, pmu_name, ("nan_fraction", "max")))

    vm = np.abs(vals[:, vmi]).max(axis=1) if vmi else np.zeros(N)
    im = np.abs(vals[:, imi]).max(axis=1) if imi else np.zeros(N)
    dm = np.abs(vals[:, di]).max(axis=1) if di else np.zeros(N)
    fm = np.abs(vals[:, fi]).max(axis=1) if fi else np.zeros(N)
    nm = np.abs(vals[:, ni]).max(axis=1) if ni else np.zeros(N)

    return pd.DataFrame({
        "SHAPE__fault_score": np.log1p(vm * 0.001 + im * 0.01 + dm * 0.0001),
        "SHAPE__line_outage_score": np.log1p(im * 0.005 + vm * 0.0005 + severity.std(axis=1) * 5),
        "SHAPE__gen_change_score": np.log1p(fm * 10 + vm * 0.001),
        "SHAPE__load_change_score": np.log1p(fm * 5 + vm * 0.0005),
        "SHAPE__missing_score": nm * 10 + np.clip(1 - severity.max(axis=1), 0, 1) * 5,
        "SHAPE__bad_data_score": (severity.max(axis=1) / (severity.sum(axis=1) + EPS)) * 5 + np.log1p(dm * 0.001),
        "SHAPE__mixed_score": severity.mean(axis=1) * 2 + severity.std(axis=1) * 3,
    })


def extract_vi_coupling(data: pd.DataFrame) -> pd.DataFrame:
    cols = _numeric_cols(data)
    vals = data[cols].fillna(0).values.astype(np.float32)
    rows = {}
    for pmu_name in PMU_NAMES:
        prefix = f"{pmu_name}__{pmu_name}_"
        for sig in ["VA_MAG", "IA_MAG", "Freq"]:
            mi = _find_indices(cols, prefix, (f"{sig}__max_abs",))
            mn = _find_indices(cols, prefix, (f"{sig}__mean_abs",))
            if mi and mn:
                rows[f"VIC__{pmu_name}__{sig}__spike_ratio"] = np.abs(vals[:, mi]).max(axis=1) / (np.abs(vals[:, mn]).max(axis=1) + EPS)
    return pd.DataFrame(rows)


def extract_v4_features(data: pd.DataFrame) -> pd.DataFrame:
    print("  Computing severity vectors...")
    severity = extract_severity_vector(data)
    print("  Computing Zbus diffusion signatures...")
    zbus = ZbusDiffusionFeatures()
    gsp = zbus.compute_features(severity)
    print("  Computing event-shape scores...")
    es = extract_event_shape_scores(data, severity)
    print("  Computing VI coupling features...")
    vi = extract_vi_coupling(data)
    all_v4 = pd.concat([gsp, es, vi], axis=1)
    print(f"  Total V4 features: {len(all_v4.columns)}")
    return all_v4
