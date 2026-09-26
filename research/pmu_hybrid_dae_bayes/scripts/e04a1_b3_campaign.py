"""Execute selected-lag cached RTS on all 100 TEST trajectories."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.linalg import expm
from pmu_hybrid.e04a_data import observed_measurements, evaluation_ground_truth
from pmu_hybrid.e04a_estimator import LinearGaussianModel, fixed_lag_filter_fast_multi, interleaved_to_complex, wrapped_angle_error, _rts_covariance_cache

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"; R = ROOT / "output/results"

def main():
    d = pd.read_csv(R / "e04_pd_dataset.csv"); y0 = pd.read_csv(R / "e04_y0_pmu.csv").iloc[:, 0].to_numpy(); h0 = pd.read_csv(R / "e04_pd_hidden0.csv").iloc[:, 0].to_numpy()
    A = expm(pd.read_csv(R / "e04_A.csv").to_numpy(float) / 30); C = pd.read_csv(R / "e04_C_pmu.csv").to_numpy(float); Ct = pd.read_csv(R / "e04_C_hidden.csv").to_numpy(float)
    m = LinearGaussianModel(A, C, np.eye(114) * 1e-6, np.eye(32) * 1e-6, np.eye(114) * 1e-2); rows = []; cache = _rts_covariance_cache(m, 91)
    for traj, g in d[d.split == "TEST"].groupby("traj", sort=True):
        g = g.sort_values("frame"); y = np.vstack([observed_measurements(r) - y0 for _, r in g.iterrows()]); t = interleaved_to_complex(np.vstack([evaluation_ground_truth(r) for _, r in g.iterrows()]))
        t0 = time.perf_counter(); multi = fixed_lag_filter_fast_multi(m, y, [0, 3, 10, 30, 60], cache=cache); elapsed_total = time.perf_counter() - t0
        for L in [0, 3, 10, 30, 60]:
            out = multi[L]; elapsed = elapsed_total / 5; idx = [i for i, q in enumerate(out) if q is not None]
            p = np.vstack([h0 + Ct @ out[i][0] for i in idx]); z = interleaved_to_complex(p); tt = t[idx]; e = z - tt
            rows.append({"traj": traj, "lag_frames": L, "lag_seconds": L / 30, "complex_rmse": float(np.sqrt(np.mean(abs(e) ** 2))), "vm_rmse": float(np.sqrt(np.mean((abs(z) - abs(tt)) ** 2))), "angle_rmse": float(np.sqrt(np.mean(np.rad2deg(wrapped_angle_error(np.angle(z), np.angle(tt))) ** 2))), "TVE_fraction": float(np.mean(abs(e) / abs(tt))), "TVE_percent": float(100 * np.mean(abs(e) / abs(tt))), "runtime_s_per_frame": elapsed / len(y)})
    pd.DataFrame(rows).to_csv(R / "e04a_b3_lag_sweep.csv", index=False)

if __name__ == "__main__": main()
