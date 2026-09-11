"""Create the preregistered E04-A1 diagnostic plots from frozen outputs."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
R, P = ROOT / "output/results", ROOT / "output/plots"
P.mkdir(exist_ok=True)

def save(name):
    plt.tight_layout(); plt.savefig(P / name, dpi=160); plt.close()

def main():
    s = pd.read_csv(R / "e04a_b0_b1_b2_summary.csv")
    plt.figure(); plt.bar(s.method, s.TVE_percent); plt.ylabel("TVE (%)"); save("e04a_method_comparison.png")
    b = pd.read_csv(R / "e04a_full_per_bus.csv"); plt.figure()
    for m, g in b[b.method.isin(["B1_SNAPSHOT_WLS", "B2_KALMAN"])].groupby("method"):
        plt.plot(g.hidden_bus, g.TVE_percent, "o-", label=m)
    plt.xlabel("hidden bus"); plt.ylabel("TVE (%)"); plt.legend(); save("e04a_per_bus_tve.png")
    e = pd.read_csv(R / "e04a1_e03_vs_e04.csv"); plt.figure(); plt.scatter(e.functional_residual, e.B2_TVE); plt.xlabel("E03 functional residual"); plt.ylabel("B2 TVE"); save("e04a_e03_vs_e04.png")
    u = pd.read_csv(R / "e04a_uncertainty.csv"); plt.figure(); plt.scatter(u.predicted_variance, u.coverage_re.astype(int), s=2); plt.xlabel("predicted variance"); plt.ylabel("Re coverage"); save("e04a_coverage.png")
    i = pd.read_csv(R / "e04a_innovations.csv"); plt.figure(); plt.acorr(i.innovation_norm - i.innovation_norm.mean(), maxlags=60); plt.xlabel("lag"); plt.ylabel("innovation ACF"); save("e04a_innovation_acf.png")
    plt.figure(); plt.plot([0], [s[s.method == "B2_KALMAN"].TVE_percent.iloc[0]], "o"); plt.xlabel("lag (frames)"); plt.ylabel("TVE (%)"); save("e04a_lag_tradeoff.png")
    rt = pd.read_csv(R / "e04a_runtime.csv"); plt.figure(); plt.bar(rt.method, rt.p95); plt.axhline(0.03333, color="r", linestyle="--"); plt.ylabel("seconds/frame (p95)"); save("e04a_runtime_vs_lag.png")

if __name__ == "__main__": main()
