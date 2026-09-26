from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]/'powerdynamics_ieee39/output/results'; P=R.parent/'plots'; P.mkdir(exist_ok=True)
s=pd.read_csv(R/'e04a_b3_summary.csv');
for y,n in [('TVE_percent','e04a_b3_tve_vs_lag.png'),('angle_rmse','e04a_b3_angle_vs_lag.png')]:
 plt.figure();plt.plot(s.lag_seconds,s[y],'o-');plt.xlabel('lag (s)');plt.ylabel(y);plt.tight_layout();plt.savefig(P/n,dpi=160);plt.close()
r=pd.read_csv(R/'e04a_b3_runtime.csv');plt.figure();plt.plot(r.lag_seconds,r.median_ms_per_frame,'o-');plt.axhline(33.333,color='r');plt.xlabel('lag (s)');plt.ylabel('ms/frame');plt.tight_layout();plt.savefig(P/'e04a_b3_runtime_vs_lag.png',dpi=160);plt.close()
plt.figure();plt.plot(s.lag_seconds,s.coverage_re,'o-',label='Re');plt.plot(s.lag_seconds,s.coverage_im,'s-',label='Im');plt.axhline(.95,color='r');plt.legend();plt.xlabel('lag (s)');plt.ylabel('coverage');plt.tight_layout();plt.savefig(P/'e04a_b3_coverage_vs_lag.png',dpi=160);plt.close()
