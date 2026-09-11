"""Freeze E03 predictions before inspecting E04 reconstruction errors."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'output/results'
h=pd.read_csv(out/'pd_observability_horizons.csv'); hidden=[b for b in range(1,40) if b not in [2,5,6,10,19,22,29,39]]
rows=[]
for bus in hidden:
    r=h.iloc[-1]; rows.append({'bus':bus,'e03_horizon_frames':int(r.horizon),'e03_functional_residual_global':float(r.functional_output_residual),'e03_information_bound_sigma1e3':float(r.noise_information_bound_sigma1e3),'e03_worst_hidden_bus_180':int(r.worst_hidden_bus),'e03_predicted_weak':bool(bus==int(r.worst_hidden_bus)),'preregistered_before_e04':True})
pd.DataFrame(rows).to_csv(out/'e04_preregistered_e03_predictions.csv',index=False)
print(f'preregistered {len(rows)} hidden buses')
