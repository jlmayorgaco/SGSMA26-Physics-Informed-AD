"""
Debug script: print pre-event mean voltage magnitude in p.u. for each bus in SIM_0001.
Shows both hypotheses side by side:
  Hypothesis A — nominal base (BUS_BASE_KV_LL from network model)
  Hypothesis B — measurement override: buses 20 and 30-38 use 345 kV base
"""
from __future__ import annotations

from src.utils.data_loader import list_available_buses, load_synth_bus_data
from src.utils.per_unit import (
    BUS_BASE_KV_LL,
    BUS_MEASUREMENT_BASE_KV_LL_OVERRIDE_345,
    pre_event_mean_pu,
)

SCENARIO_ID = "SIM_0001"

buses = list_available_buses(scenario_id=SCENARIO_ID)

print(f"Scenario: {SCENARIO_ID}")
print(f"Pre-event mean voltage magnitude in p.u. — both hypotheses\n")
print(
    f"{'Bus':>5}  {'Nom. base [kV]':>14}  {'Meas. base [kV]':>15}  "
    f"{'HypA |V| p.u.':>14}  {'HypB |V| p.u.':>14}"
)
print("-" * 74)

for bus_id in buses:
    df = load_synth_bus_data(scenario_id=SCENARIO_ID, bus_id=bus_id)

    # Use simple phase-average magnitude as the representative scalar magnitude
    va_mag = df["VA_MAG"].to_numpy()
    vb_mag = df["VB_MAG"].to_numpy()
    vc_mag = df["VC_MAG"].to_numpy()
    mag = (va_mag + vb_mag + vc_mag) / 3.0

    event = df["Event"].to_numpy()

    pu_a = pre_event_mean_pu(magnitude=mag, event=event, bus_id=bus_id, mode="nominal")
    pu_b = pre_event_mean_pu(magnitude=mag, event=event, bus_id=bus_id, mode="override_345")

    nom_kv = BUS_BASE_KV_LL[bus_id]
    meas_kv = BUS_MEASUREMENT_BASE_KV_LL_OVERRIDE_345[bus_id]

    print(
        f"{bus_id:>5}  {nom_kv:>14.1f}  {meas_kv:>15.1f}  "
        f"{pu_a:>14.6f}  {pu_b:>14.6f}"
    )
