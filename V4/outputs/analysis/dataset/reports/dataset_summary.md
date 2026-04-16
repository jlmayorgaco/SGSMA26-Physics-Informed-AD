# Dataset summary

Input directory: `C:\Users\walla\Documents\Github\SGSMA26-Physics-Informed-AD\V4\data\RAW0001`

Bus count: **8**

Buses: Bus2, Bus5, Bus6, Bus10, Bus19, Bus22, Bus29, Bus39

## Event dictionary

- `0` → **Normal operation**: No disturbance. The system is in steady state or near-steady state. Minor natural fluctuations may be present.
- `1` → **Fault**: A short-circuit event (e.g., three-phase-to-ground). Produces sudden voltage sags and current spikes, typically lasting a few cycles to several seconds.
- `2` → **Line outage**: A transmission line is disconnected (tripped or opened). Causes power-flow redistribution and voltage/angle shifts across the network.
- `3` → **Generation change/outage**: A generator changes its MW output (step change) or trips offline entirely. Causes frequency deviation and system-wide power-flow redistribution.
- `4` → **Load change/drop**: A load suddenly increases, decreases, or disconnects. Similar to generation change but typically produces smaller frequency excursions.
- `5` → **Missing data**: PMU frame missing due to communication failure. All measurements are NaN; DATA_PRESENT = 0. No physical event is occurring.
- `6` → **Missing data + physical event**: Missing data at one PMU concurrent with a physical event elsewhere. Measurements are NaN at the affected PMU; the physical event must be inferred from other PMUs.
- `7` → **Bad data**: Corrupted measurement frame(s): non-physical spikes, jumps, or inconsistent values. The PMU reports data, but the values are unreliable.
- `8` → **Unknown event**: An abnormal pattern that does not match labels 1–7. This is an open-set class for ambiguous or unusual disturbances.
