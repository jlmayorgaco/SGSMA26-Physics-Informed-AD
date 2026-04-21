from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PMU_BUSES = ["39", "29", "10", "22", "19", "2", "5", "6"]


def create_synthetic_pmu_csvs(base_dir: Path, n: int = 8, missing_bus: str | None = None) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    t = np.round(np.arange(n) * (1.0 / 30.0), 3)
    for bus in PMU_BUSES:
        data_present = np.ones(n, dtype=int)
        if missing_bus is not None and bus == missing_bus:
            data_present[:] = 0
        df = pd.DataFrame(
            {
                "TIMESTAMP": t,
                f"BUS{bus}_VA_ANG": np.zeros(n),
                f"BUS{bus}_VA_MAG": np.full(n, 200000.0),
                f"BUS{bus}_VB_ANG": np.full(n, -120.0),
                f"BUS{bus}_VB_MAG": np.full(n, 200000.0),
                f"BUS{bus}_VC_ANG": np.full(n, 120.0),
                f"BUS{bus}_VC_MAG": np.full(n, 200000.0),
                f"BUS{bus}_IA_ANG": np.zeros(n),
                f"BUS{bus}_IA_MAG": np.full(n, 500.0),
                f"BUS{bus}_IB_ANG": np.full(n, -120.0),
                f"BUS{bus}_IB_MAG": np.full(n, 500.0),
                f"BUS{bus}_IC_ANG": np.full(n, 120.0),
                f"BUS{bus}_IC_MAG": np.full(n, 500.0),
                f"BUS{bus}_Freq": np.full(n, 60.0),
                f"BUS{bus}_ROCOF": np.zeros(n),
                "DATA_PRESENT": data_present,
                "Event": np.zeros(n, dtype=int),
            }
        )
        df.to_csv(base_dir / f"Bus{bus}_Competition_Data_nanmask.csv", index=False)
    return base_dir

