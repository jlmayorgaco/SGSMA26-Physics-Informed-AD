from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_synthetic_scenario(root: Path, scenario_id: str = "SIMX", *, abnormal_start_idx: int = 20) -> Path:
    scenario_dir = root / scenario_id
    pmu_dir = scenario_dir / "pmu"
    labels_dir = scenario_dir / "labels"
    metadata_dir = scenario_dir / "metadata"
    pmu_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    n = 80
    ts = np.arange(n, dtype=float) * 0.02
    event = np.zeros(n, dtype=int)
    event[abnormal_start_idx:] = 5
    data_present = np.ones(n, dtype=float)
    data_present[abnormal_start_idx + 5 : abnormal_start_idx + 10] = 0.0
    base_mag = np.linspace(220000.0, 221000.0, n)
    base_ang = np.linspace(-5.0, 5.0, n)

    for bus in (10, 39):
        frame = pd.DataFrame(
            {
                "TIMESTAMP": ts,
                f"BUS{bus}_VA_MAG": base_mag + bus,
                f"BUS{bus}_VA_ANG": base_ang + bus * 0.01,
                f"BUS{bus}_VB_MAG": base_mag + bus + 5.0,
                f"BUS{bus}_VB_ANG": base_ang + 120.0 + bus * 0.01,
                f"BUS{bus}_VC_MAG": base_mag + bus + 8.0,
                f"BUS{bus}_VC_ANG": base_ang - 120.0 + bus * 0.01,
                f"BUS{bus}_Freq": 60.0 + 0.01 * np.sin(np.linspace(0, 2, n)),
                f"BUS{bus}_ROCOF": 0.001 * np.cos(np.linspace(0, 2, n)),
                "DATA_PRESENT": data_present,
                "Event": event,
            }
        )
        frame.to_csv(pmu_dir / f"Bus{bus}_Competition_Data_sim.csv", index=False)

    labels = pd.DataFrame(
        {
            "TIMESTAMP": ts,
            "EVENT": event,
            "IS_PHYSICAL_EVENT": np.zeros(n, dtype=bool),
            "IS_CYBER_EVENT": event > 0,
            "IS_CONCURRENT_EVENT": np.zeros(n, dtype=bool),
            "CYBER_SUBTYPE": ["full_dropout" if x > 0 else "" for x in event],
            "TARGET_PMU": ["BUS39" if x > 0 else "" for x in event],
            "DIFFICULTY_LEVEL": ["easy"] * n,
        }
    )
    labels.to_csv(labels_dir / "event_frame_labels.csv", index=False)

    (metadata_dir / "scenario_scoring.json").write_text(
        '{"estimator_difficulty_score": 0.2, "detector_difficulty_score": 0.3, "overall_training_value_score": 0.5}',
        encoding="utf-8",
    )
    return scenario_dir

