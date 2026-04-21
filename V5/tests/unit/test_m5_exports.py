from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metadata.exports import export_metadata_bundle


def _clean_dir(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        else:
            p.rmdir()
    path.rmdir()


def test_export_files_created_and_valid() -> None:
    out = Path("tests/fixtures/_tmp_m5_exports")
    out.mkdir(parents=True, exist_ok=True)
    try:
        y = np.array([[1 + 1j, 0], [0, 2 + 2j]], dtype=complex)
        z = np.array([[1 - 1j, 0], [0, 0.5 - 0.5j]], dtype=complex)
        summary = export_metadata_bundle(
            output_dir=out,
            system_metadata={"system_name": "x"},
            pmu_meta={"buses": [{"bus_label_canonical": "BUS1"}], "pmu_map": [{"pmu_id": 1, "bus_label_canonical": "BUS1"}]},
            raw_meta={"raw_sections_summary": {"bus_count": 2}},
            ybus=y,
            ybus_order=["BUS1", "BUS2"],
            ybus_info={},
            zbus=z,
            zbus_info={},
            electrical_distance=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            validation_report={"ok": True},
        )
        assert (out / "metadata_system.json").exists()
        assert (out / "metadata_buses.csv").exists()
        assert (out / "pmu_mapping.csv").exists()
        assert (out / "ybus_real.json").exists()
        assert (out / "zbus_real.json").exists()
        assert (out / "validation_report.json").exists()
        # JSON valid
        json.loads((out / "metadata_system.json").read_text(encoding="utf-8"))
        pd.read_csv(out / "metadata_buses.csv")
        assert "ybus_bus_order.json" in summary["files_written"]
    finally:
        _clean_dir(out)
