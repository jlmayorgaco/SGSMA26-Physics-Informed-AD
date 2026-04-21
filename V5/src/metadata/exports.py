"""Artifact exports for M5 metadata/model bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _matrix_payload(mat: np.ndarray, bus_order: list[str], component: str) -> dict:
    if component == "real":
        arr = np.real(mat)
    elif component == "imag":
        arr = np.imag(mat)
    elif component == "mag":
        arr = np.abs(mat)
    elif component == "angle_deg":
        arr = np.rad2deg(np.angle(mat))
    else:
        raise ValueError(f"Unknown matrix component: {component}")
    return {"bus_order": list(bus_order), "matrix": arr.tolist()}


def export_metadata_bundle(
    output_dir: str | Path,
    system_metadata: dict,
    pmu_meta: dict,
    raw_meta: dict,
    ybus: np.ndarray,
    ybus_order: list[str],
    ybus_info: dict,
    zbus: np.ndarray,
    zbus_info: dict,
    electrical_distance: np.ndarray,
    validation_report: dict,
) -> dict:
    """Write all required M5 JSON/CSV/NPZ artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "metadata_system.json").write_text(json.dumps(system_metadata, indent=2), encoding="utf-8")
    pd.DataFrame(pmu_meta.get("buses", [])).to_csv(out / "metadata_buses.csv", index=False)
    pd.DataFrame(pmu_meta.get("pmu_map", [])).to_csv(out / "pmu_mapping.csv", index=False)
    (out / "raw_sections_summary.json").write_text(
        json.dumps(raw_meta.get("raw_sections_summary", {}), indent=2), encoding="utf-8"
    )

    for comp in ["real", "imag", "mag", "angle_deg"]:
        (out / f"ybus_{comp}.json").write_text(
            json.dumps(_matrix_payload(ybus, ybus_order, comp), indent=2), encoding="utf-8"
        )
        (out / f"zbus_{comp}.json").write_text(
            json.dumps(_matrix_payload(zbus, ybus_order, comp), indent=2), encoding="utf-8"
        )
    (out / "ybus_bus_order.json").write_text(json.dumps({"bus_order": ybus_order}, indent=2), encoding="utf-8")
    (out / "zbus_bus_order.json").write_text(json.dumps({"bus_order": ybus_order}, indent=2), encoding="utf-8")
    (out / "electrical_distance_matrix.json").write_text(
        json.dumps({"bus_order": ybus_order, "matrix": np.asarray(electrical_distance, dtype=float).tolist()}, indent=2),
        encoding="utf-8",
    )
    (out / "validation_report.json").write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

    np.savez_compressed(out / "ybus_complex.npz", ybus=ybus, bus_order=np.array(ybus_order, dtype=object))
    np.savez_compressed(out / "zbus_complex.npz", zbus=zbus, bus_order=np.array(ybus_order, dtype=object))

    return {
        "output_dir": str(out),
        "ybus_shape": list(np.asarray(ybus).shape),
        "zbus_shape": list(np.asarray(zbus).shape),
        "bus_count": len(ybus_order),
        "files_written": sorted([p.name for p in out.iterdir() if p.is_file()]),
        "ybus_info": ybus_info,
        "zbus_info": zbus_info,
    }
