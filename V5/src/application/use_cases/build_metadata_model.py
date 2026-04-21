"""Use case for building M5 metadata/model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.metadata.bus_mapping import build_bus_alignment
from src.metadata.electrical_distance import build_electrical_distance_matrix
from src.metadata.exports import export_metadata_bundle
from src.metadata.pmu_location_parser import parse_pmu_location_file
from src.metadata.raw_parser import parse_raw_file
from src.metadata.validation import validate_metadata_consistency
from src.metadata.ybus_builder import build_ybus
from src.metadata.zbus_builder import build_zbus


def _build_system_metadata(pmu_meta: dict, raw_meta: dict, ybus_info: dict, zbus_info: dict) -> dict[str, Any]:
    return {
        "system_name": pmu_meta.get("system_name", "IEEE 39 Bus"),
        "base_mva": float(pmu_meta.get("base_mva", raw_meta.get("base_mva", 100.0))),
        "rated_frequency_hz": float(pmu_meta.get("rated_frequency_hz", 60.0)),
        "bus_count": int(len(pmu_meta.get("buses", []))),
        "generator_count": int(len(raw_meta.get("generators", []))),
        "line_count": int(len(raw_meta.get("branches", []))),
        "transformer_count": int(len(raw_meta.get("transformers", []))),
        "buses": pmu_meta.get("buses", []),
        "pmu_bus_ids": pmu_meta.get("pmu_bus_ids", []),
        "pmu_map": pmu_meta.get("pmu_map", []),
        "raw_sections_summary": raw_meta.get("raw_sections_summary", {}),
        "ybus_metadata": ybus_info,
        "zbus_metadata": zbus_info,
    }


def run_build_metadata_model_use_case(
    pmu_location_path: str | Path,
    raw_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Parse/validate/build/export complete M5 metadata model bundle."""
    pmu_path = Path(pmu_location_path)
    raw_file = Path(raw_path)
    out = Path(output_dir)

    if not pmu_path.exists():
        raise FileNotFoundError(f"PMU location file not found: {pmu_path}")
    if not raw_file.exists():
        raise FileNotFoundError(f"RAW file not found: {raw_file}")

    pmu_meta = parse_pmu_location_file(pmu_path)
    raw_meta = parse_raw_file(raw_file)
    alignment = build_bus_alignment(pmu_meta, raw_meta)

    ybus, bus_order, ybus_info = build_ybus(raw_meta)
    zbus, zbus_info = build_zbus(ybus)
    z_status = {"built": True, "invertible": bool(zbus_info.get("invertible", False))}
    ybus_info = {**ybus_info, "zbus_status": z_status}

    distance = build_electrical_distance_matrix(zbus)
    validation = validate_metadata_consistency(pmu_meta, raw_meta, ybus_info=ybus_info)
    system_metadata = _build_system_metadata(pmu_meta, raw_meta, ybus_info, zbus_info)

    export_summary = export_metadata_bundle(
        output_dir=out,
        system_metadata=system_metadata,
        pmu_meta=pmu_meta,
        raw_meta=raw_meta,
        ybus=ybus,
        ybus_order=bus_order,
        ybus_info=ybus_info,
        zbus=zbus,
        zbus_info=zbus_info,
        electrical_distance=distance,
        validation_report=validation,
    )
    return {
        "output_dir": str(out),
        "pmu_location_path": str(pmu_path),
        "raw_path": str(raw_file),
        "bus_alignment_ok": alignment["aligned_ok"],
        "ready_for_estimation": validation.get("ready_for_estimation", False),
        "ybus_shape": list(ybus.shape),
        "zbus_shape": list(zbus.shape),
        "files_written": export_summary["files_written"],
    }
