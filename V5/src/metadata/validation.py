"""Consistency validation between PMU metadata and RAW model."""

from __future__ import annotations

from src.metadata.bus_mapping import build_bus_alignment, canonicalize_bus_label


EXPECTED_PMU_BUSES = {"BUS39", "BUS29", "BUS10", "BUS22", "BUS19", "BUS2", "BUS5", "BUS6"}


def _bus_lookup(records: list[dict]) -> dict[str, dict]:
    return {canonicalize_bus_label(r["bus_label_canonical"]): r for r in records}


def validate_metadata_consistency(pmu_meta: dict, raw_meta: dict, ybus_info: dict | None = None) -> dict:
    """Build structured validation report for M5 artifacts."""
    alignment = build_bus_alignment(pmu_meta, raw_meta)
    pmu_bus_set = set(pmu_meta.get("pmu_bus_ids", []))
    pmu_mapping_ok = pmu_bus_set == EXPECTED_PMU_BUSES

    base_mva_ok = abs(float(pmu_meta.get("base_mva", 0.0)) - float(raw_meta.get("base_mva", 0.0))) < 1e-9
    rated_frequency_hz_ok = abs(float(pmu_meta.get("rated_frequency_hz", 60.0)) - 60.0) < 1e-9

    pmu_buses = _bus_lookup(pmu_meta.get("buses", []))
    raw_buses = _bus_lookup(raw_meta.get("buses", []))

    kv_mismatch = []
    type_mismatch = []
    type_map = {1: "PQ", 2: "PV", 3: "SWING"}
    for b in alignment["common_bus_set"]:
        pmu_rec = pmu_buses[b]
        raw_rec = raw_buses[b]
        kv1 = pmu_rec.get("kv_ll")
        kv2 = raw_rec.get("kv_ll")
        if kv1 is not None and kv2 is not None and abs(float(kv1) - float(kv2)) > 1e-6:
            kv_mismatch.append({"bus": b, "pmu_kv": kv1, "raw_kv": kv2})
        pmu_t = str(pmu_rec.get("bus_type_name", "")).upper()
        raw_t = type_map.get(int(raw_rec.get("bus_type_raw", 0)), "UNKNOWN")
        if pmu_t != raw_t:
            type_mismatch.append({"bus": b, "pmu_type": pmu_t, "raw_type": raw_t})

    bus_kv_ok = len(kv_mismatch) == 0
    bus_type_ok = len(type_mismatch) == 0
    ybus_built_ok = bool(ybus_info and ybus_info.get("shape"))
    zbus_built_ok = bool(ybus_info and ybus_info.get("zbus_status", {}).get("built", False))

    ready_for_estimation = (
        alignment["aligned_ok"]
        and pmu_mapping_ok
        and base_mva_ok
        and rated_frequency_hz_ok
        and ybus_built_ok
    )

    return {
        "metadata_vs_raw_bus_set_ok": alignment["aligned_ok"],
        "pmu_mapping_ok": pmu_mapping_ok,
        "base_mva_ok": base_mva_ok,
        "rated_frequency_hz_ok": rated_frequency_hz_ok,
        "bus_kv_ok": bus_kv_ok,
        "bus_type_ok": bus_type_ok,
        "ybus_built_ok": ybus_built_ok,
        "zbus_built_ok": zbus_built_ok,
        "ready_for_estimation": ready_for_estimation,
        "bus_alignment": alignment,
        "mismatches": {
            "kv_mismatch": kv_mismatch,
            "bus_type_mismatch": type_mismatch,
        },
        "notes": [
            "Metadata is necessary but not sufficient for full state estimation.",
            "Full estimation also depends on load/generator injections, correct branch/transformer model,",
            "current measurement mapping, and chosen state-estimation formulation.",
        ],
    }
