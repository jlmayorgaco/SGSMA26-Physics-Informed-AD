"""Parser for PMUbus_ Location.txt metadata file."""

from __future__ import annotations

from pathlib import Path
import re

from src.metadata.bus_mapping import canonicalize_bus_label, numeric_bus_sort_key
from src.metadata.models import BusRecord, PMUMapEntry


EXPECTED_PMU_MAP = {
    1: "BUS39",
    2: "BUS29",
    3: "BUS10",
    4: "BUS22",
    5: "BUS19",
    6: "BUS2",
    7: "BUS5",
    8: "BUS6",
}

BUS_TYPE_MAP = {1: "PQ", 2: "PV", 3: "SWING"}


def _parse_csvish(line: str) -> list[str]:
    return [p.strip() for p in line.split(",")]


def parse_pmu_location_file(path: str | Path) -> dict:
    """Parse PMU location metadata into canonical intermediate dict."""
    src = Path(path)
    lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Empty PMU location file: {src}")

    # File format:
    # MVA base
    # 100
    # Rated Frequency 60 HZ
    # 39 Bus Load flow
    # ...
    try:
        base_mva = float(lines[1].strip())
    except Exception as exc:
        raise ValueError("Invalid PMU file header; expected base MVA on line 2.") from exc
    system_name = lines[3].strip() if len(lines) > 3 else "IEEE 39 Bus"

    rated_frequency_hz = 60.0
    for line in lines[:6]:
        m = re.search(r"(\d+(?:\.\d+)?)\s*HZ", line.upper())
        if m:
            rated_frequency_hz = float(m.group(1))
            break

    buses: list[BusRecord] = []
    explicit_pmu_entries: list[PMUMapEntry] = []
    in_bus_section = False
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if "BUS NUMBER" in raw.upper() and "PMU" in raw.upper():
            in_bus_section = True
            continue
        if not in_bus_section:
            continue
        if not raw.startswith("'"):
            continue
        parts = _parse_csvish(raw)
        if len(parts) < 6:
            continue
        bus_label_original = parts[0].strip().strip("'").strip('"')
        bus_label_canonical = canonicalize_bus_label(bus_label_original)
        bus_type_raw = int(float(parts[2]))
        bus_type_name = BUS_TYPE_MAP.get(bus_type_raw, "UNKNOWN")
        kv_ll = float(parts[1])
        v_pu = float(parts[3])
        theta_deg = float(parts[4])

        pmu_id = None
        for token in parts[5:]:
            m = re.search(r"PMU\s*(\d+)", token.upper())
            if m:
                pmu_id = int(m.group(1))
                break
            token_clean = token.strip()
            if token_clean.isdigit():
                v = int(token_clean)
                if 1 <= v <= 8:
                    pmu_id = v
                    break

        bus_num_match = re.search(r"(\d+)", bus_label_canonical)
        bus_id_numeric = int(bus_num_match.group(1)) if bus_num_match else None
        buses.append(
            BusRecord(
                bus_id_numeric=bus_id_numeric,
                bus_label_original=bus_label_original,
                bus_label_canonical=bus_label_canonical,
                kv_ll=kv_ll,
                bus_type_raw=bus_type_raw,
                bus_type_name=bus_type_name,
                v_pu=v_pu,
                theta_deg=theta_deg,
                pmu_id=pmu_id,
                has_pmu=pmu_id is not None,
            )
        )
        if pmu_id is not None:
            explicit_pmu_entries.append(PMUMapEntry(pmu_id=pmu_id, bus_label_canonical=bus_label_canonical))

    if not explicit_pmu_entries:
        explicit_pmu_entries = [
            PMUMapEntry(pmu_id=k, bus_label_canonical=v) for k, v in sorted(EXPECTED_PMU_MAP.items(), key=lambda x: x[0])
        ]

    pmu_bus_ids = sorted([e.bus_label_canonical for e in explicit_pmu_entries], key=numeric_bus_sort_key)
    buses_sorted = sorted(buses, key=lambda b: numeric_bus_sort_key(b.bus_label_canonical))

    return {
        "system_name": system_name,
        "base_mva": base_mva,
        "rated_frequency_hz": rated_frequency_hz,
        "buses": [
            {
                "bus_id_numeric": b.bus_id_numeric,
                "bus_label_original": b.bus_label_original,
                "bus_label_canonical": b.bus_label_canonical,
                "kv_ll": b.kv_ll,
                "bus_type_raw": b.bus_type_raw,
                "bus_type_name": b.bus_type_name,
                "v_pu": b.v_pu,
                "theta_deg": b.theta_deg,
                "pmu_id": b.pmu_id,
                "has_pmu": b.has_pmu,
            }
            for b in buses_sorted
        ],
        "pmu_map": [{"pmu_id": e.pmu_id, "bus_label_canonical": e.bus_label_canonical} for e in explicit_pmu_entries],
        "pmu_bus_ids": pmu_bus_ids,
        "bus_count": len(buses_sorted),
    }
