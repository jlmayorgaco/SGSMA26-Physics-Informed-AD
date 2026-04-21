"""Lightweight project-specific parser for IEEE_39_Bus_Power_System.raw."""

from __future__ import annotations

from pathlib import Path

from src.metadata.bus_mapping import canonicalize_bus_label, numeric_bus_sort_key


def _clean_line(line: str) -> str:
    return line.strip()


def _strip_comment(line: str) -> str:
    if "/" in line:
        return line.split("/", 1)[0].strip()
    return line.strip()


def _split_fields(line: str) -> list[str]:
    return [p.strip() for p in line.split(",")]


def parse_raw_file(path: str | Path) -> dict:
    """Parse the portions of RAW required for M5 (bus/load/gen/branch/xfmr/shunt summary)."""
    src = Path(path)
    lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Empty RAW file: {src}")

    header = _split_fields(_strip_comment(lines[0]))
    if len(header) < 2:
        raise ValueError("Invalid RAW header line; expected base MVA in column 2.")
    base_mva = float(header[1])

    buses: list[dict] = []
    loads: list[dict] = []
    generators: list[dict] = []
    branches: list[dict] = []
    transformers: list[dict] = []
    fixed_shunts: list[dict] = []

    section = "bus"
    for raw_line in lines[1:]:
        line = _clean_line(raw_line)
        if not line:
            continue
        low = line.lower()
        if low.startswith("0 /end bus section"):
            section = "load"
            continue
        if low.startswith("0 /end load section"):
            section = "generator"
            continue
        if low.startswith("0 /end source section") or low.startswith("0 /end generator section"):
            section = "branch"
            continue
        if line == "0":
            # Remaining trailing sections are ignored for this project parser.
            if section == "branch":
                section = "done"
            continue
        if section == "done":
            continue

        content = _strip_comment(line)
        parts = _split_fields(content)
        if section == "bus":
            if len(parts) < 10:
                continue
            bus_label = parts[1].strip().strip("'").strip('"')
            bus_canon = canonicalize_bus_label(bus_label)
            buses.append(
                {
                    "bus_id_numeric": int(float(parts[0])),
                    "bus_label_original": bus_label,
                    "bus_label_canonical": bus_canon,
                    "kv_ll": float(parts[2]),
                    "bus_type_raw": int(float(parts[3])),
                    "v_pu": float(parts[8]),
                    "theta_deg": float(parts[9]),
                }
            )
        elif section == "load":
            if len(parts) < 7:
                continue
            bus_num = int(float(parts[0]))
            bus_label = f"BUS{bus_num}"
            loads.append(
                {
                    "bus_label_canonical": canonicalize_bus_label(bus_label),
                    "load_id": parts[1].strip().strip("'").strip('"'),
                    "status": int(float(parts[2])) if parts[2] else 1,
                    "p_mw": float(parts[5]),
                    "q_mvar": float(parts[6]),
                }
            )
        elif section == "generator":
            if len(parts) < 5:
                continue
            bus_num = int(float(parts[0]))
            bus_label = f"BUS{bus_num}"
            generators.append(
                {
                    "bus_label_canonical": canonicalize_bus_label(bus_label),
                    "gen_id": str(parts[1]).strip().strip("'").strip('"'),
                    "p_mw": float(parts[2]),
                    "q_mvar": float(parts[3]),
                    "status": int(float(parts[14])) if len(parts) > 14 and parts[14] else 1,
                }
            )
        elif section == "branch":
            if len(parts) < 6:
                continue
            i = int(float(parts[0]))
            j = int(float(parts[1]))
            ck = str(parts[2]).strip().strip("'").strip('"')
            r = float(parts[3])
            x = float(parts[4])
            b = float(parts[5]) if len(parts) > 5 and parts[5] else 0.0
            tap = float(parts[8]) if len(parts) > 8 and parts[8] else 0.0
            shift = float(parts[9]) if len(parts) > 9 and parts[9] else 0.0
            status = int(float(parts[15])) if len(parts) > 15 and parts[15] else 1
            rec = {
                "from_bus": canonicalize_bus_label(f"BUS{i}"),
                "to_bus": canonicalize_bus_label(f"BUS{j}"),
                "circuit_id": ck,
                "r_pu": r,
                "x_pu": x,
                "b_pu": b,
                "tap_ratio": tap,
                "phase_shift_deg": shift,
                "status": status,
            }
            branches.append(rec)
            if (tap not in (0.0, 1.0)) or abs(shift) > 0.0:
                transformers.append(
                    {
                        "from_bus": rec["from_bus"],
                        "to_bus": rec["to_bus"],
                        "circuit_id": rec["circuit_id"],
                        "r_pu": rec["r_pu"],
                        "x_pu": rec["x_pu"],
                        "tap_ratio": 1.0 if tap == 0.0 else tap,
                        "phase_shift_deg": rec["phase_shift_deg"],
                        "status": rec["status"],
                    }
                )

    buses_sorted = sorted(buses, key=lambda b: numeric_bus_sort_key(b["bus_label_canonical"]))
    return {
        "base_mva": base_mva,
        "buses": buses_sorted,
        "loads": loads,
        "generators": generators,
        "branches": branches,
        "transformers": transformers,
        "fixed_shunts": fixed_shunts,
        "raw_sections_summary": {
            "bus_count": len(buses_sorted),
            "load_count": len(loads),
            "generator_count": len(generators),
            "branch_count": len(branches),
            "transformer_count": len(transformers),
            "fixed_shunt_count": len(fixed_shunts),
        },
    }
