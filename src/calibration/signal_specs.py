"""Signal specification contract for m3 raw event-0 calibration."""

from __future__ import annotations


def build_signal_specs() -> list[dict]:
    """Build legacy-compatible signal specification entries."""
    specs = []
    for ph in ["A", "B", "C"]:
        specs.append(
            {
                "signal_key": f"V{ph}_mag",
                "raw_suffix": f"V{ph}_MAG",
                "sim_source": "voltage_mag",
                "support_status": "supported_direct",
                "raw_representation": "raw voltage magnitude in physical dataset units",
                "recommended": True,
                "notes": "ANDES positive-sequence Bus.v mapped to raw magnitude using event-0 calibration.",
            }
        )
    for ph in ["A", "B", "C"]:
        specs.append(
            {
                "signal_key": f"I{ph}_mag",
                "raw_suffix": f"I{ph}_MAG",
                "sim_source": "current_mag_selected",
                "support_status": "supported_derived",
                "raw_representation": "selected positive-sequence equivalent current magnitude in A",
                "recommended": False,
                "notes": "Mapping selected per bus from injection, incident branches, dominant branch, and generator candidates.",
            }
        )
    specs.extend(
        [
            {
                "signal_key": "Frequency",
                "raw_suffix": "Freq",
                "sim_source": "frequency",
                "support_status": "supported_derived",
                "raw_representation": "raw PMU frequency in Hz",
                "recommended": True,
                "notes": "Derived from ANDES bus-angle speed, then calibrated in raw units.",
            },
            {
                "signal_key": "ROCOF",
                "raw_suffix": "ROCOF",
                "sim_source": "rocof",
                "support_status": "supported_derived",
                "raw_representation": "raw PMU ROCOF",
                "recommended": False,
                "notes": "Derived from frequency gradient; often too noisy/flat in normal operation.",
            },
        ]
    )
    for ph in ["A", "B", "C"]:
        specs.append(
            {
                "signal_key": f"V{ph}_ang_delta",
                "raw_suffix": f"V{ph}_ANG",
                "sim_source": "voltage_angle_delta",
                "support_status": "experimental_relative_only",
                "raw_representation": "wrap(raw voltage angle - local event-0 median) in degrees",
                "recommended": False,
                "notes": "Raw absolute PMU angle is unsupported; only wrapped local delta is evaluated as experimental.",
            }
        )
    for ph in ["A", "B", "C"]:
        specs.append(
            {
                "signal_key": f"I{ph}_ang",
                "raw_suffix": f"I{ph}_ANG",
                "sim_source": None,
                "support_status": "unsupported_raw_absolute",
                "raw_representation": "raw current absolute angle",
                "recommended": False,
                "notes": "Current absolute angle has no robust physical mapping in this positive-sequence model.",
            }
        )
    return specs


SIGNAL_SPECS = build_signal_specs()
