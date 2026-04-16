from __future__ import annotations

# Canonical PMU measurement columns used across the project.
MEASUREMENT_COLUMNS: list[str] = [
    "VA_mag",
    "VA_ang",
    "VB_mag",
    "VB_ang",
    "VC_mag",
    "VC_ang",
    "IA_mag",
    "IA_ang",
    "IB_mag",
    "IB_ang",
    "IC_mag",
    "IC_ang",
    "Frequency",
    "ROCOF",
]

# Optional metadata columns present in the hackathon CSVs.
OPTIONAL_COLUMNS: list[str] = [
    "DATA_PRESENT",
    "Event",
]

# Angle-like columns. These may require unwrap / circular handling.
ANGLE_COLUMNS: set[str] = {
    "VA_ang",
    "VB_ang",
    "VC_ang",
    "IA_ang",
    "IB_ang",
    "IC_ang",
}

# Magnitude-like columns.
MAGNITUDE_COLUMNS: list[str] = [
    "VA_mag",
    "VB_mag",
    "VC_mag",
    "IA_mag",
    "IB_mag",
    "IC_mag",
]

# Voltage-only and current-only convenience groups.
VOLTAGE_COLUMNS: list[str] = [
    "VA_mag",
    "VA_ang",
    "VB_mag",
    "VB_ang",
    "VC_mag",
    "VC_ang",
]

CURRENT_COLUMNS: list[str] = [
    "IA_mag",
    "IA_ang",
    "IB_mag",
    "IB_ang",
    "IC_mag",
    "IC_ang",
]

# Three-phase groupings for reusable loops.
PHASE_VOLTAGE_MAG_COLUMNS: list[str] = ["VA_mag", "VB_mag", "VC_mag"]
PHASE_VOLTAGE_ANG_COLUMNS: list[str] = ["VA_ang", "VB_ang", "VC_ang"]

PHASE_CURRENT_MAG_COLUMNS: list[str] = ["IA_mag", "IB_mag", "IC_mag"]
PHASE_CURRENT_ANG_COLUMNS: list[str] = ["IA_ang", "IB_ang", "IC_ang"]

# Common channels typically used in global multi-bus comparisons.
CORE_CROSS_BUS_COLUMNS: list[str] = [
    "VA_mag",
    "IA_mag",
    "Frequency",
    "ROCOF",
]

# Canonical time column name.
TIMESTAMP_COLUMN: str = "TIMESTAMP"

# Default CSV glob used in the RAW PMU folders.
DEFAULT_INPUT_PATTERN: str = "Bus*_Competition_Data*.csv"

# Small numerical guard values.
EPS: float = 1e-12
DEFAULT_MIN_VALID_POINTS: int = 3