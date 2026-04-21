"""Constants for the M9 scenario simulator."""

from __future__ import annotations

PMU_BUSES_OFFICIAL: list[str] = ["BUS39", "BUS29", "BUS10", "BUS22", "BUS19", "BUS2", "BUS5", "BUS6"]

PMU_MEASUREMENT_SUFFIXES: list[str] = [
    "VA_ANG",
    "VA_MAG",
    "VB_ANG",
    "VB_MAG",
    "VC_ANG",
    "VC_MAG",
    "IA_ANG",
    "IA_MAG",
    "IB_ANG",
    "IB_MAG",
    "IC_ANG",
    "IC_MAG",
    "Freq",
    "ROCOF",
]

EVENT_LABELS: dict[int, str] = {
    0: "normal",
    1: "fault",
    2: "line_outage",
    3: "generation_change_or_outage",
    4: "load_change_or_drop",
    5: "missing_data",
    6: "missing_data_plus_physical_event",
    7: "bad_data",
    8: "unknown_or_composite",
}

PHYSICAL_EVENT_LABELS = {"fault": 1, "line_outage": 2, "generation_change": 3, "generation_outage": 3, "load_change": 4, "load_drop": 4}
MISSING_EVENT_TYPES = {"missing_data", "full_dropout", "burst_dropout", "periodic_dropout", "partial_channel_dropout"}
BAD_DATA_EVENT_TYPES = {
    "bad_data",
    "spike",
    "bias",
    "drift",
    "stuck_at_last_value",
    "gain_error",
    "clipping",
    "replay_window",
    "channel_swap",
    "angle_wrap_corruption",
    "fixed_delay",
    "variable_delay",
    "timestamp_jitter",
    "duplicated_frames",
    "frame_reordering",
}

DEFAULT_RATED_FREQUENCY_HZ = 60.0
DEFAULT_MVA_BASE = 100.0
