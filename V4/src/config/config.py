from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_INPUT_PATTERN
from .enums import EventId


DEFAULT_EVENT_LABELS: dict[int, str] = {
    EventId.NORMAL_OPERATION: "Normal operation",
    EventId.FAULT: "Fault",
    EventId.LINE_OUTAGE: "Line outage",
    EventId.GENERATION_CHANGE_OUTAGE: "Generation change/outage",
    EventId.LOAD_CHANGE_DROP: "Load change/drop",
    EventId.MISSING_DATA: "Missing data",
    EventId.MISSING_DATA_PLUS_PHYSICAL_EVENT: "Missing data + physical event",
    EventId.BAD_DATA: "Bad data",
    EventId.UNKNOWN_EVENT: "Unknown event",
}

DEFAULT_EVENT_DESCRIPTIONS: dict[int, str] = {
    EventId.NORMAL_OPERATION: (
        "No disturbance. The system is in steady state or near-steady state. "
        "Minor natural fluctuations may be present."
    ),
    EventId.FAULT: (
        "A short-circuit event (e.g., three-phase-to-ground). Produces sudden "
        "voltage sags and current spikes, typically lasting a few cycles to several seconds."
    ),
    EventId.LINE_OUTAGE: (
        "A transmission line is disconnected (tripped or opened). Causes power-flow "
        "redistribution and voltage/angle shifts across the network."
    ),
    EventId.GENERATION_CHANGE_OUTAGE: (
        "A generator changes its MW output (step change) or trips offline entirely. "
        "Causes frequency deviation and system-wide power-flow redistribution."
    ),
    EventId.LOAD_CHANGE_DROP: (
        "A load suddenly increases, decreases, or disconnects. Similar to generation "
        "change but typically produces smaller frequency excursions."
    ),
    EventId.MISSING_DATA: (
        "PMU frame missing due to communication failure. All measurements are NaN; "
        "DATA_PRESENT = 0. No physical event is occurring."
    ),
    EventId.MISSING_DATA_PLUS_PHYSICAL_EVENT: (
        "Missing data at one PMU concurrent with a physical event elsewhere. "
        "Measurements are NaN at the affected PMU; the physical event must be inferred from other PMUs."
    ),
    EventId.BAD_DATA: (
        "Corrupted measurement frame(s): non-physical spikes, jumps, or inconsistent values. "
        "The PMU reports data, but the values are unreliable."
    ),
    EventId.UNKNOWN_EVENT: (
        "An abnormal pattern that does not match labels 1–7. "
        "This is an open-set class for ambiguous or unusual disturbances."
    ),
}


@dataclass(slots=True)
class AnalysisConfig:
    """
    Main runtime configuration for the RAW PMU scenario analyzer.
    """
    input_dir: Path
    output_dir: Path
    pattern: str = DEFAULT_INPUT_PATTERN

    # Time-window settings
    event_context_seconds: float = 1.0
    trend_window_seconds: float = 1.0
    event_baseline_seconds: float = 1.0
    event_post_seconds: float = 1.0

    # Detection / analysis thresholds
    artifact_z_threshold: float = 6.0

    # Plot configuration
    plots_dpi: int = 220
    plots_zoom_dpi: int = 400

    # Export toggles
    save_per_bus_json: bool = True
    save_event_bus_csvs: bool = True

    def validate(self) -> None:
        if self.event_context_seconds < 0:
            raise ValueError("event_context_seconds must be >= 0")
        if self.trend_window_seconds <= 0:
            raise ValueError("trend_window_seconds must be > 0")
        if self.event_baseline_seconds < 0:
            raise ValueError("event_baseline_seconds must be >= 0")
        if self.event_post_seconds < 0:
            raise ValueError("event_post_seconds must be >= 0")
        if self.artifact_z_threshold <= 0:
            raise ValueError("artifact_z_threshold must be > 0")
        if self.plots_dpi <= 0 or self.plots_zoom_dpi <= 0:
            raise ValueError("Plot DPI values must be > 0")

    @property
    def output_dataset_dir(self) -> Path:
        return self.output_dir / "dataset"

    @property
    def output_general_dir(self) -> Path:
        return self.output_dir / "general"

    @property
    def output_buses_dir(self) -> Path:
        return self.output_dir / "buses"

    @property
    def output_events_dir(self) -> Path:
        return self.output_dir / "events"