"""RAW-informed cyber layer orchestrator for Event 5 and Event 7."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.detectors.shared.preprocessing.pmu_alignment import align_scenario_pmu_dir
from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.event5_dropout_process import Event5DropoutProcess
from src.simulation.m9.event7_corruption_process import Event7CorruptionProcess
from src.simulation.m9.robust_noise_model import RobustNoiseModel


@dataclass(slots=True)
class CyberIntervalSpec:
    """Cyber interval applied by the RAW-informed layer."""

    kind: str  # "event5" or "event7"
    start_time_s: float
    end_time_s: float
    target_pmus: list[str]
    subtype: str
    event_label_if_no_physical: int
    event_label_if_physical: int


def _measurement_columns(bus: str, frame: pd.DataFrame) -> list[str]:
    return [f"{bus}_{suffix}" for suffix in PMU_MEASUREMENT_SUFFIXES if f"{bus}_{suffix}" in frame.columns]


def load_scenario_pmu_frames(scenario_dir: Path) -> dict[str, pd.DataFrame]:
    pmu_dir = scenario_dir / "pmu"
    if not pmu_dir.exists():
        raise FileNotFoundError(f"Missing PMU dir: {pmu_dir}")
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(pmu_dir.glob("Bus*_Competition_Data_sim.csv")):
        name = path.stem
        bus_token = "".join(ch for ch in name.split("_", maxsplit=1)[0] if ch.isdigit())
        bus = f"BUS{int(bus_token)}" if bus_token else "BUSUNK"
        frames[bus] = pd.read_csv(path)
    if not frames:
        raise RuntimeError(f"No PMU simulation CSV files found under {pmu_dir}")
    return frames


def save_scenario_pmu_frames(scenario_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    pmu_dir = scenario_dir / "pmu"
    pmu_dir.mkdir(parents=True, exist_ok=True)
    for bus, frame in frames.items():
        token = int("".join(ch for ch in bus if ch.isdigit()))
        path = pmu_dir / f"Bus{token}_Competition_Data_sim.csv"
        frame.to_csv(path, index=False)


@dataclass(slots=True)
class RawInformedCyberLayer:
    """Apply Event 5 / Event 7 stochastic processes over PMU frames."""

    event5_params: dict[str, Any]
    event7_params: dict[str, Any]
    event0_noise_baseline: dict[str, Any]
    seed: int = 12345
    apply_shared_noise: bool = True
    _event5: Event5DropoutProcess = field(init=False, repr=False)
    _event7: Event7CorruptionProcess = field(init=False, repr=False)
    _noise: RobustNoiseModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._event5 = Event5DropoutProcess(self.event5_params, seed=self.seed + 11)
        self._event7 = Event7CorruptionProcess(
            self.event7_params,
            noise_baseline=self.event0_noise_baseline,
            seed=self.seed + 23,
        )
        self._noise = RobustNoiseModel(self.event0_noise_baseline, seed=self.seed + 31)

    def apply_to_frames(
        self,
        *,
        pmu_frames: dict[str, pd.DataFrame],
        intervals: list[CyberIntervalSpec],
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        out = {bus: frame.copy() for bus, frame in pmu_frames.items()}
        latent_rows: list[dict[str, Any]] = []

        for bus, frame in list(out.items()):
            channels = _measurement_columns(bus, frame)
            if self.apply_shared_noise and channels:
                out[bus] = self._noise.apply_to_frame(
                    bus=bus,
                    frame=out[bus],
                    columns=channels,
                    indices=None,
                    scale_multiplier=0.08,
                )

            for index, interval in enumerate(intervals):
                if bus not in interval.target_pmus:
                    continue
                ts = pd.to_numeric(out[bus]["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
                mask = (ts >= float(interval.start_time_s)) & (ts <= float(interval.end_time_s))
                if not np.any(mask):
                    continue
                idx = np.where(mask)[0]
                start_idx = int(idx[0])
                end_idx = int(idx[-1])
                physical = pd.to_numeric(out[bus].get("Event", 0), errors="coerce").fillna(0).to_numpy(dtype=int)
                physical_mask = physical.astype(int) != 0
                if interval.kind == "event5":
                    updated, latent = self._event5.apply(
                        bus=bus,
                        frame=out[bus],
                        start_idx=start_idx,
                        end_idx=end_idx,
                        physical_event_mask=physical_mask,
                        event_label_when_physical=int(interval.event_label_if_physical),
                    )
                    out[bus] = updated
                    for row in latent:
                        row["interval_index"] = index
                        row["kind"] = "event5"
                        row["subtype"] = interval.subtype
                    latent_rows.extend(latent)
                elif interval.kind == "event7":
                    updated, latent = self._event7.apply(
                        bus=bus,
                        frame=out[bus],
                        start_idx=start_idx,
                        end_idx=end_idx,
                        physical_event_mask=physical_mask,
                        event_label_when_physical=int(interval.event_label_if_physical),
                    )
                    out[bus] = updated
                    for row in latent:
                        row["interval_index"] = index
                        row["kind"] = "event7"
                        row["subtype"] = interval.subtype
                    latent_rows.extend(latent)

        latent = pd.DataFrame(latent_rows)
        return out, latent


def _cyber_type_for_event(event: int) -> tuple[str, str]:
    if int(event) in {5, 6}:
        return "missing_data", "raw_informed_dropout"
    if int(event) in {7, 8}:
        return "bad_data", "raw_informed_corruption"
    return "", ""


def _bool_from_event(event: int) -> tuple[bool, bool, bool]:
    event_int = int(event)
    is_abnormal = event_int != 0
    is_physical = event_int in {1, 2, 3, 4, 6, 8}
    is_cyber = event_int in {5, 6, 7, 8}
    return is_abnormal, is_physical, is_cyber


def rebuild_labels_from_pmu(
    *,
    scenario_dir: Path,
    intervals: list[CyberIntervalSpec],
    latent_trace_path: Path | None = None,
) -> None:
    pmu_dir = scenario_dir / "pmu"
    merged = align_scenario_pmu_dir(pmu_dir)
    merged["TIMESTAMP"] = pd.to_numeric(merged["TIMESTAMP"], errors="coerce")
    merged = merged.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP").reset_index(drop=True)
    labels = pd.DataFrame()
    labels["TIMESTAMP"] = merged["TIMESTAMP"].astype(float)
    labels["EVENT"] = pd.to_numeric(merged["EVENT"], errors="coerce").fillna(0).astype(int)
    rows: list[dict[str, Any]] = []
    for _, row in labels.iterrows():
        event = int(row["EVENT"])
        cyber_type, subtype = _cyber_type_for_event(event)
        is_abnormal, is_physical, is_cyber = _bool_from_event(event)
        ts = float(row["TIMESTAMP"])
        target_pmu = ""
        for interval in intervals:
            if ts >= float(interval.start_time_s) and ts <= float(interval.end_time_s):
                target_pmu = ";".join(interval.target_pmus)
                if not subtype:
                    subtype = interval.subtype
                break
        rows.append(
            {
                "TIMESTAMP": ts,
                "EVENT": event,
                "IS_ABNORMAL": bool(is_abnormal),
                "IS_PHYSICAL_EVENT": bool(is_physical),
                "IS_CYBER_EVENT": bool(is_cyber),
                "IS_CONCURRENT_EVENT": bool(event in {6, 8}),
                "PHYSICAL_EVENT_TYPE": "" if event in {0, 5, 7} else "physical_event",
                "CYBER_EVENT_TYPE": cyber_type,
                "CYBER_SUBTYPE": subtype,
                "TARGET_PMU": target_pmu,
                "TARGET_BUS": "",
                "TARGET_LINE": "",
            }
        )
    labels_frame = pd.DataFrame(rows)
    labels_dir = scenario_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_frame.to_csv(labels_dir / "event_frame_labels.csv", index=False)

    interval_rows: list[dict[str, Any]] = []
    for index, interval in enumerate(intervals):
        interval_rows.append(
            {
                "EVENT_ID": f"RAWCY{index + 1:03d}",
                "EVENT": int(interval.event_label_if_no_physical),
                "EVENT_FAMILY": "cyber",
                "EVENT_TYPE": "missing_data" if interval.kind == "event5" else "bad_data",
                "SUBTYPE": interval.subtype,
                "START_TIME_S": float(interval.start_time_s),
                "END_TIME_S": float(interval.end_time_s),
                "TARGET_BUS": "",
                "TARGET_LINE": "",
                "TARGET_PMU": ";".join(interval.target_pmus),
            }
        )
    pd.DataFrame(interval_rows).to_csv(labels_dir / "event_intervals.csv", index=False)
    if not (labels_dir / "localization_targets.csv").exists():
        pd.DataFrame(
            columns=[
                "EVENT_ID",
                "EVENT",
                "LOCALIZATION_TYPE",
                "TARGET_BUS",
                "TARGET_LINE",
                "START_TIME_S",
                "END_TIME_S",
            ]
        ).to_csv(labels_dir / "localization_targets.csv", index=False)

    if latent_trace_path is not None and latent_trace_path.exists():
        meta_dir = scenario_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        latent_rel = str(latent_trace_path.resolve())
        manifest_path = scenario_dir / "scenario_manifest.json"
        if manifest_path.exists():
            manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest_dict = {"scenario_id": scenario_dir.name}
        manifest_dict["raw_informed_cyber"] = {
            "enabled": True,
            "latent_trace": latent_rel,
            "labels_rebuilt_from_pmu": True,
        }
        manifest_path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")

    # Keep all-bus labels aligned with the updated frame labels.
    all_bus_truth_path = scenario_dir / "all_buses" / "all_bus_truth.csv"
    if all_bus_truth_path.exists():
        all_bus = pd.read_csv(all_bus_truth_path)
        mapped = labels_frame.drop_duplicates("TIMESTAMP").set_index("TIMESTAMP")["EVENT"]
        all_bus["EVENT"] = pd.to_numeric(all_bus["TIMESTAMP"], errors="coerce").map(mapped).fillna(0).astype(int)
        all_bus.to_csv(all_bus_truth_path, index=False)
    full_state_target_path = scenario_dir / "all_buses" / "full_state_target.csv"
    if full_state_target_path.exists():
        target = pd.read_csv(full_state_target_path)
        mapped = labels_frame.drop_duplicates("TIMESTAMP").set_index("TIMESTAMP")["EVENT"]
        target["EVENT"] = pd.to_numeric(target["TIMESTAMP"], errors="coerce").map(mapped).fillna(0).astype(int)
        target.to_csv(full_state_target_path, index=False)


def apply_layer_to_scenario_dir(
    *,
    scenario_dir: Path,
    layer: RawInformedCyberLayer,
    intervals: list[CyberIntervalSpec],
    latent_output_path: Path,
) -> pd.DataFrame:
    frames = load_scenario_pmu_frames(scenario_dir)
    updated, latent = layer.apply_to_frames(pmu_frames=frames, intervals=intervals)
    save_scenario_pmu_frames(scenario_dir, updated)
    latent_output_path.parent.mkdir(parents=True, exist_ok=True)
    latent.to_csv(latent_output_path, index=False)
    rebuild_labels_from_pmu(scenario_dir=scenario_dir, intervals=intervals, latent_trace_path=latent_output_path)
    return latent
