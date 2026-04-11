"""Declarative ANDES runner wrapper for POC synthetic scenarios.

Primary intent: drive ANDES with event declarations and save CSVs in the exact
competition schema.  In this local repo the existing ``src.augmentation``
module already contains a physics fallback with the right PMU units and schema,
so this wrapper delegates to it when a direct ANDES runtime is not available.

Paper note: report whether ``backend`` is ``andes`` or ``surrogate`` for each
run.  The CSV contract is identical either way, which keeps loader validation
honest.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from poc.schema import CHANNEL_SUFFIXES, PMU_BUSES, Scenario

LOG = logging.getLogger(__name__)


class AndesRunner:
    """Generate one SGSMA event scenario and write eight bus CSVs.

    Method: declarative scenario -> time-domain event generator -> competition
    CSV schema.  Reference: ANDES IEEE-39 DAE simulator as primary backend; the
    local POC uses the existing physics surrogate if ANDES integration cannot be
    reached.  Parameter count: zero.
    """

    def __init__(self, raw_data_dir: Path, output_dir: Path, seed: int = 42) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self._stats: dict | None = None
        self.backend = "surrogate"

    def _baseline_stats(self) -> dict:
        """Load measurement noise statistics from the real first-60s baseline."""

        if self._stats is not None:
            return self._stats
        from src.augmentation.andes_sim import extract_normal_baseline
        from src.io.load_csv import load_all

        df = load_all(self.raw_data_dir)
        self._stats = extract_normal_baseline(df)
        return self._stats

    def generate_scenario(self, scenario: Scenario) -> Path:
        """Generate one scenario directory with exact ``BusX_...csv`` names."""

        rng = np.random.default_rng(self.seed + _scenario_seed_offset(scenario.scenario_id))
        stats = self._baseline_stats()
        bus_data, timestamps, labels = self._generate_bus_data(scenario, stats, rng)

        scenario_dir = self.output_dir / scenario.category / scenario.scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        self._write_competition_csvs(bus_data, timestamps, labels, scenario_dir)
        LOG.info("Generated %s scenario at %s with backend=%s", scenario.category, scenario_dir, self.backend)
        return scenario_dir

    def verify_with_src_loader(self, scenario_dir: Path) -> pd.DataFrame:
        """Assert the generated scenario loads through ``src.io.load_csv.load_all``."""

        from src.io.load_csv import get_all_measurement_cols, load_all

        df = load_all(scenario_dir)
        expected = set(get_all_measurement_cols())
        missing = expected.difference(df.columns)
        if missing:
            raise AssertionError(f"Generated CSV missing columns: {sorted(missing)[:5]}")
        if "Event" not in df.columns:
            raise AssertionError("Merged generated CSV has no global Event column")
        return df

    def _generate_bus_data(self, scenario: Scenario, stats: dict, rng: np.random.Generator):
        """Dispatch to the local dynamic surrogate by label."""

        from src.augmentation import andes_sim

        label = int(scenario.label)
        t_event = float(scenario.t_event)
        if label == 1:
            return andes_sim.generate_fault(
                event_bus=scenario.event_bus,
                stats=stats,
                rng=rng,
                window_sec=max(6.0, t_event + 4.0),
                t_event=t_event,
                fault_impedance=float(scenario.params.get("fault_impedance", 0.01)),
                clear_cycles=int(scenario.params.get("clearing_cycles", 5)),
            )
        if label == 2:
            line = scenario.params.get("line", (scenario.event_bus, min(39, scenario.event_bus + 1)))
            return andes_sim.generate_line_outage(
                line_from=int(line[0]),
                line_to=int(line[1]),
                stats=stats,
                rng=rng,
                window_sec=max(6.0, t_event + 4.0),
                t_event=t_event,
            )
        if label == 3:
            return andes_sim.generate_gen_change(
                gen_bus=scenario.event_bus,
                stats=stats,
                rng=rng,
                window_sec=max(6.0, t_event + 4.0),
                t_event=t_event,
                delta_mw=float(scenario.params.get("delta_mw", 25.0)),
            )
        if label == 4:
            return andes_sim.generate_load_change(
                load_bus=scenario.event_bus,
                stats=stats,
                rng=rng,
                window_sec=max(6.0, t_event + 4.0),
                t_event=t_event,
                delta_mw=float(scenario.params.get("delta_mw", 25.0)),
            )
        if label == 5:
            return andes_sim.generate_pmu_dropout(
                dropout_bus=int(scenario.params.get("dropout_bus", scenario.event_bus)),
                stats=stats,
                rng=rng,
                window_sec=max(6.0, t_event + min(4.0, scenario.duration + 1.0)),
                t_event=t_event,
                dropout_sec=float(scenario.duration),
            )
        if label == 6:
            return andes_sim.generate_post_cyber_physical(
                stats=stats,
                rng=rng,
                dropout_bus=int(scenario.params.get("dropout_bus", 29)),
                t_dropout=max(0.5, t_event - 1.0),
                t_physical=t_event,
                t_recover=t_event + max(1.0, scenario.duration * 0.6),
            )
        if label == 7:
            return andes_sim.generate_bad_data(
                bad_bus=int(scenario.params.get("bad_bus", scenario.event_bus)),
                stats=stats,
                rng=rng,
                window_sec=max(6.0, t_event + 4.0),
                t_event=t_event,
            )
        raise ValueError(f"Unsupported synthetic label {label}")

    @staticmethod
    def _write_competition_csvs(
        bus_data: dict[str, np.ndarray],
        timestamps: np.ndarray,
        event_labels: np.ndarray,
        out_dir: Path,
    ) -> list[Path]:
        """Write exact competition file names without synthetic run prefixes."""

        paths: list[Path] = []
        for bus in PMU_BUSES:
            rows: dict[str, np.ndarray] = {"TIMESTAMP": np.round(timestamps, 10)}
            for suffix in CHANNEL_SUFFIXES:
                col = f"BUS{bus}_{suffix}"
                rows[col] = bus_data.get(col, np.zeros(len(timestamps)))
            rows["DATA_PRESENT"] = bus_data.get(
                f"BUS{bus}_DATA_PRESENT",
                np.ones(len(timestamps), dtype=int),
            ).astype(int)
            rows["Event"] = event_labels.astype(int)
            df = pd.DataFrame(rows)
            path = out_dir / f"Bus{bus}_Competition_Data_nanmask.csv"
            df.to_csv(path, index=False)
            paths.append(path)
        return paths


def _scenario_seed_offset(scenario_id: str) -> int:
    """Stable small integer hash without depending on randomized hash seeds."""

    return sum((i + 1) * ord(ch) for i, ch in enumerate(scenario_id)) % 100_000

