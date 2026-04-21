"""Loader for synchronized PMU CSV bus files."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from src.domain.topology import bus_token


REQUIRED_SUFFIXES = [
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


class PmuCsvLoader:
    """Load per-bus PMU CSV files and enforce schema consistency."""

    def __init__(self, pmu_data_dir: str | Path) -> None:
        self.pmu_data_dir = Path(pmu_data_dir)
        if not self.pmu_data_dir.exists():
            raise FileNotFoundError(f"PMU data directory does not exist: {self.pmu_data_dir}")

    @staticmethod
    def _bus_token(bus_id: str | int) -> str:
        return bus_token(bus_id)

    def _find_file_for_bus(self, bus_id: str | int) -> Path:
        token = self._bus_token(bus_id)
        patterns = [
            f"Bus{token}_Competition_Data_nanmask.csv",
            f"Bus{token}_normalized.csv",
            f"Bus{token}.csv",
            f"Bus{token}_*.csv",
        ]
        for pattern in patterns:
            matches = sorted(self.pmu_data_dir.glob(pattern))
            if matches:
                return matches[0]
        regex = re.compile(rf"^Bus{re.escape(token)}(?:_|\.|$)", re.IGNORECASE)
        candidates = sorted(p for p in self.pmu_data_dir.glob("Bus*.csv") if regex.match(p.name))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"No PMU CSV found for bus {token} under {self.pmu_data_dir}")

    @staticmethod
    def _required_columns(bus_id: str) -> list[str]:
        return ["TIMESTAMP", "DATA_PRESENT", "Event"] + [f"BUS{bus_id}_{s}" for s in REQUIRED_SUFFIXES]

    def load_bus(self, bus_id: str | int) -> pd.DataFrame:
        """Load, clean and validate one bus PMU dataframe."""
        token = self._bus_token(bus_id)
        path = self._find_file_for_bus(token)
        df = pd.read_csv(path)
        if "TIMESTAMP" not in df.columns:
            raise ValueError(f"TIMESTAMP column missing in {path}")
        df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce").round(3)
        df = df.dropna(subset=["TIMESTAMP"]).drop_duplicates(subset=["TIMESTAMP"]).sort_values("TIMESTAMP")
        missing = [c for c in self._required_columns(token) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")
        return df.reset_index(drop=True)

    def load_all(self, pmu_bus_ids: list[str | int]) -> dict[str, pd.DataFrame]:
        """Load all PMU buses into a bus_id->DataFrame mapping."""
        out: dict[str, pd.DataFrame] = {}
        for bus in pmu_bus_ids:
            key = self._bus_token(bus)
            out[key] = self.load_bus(key)
        return out
