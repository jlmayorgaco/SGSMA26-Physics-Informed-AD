from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

TIMESTAMP_COL = "TIMESTAMP"
DATA_PRESENT_COL = "DATA_PRESENT"
EVENT_COL = "Event"


def _default_project_root() -> Path:
    """
    Resolve project root assuming this file lives at:
    src/utils/data_loader.py
    """
    return Path(__file__).resolve().parents[2]


def _resolve_project_root(project_root: str | Path | None) -> Path:
    """
    Resolve the project root path.
    """
    if project_root is None:
        return _default_project_root()
    return Path(project_root)


def _build_scenario_dir(
    scenario_id: str,
    project_root: str | Path | None = None,
    synthetic_dir: str = "data/synthetic",
) -> Path:
    """
    Build the scenario directory path.

    Expected layout:
        <project_root>/<synthetic_dir>/<scenario_id>/
    """
    root = _resolve_project_root(project_root)
    return root / synthetic_dir / scenario_id


def _build_scenario_json_path(
    scenario_id: str,
    project_root: str | Path | None = None,
    synthetic_dir: str = "data/synthetic",
) -> Path:
    """
    Build the scenario JSON path.

    Expected layout:
        <project_root>/<synthetic_dir>/<scenario_id>/<scenario_id>.json
    """
    scenario_dir = _build_scenario_dir(
        scenario_id=scenario_id,
        project_root=project_root,
        synthetic_dir=synthetic_dir,
    )
    return scenario_dir / f"{scenario_id}.json"


def _load_json(json_path: Path) -> dict[str, Any]:
    """
    Load a JSON file from disk.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Synthetic JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _extract_bus_id_from_filename(path_str: str) -> int:
    """
    Extract bus number from filenames such as:
        csv/PMU_Bus39_Competition_Data_nanmask.csv
        csv/NON_PMU_Bus1_Competition_Data_nanmask.csv
    """
    match = re.search(r"Bus(\d+)", Path(path_str).name)
    if not match:
        raise ValueError(f"Could not extract bus id from filename: {path_str}")
    return int(match.group(1))


def _validate_required_columns(df: pd.DataFrame, source_name: str) -> None:
    """
    Validate minimal required columns.
    """
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Missing required column '{TIMESTAMP_COL}' in source: {source_name}"
        )


def _read_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read a CSV file and validate minimal schema.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _validate_required_columns(df, str(csv_path))
    return df


def _rename_bus_columns_for_merged_dataframe(
    df: pd.DataFrame,
    bus_id: int,
) -> pd.DataFrame:
    """
    Rename generic columns that would collide in the merged dataframe.

    Measurement columns are already bus-prefixed, e.g.:
        BUS2_VA_MAG, BUS39_Freq, ...

    Only generic columns need renaming:
        DATA_PRESENT -> BUS{bus_id}_DATA_PRESENT
        Event        -> BUS{bus_id}_Event
    """
    rename_map: dict[str, str] = {}

    if DATA_PRESENT_COL in df.columns:
        rename_map[DATA_PRESENT_COL] = f"BUS{bus_id}_{DATA_PRESENT_COL}"

    if EVENT_COL in df.columns:
        rename_map[EVENT_COL] = f"BUS{bus_id}_{EVENT_COL}"

    return df.rename(columns=rename_map)


def _rename_bus_columns_for_single_bus_dataframe(
    df: pd.DataFrame,
    bus_id: int,
) -> pd.DataFrame:
    """
    Rename generic columns for a single-bus dataframe.

    Event remains global-like for convenience.
    DATA_PRESENT is temporarily bus-prefixed, then optionally stripped later.
    """
    rename_map: dict[str, str] = {}

    if DATA_PRESENT_COL in df.columns:
        rename_map[DATA_PRESENT_COL] = f"BUS{bus_id}_{DATA_PRESENT_COL}"

    return df.rename(columns=rename_map)


def _get_csv_file_map(
    scenario_id: str,
    project_root: str | Path | None = None,
    synthetic_dir: str = "data/synthetic",
) -> dict[str, str]:
    """
    Return the bus-to-CSV mapping from the scenario metadata JSON.
    """
    metadata = load_synth_metadata(
        scenario_id=scenario_id,
        project_root=project_root,
        synthetic_dir=synthetic_dir,
    )

    csv_files = metadata.get("csv_files")
    if not csv_files:
        raise ValueError(
            f"No 'csv_files' mapping found in scenario metadata for {scenario_id}"
        )

    return csv_files


def _get_bus_csv_path(
    scenario_id: str,
    bus_id: int,
    project_root: str | Path | None = None,
    synthetic_dir: str = "data/synthetic",
) -> Path:
    """
    Resolve the CSV path for one bus within a scenario.
    """
    csv_files = _get_csv_file_map(
        scenario_id=scenario_id,
        project_root=project_root,
        synthetic_dir=synthetic_dir,
    )

    bus_key = str(bus_id)
    if bus_key not in csv_files:
        available_buses = sorted(int(key) for key in csv_files.keys())
        raise ValueError(
            f"Bus {bus_id} not found in scenario '{scenario_id}'. "
            f"Available buses: {available_buses}"
        )

    scenario_dir = _build_scenario_dir(
        scenario_id=scenario_id,
        project_root=project_root,
        synthetic_dir=synthetic_dir,
    )
    return scenario_dir / csv_files[bus_key]


def _sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a dataframe by TIMESTAMP.
    """
    if TIMESTAMP_COL in df.columns:
        return df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return df.reset_index(drop=True)


def _build_global_event_column(
    merged_df: pd.DataFrame,
    event_columns: list[str],
    validate_event_consistency: bool,
) -> pd.DataFrame:
    """
    Build one global Event column from per-bus event columns.

    Assumes all per-bus event columns should match row-wise.
    """
    if not event_columns:
        return merged_df

    if validate_event_consistency:
        inconsistent_mask = merged_df[event_columns].nunique(axis=1) > 1

        if inconsistent_mask.any():
            bad_rows = merged_df.loc[
                inconsistent_mask,
                [TIMESTAMP_COL, *event_columns],
            ].head(10)

            raise ValueError(
                "Per-bus Event columns are not consistent across buses.\n"
                f"First inconsistent rows:\n{bad_rows.to_string(index=False)}"
            )

    merged_df[EVENT_COL] = merged_df[event_columns[0]]
    merged_df = merged_df.drop(columns=event_columns)

    return merged_df


def load_synth_metadata(
    scenario_id: str = "SIM_0001",
    project_root: str | Path | None = None,
    synthetic_dir: str = "data/synthetic",
) -> dict[str, Any]:
    """
    Load scenario metadata JSON.

    Returns
    -------
    dict[str, Any]
        Parsed JSON metadata for the scenario.
    """
    json_path = _build_scenario_json_path(
        scenario_id=scenario_id,
        project_root=project_root,
        synthetic_dir=synthetic_dir,
    )
    return _load_json(json_path)


def list_available_buses(
    scenario_id: str = "SIM_0001",
    project_root: str | Path | None = None,
    synthetic_dir: str = "data/synthetic",
) -> list[int]:
    """
    List all available bus IDs in the scenario.
    """
    csv_files = _get_csv_file_map(
        scenario_id=scenario_id,
        project_root=project_root,
        synthetic_dir=synthetic_dir,
    )
    return sorted(int(key) for key in csv_files.keys())


def load_synth_data(
    scenario_id: str = "SIM_0001",
    project_root: str | Path | None = None,
    synthetic_dir: str = "data/synthetic",
    validate_event_consistency: bool = True,
    sort_by_timestamp: bool = True,
) -> pd.DataFrame:
    """
    Load one synthetic scenario into a single wide dataframe.

    Parameters
    ----------
    scenario_id
        Scenario name, e.g. 'SIM_0001'.
    project_root
        Root path of the repository/project. If None, inferred automatically.
    synthetic_dir
        Directory relative to project_root where scenario folders live.
    validate_event_consistency
        If True, verifies that all per-bus Event columns are identical row-wise.
    sort_by_timestamp
        If True, sorts the final dataframe by TIMESTAMP.

    Returns
    -------
    pd.DataFrame
        Wide dataframe merged by TIMESTAMP across all bus CSVs.

        Includes:
        - bus-prefixed measurements like BUS2_VA_MAG, BUS39_Freq, etc.
        - bus-specific availability columns like BUS29_DATA_PRESENT
        - one global Event column
    """
    metadata = load_synth_metadata(
        scenario_id=scenario_id,
        project_root=project_root,
        synthetic_dir=synthetic_dir,
    )

    csv_files = metadata.get("csv_files")
    if not csv_files:
        raise ValueError(
            f"No 'csv_files' mapping found in scenario metadata for {scenario_id}"
        )

    merged_df: pd.DataFrame | None = None
    event_columns: list[str] = []

    for _, relative_csv_path in sorted(csv_files.items(), key=lambda item: int(item[0])):
        bus_id = _extract_bus_id_from_filename(relative_csv_path)

        scenario_dir = _build_scenario_dir(
            scenario_id=scenario_id,
            project_root=project_root,
            synthetic_dir=synthetic_dir,
        )
        csv_path = scenario_dir / relative_csv_path

        df_bus = _read_csv(csv_path)
        df_bus = _rename_bus_columns_for_merged_dataframe(df_bus, bus_id)

        event_column = f"BUS{bus_id}_{EVENT_COL}"
        if event_column in df_bus.columns:
            event_columns.append(event_column)

        if merged_df is None:
            merged_df = df_bus
        else:
            merged_df = merged_df.merge(df_bus, on=TIMESTAMP_COL, how="inner")

    if merged_df is None:
        raise RuntimeError(f"No CSV files could be loaded for scenario '{scenario_id}'")

    merged_df = _build_global_event_column(
        merged_df=merged_df,
        event_columns=event_columns,
        validate_event_consistency=validate_event_consistency,
    )

    if sort_by_timestamp:
        merged_df = _sort_by_timestamp(merged_df)

    return merged_df


def load_synth_bus_data(
    scenario_id: str = "SIM_0001",
    bus_id: int = 1,
    project_root: str | Path | None = None,
    synthetic_dir: str = "data/synthetic",
    sort_by_timestamp: bool = True,
    strip_bus_prefix: bool = True,
) -> pd.DataFrame:
    """
    Load one bus only from one scenario.

    Parameters
    ----------
    scenario_id
        Scenario name, e.g. 'SIM_0001'.
    bus_id
        Bus number to load.
    project_root
        Root path of the repository/project. If None, inferred automatically.
    synthetic_dir
        Directory relative to project_root where scenario folders live.
    sort_by_timestamp
        If True, sorts the dataframe by TIMESTAMP.
    strip_bus_prefix
        If True, converts columns like BUS1_VA_ANG -> VA_ANG for single-bus use.

    Returns
    -------
    pd.DataFrame
        Single-bus dataframe containing:
        - TIMESTAMP
        - electrical measurements
        - DATA_PRESENT
        - Event
    """
    csv_path = _get_bus_csv_path(
        scenario_id=scenario_id,
        bus_id=bus_id,
        project_root=project_root,
        synthetic_dir=synthetic_dir,
    )

    df_bus = _read_csv(csv_path)
    df_bus = _rename_bus_columns_for_single_bus_dataframe(df_bus, bus_id)

    if strip_bus_prefix:
        df_bus = _strip_bus_prefix_from_columns(df_bus, bus_id)

    if sort_by_timestamp:
        df_bus = _sort_by_timestamp(df_bus)

    return df_bus

def load_synth_bus_columns(
    scenario_id: str = "SIM_0001",
    bus_id: int = 1,
    columns: list[str] | None = None,
    project_root: str | Path | None = None,
    synthetic_dir: str = "data/synthetic",
    sort_by_timestamp: bool = True,
    strip_bus_prefix: bool = True,
) -> pd.DataFrame:
    """
    Load one bus and keep only selected columns.
    """
    df_bus = load_synth_bus_data(
        scenario_id=scenario_id,
        bus_id=bus_id,
        project_root=project_root,
        synthetic_dir=synthetic_dir,
        sort_by_timestamp=sort_by_timestamp,
        strip_bus_prefix=strip_bus_prefix,
    )

    if columns is None:
        return df_bus.copy()

    missing_columns = [column for column in columns if column not in df_bus.columns]
    if missing_columns:
        raise ValueError(
            f"Requested columns not found for bus {bus_id}: {missing_columns}\n"
            f"Available columns: {df_bus.columns.tolist()}"
        )

    return df_bus[columns].copy()


def _strip_bus_prefix_from_columns(df: pd.DataFrame, bus_id: int) -> pd.DataFrame:
    """
    Convert single-bus columns from:
        BUS1_VA_ANG -> VA_ANG
        BUS1_Freq -> Freq
        BUS1_DATA_PRESENT -> DATA_PRESENT

    Keeps TIMESTAMP and Event unchanged.
    """
    bus_prefix = f"BUS{bus_id}_"
    rename_map: dict[str, str] = {}

    for column in df.columns:
        if column.startswith(bus_prefix):
            rename_map[column] = column[len(bus_prefix):]

    return df.rename(columns=rename_map)