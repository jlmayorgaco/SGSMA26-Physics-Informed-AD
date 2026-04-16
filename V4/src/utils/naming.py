from __future__ import annotations

import re
from pathlib import Path


def infer_bus_id(path: Path) -> str:
    match = re.search(r"Bus(\d+)", path.stem, re.IGNORECASE)
    if match:
        return f"Bus{int(match.group(1))}"
    return path.stem


def normalize_raw_column_name(col: str) -> str:
    col = str(col).replace("\ufeff", "").strip()
    col = re.sub(r"\s+", "_", col)
    return col


def safe_slug(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    return text


def event_folder_name(event_index: int, label: str) -> str:
    return f"event_{event_index:04d}_{safe_slug(label)}"