from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PipelineResult:
    """Compact result payload written by each operational pipeline."""

    name: str
    status: str
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = {key: str(Path(value)) for key, value in self.outputs.items()}
        return payload
