"""Atomic, auditable manifests for every executable campaign unit."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def git_commit(repository_root: Path) -> str:
    """Return the exact local commit or an explicit unavailable marker."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"
    return result.stdout.strip()


def write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON so interrupted work never looks complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def base_manifest(repository_root: Path, *, seed: int | None = None) -> dict[str, Any]:
    """Create common immutable context shared by all phase manifests."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(repository_root),
        "seed": seed,
        "external_holdout_access": "FORBIDDEN_BEFORE_FREEZE",
        "truth_access": "POST_HOC_EVALUATION_ONLY",
    }
