"""ANDES backend gate for later dynamic scenarios; it never substitutes a surrogate."""

from __future__ import annotations

import json
from pathlib import Path


class StaticParityGateError(RuntimeError):
    """Raised instead of running a DAE campaign with an unresolved static model."""


def require_static_parity(root: Path) -> None:
    """Enforce G0 before scenario simulation generates DAE evidence."""
    manifest = root / "output" / "manifests" / "static_parity.json"
    if not manifest.exists():
        raise StaticParityGateError("G0 has not been run")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise StaticParityGateError("G0 static parity is not PASS; ANDES scenario simulation is blocked")
