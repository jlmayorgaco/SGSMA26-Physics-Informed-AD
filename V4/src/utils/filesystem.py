from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        ensure_dir(path)


def build_subdir(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    ensure_dir(path)
    return path


def write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding=encoding)