"""Build and audit the SGSMA submission zip from a clean manifest."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


DEFAULT_ZIP = Path("submission_sgsma2026.zip")
DIRS_TO_INCLUDE = [
    Path("src"),
    Path("tests"),
    Path("predictions"),
]
OPTIONAL_DIRS_TO_INCLUDE = [
    Path("report/engineering_review"),
]
FILES_TO_INCLUDE = [
    Path("report/sgsma2026_report.pdf"),
    Path("report/sgsma2026_report.tex"),
    Path("README.md"),
    Path("requirements.txt"),
    Path("pyproject.toml"),
    Path("Makefile"),
    Path("CLAUDE.md"),
    Path("RESUMEN.md"),
]
OPTIONAL_FILES = [
    Path("report/confusion_matrix_sample.csv"),
    Path("report/model_benchmark.csv"),
    Path("report/model_benchmark.json"),
    Path("report/sample_level_metrics.json"),
    Path("report/submission_validation.json"),
    Path("experiments/andes_fault_bus7/report.md"),
    Path("experiments/andes_fault_bus7/metrics.csv"),
    Path("experiments/andes_fault_bus7/metadata.json"),
    Path("experiments/andes_fault_bus7/figures/bus7_vm_reconstruction.png"),
    Path("experiments/andes_fault_bus7/figures/hidden_bus_post_fault_vm_rmse.png"),
]
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".aux",
    ".log",
    ".out",
    ".synctex.gz",
    ".fdb_latexmk",
    ".fls",
}


def _is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_PARTS:
        return True
    name = path.name
    return any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def _iter_manifest(root: Path) -> list[Path]:
    files: list[Path] = []
    for d in DIRS_TO_INCLUDE:
        base = root / d
        if not base.exists():
            raise FileNotFoundError(f"Required directory missing: {d}")
        for p in base.rglob("*"):
            rel = p.relative_to(root)
            if p.is_file() and not _is_excluded(rel):
                files.append(rel)

    for f in FILES_TO_INCLUDE:
        if not (root / f).exists():
            raise FileNotFoundError(f"Required file missing: {f}")
        if not _is_excluded(f):
            files.append(f)

    for f in OPTIONAL_FILES:
        if (root / f).exists() and not _is_excluded(f):
            files.append(f)

    for d in OPTIONAL_DIRS_TO_INCLUDE:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            rel = p.relative_to(root)
            if p.is_file() and not _is_excluded(rel):
                files.append(rel)

    return sorted(set(files), key=lambda p: p.as_posix())


def audit_zip(zip_path: Path | str) -> dict:
    """Return an audit report for required and forbidden zip contents."""
    zip_path = Path(zip_path)
    required = {
        "src/estimator/topology_state.py",
        "src/detector/bad_data.py",
        "src/classifier/rules.py",
        "src/pipeline/andes_fault_reconstruction.py",
        "src/pipeline/benchmark_models.py",
        "src/pipeline/generate_engineering_review.py",
        "src/pipeline/validate_submission.py",
        "tests/test_topology_state.py",
        "predictions/submission.csv",
        "report/sgsma2026_report.pdf",
        "report/engineering_review/detection_and_scenario_audit.json",
        "report/engineering_review/ieee39_topology.json",
        "report/engineering_review/per_bus_sample_metrics.json",
        "README.md",
    }
    errors: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    missing = sorted(required - names)
    if missing:
        errors.append(f"Missing required files: {missing}")
    forbidden = sorted(
        n for n in names
        if "__pycache__" in n or n.endswith((".pyc", ".pyo", ".aux", ".log", ".synctex.gz"))
    )
    if forbidden:
        errors.append(f"Forbidden cache/build files present: {forbidden[:20]}")
    return {
        "ok": not errors,
        "zip": str(zip_path),
        "n_files": len(names),
        "errors": errors,
        "missing": missing,
        "forbidden_count": len(forbidden),
    }


def build_zip(root: Path | str = Path("."), zip_path: Path | str = DEFAULT_ZIP) -> dict:
    """Create the submission zip and return an audit report."""
    root = Path(root).resolve()
    zip_path = Path(zip_path)
    if not zip_path.is_absolute():
        zip_path = root / zip_path
    files = _iter_manifest(root)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            zf.write(root / rel, rel.as_posix())
    report = audit_zip(zip_path)
    report["manifest_files"] = len(files)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    report = audit_zip(args.zip) if args.audit_only else build_zip(args.root, args.zip)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
