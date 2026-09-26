"""Reproduce the fitted-candidate audit reported in the paper."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import joblib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.candidate_ontology import CandidateOntology, canonicalize_frozen_location


DEFAULT_ARCHIVE = ROOT / "sgsma_2026_final_submission.zip"
DEFAULT_TOPOLOGY = ROOT / "data" / "topology" / "ieee39" / "branches_physical.csv"
DEFAULT_OUTPUT = ROOT / "paper" / "evidence" / "revised" / "candidate_audit.json"
MODEL_MEMBER = "models/localizer/final_dynamic_localizers.joblib"


def _load_bundle(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as handle:
        payload = handle.read(MODEL_MEMBER)
    return joblib.load(io.BytesIO(payload))


def audit(archive: Path, topology: Path) -> dict[str, object]:
    ontology = CandidateOntology.from_branch_csv(topology)
    bundle = _load_bundle(archive)
    localizers = bundle["localizers"]

    def classes(key: str) -> list[str]:
        model = localizers[key]
        estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
        return [str(value) for value in estimator.classes_]

    line_audit = ontology.audit_fitted_classes(2, classes("LINE:event2"))
    generation_fitted = classes("BUS:event3")
    generation_reported = [canonicalize_frozen_location(3, value) for value in generation_fitted]
    generation_audit = ontology.audit_fitted_classes(3, generation_reported)
    load_audit = ontology.audit_fitted_classes(4, classes("BUS:event4"))

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return {
        "archive": str(archive.resolve()),
        "archive_sha256": digest,
        "model_member": MODEL_MEMBER,
        "topology": str(topology.resolve()),
        "topology_partition": {
            "physical_branches": len(ontology.transmission_lines) + len(ontology.transformers),
            "transmission_lines": len(ontology.transmission_lines),
            "transformers": len(ontology.transformers),
        },
        "line_event": line_audit,
        "generation_event_after_report_bus_mapping": generation_audit,
        "load_event": load_audit,
        "pmu_sites": list(ontology.pmu_sites),
        "interpretation": (
            "Generation labels admit a one-to-one reporting conversion. "
            "The missing line classes require regenerated training data and cannot be repaired by relabeling."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.archive, args.topology)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
