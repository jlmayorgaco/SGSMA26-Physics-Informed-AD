"""Run the bounded G0 static repair and leave all downstream phases blocked."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmu_hybrid.cases import g0_audit, parity


def run(root: Path) -> dict[str, object]:
    root = root.resolve()
    ybus_summary = g0_audit.run(root)
    parity_summary = parity.run(root)
    reports = root / "output" / "reports"
    registry_path = reports / "experiment_registry.csv"
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
    else:
        registry = pd.read_csv(root / "configs" / "experiment_registry.csv")
    registry.loc[registry.experiment_id.eq("E01"), ["status", "result_status", "status_detail"]] = [
        parity_summary["status"], parity_summary["status"], "canonical ANDES workbook translation audited by G0-A/G0-B",
    ]
    # This run intentionally executes no measurement, DAE, estimator,
    # localization, Bayesian or ML experiment.
    registry.loc[registry.experiment_id.isin(["E02", "E03"]), ["status", "result_status", "status_detail"]] = [
        "SKIPPED_WITH_REASON", "BLOCKED_BY_G0", "not executed during bounded G0 repair",
    ]
    registry.to_csv(registry_path, index=False)
    status = "PASS" if ybus_summary["status"] == "PASS" and parity_summary["status"] == "PASS" else "FAIL"
    (reports / "g0_parity_repair_v2.md").write_text(
        "# G0 static parity repair\n\n"
        "## A. Authority and scope\n\n"
        "ANDES `ieee39_full.xlsx` is the canonical source. pandapower `case39()` was retained only as an external reference. This run is static-only; downstream phases were not executed.\n\n"
        "## B. G0-A canonical inventory\n\n"
        "The six `g0_canonical_*_table.csv` artifacts record buses, branches, transformers, shunts, generators and loads directly from the workbook before PF.\n\n"
        "## C. G0-B Ybus audit\n\n"
        "Ybus was reconstructed from translated pandapower element tables before `runpp`. Primitive line charging, transformer taps/phase, and bus shunt signs are recorded in the G0 result tables.\n\n"
        "## D. Semantic repairs\n\n"
        "Transformer tap positions now use `sign(tap-1)` with an absolute step and explicit Ratio changer semantics. Native ANDES injections include static shunts. Canonical PF injections use the complete solved Ybus balance because pandapower `res_bus` excludes admittance shunts.\n\n"
        "## E. Numerical evidence\n\n"
        "Recorded pre-repair baseline: canonical PF gate **FAIL** (max |dV|=0.07688463 pu, max |dVa|=1.515304 deg, max |dQ|=171.90936 MVAr, max branch-terminal error=174.623858 MVA).\n\n"
        f"Ybus audit: `{ybus_summary['status']}`; canonical PF parity: `{parity_summary['status']}`.\n\n"
        "## F. Regression coverage\n\n"
        "The network, converter and audit tests are run independently of dynamic experiments.\n\n"
        "## G. Downstream registry\n\n"
        "E02/E03 and all later phases remain blocked or pending; no downstream execution was performed.\n\n"
        "## H. Gate\n\n"
        f"G0 = **{status}**\n",
        encoding="utf-8",
    )
    return {"status": status, "ybus": ybus_summary, "parity": parity_summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    print(result["status"])
    raise SystemExit(0 if result["status"] == "PASS" else 2)
