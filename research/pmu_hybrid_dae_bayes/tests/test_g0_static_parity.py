from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_canonical_transformers_keep_ratio_tap_semantics():
    pytest.importorskip("pandapower")
    from pmu_hybrid.cases.ieee39_pandapower import solve_canonical_from_andes
    from pmu_hybrid.cases.g0_audit import _pp_ybus_before_pf

    net, _ = solve_canonical_from_andes(return_net=True)
    assert set(net.trafo.tap_changer_type.dropna()) == {"Ratio"}
    assert net.trafo.tap_pos.iloc[0] == 1
    assert net.trafo.tap_pos.iloc[1] == -1
    _, ppc, _ = _pp_ybus_before_pf(net)
    # The first transformer is ANDES Line_35: tap=1.025 on the HV side.
    assert ppc["branch"][0, 8] == pytest.approx(1.025, abs=1e-12)


def test_g0_audit_reconstructs_source_ybus():
    pytest.importorskip("andes")
    pytest.importorskip("pandapower")
    from pmu_hybrid.cases.g0_audit import run

    test_root = Path(__file__).resolve().parents[1] / "output" / "g0_test_artifacts"
    summary = run(test_root)
    assert summary["status"] == "PASS"
    assert summary["max_source_formula_ybus_abs_error"] < 1e-10
    assert summary["max_ybus_abs_error"] < 1e-10
    for name in (
        "g0_canonical_bus_table.csv",
        "g0_canonical_branch_table.csv",
        "g0_canonical_transformer_table.csv",
        "g0_canonical_shunt_table.csv",
        "g0_canonical_generator_table.csv",
        "g0_canonical_load_table.csv",
        "g0_ybus_matrix_diff.csv",
        "g0_branch_primitive_admittances.csv",
        "g0_transformer_audit.csv",
        "g0_shunt_charging_audit.csv",
    ):
        assert (test_root / "output" / "results" / name).exists()
