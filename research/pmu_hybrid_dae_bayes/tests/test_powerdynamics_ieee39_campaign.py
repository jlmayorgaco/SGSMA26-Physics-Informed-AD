from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"

def test_pd_g0_and_g1_artifacts_pass():
    g0 = json.loads((ROOT / "output/results/pd_g0_static_parity.json").read_text())
    g1 = json.loads((ROOT / "output/results/pd_g1_pmu_summary.json").read_text())
    assert g0["PD-G0"] == "PASS"
    assert g1["pd_g1"] == "PASS"
    assert g1["existing"] == g1["requested"]
    pmu = list(csv.DictReader((ROOT / "output/results/pd_g1_pmu_branch_audit.csv").open()))
    assert len(pmu) == 8 and all(float(r["current_error_mva"]) < 1e-6 for r in pmu)

def test_pd_linearization_all_cases_are_second_order():
    rows = list(csv.DictReader((ROOT / "output/results/pd_linearization_convergence.csv").open()))
    assert rows and all(r["retcode"] == "Success" and r["finite"] == "true" for r in rows)
    for scenario in sorted({r["scenario"] for r in rows}):
        vals = [float(r["error_over_epsilon2"]) for r in rows if r["scenario"] == scenario]
        assert max(vals) / min(vals) < 2.0

def test_pd_descriptor_and_mapping_inventory():
    inv = list(csv.DictReader((ROOT / "output/results/pd_descriptor_inventory.csv").open()))
    mapping = list(csv.DictReader((ROOT / "output/results/pd_machine_controller_map.csv").open()))
    assert len(inv) == 192 and sum(r["kind"] == "differential" for r in inv) == 114
    assert sum(r["kind"] == "algebraic_zero_mass" for r in inv) == 78
    assert len(mapping) == 10 and sum(r["avr_id"] == "NONE" for r in mapping) == 1

def test_pd_observability_horizons_emitted():
    rows = list(csv.DictReader((ROOT / "output/results/pd_observability_horizons.csv").open()))
    assert [int(r["horizon"]) for r in rows] == [0, 3, 10, 30, 60, 120, 180]
    assert int(rows[-1]["rank"]) > int(rows[0]["rank"])
    assert all(float(r["functional_output_residual"]) >= 0 for r in rows)

def test_pd_nonlinear_suspicious_modes_completed():
    text = (ROOT / "output/results/pd_nonlinear_probe.txt").read_text()
    assert "probe_count=2" in text and text.count("retcode=Success") == 2
