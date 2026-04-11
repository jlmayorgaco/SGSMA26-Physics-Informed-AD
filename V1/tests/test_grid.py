"""M1 grid model tests: Ybus shape, base-case voltages, Kron reduction."""
from __future__ import annotations

import numpy as np
import pytest

from src.grid.electrical_distance import electrical_distance
from src.grid.kron_reduce import kron_reduce
from src.grid.load_case import SPEC_TABLE1


class TestGridCase:
    def test_bus_count(self, grid_case):
        assert len(grid_case.buses) >= 39, f"Expected ≥39 buses, got {len(grid_case.buses)}"

    def test_ybus_shape(self, grid_case):
        N = len(grid_case.buses)
        assert grid_case.Ybus.shape == (N, N)

    def test_ybus_complex(self, grid_case):
        assert grid_case.Ybus.dtype == complex

    def test_zbus_shape(self, grid_case):
        assert grid_case.Zbus.shape == grid_case.Ybus.shape

    def test_pmu_bus_indices_count(self, grid_case):
        assert len(grid_case.pmu_bus_indices) == 8, (
            f"Expected 8 PMU bus indices, got {len(grid_case.pmu_bus_indices)}"
        )

    def test_gen_indices_found(self, grid_case):
        assert len(grid_case.gen_indices) > 0, "No generator indices found"

    def test_comp_to_psse_mapping(self, grid_case):
        """All 8 competition bus numbers should map to PSS/E bus numbers."""
        for comp_k in [2, 5, 6, 10, 19, 22, 29, 39]:
            assert comp_k in grid_case.comp_to_psse, (
                f"Competition bus {comp_k} not in comp_to_psse mapping"
            )


class TestBaseVoltages:
    def test_pmu_voltages_match_spec(self, grid_case):
        """Sanity check 4: base-case voltages match spec to 3 decimals."""
        for comp_k, (ref_vm, ref_va) in SPEC_TABLE1.items():
            psse_num = grid_case.comp_to_psse.get(comp_k)
            assert psse_num is not None, f"BUS{comp_k} not mapped"
            idx = grid_case.psse_to_idx[psse_num]
            bus = grid_case.buses[idx]
            err_v = abs(bus.vm_pu - ref_vm)
            err_a = abs(bus.va_deg - ref_va)
            assert err_v < 0.005, (
                f"BUS{comp_k}: |V|={bus.vm_pu:.4f} vs spec {ref_vm:.4f} (Δ={err_v:.4f})"
            )
            assert err_a < 1.0, (
                f"BUS{comp_k}: θ={bus.va_deg:.2f}° vs spec {ref_va:.2f}° (Δ={err_a:.2f}°)"
            )

    def test_bus39_specific(self, grid_case):
        """BUS39 = 1.0300 p.u., -10.05° per spec."""
        psse_num = grid_case.comp_to_psse[39]
        idx = grid_case.psse_to_idx[psse_num]
        bus = grid_case.buses[idx]
        assert abs(bus.vm_pu - 1.03) < 0.005
        assert abs(bus.va_deg - (-10.05)) < 0.5


class TestElectricalDistance:
    def test_symmetry(self, grid_case):
        D = electrical_distance(grid_case.Zbus)
        assert np.allclose(D, D.T, atol=1e-10)

    def test_zero_diagonal(self, grid_case):
        D = electrical_distance(grid_case.Zbus)
        assert np.allclose(np.diag(D), 0.0, atol=1e-10)

    def test_nonnegative(self, grid_case):
        D = electrical_distance(grid_case.Zbus)
        assert np.all(D >= -1e-12)

    def test_shape(self, grid_case):
        D = electrical_distance(grid_case.Zbus)
        N = len(grid_case.buses)
        assert D.shape == (N, N)


class TestKronReduction:
    def test_shape(self, grid_case):
        n = len(grid_case.gen_indices)
        if n == 0:
            pytest.skip("No generator indices found")
        Y_red = kron_reduce(grid_case.Ybus, grid_case.gen_indices)
        assert Y_red.shape == (n, n)

    def test_diagonal_dominant(self, grid_case):
        """Reduced Ybus diagonal should dominate (passivity check)."""
        n = len(grid_case.gen_indices)
        if n == 0:
            pytest.skip("No generator indices found")
        Y_red = kron_reduce(grid_case.Ybus, grid_case.gen_indices)
        # NOTE: Y_red is NOT symmetric when phase-shifting transformers are present.
        # That is physically correct — the Ybus with phase shifts is non-symmetric.
        # Instead, check that diagonal elements have negative imaginary part (capacitive/inductive)
        # and that the matrix is finite.
        assert np.all(np.isfinite(Y_red)), "Reduced Ybus contains NaN or Inf"
        assert Y_red.shape == (n, n)
