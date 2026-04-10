"""M2 tests: swing model and measurement function."""
from __future__ import annotations

import numpy as np
import pytest

from src.dynamics.parameters import GEN_PSSE_BUSES, S_BASE, load_params
from src.dynamics.swing import N_GEN, STATE_DIM, SwingModel, build_model
from src.dynamics.measurement import (
    N_CHAN, N_PMU, F0, V_BASE_LN, I_BASE,
    build_measurement_model,
)
from src.grid.kron_reduce import kron_reduce


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gen_params():
    return load_params()


@pytest.fixture(scope="module")
def swing(grid_case, gen_params):
    """Build the SwingModel from the grid case and ANDES parameters."""
    # Kron-reduce Ybus to generator terminal buses
    Y_red = kron_reduce(grid_case.Ybus, grid_case.gen_indices)

    # Extract base-case terminal bus angles and magnitudes
    gen_angles_deg = []
    gen_vmag_pu = []
    for psse_bus in GEN_PSSE_BUSES:
        idx = grid_case.psse_to_idx.get(psse_bus)
        if idx is not None:
            bus = grid_case.buses[idx]
            gen_angles_deg.append(bus.va_deg)
            gen_vmag_pu.append(bus.vm_pu)
        else:
            gen_angles_deg.append(0.0)
            gen_vmag_pu.append(1.0)

    return build_model(gen_params, Y_red, gen_angles_deg, gen_vmag_pu)


@pytest.fixture(scope="module")
def meas_model(swing, grid_case):
    """Build the MeasurementModel."""
    N = len(grid_case.buses)
    bus_theta0_rad = np.deg2rad(np.array([b.va_deg for b in grid_case.buses]))
    bus_vmag0_pu = np.array([b.vm_pu for b in grid_case.buses])

    return build_measurement_model(
        swing_model=swing,
        Ybus=grid_case.Ybus,
        pmu_indices=grid_case.pmu_bus_indices,
        gen_indices=grid_case.gen_indices,
        bus_theta0_rad=bus_theta0_rad,
        bus_vmag0_pu=bus_vmag0_pu,
    )


# ── Generator Parameter Tests ─────────────────────────────────────────────────

class TestGenParams:
    def test_shape(self, gen_params):
        assert gen_params.H.shape == (N_GEN,)
        assert gen_params.D.shape == (N_GEN,)
        assert gen_params.xd1_sys.shape == (N_GEN,)

    def test_H_positive(self, gen_params):
        assert np.all(gen_params.H > 0), "All inertia constants must be positive"

    def test_H_range(self, gen_params):
        """H values should be in the range 2-60 s (Gen 10 has H=50)."""
        assert np.all(gen_params.H >= 2.0), f"Min H too low: {gen_params.H.min():.2f}"
        assert np.all(gen_params.H <= 60.0), f"Max H too high: {gen_params.H.max():.2f}"

    def test_xd1_sys_positive(self, gen_params):
        assert np.all(gen_params.xd1_sys > 0)

    def test_psse_buses_count(self, gen_params):
        assert len(gen_params.psse_buses) == N_GEN


# ── Swing Model Tests ─────────────────────────────────────────────────────────

class TestSwingModel:
    def test_x0_shape(self, swing):
        assert swing.x0().shape == (STATE_DIM,)

    def test_x0_omega_zero(self, swing):
        x0 = swing.x0()
        assert np.allclose(x0[N_GEN: 2 * N_GEN], 0.0), "Base-case ω must be zero"

    def test_Pe_equals_Pm_at_base(self, swing):
        """At x0, P_e,i = P_m,i exactly (by construction in build_model)."""
        Pe = swing.electric_power(swing.delta0)
        np.testing.assert_allclose(Pe, swing.Pm0, rtol=1e-10, atol=1e-12,
            err_msg="Electrical power must equal mechanical power at base case")

    def test_f_zero_at_equilibrium(self, swing):
        """dx/dt must be zero at the base-case state."""
        x0 = swing.x0()
        dxdt = swing.f(x0)
        np.testing.assert_allclose(dxdt, 0.0, atol=1e-10,
            err_msg="dxdt must be zero at equilibrium")

    def test_rk4_step_preserves_equilibrium(self, swing):
        """One RK4 step at equilibrium returns the same state."""
        x0 = swing.x0()
        x1 = swing.step(x0, 1.0 / 30.0)
        np.testing.assert_allclose(x1, x0, atol=1e-10)

    def test_bounded_90_minutes(self, swing):
        """Open-loop simulation over 90 min stays within safe bounds.

        Per M2 spec: zero process noise, base-case initial conditions.
        Rotor angles should stay within ±5° (≈ 0.087 rad) of base case.
        Speed deviations should stay < 0.1 rad/s.
        """
        x0 = swing.x0()
        dt = 1.0 / 30.0             # 30 fps (approximate)
        n_steps = int(90 * 60 / dt)  # 90 minutes

        # Simulate in chunks to save memory — only track max deviation
        x = x0.copy()
        delta0 = swing.delta0.copy()
        max_delta_dev = 0.0
        max_omega = 0.0

        # Subsample to 1-second chunks for efficiency (still 5400 steps total)
        chunk = 30  # 1 second worth
        for _ in range(int(n_steps / chunk)):
            for _ in range(chunk):
                x = swing.step(x, dt)
            delta_dev = np.abs(x[:N_GEN] - delta0)
            max_delta_dev = max(max_delta_dev, delta_dev.max())
            max_omega = max(max_omega, np.abs(x[N_GEN: 2*N_GEN]).max())

        assert max_delta_dev < 0.001, (
            f"Rotor angle deviated {np.rad2deg(max_delta_dev):.4f}° from base case"
            f" over 90 min — expected < 0.057° (1e-3 rad)"
        )
        assert max_omega < 0.001, (
            f"Speed deviation {max_omega:.6f} rad/s exceeds bound — expected < 1e-3"
        )


# ── Measurement Model Tests ───────────────────────────────────────────────────

class TestMeasurementModel:
    def test_h_shape(self, swing, meas_model):
        x0 = swing.x0()
        y = meas_model.h(np.array(x0))
        assert y.shape == (N_PMU, N_CHAN), (
            f"Expected ({N_PMU}, {N_CHAN}), got {y.shape}"
        )

    def test_h_finite(self, swing, meas_model):
        x0 = swing.x0()
        y = np.array(meas_model.h(np.array(x0)))
        assert np.all(np.isfinite(y)), "h(x0) contains NaN or Inf"

    def test_frequency_at_base_case(self, swing, meas_model):
        """At base case, frequency must be exactly 60 Hz."""
        x0 = swing.x0()
        y = np.array(meas_model.h(np.array(x0)))
        freq = y[:, 12]
        np.testing.assert_allclose(freq, F0, atol=1e-8,
            err_msg="Frequency must be 60 Hz at base case")

    def test_rocof_at_base_case(self, swing, meas_model):
        """At base case (ω=0), ROCOF must be 0."""
        x0 = swing.x0()
        y = np.array(meas_model.h(np.array(x0)))
        rocof = y[:, 13]
        np.testing.assert_allclose(rocof, 0.0, atol=1e-8,
            err_msg="ROCOF must be 0 at base case (ω=0)")

    def test_voltage_magnitudes_near_pf(self, swing, meas_model):
        """At base case, VA_MAG should match power-flow values within 1%."""
        x0 = swing.x0()
        y = np.array(meas_model.h(np.array(x0)))
        va_mag = y[:, 1]   # volts
        # Compare to expected: vmag0 * V_BASE_LN
        expected = meas_model.vmag0 * V_BASE_LN
        np.testing.assert_allclose(va_mag, expected, rtol=1e-6)

    def test_three_phase_symmetry(self, swing, meas_model):
        """VB_ANG = VA_ANG - 120°, VC_ANG = VA_ANG + 120°."""
        x0 = swing.x0()
        y = np.array(meas_model.h(np.array(x0)))
        VA = y[:, 0]
        VB = y[:, 2]
        VC = y[:, 4]
        np.testing.assert_allclose(VB, VA - 120.0, atol=1e-8)
        np.testing.assert_allclose(VC, VA + 120.0, atol=1e-8)

    def test_current_magnitudes_positive(self, swing, meas_model):
        """Current magnitudes must be positive at base case."""
        x0 = swing.x0()
        y = np.array(meas_model.h(np.array(x0)))
        IA_mag = y[:, 7]
        assert np.all(IA_mag > 0), "Current magnitudes must be positive"

    def test_match_csv_voltage_magnitude(self, merged_df, swing, meas_model):
        """h(x0) voltage magnitudes should be within 5% of CSV row 0 values."""
        x0 = swing.x0()
        y = np.array(meas_model.h(np.array(x0)))
        va_mag_pred = y[:, 1]   # volts at 8 PMU buses

        from src.io.load_csv import PMU_BUSES
        row0 = merged_df.iloc[0]
        for i, bus in enumerate(PMU_BUSES):
            csv_col = f"BUS{bus}_VA_MAG"
            if csv_col not in merged_df.columns:
                continue
            csv_val = float(row0[csv_col])
            pred_val = float(va_mag_pred[i])
            if np.isnan(csv_val):
                continue
            err_frac = abs(pred_val - csv_val) / csv_val
            assert err_frac < 0.05, (
                f"BUS{bus} VA_MAG: predicted {pred_val:.0f} V vs CSV {csv_val:.0f} V "
                f"({100*err_frac:.1f}% error, threshold 5%)"
            )

    def test_h_flat_shape(self, swing, meas_model):
        x0 = swing.x0()
        y_flat = np.array(meas_model.h_flat(np.array(x0)))
        assert y_flat.shape == (N_PMU * N_CHAN,), (
            f"Expected ({N_PMU * N_CHAN},), got {y_flat.shape}"
        )
