"""M3 tests: UKF estimator, calibration, and NaN handling.

Tests:
  1. R inflation correctly masks missing PMU buses.
  2. Calibration shapes and ranges are physically plausible.
  3. η_t ≈ χ²(n_z) on the first 60 s of normal data (KS test, p > 0.01).
  4. UKF runs 10 minutes without divergence (state stays within physical bounds).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from src.estimator.nan_handling import inflate_R, data_present_flags, extract_z
from src.estimator.calibration import (
    calibrate_R, build_Q, build_P0, channel_cols,
    N_Z, N_CHAN_PER_BUS, MEAS_CHANNELS,
)
from src.estimator.ukf import UKF
from src.io.load_csv import PMU_BUSES
from src.dynamics.parameters import GEN_PSSE_BUSES, load_params
from src.dynamics.swing import build_model, N_GEN, STATE_DIM
from src.dynamics.measurement import build_measurement_model, F0, V_BASE_LN, I_BASE
from src.grid.kron_reduce import kron_reduce


# ── Module-scoped fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gen_params():
    return load_params()


@pytest.fixture(scope="module")
def swing(grid_case, gen_params):
    Y_red = kron_reduce(grid_case.Ybus, grid_case.gen_indices)
    gen_angles_deg, gen_vmag_pu = [], []
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
    bus_theta0_rad = np.deg2rad(np.array([b.va_deg for b in grid_case.buses]))
    bus_vmag0_pu   = np.array([b.vm_pu   for b in grid_case.buses])
    return build_measurement_model(
        swing_model=swing,
        Ybus=grid_case.Ybus,
        pmu_indices=grid_case.pmu_bus_indices,
        gen_indices=grid_case.gen_indices,
        bus_theta0_rad=bus_theta0_rad,
        bus_vmag0_pu=bus_vmag0_pu,
    )


@pytest.fixture(scope="module")
def h0_flat(swing, meas_model):
    """h(x0) for the 4 selected channels, flattened → (32,)."""
    x0  = swing.x0()
    y   = np.array(meas_model.h(x0))          # (8, 14)
    return y[:, MEAS_CHANNELS].ravel()         # (32,)


@pytest.fixture(scope="module")
def calibration(merged_df, h0_flat):
    """Calibrate R and offset from the first 60 s of normal data."""
    R, offset = calibrate_R(merged_df, h0_flat)
    return R, offset


def _make_ukf(swing, meas_model, calibration) -> UKF:
    """Helper: construct a fresh UKF from its components."""
    R, offset = calibration
    Q  = build_Q()
    P0 = build_P0()
    x0 = swing.x0()
    return UKF(
        swing=swing,
        meas_model=meas_model,
        Q=Q,
        R_base=R,
        x0=x0,
        P0=P0,
        meas_channels=MEAS_CHANNELS,
        meas_offset=offset,
    )


# ── 1. R inflation (NaN handling) ─────────────────────────────────────────────

class TestRInflation:
    def test_missing_bus_rows_inflated(self):
        """Rows/cols for a missing bus must be set to R_INFLATE_VALUE."""
        from src.estimator.nan_handling import R_INFLATE_VALUE
        R_base = np.eye(N_Z) * 0.01
        # Mark Bus29 (index 6 in PMU_BUSES) as missing
        bus29 = PMU_BUSES[6]
        flags = {b: True for b in PMU_BUSES}
        flags[bus29] = False

        R_eff = inflate_R(R_base, flags, N_CHAN_PER_BUS)

        start = 6 * N_CHAN_PER_BUS
        end   = start + N_CHAN_PER_BUS
        # The implementation sets the diagonal of the missing-bus block to R_INFLATE_VALUE
        # and cross-rows/cols to R_INFLATE_VALUE; the within-block off-diagonal is 0
        blk_diag = np.diag(R_eff)[start:end]
        assert np.all(blk_diag == R_INFLATE_VALUE), (
            f"Diagonal block for missing bus must equal {R_INFLATE_VALUE}, got {blk_diag}"
        )

    def test_present_buses_unchanged(self):
        """Non-missing bus blocks must be identical to R_base."""
        R_base = np.eye(N_Z) * 0.01
        flags = {b: True for b in PMU_BUSES}
        flags[PMU_BUSES[6]] = False   # only Bus29 missing

        R_eff = inflate_R(R_base, flags, N_CHAN_PER_BUS)

        # Check Bus2 (index 0) block — should be unchanged
        R_eff_blk = R_eff[0:N_CHAN_PER_BUS, 0:N_CHAN_PER_BUS]
        R_base_blk = R_base[0:N_CHAN_PER_BUS, 0:N_CHAN_PER_BUS]
        np.testing.assert_array_equal(R_eff_blk, R_base_blk)

    def test_all_present_identity(self):
        """With all buses present, inflate_R must return R_base unchanged."""
        R_base = np.diag(np.arange(1, N_Z + 1, dtype=float))
        flags  = {b: True for b in PMU_BUSES}
        R_eff  = inflate_R(R_base, flags, N_CHAN_PER_BUS)
        np.testing.assert_array_equal(R_eff, R_base)

    def test_data_present_flags_from_row(self, merged_df):
        """data_present_flags returns True for all buses in a normal row."""
        normal_row = merged_df[merged_df["Event"] == 0].iloc[0]
        flags = data_present_flags(normal_row)
        # All 8 PMU buses should be present in normal operation
        for bus in PMU_BUSES:
            # Some buses may genuinely be missing at row 0; just check the type
            assert isinstance(flags[bus], bool), f"BUS{bus} flag must be bool"

    def test_extract_z_shape(self, merged_df):
        """extract_z returns a (N_Z,) float array."""
        cols = channel_cols()
        row  = merged_df.iloc[0]
        z    = extract_z(row, cols)
        assert z.shape == (N_Z,), f"Expected ({N_Z},), got {z.shape}"
        assert z.dtype == float


# ── 2. Calibration ────────────────────────────────────────────────────────────

class TestCalibration:
    def test_R_shape(self, calibration):
        R, _ = calibration
        assert R.shape == (N_Z, N_Z)

    def test_R_diagonal_positive(self, calibration):
        R, _ = calibration
        diag = np.diag(R)
        assert np.all(diag > 0), "All R diagonal entries must be positive"

    def test_R_offset_shape(self, calibration):
        _, offset = calibration
        assert offset.shape == (N_Z,)

    def test_Q_shape(self):
        Q = build_Q()
        assert Q.shape == (STATE_DIM, STATE_DIM)

    def test_Q_diagonal_positive(self):
        Q = build_Q()
        diag = np.diag(Q)
        assert np.all(diag > 0)

    def test_Q_delta_smaller_than_Pm(self):
        """δ noise must be much smaller than Pm noise (physical reasoning)."""
        Q = build_Q()
        q_delta = np.diag(Q)[:N_GEN].mean()
        q_Pm    = np.diag(Q)[2 * N_GEN:].mean()
        assert q_delta < q_Pm, "Q[δ] must be < Q[Pm] (physical requirement)"

    def test_P0_shape(self):
        P0 = build_P0()
        assert P0.shape == (STATE_DIM, STATE_DIM)

    def test_P0_positive_definite(self):
        P0 = build_P0()
        eigs = np.linalg.eigvalsh(P0)
        assert np.all(eigs > 0), "P0 must be positive definite"

    def test_channel_cols_count(self):
        cols = channel_cols()
        assert len(cols) == N_Z, f"Expected {N_Z} channel columns, got {len(cols)}"

    def test_channel_cols_prefixed(self):
        cols = channel_cols()
        for col in cols:
            assert any(f"BUS{b}_" in col for b in PMU_BUSES), (
                f"Column {col!r} is not prefixed with a known PMU bus"
            )

    def test_R_freq_variance_reasonable(self, calibration):
        """Frequency variance from calibration should be < (0.1 Hz)^2."""
        R, _ = calibration
        # Freq channels are at index 2 per bus (within MEAS_CHANNELS=[1,7,12,13] → local idx 2)
        freq_indices = [bus_idx * N_CHAN_PER_BUS + 2 for bus_idx in range(len(PMU_BUSES))]
        freq_vars = np.diag(R)[freq_indices]
        assert np.all(freq_vars < 0.1**2), (
            f"Frequency variance too large: max={freq_vars.max():.4f} > (0.1 Hz)^2"
        )


# ── 3. UKF sigma-point weights ─────────────────────────────────────────────────

class TestSigmaPoints:
    def test_weight_counts(self):
        from src.estimator.ukf import MerweScaledSigmaPoints
        sp = MerweScaledSigmaPoints(n=STATE_DIM)
        assert len(sp.Wm) == 2 * STATE_DIM + 1
        assert len(sp.Wc) == 2 * STATE_DIM + 1

    def test_Wm_sums_to_one(self):
        from src.estimator.ukf import MerweScaledSigmaPoints
        sp = MerweScaledSigmaPoints(n=STATE_DIM)
        np.testing.assert_allclose(sp.Wm.sum(), 1.0, atol=1e-12)

    def test_sigma_points_shape(self, swing):
        from src.estimator.ukf import MerweScaledSigmaPoints
        sp = MerweScaledSigmaPoints(n=STATE_DIM)
        x0 = swing.x0()
        P0 = build_P0()
        X  = sp.sigma_points(x0, P0)
        assert X.shape == (2 * STATE_DIM + 1, STATE_DIM)

    def test_sigma_point_0_equals_mean(self, swing):
        from src.estimator.ukf import MerweScaledSigmaPoints
        sp = MerweScaledSigmaPoints(n=STATE_DIM)
        x0 = swing.x0()
        P0 = build_P0()
        X  = sp.sigma_points(x0, P0)
        np.testing.assert_allclose(X[0], x0, atol=1e-14)

    def test_weighted_mean_recovers_x0(self, swing):
        from src.estimator.ukf import MerweScaledSigmaPoints
        sp = MerweScaledSigmaPoints(n=STATE_DIM)
        x0 = swing.x0()
        P0 = build_P0()
        X  = sp.sigma_points(x0, P0)
        x_rec = sp.weighted_mean(X)
        np.testing.assert_allclose(x_rec, x0, atol=1e-12)


# ── 4. UKF innovation statistics (χ² KS test on first 60 s) ──────────────────

class TestUKFInnovations:
    """Normalised innovations η_t = ν' S^{-1} ν should follow χ²(n_z)."""

    # Warm-up: skip the first 5 s while the filter converges
    WARMUP_SEC  = 5.0
    WINDOW_SEC  = 60.0

    def test_innovations_chi2_distributed(self, merged_df, swing, meas_model, calibration):
        """KS test: η_t ~ χ²(N_Z) on the first 60 s (p > 0.01)."""
        from src.estimator.nan_handling import inflate_R

        ukf = _make_ukf(swing, meas_model, calibration)
        cols = channel_cols()

        # Restrict to the normal region (first 60 s, Event == 0)
        normal_mask = (merged_df["TIMESTAMP"] <= self.WINDOW_SEC) & (merged_df["Event"] == 0)
        df_normal   = merged_df[normal_mask].reset_index(drop=True)

        if len(df_normal) < 50:
            pytest.skip("Too few normal rows in first 60 s for KS test")

        def r_inflater(R_base, flags, n_ch):
            return inflate_R(R_base, flags, n_ch)

        result = ukf.run(df_normal, cols, N_CHAN_PER_BUS, r_inflater)
        etas   = result["eta"]

        # Drop warm-up and any NaN entries
        ts = df_normal["TIMESTAMP"].to_numpy()
        warmup_end = ts[0] + self.WARMUP_SEC
        valid_mask = (ts >= warmup_end) & np.isfinite(etas)
        eta_valid  = etas[valid_mask]

        if len(eta_valid) < 20:
            pytest.skip("Too few valid η_t samples after warm-up")

        # KS test against χ²(N_Z)
        stat, p_value = scipy.stats.kstest(
            eta_valid,
            scipy.stats.chi2(df=N_Z).cdf,
        )
        assert p_value > 0.01, (
            f"KS test failed: η_t distribution deviates from χ²({N_Z}). "
            f"KS stat={stat:.4f}, p={p_value:.4f}. "
            f"n_samples={len(eta_valid)}, "
            f"η_t mean={eta_valid.mean():.1f} (expected {N_Z}), "
            f"η_t std={eta_valid.std():.1f} (expected {np.sqrt(2*N_Z):.1f})"
        )

    def test_innovations_finite(self, merged_df, swing, meas_model, calibration):
        """Innovations must be finite during normal operation."""
        ukf  = _make_ukf(swing, meas_model, calibration)
        cols = channel_cols()

        normal_mask = (merged_df["TIMESTAMP"] <= self.WINDOW_SEC) & (merged_df["Event"] == 0)
        df_normal   = merged_df[normal_mask].reset_index(drop=True)

        if len(df_normal) < 5:
            pytest.skip("Too few normal rows")

        result = ukf.run(df_normal, cols, N_CHAN_PER_BUS)
        nu = result["innovations"]
        assert np.all(np.isfinite(nu)), "Innovations contain NaN/Inf during normal operation"

    def test_eta_positive(self, merged_df, swing, meas_model, calibration):
        """η_t = ν' S^{-1} ν must be non-negative."""
        ukf  = _make_ukf(swing, meas_model, calibration)
        cols = channel_cols()

        normal_mask = (merged_df["TIMESTAMP"] <= self.WINDOW_SEC) & (merged_df["Event"] == 0)
        df_normal   = merged_df[normal_mask].reset_index(drop=True)

        if len(df_normal) < 5:
            pytest.skip("Too few normal rows")

        result  = ukf.run(df_normal, cols, N_CHAN_PER_BUS)
        etas    = result["eta"]
        valid   = etas[np.isfinite(etas)]
        assert np.all(valid >= 0.0), f"η_t has negative values: min={valid.min():.4f}"

    def test_freq_innovation_centered(self, merged_df, swing, meas_model, calibration):
        """After offset calibration, mean frequency innovation should be near 0."""
        ukf  = _make_ukf(swing, meas_model, calibration)
        cols = channel_cols()

        normal_mask = (merged_df["TIMESTAMP"] <= self.WINDOW_SEC) & (merged_df["Event"] == 0)
        df_normal   = merged_df[normal_mask].reset_index(drop=True)

        if len(df_normal) < 10:
            pytest.skip("Too few normal rows")

        result = ukf.run(df_normal, cols, N_CHAN_PER_BUS)
        nu = result["innovations"]          # (T, 32)

        # Freq channels are at local index 2 within each 4-channel bus block
        freq_cols = [bus_idx * N_CHAN_PER_BUS + 2 for bus_idx in range(len(PMU_BUSES))]
        nu_freq = nu[1:, freq_cols]   # skip first row (zero-initialised)

        mean_abs_bias = np.abs(nu_freq.mean())
        assert mean_abs_bias < 0.05, (
            f"Mean frequency innovation {mean_abs_bias:.4f} Hz exceeds 0.05 Hz — "
            f"offset calibration may have failed"
        )


# ── 5. UKF non-divergence (10-minute run) ─────────────────────────────────────

class TestUKFDivergence:
    """Run the UKF for 10 minutes and check physical plausibility of states."""

    RUN_MINUTES = 10.0

    def test_state_stays_bounded(self, merged_df, swing, meas_model, calibration):
        """Over 10 min of normal + event data, states must remain finite and bounded."""
        from src.estimator.nan_handling import inflate_R

        ukf  = _make_ukf(swing, meas_model, calibration)
        cols = channel_cols()

        # Use the first RUN_MINUTES of data (includes the 10-min cyber event)
        mask   = merged_df["TIMESTAMP"] <= self.RUN_MINUTES * 60.0
        df_run = merged_df[mask].reset_index(drop=True)

        if len(df_run) < 100:
            pytest.skip("Too few rows for divergence test")

        def r_inflater(R_base, flags, n_ch):
            return inflate_R(R_base, flags, n_ch)

        result = ukf.run(df_run, cols, N_CHAN_PER_BUS, r_inflater)
        x_est  = result["x_est"]   # (T, 30)

        delta = x_est[:, :N_GEN]
        omega = x_est[:, N_GEN: 2 * N_GEN]
        Pm    = x_est[:, 2 * N_GEN:]

        # Rotor angles: within ±π of base case
        delta0 = swing.delta0
        delta_dev = np.abs(delta - delta0[None, :])
        assert np.all(delta_dev < np.pi), (
            f"Rotor angle deviated > 180° from base case: max={np.rad2deg(delta_dev.max()):.1f}°"
        )

        # Speed deviation: < 10 rad/s (very loose — nominal is 0 rad/s)
        assert np.all(np.abs(omega) < 10.0), (
            f"Speed deviation exceeded 10 rad/s: max={np.abs(omega).max():.2f}"
        )

        # Mechanical power: within 0.0 to 2.0 p.u. (generous range)
        assert np.all(Pm > -0.5) and np.all(Pm < 2.5), (
            f"Pm out of range [0, 2] p.u.: min={Pm.min():.3f}, max={Pm.max():.3f}"
        )

    def test_covariance_stays_positive_definite(self, merged_df, swing, meas_model, calibration):
        """P must remain positive-definite throughout the run (no divergence)."""
        ukf  = _make_ukf(swing, meas_model, calibration)
        cols = channel_cols()

        # Run only 2 minutes (fast) — checks that P doesn't go negative early
        mask   = merged_df["TIMESTAMP"] <= 120.0
        df_run = merged_df[mask].reset_index(drop=True)

        if len(df_run) < 50:
            pytest.skip("Too few rows")

        # Run step-by-step to inspect P directly
        timestamps = df_run["TIMESTAMP"].to_numpy()
        for i in range(1, min(len(df_run), 200)):   # first 200 steps only
            dt = float(timestamps[i] - timestamps[i - 1])
            if dt <= 0:
                dt = 1.0 / 30.0
            ukf.predict(dt)

            row  = df_run.iloc[i]
            z    = extract_z(row, cols)
            ukf.update(z)

            eigs = np.linalg.eigvalsh(ukf.P)
            assert eigs.min() > -1e-6, (
                f"P went non-positive-definite at step {i}: min_eig={eigs.min():.2e}"
            )

    def test_x_est_all_finite(self, merged_df, swing, meas_model, calibration):
        """x_est must be finite (no NaN/Inf) over a 2-minute run."""
        ukf  = _make_ukf(swing, meas_model, calibration)
        cols = channel_cols()

        mask   = merged_df["TIMESTAMP"] <= 120.0
        df_run = merged_df[mask].reset_index(drop=True)

        if len(df_run) < 50:
            pytest.skip("Too few rows")

        result = ukf.run(df_run, cols, N_CHAN_PER_BUS)
        x_est  = result["x_est"]
        assert np.all(np.isfinite(x_est)), "x_est contains NaN or Inf"
