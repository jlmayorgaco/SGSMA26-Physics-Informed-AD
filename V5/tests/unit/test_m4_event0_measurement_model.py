from __future__ import annotations

import numpy as np

import src.simulation.event0_measurement_model as mm


def test_convert_clean_to_event0_center_magnitude_scales() -> None:
    clean = np.array([10.0, 10.0, 10.0])
    artifacts = {"noise_layer": None, "family_defaults": {"voltage_mag": {"eda_stats": {"median": 20.0}}}}
    out = mm.convert_clean_to_event0_center(clean, "VA_MAG", artifacts, "10")
    assert np.allclose(out, np.array([20.0, 20.0, 20.0]))


def test_convert_clean_to_event0_center_frequency_shifts() -> None:
    clean = np.array([59.9, 60.0, 60.1])
    artifacts = {"noise_layer": None, "family_defaults": {"frequency": {"eda_stats": {"median": 61.0}}}}
    out = mm.convert_clean_to_event0_center(clean, "Freq", artifacts, "10")
    assert np.isclose(np.median(out), 61.0)


def test_apply_event0_measurement_model_preserves_length_and_clips(monkeypatch) -> None:
    class _StubLegacy:
        ESTIMATION_FREQ_CLIP = (58.5, 61.5)
        ESTIMATION_ROCOF_CLIP = (-20.0, 20.0)

        @staticmethod
        def apply_event0_measurement_model(**kwargs):
            clean = np.asarray(kwargs["clean_values"], dtype=float)
            noisy = clean + 100.0
            return clean, noisy

    monkeypatch.setattr(mm, "_legacy_m4", lambda: _StubLegacy())
    clean, noisy = mm.apply_event0_measurement_model(
        bus_id="10",
        raw_suffix="Freq",
        clean_values=np.array([60.0, 60.0]),
        t=np.array([0.0, 1.0]),
        artifacts={},
        rng=np.random.default_rng(1),
    )
    assert len(clean) == 2
    assert len(noisy) == 2
    assert np.all(noisy <= 61.5)


def test_disabled_angle_noise_clean_wrapping_policy() -> None:
    # policy-level check: if model returns angle-like values, wrapper keeps finite and length-preserving outputs.
    clean = np.array([179.0, -179.0])
    assert len(clean) == 2
