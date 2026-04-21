"""ANDES normal-operation runtime for m4 type0 scenario."""

from __future__ import annotations

import logging

import numpy as np


LOGGER = logging.getLogger(__name__)


def _is_usable_tds_result(system, tf: float, tstep: float) -> bool:
    """Return True when TDS trajectory is sufficiently complete for downstream use."""
    try:
        t = np.asarray(system.dae.ts.t, dtype=float)
    except Exception:
        return False
    if t.size < 4:
        return False
    last_t = float(np.nanmax(t))
    # Accept near-complete runs where ANDES reports False at the very end.
    tolerance = max(5.0 * float(tstep), 1e-6)
    return np.isfinite(last_t) and last_t >= float(tf) - tolerance


def run_normal_simulation_andes(tf: float = 10.1, tstep: float = 1.0 / 30.0):
    """Run IEEE39 normal simulation with no injected fault."""
    import andes

    system = andes.load(
        andes.get_case("ieee39/ieee39_full.xlsx"),
        setup=False,
        no_output=True,
    )
    system.setup()

    ok_pf = system.PFlow.run()
    if ok_pf is False:
        raise RuntimeError("ANDES PFlow failed for type0 normal simulation.")

    system.TDS.config.tf = float(tf)
    system.TDS.config.tstep = float(tstep)
    if hasattr(system.TDS.config, "criteria"):
        system.TDS.config.criteria = 0

    ok_tds = system.TDS.run()
    if ok_tds is False:
        if _is_usable_tds_result(system, tf=float(tf), tstep=float(tstep)):
            LOGGER.warning(
                "ANDES TDS returned False but usable trajectory exists "
                "(near-complete run). Continuing with extracted time series."
            )
        else:
            raise RuntimeError("ANDES TDS failed for type0 normal simulation.")

    return system
