"""ANDES normal-operation runtime for m4 type0 scenario."""

from __future__ import annotations


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
        raise RuntimeError("ANDES TDS failed for type0 normal simulation.")

    return system
