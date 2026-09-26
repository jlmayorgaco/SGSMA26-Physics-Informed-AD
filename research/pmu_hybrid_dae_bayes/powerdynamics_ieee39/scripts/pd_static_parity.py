"""Translate frozen PowerDynamics IEEE39 static data into pandapower."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output" / "provenance" / "powerdynamics_original"
RESULTS = ROOT / "output" / "results"
REPORTS = ROOT / "output" / "reports"
RESULTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def build_ppc():
    bus = pd.read_csv(DATA / "bus.csv")
    branch = pd.read_csv(DATA / "branch.csv")
    loads = pd.read_csv(DATA / "load.csv").set_index("bus")
    base = 100.0
    pb = np.zeros((len(bus), 13), float)
    pb[:, 0] = bus.bus
    pb[:, 1] = bus.bus_type.map({"PQ": 1, "PV": 2, "Slack": 3})
    for k, row in bus.iterrows():
        if int(row.bus) in loads.index:
            pb[k, 2] = -float(loads.loc[int(row.bus), "Pset"]) * base
            pb[k, 3] = -float(loads.loc[int(row.bus), "Qset"]) * base
        pb[k, 7] = float(row.V) if np.isfinite(row.V) else 1.0
        pb[k, 9] = float(row.base_kv)
        pb[k, 11] = 1.1
        pb[k, 12] = 0.9
    gr = []
    for _, row in bus.iterrows():
        if row.bus_type not in ("PV", "Slack"):
            continue
        lp = float(loads.loc[int(row.bus), "Pset"]) if int(row.bus) in loads.index else 0.0
        gr.append([row.bus, (float(row.P) - lp) * base, 0.0, 1e4, -1e4, float(row.V) if np.isfinite(row.V) else 1.0, 1.0, 1, 1e4, -1e4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    pg = np.asarray(gr, float)
    pbr = np.zeros((len(branch), 21), float)
    pbr[:, 0] = branch.src_bus
    pbr[:, 1] = branch.dst_bus
    pbr[:, 2] = branch.R
    pbr[:, 3] = branch.X
    pbr[:, 4] = branch.B_src + branch.B_dst
    pbr[:, 5:7] = 100.0
    pbr[:, 8] = branch.r_src.where(branch.r_src != 0, 1.0)
    pbr[:, 10] = 1.0
    pbr[:, 11] = -360.0
    pbr[:, 12] = 360.0
    # PowerDynamics applies r_src on the source side.  pandapower's trafo
    # converter canonicalizes the high-voltage side, so preserve the exact
    # two-port primitive by transforming low-to-high rows before conversion.
    kv = bus.set_index("bus").base_kv
    tr = branch.transformer.to_numpy(dtype=bool)
    low = np.array([float(kv[int(r.src_bus)]) < float(kv[int(r.dst_bus)]) for r in branch.itertuples()])
    for k in np.where(tr & low)[0]:
        ratio = float(branch.iloc[k].r_src) or 1.0
        pbr[k, 2:4] /= ratio**2
        pbr[k, 8] = ratio
    for k in np.where(tr & ~low)[0]:
        ratio = float(branch.iloc[k].r_src) or 1.0
        pbr[k, 8] = 1.0 / ratio
    return {"version": "2", "baseMVA": base, "bus": pb, "gen": pg, "branch": pbr}, bus, branch


def pd_ybus(branch):
    y = np.zeros((39, 39), complex)
    for r in branch.itertuples(index=False):
        i, j = int(r.src_bus) - 1, int(r.dst_bus) - 1
        ys = 1 / complex(float(r.R), float(r.X))
        tap = float(r.r_src) or 1.0
        y[i, i] += (ys + complex(float(r.G_src), float(r.B_src))) * tap**2
        y[j, j] += ys + complex(float(r.G_dst), float(r.B_dst))
        y[i, j] += -ys * tap
        y[j, i] += -ys * tap
    return y


def main():
    import pandapower as pp
    from pandapower.converter.pypower import from_ppc
    ppc, bus, branch = build_ppc()
    net = from_ppc(ppc, f_hz=60.0, validate_conversion=False)
    pp.runpp(net, calculate_voltage_angles=True, tolerance_mva=1e-9, init="flat")
    if not bool(net.converged):
        raise RuntimeError("same-case pandapower conversion did not converge")
    ypp = net._ppc["internal"]["Ybus"].toarray()
    ypd = pd_ybus(branch)
    pds = pd.read_csv(RESULTS / "pd_static_solution.csv").set_index("bus").loc[range(1, 40)]
    vpd = pds.u_r.to_numpy() + 1j * pds.u_i.to_numpy()
    vpp = net.res_bus.vm_pu.to_numpy() * np.exp(1j * np.deg2rad(net.res_bus.va_degree.to_numpy()))
    spp = vpp * np.conj(ypp @ vpp) * 100.0
    spd = -(pds.P.to_numpy() + 1j * pds.Q.to_numpy()) * 100.0
    out = pd.DataFrame({"bus": range(1, 40), "pd_vm_pu": abs(vpd), "pp_vm_pu": abs(vpp), "pd_va_deg": np.rad2deg(np.angle(vpd)), "pp_va_deg": np.rad2deg(np.angle(vpp)), "pd_p_inj_mw": spd.real, "pp_p_inj_mw": spp.real, "pd_q_inj_mvar": spd.imag, "pp_q_inj_mvar": spp.imag})
    out["pd_va_aligned_deg"] = out.pd_va_deg - out.loc[out.bus == 31, "pd_va_deg"].iloc[0]
    out["pp_va_aligned_deg"] = out.pp_va_deg - out.loc[out.bus == 31, "pp_va_deg"].iloc[0]
    out["abs_vm_error_pu"] = abs(out.pd_vm_pu - out.pp_vm_pu)
    out["abs_va_error_deg"] = abs(out.pd_va_aligned_deg - out.pp_va_aligned_deg)
    out["abs_p_error_mw"] = abs(out.pd_p_inj_mw - out.pp_p_inj_mw)
    out["abs_q_error_mvar"] = abs(out.pd_q_inj_mvar - out.pp_q_inj_mvar)
    out.to_csv(RESULTS / "pd_same_case_bus_parity.csv", index=False)
    # pandapower canonicalizes transformer rows (and may reverse HV/LV
    # orientation). Match the internal terminal operators by endpoint pair.
    yf = net._ppc["internal"]["Yf"].toarray()
    yt = net._ppc["internal"]["Yt"].toarray()
    internal_branch = net._ppc["branch"]
    flow = []
    for k, r in enumerate(branch.itertuples(index=False)):
        i, j = int(r.src_bus) - 1, int(r.dst_bus) - 1
        ys = 1 / complex(float(r.R), float(r.X)); tap = float(r.r_src) or 1.0
        yff = (ys + complex(float(r.G_src), float(r.B_src))) * tap**2; ytt = ys + complex(float(r.G_dst), float(r.B_dst)); yft = -ys * tap
        sf = vpd[i] * np.conj(yff * vpd[i] + yft * vpd[j]) * 100.0
        st = vpd[j] * np.conj(yft * vpd[i] + ytt * vpd[j]) * 100.0
        matches = np.where(((internal_branch[:, 0] == i) & (internal_branch[:, 1] == j)) | ((internal_branch[:, 0] == j) & (internal_branch[:, 1] == i)))[0]
        if len(matches) != 1:
            raise RuntimeError(f"could not uniquely map branch {k+1} ({r.src_bus},{r.dst_bus}) to pandapower internal branch")
        q = int(matches[0])
        if int(internal_branch[q, 0]) == i:
            pp_sf = vpp[i] * np.conj(yf[q] @ vpp) * 100.0
            pp_st = vpp[j] * np.conj(yt[q] @ vpp) * 100.0
        else:
            pp_sf = vpp[i] * np.conj(yt[q] @ vpp) * 100.0
            pp_st = vpp[j] * np.conj(yf[q] @ vpp) * 100.0
        err = np.sqrt((sf.real-pp_sf.real)**2 + (sf.imag-pp_sf.imag)**2 + (st.real-pp_st.real)**2 + (st.imag-pp_st.imag)**2)
        flow.append({"branch": k+1, "from_bus": r.src_bus, "to_bus": r.dst_bus, "pd_p_from_mw": sf.real, "pd_q_from_mvar": sf.imag, "pd_p_to_mw": st.real, "pd_q_to_mvar": st.imag, "pp_p_from_mw": pp_sf.real, "pp_q_from_mvar": pp_sf.imag, "pp_p_to_mw": pp_st.real, "pp_q_to_mvar": pp_st.imag, "abs_error_mva": err})
    flows = pd.DataFrame(flow)
    flows.to_csv(RESULTS / "pd_same_case_terminal_flows.csv", index=False)
    summary = {"pandapower_version": pp.__version__, "converged": True, "max_delta_y": float(abs(ypd-ypp).max()), "relative_frobenius_y": float(np.linalg.norm(ypd-ypp)/np.linalg.norm(ypd)), "max_vm_error_pu": float(out.abs_vm_error_pu.max()), "max_va_error_deg": float(out.abs_va_error_deg.max()), "max_p_error_mw": float(out.abs_p_error_mw.max()), "max_q_error_mvar": float(out.abs_q_error_mvar.max()), "max_terminal_flow_error_mva": float(flows.abs_error_mva.max()), "loss_pd_mw": float(sum(x["pd_p_from_mw"]+x["pd_p_to_mw"] for x in flow)), "loss_pp_mw": float(sum(x["pp_p_from_mw"]+x["pp_p_to_mw"] for x in flow))}
    summary["PD-G0"] = "PASS" if summary["max_delta_y"] < 1e-8 and summary["max_vm_error_pu"] < 1e-8 and summary["max_va_error_deg"] < 1e-6 and summary["max_p_error_mw"] < 1e-5 and summary["max_q_error_mvar"] < 1e-5 and summary["max_terminal_flow_error_mva"] < 1e-5 else "FAIL"
    (RESULTS / "pd_g0_static_parity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORTS / "pd_g0_static_parity.md").write_text("# PowerDynamics → equivalent pandapower same-case parity\n\nThe source is the frozen PowerDynamics CSV set, not `pandapower.networks.case39()`. Angles are aligned to bus 31.\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
