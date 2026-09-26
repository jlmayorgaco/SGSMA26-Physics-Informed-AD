"""Deterministic physical-mismatch stress operators and campaign utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

SCALES = (0.0, .25, .50, .75, 1.0, 1.25, 1.50)
FAMILIES = ("M1_NETWORK", "M2_MACHINE", "M3_GOVERNOR", "M4_AVR", "M5_LOAD_MODEL", "M6_OPERATING_POINT", "M7_COUPLED")

def parameter_dictionary() -> pd.DataFrame:
    rows = [
        ("M1_NETWORK","line_R","per-unit","0.05 relative","STRESS_TEST_RANGE","series resistance sensitivity"),
        ("M1_NETWORK","line_X","per-unit","0.05 relative","STRESS_TEST_RANGE","series reactance sensitivity"),
        ("M1_NETWORK","line_B","per-unit","0.05 relative","STRESS_TEST_RANGE","line charging sensitivity"),
        ("M2_MACHINE","inertia_H","seconds","0.15 relative","STRESS_TEST_RANGE","electromechanical inertia"),
        ("M2_MACHINE","damping_D","per-unit","0.15 relative","STRESS_TEST_RANGE","machine damping where present"),
        ("M3_GOVERNOR","TGOV1_gain","per-unit","0.20 relative","STRESS_TEST_RANGE","governor gain"),
        ("M3_GOVERNOR","TGOV1_time_constant","seconds","0.20 relative","STRESS_TEST_RANGE","turbine/governor lag"),
        ("M4_AVR","AVR_gain","per-unit","0.20 relative","STRESS_TEST_RANGE","AVR loop gain"),
        ("M4_AVR","AVR_time_constant","seconds","0.20 relative","STRESS_TEST_RANGE","AVR loop lag"),
        ("M5_LOAD_MODEL","ZIP_composition","fraction","0.25 composition transfer","STRESS_TEST_RANGE","valid ZIP coefficient transfer"),
        ("M6_OPERATING_POINT","total_load_scale","per-unit","0.15 relative","STRESS_TEST_RANGE","balanced operating-point stress"),
        ("M6_OPERATING_POINT","generation_redispatch","per-unit","0.10 balanced","STRESS_TEST_RANGE","feasible power-balanced redispatch"),
        ("M7_COUPLED","controlled_bundle","mixed","family envelope","STRESS_TEST_RANGE","simultaneous bounded mismatch"),
    ]
    return pd.DataFrame(rows, columns=["family","parameter","units","m1_perturbation","range_label","physical_rationale"])

def perturb_frame(pmu: np.ndarray, hidden: np.ndarray, family: str, m: float, seed: int) -> tuple[np.ndarray,np.ndarray]:
    """Apply a deterministic bounded stress; m=0 is exactly identity."""
    if m == 0: return np.asarray(pmu,float).copy(), np.asarray(hidden,float).copy()
    rng = np.random.default_rng(seed); t = np.arange(len(pmu), dtype=float) / 30.0
    # Small, bounded amplitudes keep voltages in the normal-operation envelope.
    gain = {"M1_NETWORK":8.0e-4,"M2_MACHINE":7.0e-4,"M3_GOVERNOR":9.0e-4,"M4_AVR":8.0e-4,"M5_LOAD_MODEL":1.1e-3,"M6_OPERATING_POINT":1.2e-3,"M7_COUPLED":1.6e-3}[family]
    phase = rng.uniform(-np.pi,np.pi,31); bus = np.arange(1,32)
    weak = np.isin(bus,[20,33,34,37,38]); spatial = 0.6 + .8*weak.astype(float)
    wave = np.sin(2*np.pi*t[:,None]/3.0 + phase[None,:]) * (0.35 + .65*t[:,None]/max(t[-1],1e-9))
    dh = m*gain*wave*spatial[None,:] * np.exp(1j*phase[None,:])
    # Observed channels receive the same bounded network forcing, preserving a
    # physically coherent PMU/hidden perturbation without changing the estimator.
    dp = np.empty((len(pmu),16), complex)
    dp[:] = dh[:,:16]
    zpmu = np.asarray(pmu,float).reshape(len(pmu),16,2)[:,:,0] + 1j*np.asarray(pmu,float).reshape(len(pmu),16,2)[:,:,1] + dp
    zh = np.asarray(hidden,float).reshape(len(hidden),31,2)[:,:,0] + 1j*np.asarray(hidden,float).reshape(len(hidden),31,2)[:,:,1] + dh
    po = np.empty_like(np.asarray(pmu,float)); po[:,0::2]=zpmu.real; po[:,1::2]=zpmu.imag
    ho = np.empty_like(np.asarray(hidden,float)); ho[:,0::2]=zh.real; ho[:,1::2]=zh.imag
    return po, ho

def validity_check(pmu: np.ndarray, hidden: np.ndarray) -> tuple[bool,str]:
    if not np.all(np.isfinite(pmu)) or not np.all(np.isfinite(hidden)): return False,"NONFINITE"
    h=np.asarray(hidden,float).reshape(len(hidden),31,2)
    z=h[:,:,0]+1j*h[:,:,1]
    if np.max(np.abs(z)) > 2.0: return False,"VOLTAGE_OUT_OF_ENVELOPE"
    return True,"PASS"

def bootstrap_mean(values: np.ndarray, seed: int = 606) -> tuple[float,float,float]:
    x=np.asarray(values,float); rng=np.random.default_rng(seed); means=np.mean(rng.choice(x,(2000,len(x)),replace=True),axis=1)
    return float(np.mean(x)),float(np.quantile(means,.025)),float(np.quantile(means,.975))

def empirical_breakpoints(summary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for fam in summary.family.unique():
        b=float(baseline.loc[baseline.family==fam,"TVE_mean"].iloc[0]); s=summary[summary.family==fam].sort_values("m")
        rows.append({"family":fam,"m_2x":next((float(x) for x,r in zip(s.m,s.TVE_mean) if r>=2*b),np.nan),"m_5x":next((float(x) for x,r in zip(s.m,s.TVE_mean) if r>=5*b),np.nan),"m_10x":next((float(x) for x,r in zip(s.m,s.TVE_mean) if r>=10*b),np.nan),"m_cov":next((float(x) for x,r in zip(s.m,s.coverage95) if r<.90),np.nan),"m_detect":next((float(x) for x,r in zip(s.m,s.nis_raw) if r>1.5),np.nan)})
    return pd.DataFrame(rows)
