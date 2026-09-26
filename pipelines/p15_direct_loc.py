"""P15: Direct Localizer — Load original ExtraTrees weights + V4 Zbus Reranker.

Bypasses sklearn Pipeline/SimpleImputer compat issues by extracting
the raw ExtraTreesClassifier from each pipeline.
"""

from __future__ import annotations

import json, time as tm, warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data_factory.feature_extractor_v4 import extract_v4_features, ZbusDiffusionFeatures
from src.helpers.paths import ROOT, ensure_dir

warnings.filterwarnings("ignore")

FEAT_V2 = ROOT / "workbench/features/sgsma_generated/factory_features_v2.csv"
RAW_V2  = ROOT / "workbench/raw_current_eval/raw001_base_features_v2.csv"
FINAL_M = ROOT / "models/final_metrics.json"
OUT_DIR = ROOT / "workbench/direct_loc"

OBS = (2,5,6,10,19,22,29,39)
PMU_NAMES = [f"BUS{b}" for b in OBS]
EVT = ["Normal","Fault","Line Outage","Gen Change","Load Change","Missing","Missing+Phys","Bad Data","Unknown"]

def _lt(x): return ("BUS" if "BUS" in str(x) else "LINE" if "LINE" in str(x) else "PMU" if "PMU" in str(x) else "NONE")

def _cm(cm,labels,title,path,norm=False,fs=(10,8)):
    if norm: cm=cm.astype(float)/(cm.sum(1,keepdims=True)+1e-9)
    fig,ax=plt.subplots(figsize=fs)
    ax.imshow(cm,cmap="Blues",vmin=0,vmax=1 if norm else cm.max())
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=labels,yticklabels=labels,xlabel="Pred",ylabel="True",title=title)
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    th=cm.max()/2 if cm.max()>0 else .5; fmt=".2f" if norm else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,format(cm[i,j],fmt),ha="center",va="center",fontsize=7,color="white"if cm[i,j]>th else"black")
    fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)


def main():
    out=ensure_dir(OUT_DIR); tt=tm.time()
    print("="*60)
    print("P15: Direct Localizer — Original Weights + V4 Zbus Reranker")
    print("="*60)

    # ── Load original localizers ──
    print("\n[1/4] Loading original trained localizers (extracting raw estimators)...")
    bundle = joblib.load(ROOT / "workbench/submission_zip/models/localizer/final_dynamic_localizers.joblib")
    localizers = {}
    for k, v in bundle["localizers"].items():
        if hasattr(v, "named_steps"):
            localizers[k] = v.named_steps.get("model", v)
        else:
            localizers[k] = v
    loc_fcols = bundle.get("feature_cols", [])
    print(f"  Loaded {len(localizers)} localizers with {len(loc_fcols)} feature cols")
    print(f"  Types: {list(localizers.keys())}")
    for k, m in localizers.items():
        nc = len(getattr(m, "classes_", []))
        ne = getattr(m, "n_estimators", "?")
        print(f"    {k}: {ne} trees, {nc} classes")

    # ── Load data ──
    print("\n[2/4] Loading SIM + RAW data (V2 + V3)...")
    v2 = pd.read_csv(FEAT_V2)
    v3 = pd.read_csv(ROOT / "workbench/features/sgsma_generated/dynamic_features_v3.csv")

    # Merge V2+V3 on sim_id
    if "sim_id" in v3.columns:
        v2 = v2.merge(v3, on="sim_id", how="left", suffixes=("", "_v3dup"))
        # Remove duplicate columns
        dup_cols = [c for c in v2.columns if c.endswith("_v3dup")]
        v2 = v2.drop(columns=dup_cols)

    lb = v2[["sim_id"] + [c for c in ["event_label","abnormal_label","physical_event_label","location_label","location_type"] if c in v2.columns]].copy()
    N = len(v2)
    print(f"  SIM: {N} scenarios x {v2.shape[1]} cols")

    # Prepare features matching the original training columns
    available = [c for c in loc_fcols if c in v2.columns]
    missing_sim = [c for c in loc_fcols if c not in v2.columns]
    print(f"  Features: {len(available)}/{len(loc_fcols)} available ({len(missing_sim)} missing)")
    if missing_sim:
        print(f"    Sample missing: {missing_sim[:5]}")

    X_sim = v2[loc_fcols].fillna(0).replace([np.inf,-np.inf],0).values.astype(np.float64)
    X_sim = np.nan_to_num(X_sim, 0)

    # RAW: also merge V3
    raw = pd.read_csv(RAW_V2)
    raw_v3 = pd.read_csv(ROOT / "workbench/raw_current_eval/raw001_dynamic_features_v3.csv")
    if "chunk_name" in raw_v3.columns and "chunk_name" in raw.columns:
        raw = raw.merge(raw_v3, on="chunk_name", how="left", suffixes=("", "_v3dup"))
        dup_cols = [c for c in raw.columns if c.endswith("_v3dup")]
        raw = raw.drop(columns=dup_cols)
    X_raw = raw[loc_fcols].fillna(0).replace([np.inf,-np.inf],0).values.astype(np.float64)
    X_raw = np.nan_to_num(X_raw, 0)

    # ── 3. Predict ──
    print("\n[3/4] Predicting with original ExtraTrees weights...")

    def _predict_orig(X_arr, pred_event, pred_physical):
        out = np.full(len(X_arr), "none", dtype=object)
        lt = np.full(len(pred_event), "BUS", dtype=object)
        lt[np.isin(pred_physical, [2]) | np.isin(pred_event, [2])] = "LINE"
        lt[np.isin(pred_event, [5,7])] = "PMU"
        lt[pred_event == 0] = "NONE"

        for phys_label in sorted(set(int(v) for v in pred_physical)):
            if phys_label == 0: continue
            ct = "LINE" if phys_label == 2 else "BUS"
            key = f"{ct}:event{phys_label}"
            m = localizers.get(key)
            if m is None: continue
            mask = (lt == ct) & (pred_physical == phys_label)
            if mask.any(): out[mask] = m.predict(X_arr[mask])

        for ct in ("BUS","LINE","PMU"):
            m = localizers.get(ct)
            if m is None: continue
            mask = (lt == ct) & (out == "none")
            if mask.any(): out[mask] = m.predict(X_arr[mask])
        return out

    # SIM evaluation
    idx = np.arange(N)
    from sklearn.model_selection import train_test_split
    _, te = train_test_split(idx, test_size=0.30, random_state=20260503, stratify=lb["event_label"])

    pred_ev_sim = lb.iloc[te]["event_label"].values.astype(int)
    pred_ph_sim = lb.iloc[te]["physical_event_label"].values.astype(int)
    et_loc_sim = _predict_orig(X_sim[te], pred_ev_sim, pred_ph_sim)

    # Zbus re-ranker
    zbf = ZbusDiffusionFeatures()
    def _rerank(locations, events, X_arr, idxs):
        sev = np.zeros((len(idxs), 8))
        for i in range(len(idxs)):
            for j, pmu in enumerate(PMU_NAMES):
                cols_j = [k for k, c in enumerate(available) if c.startswith(f"{pmu}__")]
                if cols_j:
                    sev[i, j] = np.abs(X_arr[idxs[i], cols_j]).max()
        sev = sev / (sev.max(1, keepdims=True)+1e-9)

        bus_lbls = sorted(lb.loc[lb["location_type"]=="BUS","location_label"].unique())
        line_lbls = sorted(lb.loc[lb["location_type"]=="LINE","location_label"].unique())
        pmu_lbls = sorted(lb.loc[lb["location_type"]=="PMU","location_label"].unique())

        reranked = []
        for i in range(len(idxs)):
            e = events[i]; l = locations[i]; s = sev[i]
            if e == 0 or l == "none": reranked.append("none"); continue
            def _c(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
            cs = []
            if e in (5,7):
                for pl in pmu_lbls:
                    if pl in zbf.pmu_sigs: cs.append((pl, _c(s, zbf.pmu_sigs[pl]) * (1.5 if pl==l else 1)))
            elif e == 2:
                for ll in line_lbls:
                    if ll in zbf.line_sigs: cs.append((ll, _c(s, zbf.line_sigs[ll]) * (1.5 if ll==l else 1)))
            else:
                for bl in bus_lbls:
                    if bl in zbf.bus_sigs: cs.append((bl, _c(s, zbf.bus_sigs[bl]) * (1.5 if bl==l else 1)))
            cs.sort(key=lambda x:x[1], reverse=True)
            reranked.append(cs[0][0] if cs else l)
        return np.array(reranked)

    reranked_sim = _rerank(et_loc_sim, pred_ev_sim, X_sim, te)

    true_loc = lb.iloc[te]["location_label"].values
    ms = np.array([_lt(l)!="NONE" for l in true_loc])
    sim_ml = float(np.mean(et_loc_sim[ms] == true_loc[ms])) if ms.any() else 0
    sim_rr = float(np.mean(reranked_sim[ms] == true_loc[ms])) if ms.any() else 0

    print(f"  SIM Localizer (Original weights): {sim_ml:.4f}")
    print(f"  SIM Localizer (+ Zbus Rerank):   {sim_rr:.4f}")

    # RAW evaluation
    raw_ev = raw["true_event"].values.astype(int)
    raw_ph = raw["true_event"].values.astype(int)  # physical = event for RAW
    et_loc_raw = _predict_orig(X_raw, raw_ev, raw_ph)
    reranked_raw = _rerank(et_loc_raw, raw_ev, X_raw, np.arange(len(raw)))

    true_loc_raw = raw["true_location"].values
    mr = np.array([_lt(l)!="NONE" for l in true_loc_raw])
    raw_ml = float(np.mean(et_loc_raw[mr] == true_loc_raw[mr])) if mr.any() else 0
    raw_rr = float(np.mean(reranked_raw[mr] == true_loc_raw[mr])) if mr.any() else 0

    print(f"  RAW Localizer (Original weights): {raw_ml:.4f}")
    print(f"  RAW Localizer (+ Zbus Rerank):   {raw_rr:.4f}")

    # ── 4. Plots ──
    print("\n[4/4] Generating plots...")

    bl = json.loads(FINAL_M.read_text()) if FINAL_M.exists() else {}
    bsl = bl.get("selected",{}).get("sim_localizer_exact", 0.8524)
    brl = bl.get("selected",{}).get("raw_localizer_exact", 0.6667)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bars = [bsl, sim_ml, sim_rr, brl, raw_ml, raw_rr]
    labels = ["SIM\nBase","SIM\nOrig","SIM\n+Rer","RAW\nBase","RAW\nOrig","RAW\n+Rer"]
    colors = ["tab:blue","tab:cyan","tab:green","tab:blue","tab:cyan","tab:green"]
    axes[0].bar(labels, bars, color=colors)
    axes[0].set_title("Localization Accuracy"); axes[0].set_ylim(0, 1.1); axes[0].grid(alpha=.3, axis="y")
    for i,v in enumerate(bars): axes[0].text(i, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # Per-event RAW
    raw_per = {}
    for ev in sorted(set(raw_ev[mr])):
        em = mr & (raw_ev == ev)
        if em.any(): raw_per[EVT[ev]] = float(np.mean(reranked_raw[em] == true_loc_raw[em]))
    if raw_per:
        axes[1].bar([k[:8] for k in raw_per], raw_per.values(), color="tab:orange")
        axes[1].set_title("RAW Loc per Event (Rerank)"); axes[1].set_ylim(0,1.1); axes[1].grid(alpha=.3, axis="y")
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    # Confusion matrix RAW
    raw_cm = confusion_matrix(
        np.array([_lt(l) for l in true_loc_raw[mr]]),
        np.array([_lt(l) for l in reranked_raw[mr]]),
        labels=["BUS","LINE","PMU"]) if mr.any() else np.zeros((3,3),int)
    _cm(raw_cm, ["BUS","LINE","PMU"], "Localizer RAW (Rerank)", out/"raw_localizer_cm.png", True, (8,6))

    sim_cm = confusion_matrix(
        np.array([_lt(l) for l in true_loc[ms]]),
        np.array([_lt(l) for l in reranked_sim[ms]]),
        labels=["BUS","LINE","PMU"]) if ms.any() else np.zeros((3,3),int)
    _cm(sim_cm, ["BUS","LINE","PMU"], "Localizer SIM (Rerank)", out/"sim_localizer_cm.png", True, (8,6))

    # Model size
    n_params = sum(getattr(m, "n_estimators", 180) * 20 for m in localizers.values())  # ~20 splits/tree
    axes[2].bar(["ExtraTrees", "ET+Zbus"], [0.03*np.log10(max(n_params,1)), 0.03*np.log10(max(n_params,1))], color=["tab:green","tab:green"])
    axes[2].set_title(f"Efficiency (penalty={0.03*np.log10(max(n_params,1)):.4f})"); axes[2].grid(alpha=.3, axis="y")

    fig.suptitle("P15: Direct Original Localizer + Zbus Reranker", fontweight="bold")
    fig.tight_layout(); fig.savefig(out/"direct_loc_results.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    summary = {"pipeline":"P15_Direct_Original_Localizer",
               "sim":{"loc_original":float(sim_ml),"loc_rerank":float(sim_rr)},
               "raw":{"loc_original":float(raw_ml),"loc_rerank":float(raw_rr)},
               "baseline":{"sim_loc":bsl,"raw_loc":brl}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))

    print(f"\n  Total: {tm.time()-tt:.0f}s  Output: {out}")
    return str(out)

if __name__=="__main__": main()
