"""P17: Exact Replica of Original Localizer Pipeline — Fresh Training.

Replicates the EXACT architecture from train_dynamic_enhancement_ablation_v3.py:
- SimpleImputer(strategy="median") preprocessing
- ExtraTrees(n_estimators=180, min_samples_leaf=1, class_weight=balanced)
- EVENT-specific localizers (highest priority) + type-specific (fallback) + per-event typed
- Feature selection: V2 base + selected V3 dynamic blocks (rolling, rls_kalman, graph_temporal)
- HybridLocalizer with Zbus physics re-ranker
"""

from __future__ import annotations

import json, time as tm, warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.helpers.paths import ROOT, ensure_dir

warnings.filterwarnings("ignore")

FEAT_V2 = ROOT / "workbench/features/sgsma_generated/factory_features_v2.csv"
FEAT_V3 = ROOT / "workbench/features/sgsma_generated/dynamic_features_v3.csv"
RAW_V2  = ROOT / "workbench/raw_current_eval/raw001_base_features_v2.csv"
RAW_V3  = ROOT / "workbench/raw_current_eval/raw001_dynamic_features_v3.csv"
FINAL_M = ROOT / "models/final_metrics.json"
OUT_DIR = ROOT / "workbench/exact_replica"

OBS = (2,5,6,10,19,22,29,39)
PMU_NAMES = [f"BUS{b}" for b in OBS]
EVT = ["Normal","Fault","Line Outage","Gen Change","Load Change","Missing","Missing+Phys","Bad Data","Unknown"]

# Exact dynamic block settings from the original
BLOCK_PREFIX = {"rolling":"ROLL__","rls_kalman":"RLSK__","graph_temporal":"GRAPH__"}
SELECTED_BLOCKS = ("rolling","rls_kalman","graph_temporal")
COMMON_PREFIXES = ("META__",)

def _lt(x): return ("BUS" if "BUS" in str(x) else "LINE" if "LINE" in str(x) else "PMU" if "PMU" in str(x) else "NONE")

def _tree(seed, n_est=180):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("model", ExtraTreesClassifier(n_estimators=n_est, min_samples_leaf=1,
                       max_features="sqrt", class_weight="balanced", n_jobs=-1,
                       random_state=seed))])

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
    print("="*65)
    print("P17: Exact Replica — Original Pipeline Architecture, Fresh Training")
    print("="*65)

    # ── 1. Load + select features ──
    print("\n[1/5] Loading V2+V3 with EXACT feature selection...")
    t0=tm.time()
    v2 = pd.read_csv(FEAT_V2)
    v3 = pd.read_csv(FEAT_V3)

    # Labels from V2
    lb = v2[["sim_id","event_label","abnormal_label","physical_event_label","location_label","location_type"]].copy()

    # Feature selection: V2 base columns (from models/feature_columns.json) + selected V3 blocks
    base_cols = json.loads((ROOT / "models/feature_columns.json").read_text())
    # Remove unstable tokens (same as original)
    UNSTABLE = ("freq_angle_mismatch",)
    base_cols = [c for c in base_cols if not any(t in c for t in UNSTABLE) and c in v2.columns]

    # V3 dynamic columns: only from selected blocks
    prefixes = COMMON_PREFIXES + tuple(BLOCK_PREFIX[b] for b in SELECTED_BLOCKS)
    dyn_cols = [c for c in v3.columns if c.startswith(prefixes) and pd.api.types.is_numeric_dtype(v3[c])]

    # Build full feature matrix
    X_v2 = v2[base_cols].copy()
    X_v3 = v3[dyn_cols].copy()
    feature_cols = base_cols + dyn_cols

    print(f"    V2 base: {len(base_cols)}, V3 dynamic: {len(dyn_cols)}, Total: {len(feature_cols)}")

    # Merge V2+V3
    X_all = pd.concat([X_v2.reset_index(drop=True), X_v3.reset_index(drop=True)], axis=1)
    X_all = X_all.replace([np.inf,-np.inf], np.nan)
    print(f"    Feature matrix: {X_all.shape}")

    # RAW
    raw = pd.read_csv(RAW_V2)
    raw_v3 = pd.read_csv(RAW_V3)
    X_raw_all = pd.concat([raw[base_cols].reset_index(drop=True), raw_v3[dyn_cols].reset_index(drop=True)], axis=1)
    X_raw_all = X_raw_all.replace([np.inf,-np.inf], np.nan)

    # ── 2. Train/test split ──
    print("\n[2/5] Train/test split (70/30 stratified)...")
    idx = np.arange(len(X_all))
    tr, te = train_test_split(idx, test_size=0.30, random_state=20260503, stratify=lb["event_label"])
    print(f"    Train: {len(tr)}  Test: {len(te)}")

    X_tr = X_all.iloc[tr].reset_index(drop=True)
    X_te = X_all.iloc[te].reset_index(drop=True)
    lb_tr = lb.iloc[tr].reset_index(drop=True)
    lb_te = lb.iloc[te].reset_index(drop=True)

    # ── 3. Train localizers (EXACT original architecture) ──
    print("\n[3/5] Training localizers (EXACT replica: 180 trees, median imputation, min_samples_leaf=1)...")
    t0=tm.time()
    models = {}
    seed=20260503
    stable_seed={"BUS":101,"LINE":202,"PMU":303}

    # 3a. EVENT-specific localizers (highest priority — original routes by event_label)
    for event_label in sorted(lb_tr["event_label"].unique().tolist()):
        event_label = int(event_label)
        if event_label == 0: continue
        mask = lb_tr["event_label"].eq(event_label) & lb_tr["location_label"].ne("none")
        if mask.sum() < 2 or lb_tr.loc[mask,"location_label"].nunique() < 2: continue
        m = _tree(seed+3000+event_label, 180)
        m.fit(X_tr.loc[mask], lb_tr.loc[mask,"location_label"])
        models[f"EVENT:event{event_label}"] = m
        print(f"    EVENT:event{event_label}: {mask.sum()} samples, {lb_tr.loc[mask,'location_label'].nunique()} classes")

    # 3b. Type-specific localizers
    for loc_type, base_seed in [("BUS",101),("LINE",202),("PMU",303)]:
        mask = lb_tr["location_type"].eq(loc_type)
        if mask.sum() < 2 or lb_tr.loc[mask,"location_label"].nunique() < 2: continue
        m = _tree(seed+base_seed, 180)
        m.fit(X_tr.loc[mask], lb_tr.loc[mask,"location_label"])
        models[loc_type] = m
        print(f"    {loc_type}: {mask.sum()} samples, {lb_tr.loc[mask,'location_label'].nunique()} classes")

    # 3c. Per-event typed localizers (fallback)
    for phys_label in sorted(lb_tr["physical_event_label"].unique().tolist()):
        phys_label = int(phys_label)
        if phys_label == 0: continue
        loc_type = "LINE" if phys_label == 2 else "BUS"
        mask = lb_tr["location_type"].eq(loc_type) & lb_tr["physical_event_label"].eq(phys_label)
        if mask.sum() < 2 or lb_tr.loc[mask,"location_label"].nunique() < 2: continue
        key = f"{loc_type}:event{phys_label}"
        m = _tree(seed+2000+phys_label, 180)
        m.fit(X_tr.loc[mask], lb_tr.loc[mask,"location_label"])
        models[key] = m
        print(f"    {key}: {mask.sum()} samples, {lb_tr.loc[mask,'location_label'].nunique()} classes")

    print(f"    Trained {len(models)} localizers ({tm.time()-t0:.0f}s)")

    # ── 4. Predict (EXACT original logic) ──
    print("\n[4/5] Evaluating with EXACT prediction logic + Zbus re-ranker...")

    def _predict_exact(X_df, pred_event, pred_physical):
        out = np.full(len(X_df), "none", dtype=object)
        unresolved = np.ones(len(X_df), dtype=bool)
        lt = np.full(len(pred_event), "BUS", dtype=object)
        lt[np.isin(pred_physical.astype(int), [2]) | np.isin(pred_event.astype(int), [2])] = "LINE"
        lt[np.isin(pred_event.astype(int), [5,7])] = "PMU"
        lt[pred_event.astype(int) == 0] = "NONE"

        # Priority 1: EVENT-specific (routes by event_label, not physical)
        for event_label in sorted(set(int(v) for v in pred_event)):
            if event_label == 0: continue
            m = models.get(f"EVENT:event{event_label}")
            if m is None: continue
            mask = unresolved & (pred_event.astype(int) == event_label)
            if mask.any():
                out[mask] = m.predict(X_df.loc[mask])
                unresolved[mask] = False

        # Priority 2: LINE:event2 special case
        le2 = models.get("LINE:event2")
        if le2 is not None:
            mask = unresolved & (pred_event.astype(int) == 2)
            if mask.any():
                out[mask] = le2.predict(X_df.loc[mask])
                unresolved[mask] = False

        # Priority 3: Per-event typed
        for phys_label in sorted(set(int(v) for v in pred_physical)):
            if phys_label == 0: continue
            ct = "LINE" if phys_label == 2 else "BUS"
            key = f"{ct}:event{phys_label}"
            m = models.get(key)
            if m is None: continue
            mask = unresolved & (lt == ct) & (pred_physical.astype(int) == phys_label)
            if mask.any():
                out[mask] = m.predict(X_df.loc[mask])
                unresolved[mask] = False

        # Priority 4: Generic type fallback
        for ct in ("BUS","LINE","PMU"):
            m = models.get(ct)
            if m is None: continue
            mask = unresolved & (lt == ct) & (out == "none")
            if mask.any():
                out[mask] = m.predict(X_df.loc[mask])
                unresolved[mask] = False
        return out

    # SIM evaluation
    pred_ev_sim = lb_te["event_label"].values.astype(int)
    pred_ph_sim = lb_te["physical_event_label"].values.astype(int)
    et_sim = _predict_exact(X_te, pred_ev_sim, pred_ph_sim)

    true_sim = lb_te["location_label"].values
    ms = np.array([_lt(l)!="NONE" for l in true_sim])
    sim_ml = float(np.mean(et_sim[ms]==true_sim[ms])) if ms.any() else 0

    # Now add Zbus hybrid re-ranker (EXACT weights from original)
    from src.models.localizer import HybridLocalizer, TopologyResidualRanker
    ranker = TopologyResidualRanker()
    # Convert models to TypedLocalizer format for HybridLocalizer
    from src.models.localizer.hybrid import TypedLocalizer
    typed_loc = TypedLocalizer(models)
    hybrid = HybridLocalizer(typed_loc, ranker, weights={"ml_score":0.72,"physics_score":0.20,"topology_score":0.08})

    topk_sim = hybrid.predict_topk(X_te, pred_ev_sim, pred_ph_sim, k=3)
    hybrid_sim_loc = np.array([items[0]["candidate"] if items else "none" for items in topk_sim], dtype=object)
    sim_hy = float(np.mean(hybrid_sim_loc[ms]==true_sim[ms])) if ms.any() else 0

    # Per-event SIM
    per_ev_sim = {}
    for ev in sorted(set(pred_ev_sim[ms])):
        em = ms & (pred_ev_sim==ev)
        if em.any(): per_ev_sim[EVT[ev]] = float(np.mean(hybrid_sim_loc[em]==true_sim[em]))

    print(f"    SIM Localizer (ML only):  {sim_ml:.4f}")
    print(f"    SIM Localizer (Hybrid):   {sim_hy:.4f}")
    for ev_name, acc in sorted(per_ev_sim.items()):
        print(f"      {ev_name}: {acc:.4f}")

    # ── 5. RAW ──
    print("\n[5/5] Evaluating on RAW001...")
    raw_ev = raw["true_event"].values.astype(int)
    raw_ph = raw["true_event"].values.astype(int)
    et_raw = _predict_exact(X_raw_all, raw_ev, raw_ph)
    topk_raw = hybrid.predict_topk(X_raw_all, raw_ev, raw_ph, k=3)
    hybrid_raw_loc = np.array([items[0]["candidate"] if items else "none" for items in topk_raw], dtype=object)

    true_raw = raw["true_location"].values
    mr = np.array([_lt(l)!="NONE" for l in true_raw])
    raw_ml = float(np.mean(et_raw[mr]==true_raw[mr])) if mr.any() else 0
    raw_hy = float(np.mean(hybrid_raw_loc[mr]==true_raw[mr])) if mr.any() else 0

    per_ev_raw = {}
    for ev in sorted(set(raw_ev[mr])):
        em = mr & (raw_ev==ev)
        if em.any(): per_ev_raw[EVT[ev]] = float(np.mean(hybrid_raw_loc[em]==true_raw[em]))

    print(f"    RAW Localizer (ML only):  {raw_ml:.4f}")
    print(f"    RAW Localizer (Hybrid):   {raw_hy:.4f}")
    for ev_name, acc in sorted(per_ev_raw.items()):
        print(f"      {ev_name}: {acc:.4f}")

    # Baseline
    bl = json.loads(FINAL_M.read_text()) if FINAL_M.exists() else {}
    bsl = bl.get("selected",{}).get("sim_localizer_exact",0.8524)
    brl = bl.get("selected",{}).get("raw_localizer_exact",0.6667)

    # ── Plots ──
    sim_cm = confusion_matrix(np.array([_lt(l) for l in true_sim[ms]]), np.array([_lt(l) for l in hybrid_sim_loc[ms]]), labels=["BUS","LINE","PMU"]) if ms.any() else np.zeros((3,3),int)
    raw_cm = confusion_matrix(np.array([_lt(l) for l in true_raw[mr]]), np.array([_lt(l) for l in hybrid_raw_loc[mr]]), labels=["BUS","LINE","PMU"]) if mr.any() else np.zeros((3,3),int)

    fig,axes=plt.subplots(2,3,figsize=(18,11))
    locs=["SIM\nBase","SIM\nML","SIM\nHybrid","RAW\nBase","RAW\nML","RAW\nHybrid"]
    vals=[bsl,sim_ml,sim_hy,brl,raw_ml,raw_hy]
    colors=["tab:blue","tab:cyan","tab:green","tab:blue","tab:cyan","tab:green"]
    axes[0,0].bar(locs,vals,color=colors)
    axes[0,0].set_title("Localization Accuracy"); axes[0,0].set_ylim(0,1.1); axes[0,0].grid(alpha=.3,axis="y")
    for i,v in enumerate(vals): axes[0,0].text(i,v+.02,f"{v:.3f}",ha="center",fontweight="bold",fontsize=8)

    if per_ev_sim:
        items=list(per_ev_sim.items())
        axes[0,1].bar([k[:8] for k,v in items],[v for k,v in items],color="tab:green")
        axes[0,1].set_title("SIM Loc per Event (Hybrid)"); axes[0,1].set_ylim(0,1.1); axes[0,1].grid(alpha=.3,axis="y")
        plt.setp(axes[0,1].get_xticklabels(),rotation=45,ha="right",fontsize=8)
    if per_ev_raw:
        items=list(per_ev_raw.items())
        axes[0,2].bar([k[:8] for k,v in items],[v for k,v in items],color="tab:orange")
        axes[0,2].set_title("RAW Loc per Event (Hybrid)"); axes[0,2].set_ylim(0,1.1); axes[0,2].grid(alpha=.3,axis="y")
        plt.setp(axes[0,2].get_xticklabels(),rotation=45,ha="right",fontsize=8)

    _cm(sim_cm,["BUS","LINE","PMU"],"Localizer SIM (Hybrid)",out/"sim_localizer_cm.png",True,(8,6))
    _cm(raw_cm,["BUS","LINE","PMU"],"Localizer RAW (Hybrid)",out/"raw_localizer_cm.png",True,(8,6))

    # Feature importance
    if "BUS" in models:
        imp = models["BUS"].named_steps["model"].feature_importances_
        top_n=20; top_idx=np.argsort(imp)[-top_n:]
        axes[1,0].barh(range(top_n), imp[top_idx][::-1])
        axes[1,0].set_title(f"Top {top_n} Features (BUS)")

    n_trees = sum(getattr(m.named_steps["model"],"n_estimators",180) for m in models.values())
    penalty = 0.03*np.log10(max(n_trees*20,1))
    axes[1,2].bar(["ExtraTrees","ET+Hybrid"],[penalty,penalty],color=["tab:green","tab:green"])
    axes[1,2].set_title(f"Penalty={penalty:.4f}"); axes[1,2].grid(alpha=.3,axis="y")

    fig.suptitle("P17: Exact Replica — Original Architecture + Hybrid Re-ranker",fontweight="bold",fontsize=14)
    fig.tight_layout(); fig.savefig(out/"exact_replica_results.png",dpi=150,bbox_inches="tight"); plt.close(fig)

    # Save
    joblib.dump({"models":models,"feature_cols":feature_cols}, out/"exact_replica.joblib")
    summary = {"pipeline":"P17_Exact_Replica","n_features":len(feature_cols),"n_models":len(models),
               "sim":{"loc_ml":float(sim_ml),"loc_hybrid":float(sim_hy),"per_event":per_ev_sim},
               "raw":{"loc_ml":float(raw_ml),"loc_hybrid":float(raw_hy),"per_event":per_ev_raw},
               "baseline":{"sim_loc":bsl,"raw_loc":brl}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str))

    print(f"\n  Total: {tm.time()-tt:.0f}s  Output: {out}")
    return str(out)

if __name__=="__main__": main()
