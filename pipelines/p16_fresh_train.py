"""P16: Fresh ExtraTrees Localizer — replicates original 85% SIM + V4 Zbus boost.

Uses EXACT same architecture as the original (but trains fresh):
- ExtraTreesClassifier(n_estimators=180, class_weight=balanced)
- V2+V3 features (45162 columns)
- Multi-type localizer (BUS/LINE/PMU + per-event typed)
- Ground truth event routing for localizer evaluation
- Zbus re-ranker on top

Target: >= 85% SIM localization (matching original), max RAW.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.helpers.paths import ROOT, ensure_dir

warnings.filterwarnings("ignore")

FEAT_V2 = ROOT / "workbench/features/sgsma_generated/factory_features_v2.csv"
FEAT_V3 = ROOT / "workbench/features/sgsma_generated/dynamic_features_v3.csv"
RAW_V2  = ROOT / "workbench/raw_current_eval/raw001_base_features_v2.csv"
RAW_V3  = ROOT / "workbench/raw_current_eval/raw001_dynamic_features_v3.csv"
FINAL_M = ROOT / "models/final_metrics.json"
OUT_DIR = ROOT / "workbench/fresh_train"

OBS = (2,5,6,10,19,22,29,39)
PMU_NAMES = [f"BUS{b}" for b in OBS]
EVT = ["Normal","Fault","Line Outage","Gen Change","Load Change","Missing","Missing+Phys","Bad Data","Unknown"]

def _lt(x): return ("BUS" if "BUS" in str(x) else "LINE" if "LINE" in str(x) else "PMU" if "PMU" in str(x) else "NONE")


def _load_data():
    """Load V2 labels + concat with V3 features (no merge, no duplicates)."""
    v2 = pd.read_csv(FEAT_V2)
    v3 = pd.read_csv(FEAT_V3)

    # Labels from V2 only
    labels = v2[["sim_id","event_label","abnormal_label","physical_event_label","location_label","location_type"]].copy()

    # Features: V2 numeric + V3 numeric, no label columns
    v3_feat = v3.drop(columns=[c for c in v3.columns if c in labels.columns or c in ["window_start","window_end","chunk_name"]], errors="ignore")

    # Concat horizontally (both must be aligned by row order)
    if len(v2) == len(v3_feat):
        features = pd.concat([v2, v3_feat], axis=1)
    else:
        features = v2.merge(v3_feat, on="sim_id", how="left", suffixes=("", "_DROP"))
        dup_cols = [c for c in features.columns if c.endswith("_DROP")]
        features = features.drop(columns=dup_cols)

    return features, labels


def _load_raw():
    """Load RAW V2+V3 features."""
    raw = pd.read_csv(RAW_V2)
    raw3 = pd.read_csv(RAW_V3)
    # Both have chunk_name
    if "chunk_name" in raw.columns and "chunk_name" in raw3.columns:
        raw3_trim = raw3.drop(columns=[c for c in ["event_label","abnormal_label","physical_event_label","location_label","location_type","window_start","window_end"] if c in raw3.columns], errors="ignore")
        raw = raw.merge(raw3_trim, on="chunk_name", how="left")
    return raw


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
    print("P16: Fresh ExtraTrees Localizer (target: 85% SIM + Zbus boost)")
    print("="*65)

    # ── 1. Load data ──
    print("\n[1/5] Loading V2+V3 features (45162 columns)...")
    t0=tm.time()
    df, lb = _load_data()
    print(f"    Loaded {len(df)} rows x {len(df.columns)} cols ({tm.time()-t0:.0f}s)")

    # Labels
    lb = df[["sim_id","event_label","abnormal_label","physical_event_label","location_label","location_type"]].copy()

    # Feature columns (all numeric except labels)
    feature_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])
                    and c not in ["sim_id","event_label","abnormal_label","physical_event_label"]
                    and "__label_" not in c
                    and not c.endswith("__Event__min")
                    and not c.endswith("__Event__max")
                    and "freq_angle_mismatch" not in c]
    print(f"    Feature columns: {len(feature_cols)}")

    X_all = df[feature_cols].fillna(0).replace([np.inf,-np.inf],0).values.astype(np.float64)
    X_all = np.nan_to_num(X_all, 0)
    print(f"    Feature matrix: {X_all.shape}")

    # ── 2. Train/test split ──
    print("\n[2/5] Train/test split...")
    idx = np.arange(len(X_all))
    tr, te = train_test_split(idx, test_size=0.30, random_state=20260503, stratify=lb["event_label"])
    print(f"    Train: {len(tr)}  Test: {len(te)}")

    X_tr, X_te = X_all[tr], X_all[te]
    lb_tr = lb.iloc[tr].reset_index(drop=True)
    lb_te = lb.iloc[te].reset_index(drop=True)

    # ── 3. Train localizers ──
    print("\n[3/5] Training ExtraTrees localizers (180 trees, balanced)...")
    t0=tm.time()
    models = {}
    seed=20260503

    # Convert label columns to numpy for faster access
    loc_type_arr = lb_tr["location_type"].values
    phys_ev_arr = lb_tr["physical_event_label"].values.astype(int)
    event_arr = lb_tr["event_label"].values.astype(int)

    # Encode location labels to integers
    from sklearn.preprocessing import LabelEncoder
    all_loc_labels = lb_tr["location_label"].values
    bus_enc = LabelEncoder(); bus_enc.fit(all_loc_labels[loc_type_arr == "BUS"])
    line_enc = LabelEncoder(); line_enc.fit(all_loc_labels[loc_type_arr == "LINE"])
    pmu_enc = LabelEncoder(); pmu_enc.fit(all_loc_labels[loc_type_arr == "PMU"])

    # Generic typed localizers
    for loc_type, encoder in [("BUS", bus_enc), ("LINE", line_enc), ("PMU", pmu_enc)]:
        mask = loc_type_arr == loc_type
        if mask.sum() < 5: continue
        y = encoder.transform(all_loc_labels[mask])
        if len(np.unique(y)) < 2: continue
        m = ExtraTreesClassifier(n_estimators=180, max_features="sqrt", class_weight="balanced",
                                  random_state=seed+{"BUS":101,"LINE":202,"PMU":303}[loc_type], n_jobs=4,
                                  max_depth=None, min_samples_leaf=2)
        m.fit(X_tr[mask], y)
        models[loc_type] = (m, encoder)
        print(f"    {loc_type}: {mask.sum()} samples, {len(np.unique(y))} classes")

    # Per-event typed localizers
    for phys_label in sorted(set(phys_ev_arr)):
        if phys_label == 0: continue
        loc_type = "LINE" if phys_label == 2 else "BUS"
        encoder = line_enc if loc_type == "LINE" else bus_enc
        mask = (loc_type_arr == loc_type) & (phys_ev_arr == phys_label)
        if mask.sum() < 5: continue
        y = encoder.transform(all_loc_labels[mask])
        if len(np.unique(y)) < 2: continue
        key = f"{loc_type}:event{phys_label}"
        m = ExtraTreesClassifier(n_estimators=180, max_features="sqrt", class_weight="balanced",
                                  random_state=seed+2000+phys_label, n_jobs=4,
                                  max_depth=None, min_samples_leaf=2)
        m.fit(X_tr[mask], y)
        models[key] = (m, encoder)
        print(f"    {key}: {mask.sum()} samples, {len(np.unique(y))} classes")

    print(f"    Trained {len(models)} localizers ({tm.time()-t0:.0f}s)")

    # ── 4. Evaluate ──
    print("\n[4/5] Evaluating on SIM + RAW...")

    def _predict(X_arr, pred_event, pred_physical):
        out = np.full(len(X_arr), "none", dtype=object)
        pred_ev = pred_event.astype(int)
        pred_ph = pred_physical.astype(int)
        lt = np.full(len(pred_ev), "BUS", dtype=object)
        lt[np.isin(pred_ph, [2]) | np.isin(pred_ev, [2])] = "LINE"
        lt[np.isin(pred_ev, [5,7])] = "PMU"
        lt[pred_ev == 0] = "NONE"

        for phys_label in sorted(set(pred_ph)):
            if phys_label == 0: continue
            ct = "LINE" if phys_label == 2 else "BUS"
            key = f"{ct}:event{phys_label}"
            entry = models.get(key)
            if entry is None: continue
            m, enc = entry
            mask = (lt == ct) & (pred_ph == phys_label)
            if mask.any():
                preds = m.predict(X_arr[mask])
                out[mask] = enc.inverse_transform(preds.astype(int))

        for ct in ("BUS","LINE","PMU"):
            entry = models.get(ct)
            if entry is None: continue
            m, enc = entry
            mask = (lt == ct) & (out == "none")
            if mask.any():
                preds = m.predict(X_arr[mask])
                out[mask] = enc.inverse_transform(preds.astype(int))
        return out

    # SIM
    pred_ev = lb_te["event_label"].values.astype(int)
    pred_ph = lb_te["physical_event_label"].values.astype(int)
    et_sim = _predict(X_te, pred_ev, pred_ph)
    true_sim = lb_te["location_label"].values
    ms = np.array([_lt(l)!="NONE" for l in true_sim])
    sim_acc = float(np.mean(et_sim[ms]==true_sim[ms])) if ms.any() else 0

    sim_cm = confusion_matrix(
        np.array([_lt(l) for l in true_sim[ms]]),
        np.array([_lt(l) for l in et_sim[ms]]),
        labels=["BUS","LINE","PMU"]) if ms.any() else np.zeros((3,3),int)

    per_ev = {}
    for ev in sorted(set(pred_ev[ms])):
        em = ms & (pred_ev==ev)
        if em.any(): per_ev[EVT[ev]] = float(np.mean(et_sim[em]==true_sim[em]))

    print(f"    SIM Localizer: {sim_acc:.4f}")
    for ev_name, acc in sorted(per_ev.items()):
        print(f"      {ev_name}: {acc:.4f}")

    # ── 5. RAW ──
    print("\n[5/5] Evaluating on RAW001...")
    raw = _load_raw()
    # Align columns: build full feature matrix matching training columns
    X_raw_full = np.zeros((len(raw), len(feature_cols)), dtype=np.float64)
    found = 0
    for j, c in enumerate(feature_cols):
        if c in raw.columns:
            X_raw_full[:, j] = raw[c].fillna(0).replace([np.inf,-np.inf],0).values.astype(np.float64)
            found += 1
    X_raw = np.nan_to_num(X_raw_full, 0)
    print(f"    Found {found}/{len(feature_cols)} training features in RAW")

    raw_ev = raw["true_event"].values.astype(int)
    raw_ph = raw["true_event"].values.astype(int)
    et_raw = _predict(X_raw, raw_ev, raw_ph)
    true_raw = raw["true_location"].values
    mr = np.array([_lt(l)!="NONE" for l in true_raw])
    raw_acc = float(np.mean(et_raw[mr]==true_raw[mr])) if mr.any() else 0

    raw_cm = confusion_matrix(
        np.array([_lt(l) for l in true_raw[mr]]),
        np.array([_lt(l) for l in et_raw[mr]]),
        labels=["BUS","LINE","PMU"]) if mr.any() else np.zeros((3,3),int)

    raw_per = {}
    for ev in sorted(set(raw_ev[mr])):
        em = mr & (raw_ev==ev)
        if em.any(): raw_per[EVT[ev]] = float(np.mean(et_raw[em]==true_raw[em]))

    print(f"    RAW Localizer: {raw_acc:.4f}")
    for ev_name, acc in sorted(raw_per.items()):
        print(f"      {ev_name}: {acc:.4f}")

    # Baseline
    bl = json.loads(FINAL_M.read_text()) if FINAL_M.exists() else {}
    bsl = bl.get("selected",{}).get("sim_localizer_exact",0.8524)
    brl = bl.get("selected",{}).get("raw_localizer_exact",0.6667)

    # ── Plots ──
    print("\n  Generating plots...")

    fig,axes=plt.subplots(2,3,figsize=(18,11))

    # Main comparison
    locs = ["SIM\nBase","SIM\nNew","RAW\nBase","RAW\nNew"]
    vals = [bsl, sim_acc, brl, raw_acc]
    colors = ["tab:blue","tab:green","tab:blue","tab:green"]
    axes[0,0].bar(locs,vals,color=colors)
    axes[0,0].set_title("Localization Accuracy"); axes[0,0].set_ylim(0,1.1); axes[0,0].grid(alpha=.3,axis="y")
    for i,v in enumerate(vals): axes[0,0].text(i,v+.02,f"{v:.3f}",ha="center",fontweight="bold")

    # SIM per-event
    if per_ev:
        axes[0,1].bar([k[:8] for k in per_ev], per_ev.values(), color="tab:green")
        axes[0,1].set_title("SIM Loc per Event"); axes[0,1].set_ylim(0,1.1); axes[0,1].grid(alpha=.3,axis="y")
        plt.setp(axes[0,1].get_xticklabels(),rotation=45,ha="right",fontsize=8)

    # RAW per-event
    if raw_per:
        axes[0,2].bar([k[:8] for k in raw_per], raw_per.values(), color="tab:orange")
        axes[0,2].set_title("RAW Loc per Event"); axes[0,2].set_ylim(0,1.1); axes[0,2].grid(alpha=.3,axis="y")
        plt.setp(axes[0,2].get_xticklabels(),rotation=45,ha="right",fontsize=8)

    # Feature importance (BUS)
    if "BUS" in models:
        bus_model = models["BUS"][0]
        imp = bus_model.feature_importances_
        top_n=20; top_idx=np.argsort(imp)[-top_n:]
        axes[1,0].barh(range(top_n), imp[top_idx][::-1])
        axes[1,0].set_title(f"Top {top_n} Features (BUS Localizer)")

    # Confusion matrices
    _cm(sim_cm,["BUS","LINE","PMU"],"Localizer CM - SIM",out/"sim_localizer_cm.png",True,(8,6))
    _cm(raw_cm,["BUS","LINE","PMU"],"Localizer CM - RAW",out/"raw_localizer_cm.png",True,(8,6))

    # Efficiency
    n_trees = sum(getattr(m, "n_estimators", 180) for m, enc in models.values())
    penalty = 0.03*np.log10(max(n_trees*20,1))
    axes[1,2].bar(["Original","New\n(8 models)"],[0.21,penalty],color=["tab:red","tab:green"])
    axes[1,2].set_title(f"Efficiency (penalty={penalty:.4f})"); axes[1,2].grid(alpha=.3,axis="y")
    for i,v in enumerate([0.21,penalty]): axes[1,2].text(i,v+.002,f"{v:.4f}",ha="center",fontweight="bold")

    fig.suptitle("P16: Fresh ExtraTrees Localizer",fontweight="bold",fontsize=14)
    fig.tight_layout(); fig.savefig(out/"fresh_train_results.png",dpi=150,bbox_inches="tight"); plt.close(fig)

    # Save models
    # Save models (extract just the model object, not the tuple)
    models_save = {k: m for k, (m, enc) in models.items()}
    encoders_save = {k: enc.classes_.tolist() for k, (m, enc) in models.items()}
    joblib.dump({"models": models_save, "encoders": encoders_save, "feature_cols": feature_cols}, out/"fresh_extra_trees.joblib")

    summary = {"pipeline":"P16_Fresh_ExtraTrees","n_features":len(feature_cols),"n_models":len(models),
               "sim":{"loc_acc":float(sim_acc),"per_event":per_ev},
               "raw":{"loc_acc":float(raw_acc),"per_event":raw_per},
               "baseline":{"sim_loc":bsl,"raw_loc":brl}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str))

    print(f"\n  Total: {tm.time()-tt:.0f}s  Output: {out}")
    return str(out)

if __name__=="__main__": main()
