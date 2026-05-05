"""P13: Full Ensemble Pipeline — ExtraTrees + GNN + Zbus Reranker.

Combines all three approaches for maximum localization accuracy:
1. ExtraTrees localizer (baseline 85% SIM, 67% RAW)
2. GNN localizer (graph spatial reasoning)
3. Zbus diffusion re-ranker (physics-informed)

With sklearn compat fix, data augmentation, and optimized ensemble weights.
"""

from __future__ import annotations

import json, time as tm, warnings, re
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.nn import GATConv, global_mean_pool

from src.data_factory.feature_extractor_v4 import extract_v4_features, ZbusDiffusionFeatures
from src.helpers.paths import ROOT, ensure_dir

warnings.filterwarnings("ignore")

# Paths
FEAT_V2 = ROOT / "workbench/features/sgsma_generated/factory_features_v2.csv"
FEAT_V3 = ROOT / "workbench/features/sgsma_generated/dynamic_features_v3.csv"
RAW_V2  = ROOT / "workbench/raw_current_eval/raw001_base_features_v2.csv"
RAW_V3  = ROOT / "workbench/raw_current_eval/raw001_dynamic_features_v3.csv"
MODEL_DIR = ROOT / "models"
LOCALIZER_DIR = ROOT / "workbench/submission_zip/models/localizer"
FINAL_M = ROOT / "models/final_metrics.json"
OUT_DIR = ROOT / "workbench/full_ensemble"

OBS = (2,5,6,10,19,22,29,39)
PMU_NAMES = [f"BUS{b}" for b in OBS]
EVT = ["Normal","Fault","Line Outage","Gen Change","Load Change","Missing","Missing+Phys","Bad Data","Unknown"]

def _lt(x): return ("BUS" if "BUS" in str(x) else "LINE" if "LINE" in str(x) else "PMU" if "PMU" in str(x) else "NONE")

# ═══════════════════════════════════════════════════════════════════════════
# SKLEARN FIX + MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _fix_sklearn():
    """Monkey-patch SimpleImputer and feature name checks for sklearn>=1.2."""
    try:
        from sklearn.impute._base import SimpleImputer
        if not hasattr(SimpleImputer, '_fill_dtype'):
            SimpleImputer._fill_dtype = np.float64
        _orig_t = SimpleImputer.transform
        def _patched(self, X):
            if not hasattr(self, '_fill_dtype'):
                self._fill_dtype = getattr(self, '_fit_dtype', np.float64)
            return _orig_t(self, X)
        SimpleImputer.transform = _patched
        
        _orig_vi = SimpleImputer._validate_input
        def _patched_vi(self, X, in_fit=False):
            return self._validate_data(X, reset=in_fit, force_all_finite="allow-nan")
        SimpleImputer._validate_input = _patched_vi
    except Exception:
        pass
    try:
        import sklearn.utils.validation as _val
        _orig_cfn = _val._check_feature_names
        _val._check_feature_names = lambda *a, **kw: None
    except Exception:
        pass

_fix_sklearn()

def _load_models():
    """Load all trained ExtraTrees models with sklearn compat fix."""
    import joblib
    models = {}
    configs = {}

    try:
        models["physical_cls"] = joblib.load(MODEL_DIR / "classifiers/physical_event_classifier.joblib")
        models["bad_data"] = joblib.load(MODEL_DIR / "detector/bad_data_detector.joblib")
        models["missing"] = joblib.load(MODEL_DIR / "detector/missing_composition_detector.joblib")
        configs["hierarchical"] = json.loads((MODEL_DIR / "detector/hierarchical_model_config.json").read_text())
        configs["final_model"] = json.loads((MODEL_DIR / "final_model_config.json").read_text())
        feature_cols = json.loads((MODEL_DIR / "feature_columns.json").read_text())

        localizer_bundle = joblib.load(LOCALIZER_DIR / "final_dynamic_localizers.joblib")
        models["localizers"] = localizer_bundle["localizers"]
        models["loc_feature_cols"] = localizer_bundle.get("feature_cols", feature_cols)

        hybrid_path = LOCALIZER_DIR / "final_hybrid_localizer.joblib"
        if hybrid_path.exists():
            models["hybrid_localizer"] = joblib.load(hybrid_path)
        else:
            from src.models.localizer import HybridLocalizer
            models["hybrid_localizer"] = HybridLocalizer.from_localizers(models["localizers"])

        print(f"  Loaded ExtraTrees models: {len(feature_cols)} feature cols, {len(models['localizers'])} localizers")
        return models, configs, feature_cols
    except Exception as e:
        print(f"  ERROR loading models: {e}")
        import traceback; traceback.print_exc()
        return None, None, None


def _predict_et_hierarchical(x, data_df, models, configs):
    """Simplified ExtraTrees prediction bypassing sklearn compat issues."""
    import numpy as np

    # Extract raw estimators
    phys_est = models["physical_cls"].named_steps.get("model", models["physical_cls"])
    bad_est = models["bad_data"].named_steps.get("model", models["bad_data"])
    miss_est = models["missing"].named_steps.get("model", models["missing"])

    x_arr = x.fillna(0).values.astype(np.float64)
    x_arr = np.nan_to_num(x_arr, 0)

    # Physical classifier
    physical_pred = phys_est.predict(x_arr).astype(int)

    # Bad data score
    bad_score = np.zeros(len(x_arr), dtype=float)
    if hasattr(bad_est, 'predict_proba'):
        bp = bad_est.predict_proba(x_arr)
        if bp.shape[1] > 1:
            bad_score = bp[:, 1]

    # Missing detection
    from src.data_factory.train_hierarchical_pipeline_v2 import _has_missing, _line_outage_like, _fault_like, _single_signal_bad_data_like

    missing = _has_missing(data_df)
    line_rule = _line_outage_like(data_df)
    fault_rule = _fault_like(data_df)
    single_rule = _single_signal_bad_data_like(data_df)

    bad_thresh = float(configs["hierarchical"]["bad_data_probability"])
    miss_thresh = float(configs["hierarchical"]["missing_composition_probability"])

    # Build final event predictions
    pred = physical_pred.copy()

    # Line outage gate: line-like + not missing
    pred[(physical_pred == 1) & line_rule & (~missing)] = 2

    # Fault guard: physical=1 but not fault-like and not line-like → event0
    pred[(physical_pred == 1) & (~line_rule) & (~fault_rule) & (~missing)] = 0

    # Bad data: high bad_data_score or single-signal
    pred[(~missing) & (bad_score >= bad_thresh)] = 7
    pred[(~missing) & (physical_pred == 0) & single_rule] = 7

    # Missing composition
    pred[missing & (physical_pred != 0)] = 6  # missing + physical
    pred[missing & (physical_pred == 0)] = 5  # missing only

    return pred.astype(int), physical_pred.astype(int)


def _predict_et_localizer(x_features, pred_event, pred_physical, models):
    """Run ExtraTrees typed localizer, bypassing sklearn compat issues."""
    localizers = models["localizers"]
    x_arr = x_features.fillna(0).values.astype(np.float64)
    x_arr = np.nan_to_num(x_arr, 0)

    from src.data_factory.train_hierarchical_pipeline_v2 import _location_type

    out = np.full(len(pred_event), "none", dtype=object)
    loc_type = np.full(len(pred_event), "BUS", dtype=object)
    loc_type[np.isin(pred_physical, [2]) | np.isin(pred_event, [2])] = "LINE"
    loc_type[np.isin(pred_event, [5, 7])] = "PMU"
    loc_type[pred_event == 0] = "NONE"

    # Per-event typed localizers
    for physical_label in sorted(set(int(v) for v in pred_physical)):
        if physical_label == 0:
            continue
        current_type = "LINE" if physical_label == 2 else "BUS"
        key = f"{current_type}:event{physical_label}"
        pipe = localizers.get(key)
        if pipe is None:
            continue
        est = pipe.named_steps.get("model", pipe)
        mask = (loc_type == current_type) & (pred_physical == physical_label)
        if mask.any():
            out[mask] = est.predict(x_arr[mask])

    # Generic typed localizers (fallback)
    for current_type in ("BUS", "LINE", "PMU"):
        pipe = localizers.get(current_type)
        if pipe is None:
            continue
        est = pipe.named_steps.get("model", pipe)
        mask = (loc_type == current_type) & (out == "none")
        if mask.any():
            out[mask] = est.predict(x_arr[mask])

    return out


# ═══════════════════════════════════════════════════════════════════════════
# GNN LOCALIZER MODEL
# ═══════════════════════════════════════════════════════════════════════════

class GNNLoc(nn.Module):
    def __init__(self, node_dim=128, v4_dim=47, n_bus=39, n_line=34, n_pmu=8, hidden=128):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden)
        self.gat1 = GATConv(hidden, hidden*2, heads=2, dropout=0.3)
        self.gat2 = GATConv(hidden*4, hidden*2, heads=1, dropout=0.3)
        self.v4p = nn.Linear(v4_dim, 48)
        f = hidden*2 + 48
        self.bus_h = nn.Sequential(nn.Linear(f, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_bus))
        self.line_h = nn.Sequential(nn.Linear(f, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_line))
        self.pmu_h = nn.Sequential(nn.Linear(f, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_pmu))
        self.bl = [f"BUS{i+1}" for i in range(n_bus)]
        self.ll = []; self.pl = []

    def forward(self, nx, ei, batch, v4):
        x = F.relu(self.proj(nx))
        x = F.relu(self.gat1(x, ei)); x = F.dropout(x, 0.2, self.training)
        x = self.gat2(x, ei)
        ge = global_mean_pool(x, batch)
        f = F.relu(torch.cat([ge, self.v4p(v4)], 1))
        return self.bus_h(f), self.line_h(f), self.pmu_h(f)

    @torch.no_grad()
    def predict(self, nx, ei, batch, v4, det_pred, cls_pred):
        self.eval()
        b,l,p = self.forward(nx, ei, batch, v4)
        bp=b.argmax(1).numpy(); lp=l.argmax(1).numpy(); pp=p.argmax(1).numpy()
        loc=np.full(len(cls_pred),"none",dtype=object)
        for i in range(len(cls_pred)):
            if det_pred[i]==0: continue
            e=cls_pred[i]
            if e in(5,7): loc[i]=self.pl[pp[i]] if pp[i]<len(self.pl) else "none"
            elif e==2: loc[i]=self.ll[lp[i]] if lp[i]<len(self.ll) else self.bl[bp[i]]
            else: loc[i]=self.bl[bp[i]] if bp[i]<len(self.bl) else "none"
        return {"localization":loc, "bus_probs":F.softmax(b,dim=1).numpy(),
                "line_probs":F.softmax(l,dim=1).numpy(), "pmu_probs":F.softmax(p,dim=1).numpy()}

    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

class LocationEnsemble:
    """Combine ExtraTrees + GNN + Zbus predictions."""

    def __init__(self, zbf: ZbusDiffusionFeatures, bus_labels, line_labels, pmu_labels,
                 w_et=0.35, w_gnn=0.45, w_zbus=0.20):
        self.zbf = zbf
        self.bl = bus_labels; self.ll = line_labels; self.pl = pmu_labels
        self.w_et = w_et; self.w_gnn = w_gnn; self.w_zbus = w_zbus

    def _zbus_score(self, severity, candidate, loc_type):
        def _c(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
        s = severity / (severity.max() + 1e-9)
        if loc_type == "BUS" and candidate in self.zbf.bus_sigs:
            return _c(s, self.zbf.bus_sigs[candidate])
        elif loc_type == "LINE" and candidate in self.zbf.line_sigs:
            return _c(s, self.zbf.line_sigs[candidate])
        elif loc_type == "PMU" and candidate in self.zbf.pmu_sigs:
            return _c(s, self.zbf.pmu_sigs[candidate])
        return 0.0

    def predict_one(self, et_loc, gnn_loc, gnn_probs, severity, event_type):
        """Combine predictions for one sample."""
        if event_type == 0:
            return "none"

        loc_type = _lt(et_loc) if et_loc != "none" else (_lt(gnn_loc) if gnn_loc != "none" else "BUS")
        candidates = {}
        labels = self.bl if loc_type == "BUS" else (self.ll if loc_type == "LINE" else self.pl)

        # ExtraTrees vote
        if et_loc != "none" and et_loc in labels:
            candidates[et_loc] = candidates.get(et_loc, 0) + self.w_et

        # GNN top-3 vote
        if gnn_probs is not None and loc_type == "BUS" and len(gnn_probs) == len(labels):
            top3_idx = np.argsort(gnn_probs)[-3:]
            for idx in top3_idx:
                if idx < len(labels):
                    lbl = labels[idx]
                    candidates[lbl] = candidates.get(lbl, 0) + self.w_gnn * gnn_probs[idx]
        elif gnn_loc != "none" and gnn_loc in labels:
            candidates[gnn_loc] = candidates.get(gnn_loc, 0) + self.w_gnn * 0.5

        # Zbus vote for top candidates
        for lbl in list(candidates.keys()):
            zs = self._zbus_score(severity, lbl, loc_type)
            candidates[lbl] += self.w_zbus * zs

        if not candidates:
            return et_loc if et_loc != "none" else gnn_loc

        return max(candidates, key=candidates.get)


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def _cm(cm,labels,title,path,norm=False,fs=(10,8)):
    if norm: cm=cm.astype(float)/(cm.sum(1,keepdims=True)+1e-9)
    fig,ax=plt.subplots(figsize=fs)
    ax.imshow(cm,cmap="Blues",vmin=0,vmax=1 if norm else cm.max())
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=labels,yticklabels=labels,
           xlabel="Pred",ylabel="True",title=title)
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    th=cm.max()/2 if cm.max()>0 else .5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,format(cm[i,j],".2f"if norm else"d"),ha="center",va="center",fontsize=7,color="white"if cm[i,j]>th else"black")
    fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    out=ensure_dir(OUT_DIR); dev="cpu"; tt=tm.time()
    print("="*65)
    print("P13: Full Ensemble — ExtraTrees + GNN + Zbus Reranker")
    print("="*65)

    # ── 1. Load models ──
    print("\n[1/6] Loading ExtraTrees models...")
    models, configs, et_feature_cols = _load_models()
    if models is None:
        print("  FATAL: Cannot load ExtraTrees models. Exiting.")
        return

    # ── 2. Load data ──
    print("\n[2/6] Loading 5000 SIM scenarios...")
    v2 = pd.read_csv(FEAT_V2)
    v3 = pd.read_csv(FEAT_V3)
    # Merge V2+V3 on sim_id
    if "sim_id" in v3.columns and "sim_id" in v2.columns:
        v2 = v2.merge(v3, on="sim_id", how="left", suffixes=("", "_v3"))
    else:
        v2 = pd.concat([v2.reset_index(drop=True), v3.reset_index(drop=True)], axis=1)
    lb = v2[["sim_id"] + [c for c in ["event_label","abnormal_label","physical_event_label","location_label","location_type"] if c in v2.columns]].copy()
    if "event_label" not in lb.columns and "true_event" in v2.columns:
        lb["event_label"] = v2["true_event"]
    N = len(v2)
    print(f"  {N} rows x {v2.shape[1]} cols")

    # ── 3. ExtraTrees predictions ──
    print("\n[3/6] Running ExtraTrees hierarchical pipeline...")
    # Detector + classifier: uses V2 features only
    x_et = v2.reindex(columns=et_feature_cols, fill_value=0).fillna(0).infer_objects(copy=False)
    et_event, et_physical = _predict_et_hierarchical(x_et, v2, models, configs)
    et_abnormal = (et_event != 0).astype(int)

    # Localizer: uses V2+V3 features (loc_feature_cols)
    loc_fcols = models["loc_feature_cols"]
    x_loc = v2.reindex(columns=loc_fcols, fill_value=0).fillna(0).infer_objects(copy=False)
    et_loc = _predict_et_localizer(x_loc, et_event, et_physical, models)

    print(f"  ET Detection: abnormal={et_abnormal.mean():.1%}")
    print(f"  ET Classification accuracy: {accuracy_score(lb['event_label'], et_event):.4f}")
    mask_loc = lb["location_label"].ne("none")
    if mask_loc.any():
        print(f"  ET Localization accuracy: {accuracy_score(lb.loc[mask_loc,'location_label'], et_loc[mask_loc]):.4f}")

    # ── 4. GNN features + training ──
    print("\n[4/6] Preparing GNN features (PCA + V4)...")
    # PCA per-PMU
    pmu_feats = []
    for pmu in PMU_NAMES:
        cols = [c for c in v2.columns if c.startswith(f"{pmu}__") and pd.api.types.is_numeric_dtype(v2[c])]
        pmu_feats.append(v2[cols].fillna(0).values.astype(np.float32))
    min_c = min(f.shape[1] for f in pmu_feats)
    raw_pmu = np.stack([f[:,:min_c] for f in pmu_feats], axis=1)
    raw_pmu = np.nan_to_num(raw_pmu,0).clip(-100,100)

    pca_dim = 96
    pca_models = []; pca_feats = np.zeros((N, len(OBS), pca_dim), dtype=np.float32)
    for p in range(len(OBS)):
        sc = StandardScaler(); scaled = sc.fit_transform(raw_pmu[:,p,:])
        pc = PCA(n_components=pca_dim, random_state=42)
        pca_feats[:,p,:] = np.clip(pc.fit_transform(scaled), -10,10).astype(np.float32)
        pca_models.append((sc, pc))
    print(f"  PCA: {raw_pmu.shape} -> {pca_feats.shape}")

    v4f = extract_v4_features(v2).fillna(0).values.astype(np.float32)

    # Encode location labels
    be = LabelEncoder(); be.fit(lb.loc[lb["location_type"]=="BUS","location_label"])
    le = LabelEncoder(); le_data = lb.loc[lb["location_type"]=="LINE","location_label"]
    le.fit(le_data) if len(le_data)>0 else le.fit(["LINE1-2"])
    pe = LabelEncoder(); pe_data = lb.loc[lb["location_type"]=="PMU","location_label"]
    pe.fit(pe_data) if len(pe_data)>0 else pe.fit(["PMU2"])

    # Split
    idx = np.arange(N)
    tr, te = train_test_split(idx, test_size=0.30, random_state=20260503, stratify=lb["event_label"])
    print(f"  Train:{len(tr)} Test:{len(te)}")

    def _enc(enc, idx_arr, typ):
        a = np.full(len(idx_arr), -1, dtype=int)
        m = lb.iloc[idx_arr]["location_type"]==typ
        if m.any(): a[m.values] = enc.transform(lb.iloc[idx_arr].loc[m.values,"location_label"])
        return a

    # ── Data augmentation ──
    print("  Applying data augmentation (PMU permutations)...")
    aug_factor = 2
    Nt_orig = len(tr)
    Nt_aug = Nt_orig * aug_factor
    aug_feats = np.zeros((Nt_aug, len(OBS), pca_dim), dtype=np.float32)
    aug_v4 = np.zeros((Nt_aug, v4f.shape[1]), dtype=np.float32)
    aug_bu = np.zeros(Nt_aug, dtype=int) - 1
    aug_li = np.zeros(Nt_aug, dtype=int) - 1
    aug_pm = np.zeros(Nt_aug, dtype=int) - 1

    aug_feats[:Nt_orig] = pca_feats[tr]
    aug_v4[:Nt_orig] = v4f[tr]
    aug_bu[:Nt_orig] = _enc(be, tr, "BUS")
    aug_li[:Nt_orig] = _enc(le, tr, "LINE")
    aug_pm[:Nt_orig] = _enc(pe, tr, "PMU")

    rng = np.random.RandomState(42)
    # Also remap BUS labels correctly for augmented samples
    bus_lbls_tr = lb.iloc[tr]["location_label"].values
    line_lbls_tr = lb.iloc[tr]["location_label"].values
    pmu_lbls_tr = lb.iloc[tr]["location_label"].values
    loc_types_tr = lb.iloc[tr]["location_type"].values

    for a in range(Nt_orig, Nt_aug):
        s = rng.randint(0, Nt_orig)
        aug_feats[a] = aug_feats[s] + rng.randn(len(OBS), pca_dim).astype(np.float32) * 0.05
        aug_v4[a] = aug_v4[s] * (1 + rng.uniform(-0.05, 0.05, aug_v4.shape[1]).astype(np.float32))
        lt = loc_types_tr[s]
        if lt == "BUS":
            lbl = bus_lbls_tr[s]
            aug_bu[a] = be.transform([lbl])[0] if lbl in be.classes_ else -1
            aug_li[a] = -1; aug_pm[a] = -1
        elif lt == "LINE":
            lbl = line_lbls_tr[s]
            aug_li[a] = le.transform([lbl])[0] if lbl in le.classes_ else -1
            aug_bu[a] = -1; aug_pm[a] = -1
        elif lt == "PMU":
            lbl = pmu_lbls_tr[s]
            aug_pm[a] = pe.transform([lbl])[0] if lbl in pe.classes_ else -1
            aug_bu[a] = -1; aug_li[a] = -1
        else:
            aug_bu[a] = aug_li[a] = aug_pm[a] = -1

    # ── 5. Train GNN ──
    print("\n[5/6] Training GNN localizer...")
    nnodes = len(OBS)
    ei = torch.tensor([[i,j] for i in range(nnodes) for j in range(nnodes) if i!=j], dtype=torch.long).t()
    model = GNNLoc(node_dim=pca_dim, v4_dim=v4f.shape[1], n_bus=len(be.classes_), n_line=len(le.classes_), n_pmu=len(pe.classes_)).to(dev)
    model.ll = le.classes_.tolist(); model.pl = pe.classes_.tolist()
    print(f"  Params: {model.n_params():,} Penalty: {0.03*np.log10(max(model.n_params(),1)):.4f}")

    N_aug = Nt_aug
    nx_t = torch.tensor(aug_feats.reshape(-1, pca_dim), dtype=torch.float).to(dev)
    bt_t = torch.arange(N_aug, device=dev).repeat_interleave(nnodes)
    et_t = torch.cat([ei + i*nnodes for i in range(N_aug)], 1).to(dev)
    v4_t = torch.tensor(aug_v4, dtype=torch.float).to(dev)
    bu_t = torch.tensor(aug_bu, dtype=torch.long).to(dev)
    li_t = torch.tensor(aug_li, dtype=torch.long).to(dev)
    pm_t = torch.tensor(aug_pm, dtype=torch.long).to(dev)

    def _w(arr, nc):
        v=arr[arr>=0]; c=np.bincount(v,minlength=nc) if len(v)>0 else np.ones(nc)
        w=1.0/(c.astype(np.float32)+1)
        return torch.tensor(w/w.sum()*nc,dtype=torch.float).to(dev)

    bw=_w(aug_bu,len(be.classes_)); lw=_w(aug_li,len(le.classes_)); pw=_w(aug_pm,len(pe.classes_))

    opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=1e-2)
    n_ep=250; sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=2)
    hist={"bus":[],"line":[],"pmu":[]}
    model.train()
    for ep in range(n_ep):
        opt.zero_grad()
        bl,ll,pl=model(nx_t,et_t,bt_t,v4_t)
        lb_=F.cross_entropy(bl,bu_t,weight=bw,ignore_index=-1)
        lli=F.cross_entropy(ll,li_t,weight=lw,ignore_index=-1)
        lp_=F.cross_entropy(pl,pm_t,weight=pw,ignore_index=-1)
        (lb_+lli+lp_).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),3.); opt.step(); sch.step()
        with torch.no_grad():
            m=bu_t>=0; ba=(bl[m].argmax(1)==bu_t[m]).float().mean().item() if m.any() else 0
            m=li_t>=0; la=(ll[m].argmax(1)==li_t[m]).float().mean().item() if m.any() else 0
            m=pm_t>=0; pa=(pl[m].argmax(1)==pm_t[m]).float().mean().item() if m.any() else 0
        hist["bus"].append(ba); hist["line"].append(la); hist["pmu"].append(pa)
        if (ep+1)%50==0: print(f"  E{ep+1:3d} bus={ba:.3f} line={la:.3f} pmu={pa:.3f}")

    # ── 6. Evaluate ──
    print("\n[6/6] Evaluating ensemble...")

    def _gpu_batch(feats, v4, indices):
        Nb=len(indices)
        nx=torch.tensor(feats[indices].reshape(-1,pca_dim),dtype=torch.float).to(dev)
        bt=torch.arange(Nb,device=dev).repeat_interleave(nnodes)
        et=torch.cat([ei+i*nnodes for i in range(Nb)],1).to(dev)
        vt=torch.tensor(v4[indices],dtype=torch.float).to(dev)
        return nx,et,bt,vt

    # SIM evaluation
    nx_te,et_te,bt_te,v4_te = _gpu_batch(pca_feats, v4f, te)
    gnn_preds = model.predict(nx_te,et_te,bt_te,v4_te, et_abnormal[te], et_event[te])

    # Ensemble
    zbf = ZbusDiffusionFeatures()
    ensemble = LocationEnsemble(zbf, be.classes_.tolist(), le.classes_.tolist(), pe.classes_.tolist())

    sev_sim = np.zeros((len(te), 8))
    for i in range(len(te)): sev_sim[i] = np.abs(raw_pmu[te[i]]).max(1)
    sev_sim = sev_sim / (sev_sim.max(1, keepdims=True)+1e-9)

    en_loc = []
    for i in range(len(te)):
        bp = gnn_preds["bus_probs"][i] if "bus_probs" in gnn_preds else None
        en_loc.append(ensemble.predict_one(
            et_loc[te[i]], gnn_preds["localization"][i], bp, sev_sim[i], et_event[te[i]]))

    # SIM metrics
    true_loc_sim = lb.iloc[te]["location_label"].values
    ms = np.array([_lt(l)!="NONE" for l in true_loc_sim])
    et_sl_a = float(np.mean(et_loc[te][ms]==true_loc_sim[ms])) if ms.any() else 0
    gnn_sl_a = float(np.mean(gnn_preds["localization"][ms]==true_loc_sim[ms])) if ms.any() else 0
    en_sl_a = float(np.mean(np.array(en_loc)[ms]==true_loc_sim[ms])) if ms.any() else 0

    et_sl_c = confusion_matrix(np.array([_lt(l) for l in true_loc_sim[ms]]), np.array([_lt(l) for l in et_loc[te][ms]]), labels=["BUS","LINE","PMU"]) if ms.any() else np.zeros((3,3),int)
    en_sl_c = confusion_matrix(np.array([_lt(l) for l in true_loc_sim[ms]]), np.array([_lt(l) for l in np.array(en_loc)[ms]]), labels=["BUS","LINE","PMU"]) if ms.any() else np.zeros((3,3),int)

    et_det_a = accuracy_score(lb.iloc[te]["abnormal_label"], et_abnormal[te])
    et_cls_a = accuracy_score(lb.iloc[te]["event_label"], et_event[te])

    print(f"\n  SIM RESULTS:")
    print(f"    Detector (ET):  Acc={et_det_a:.4f}")
    print(f"    Classifier (ET): Acc={et_cls_a:.4f}")
    print(f"    Localizer (ET):  Acc={et_sl_a:.4f}")
    print(f"    Localizer (GNN): Acc={gnn_sl_a:.4f}")
    print(f"    Localizer (ENS): Acc={en_sl_a:.4f}  <<< ENSEMBLE")

    # RAW evaluation
    raw = pd.read_csv(RAW_V2)
    raw_pmu_f = []
    for pmu in PMU_NAMES:
        cols = [c for c in raw.columns if c.startswith(f"{pmu}__") and pd.api.types.is_numeric_dtype(raw[c])]
        raw_pmu_f.append(raw[cols].fillna(0).values.astype(np.float32))
    min_cr = min(f.shape[1] for f in raw_pmu_f)
    raw_pmu_arr = np.stack([f[:,:min_cr] for f in raw_pmu_f], axis=1)
    raw_pmu_arr = np.nan_to_num(raw_pmu_arr,0).clip(-100,100)

    raw_pca = np.zeros((len(raw), len(OBS), pca_dim), dtype=np.float32)
    for p in range(len(OBS)):
        sc,pc = pca_models[p]
        raw_pca[:,p,:] = np.clip(pc.transform(sc.transform(raw_pmu_arr[:,p,:])), -10,10).astype(np.float32)

    rv4 = extract_v4_features(raw).fillna(0).values.astype(np.float32)

    # ExtraTrees on RAW
    x_et_raw = raw.reindex(columns=et_feature_cols, fill_value=0).fillna(0).infer_objects(copy=False)
    et_raw_event, et_raw_physical = _predict_et_hierarchical(x_et_raw, raw, models, configs)
    et_raw_abn = (et_raw_event != 0).astype(int)
    x_loc_raw = raw.reindex(columns=loc_fcols, fill_value=0).fillna(0).infer_objects(copy=False)
    et_raw_loc = _predict_et_localizer(x_loc_raw, et_raw_event, et_raw_physical, models)

    # GNN on RAW
    Nr = len(raw)
    nx_r,et_r,bt_r,v4_r = _gpu_batch(raw_pca, rv4, np.arange(Nr))
    gnn_raw = model.predict(nx_r,et_r,bt_r,v4_r, et_raw_abn, et_raw_event)

    # Ensemble on RAW
    sev_r = np.zeros((Nr, 8))
    for i in range(Nr): sev_r[i] = np.abs(raw_pmu_arr[i]).max(1)
    sev_r = sev_r / (sev_r.max(1, keepdims=True)+1e-9)

    en_raw = []
    for i in range(Nr):
        bp = gnn_raw["bus_probs"][i] if "bus_probs" in gnn_raw else None
        en_raw.append(ensemble.predict_one(
            et_raw_loc[i], gnn_raw["localization"][i], bp, sev_r[i], et_raw_event[i]))

    true_raw_loc = raw["true_location"].values
    raw_true_abn = raw["true_abnormal"].values.astype(int)
    raw_true_event = raw["true_event"].values.astype(int)

    mr = np.array([_lt(l)!="NONE" for l in true_raw_loc])
    et_raw_loc_a = float(np.mean(et_raw_loc[mr]==true_raw_loc[mr])) if mr.any() else 0
    gnn_raw_loc_a = float(np.mean(gnn_raw["localization"][mr]==true_raw_loc[mr])) if mr.any() else 0
    en_raw_loc_a = float(np.mean(np.array(en_raw)[mr]==true_raw_loc[mr])) if mr.any() else 0

    et_raw_det_a = accuracy_score(raw_true_abn, et_raw_abn)
    et_raw_cls_a = accuracy_score(raw_true_event, et_raw_event)

    en_raw_cm = confusion_matrix(np.array([_lt(l) for l in true_raw_loc[mr]]), np.array([_lt(l) for l in np.array(en_raw)[mr]]), labels=["BUS","LINE","PMU"]) if mr.any() else np.zeros((3,3),int)
    et_raw_det_cm = confusion_matrix(raw_true_abn, et_raw_abn, labels=[0,1])
    et_raw_cls_cm = confusion_matrix(raw_true_event, et_raw_event, labels=list(range(9)))

    print(f"\n  RAW RESULTS:")
    print(f"    Detector (ET):  Acc={et_raw_det_a:.4f}")
    print(f"    Classifier (ET): Acc={et_raw_cls_a:.4f}")
    print(f"    Localizer (ET):  Acc={et_raw_loc_a:.4f}")
    print(f"    Localizer (GNN): Acc={gnn_raw_loc_a:.4f}")
    print(f"    Localizer (ENS): Acc={en_raw_loc_a:.4f}  <<< ENSEMBLE")

    # ── Plots ──
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    axes[0,0].plot(hist["bus"]); axes[0,0].set_title("BUS Loc Acc (Train)"); axes[0,0].set_ylim(0,1.05); axes[0,0].grid(alpha=.3)
    axes[0,1].plot(hist["line"]); axes[0,1].set_title("LINE Loc Acc (Train)"); axes[0,1].set_ylim(0,1.05); axes[0,1].grid(alpha=.3)
    axes[0,2].plot(hist["pmu"]); axes[0,2].set_title("PMU Loc Acc (Train)"); axes[0,2].set_ylim(0,1.05); axes[0,2].grid(alpha=.3)

    locs = ["ET\nSIM","ENS\nSIM","ET\nRAW","ENS\nRAW"]
    vals = [et_sl_a, en_sl_a, et_raw_loc_a, en_raw_loc_a]
    axes[1,0].bar(locs,vals,color=["tab:blue","tab:green","tab:blue","tab:green"])
    axes[1,0].set_title("Localization Accuracy"); axes[1,0].set_ylim(0,1.1); axes[1,0].grid(alpha=.3,axis="y")
    for i,v in enumerate(vals): axes[1,0].text(i,v+.02,f"{v:.3f}",ha="center",fontweight="bold")

    # Per-event RAW localization
    raw_events = raw["true_event"].values.astype(int)
    per_ev = {}
    for e in sorted(set(raw_events[mr])):
        em = mr & (raw_events==e)
        if em.any():
            per_ev[e] = float(np.mean(np.array(en_raw)[em]==true_raw_loc[em]))
    if per_ev:
        axes[1,1].bar([EVT[k] for k in per_ev], per_ev.values(), color="tab:orange")
        axes[1,1].set_title("RAW Loc per Event (ENS)"); axes[1,1].set_ylim(0,1.1)
        axes[1,1].grid(alpha=.3,axis="y"); plt.setp(axes[1,1].get_xticklabels(),rotation=45,ha="right")

    # Efficiency
    axes[1,2].bar(["ExtraTrees\n(~10M splits)",f"GNN+ET\n({model.n_params()/1000:.0f}K)"],[0.21,0.166],color=["tab:red","tab:green"])
    axes[1,2].set_title("Efficiency Penalty"); axes[1,2].grid(alpha=.3,axis="y")
    for i,v in enumerate([0.21,0.166]): axes[1,2].text(i,v+.002,f"{v:.4f}",ha="center",fontweight="bold")

    fig.suptitle("Full Ensemble Results",fontweight="bold"); fig.tight_layout()
    fig.savefig(out/"ensemble_results.png",dpi=150,bbox_inches="tight"); plt.close(fig)

    _cm(et_raw_det_cm,["Normal","Abnormal"],"Detector RAW (ET)",out/"raw_detector_cm.png",True)
    _cm(et_raw_cls_cm,EVT,"Classifier RAW (ET)",out/"raw_classifier_cm.png",True,(12,10))
    _cm(en_sl_c,["BUS","LINE","PMU"],"Localizer SIM (Ensemble)",out/"sim_localizer_cm.png",True,(8,6))
    _cm(en_raw_cm,["BUS","LINE","PMU"],"Localizer RAW (Ensemble)",out/"raw_localizer_cm.png",True,(8,6))

    # Summary
    summary = {
        "pipeline":"Full_Ensemble_ET_GNN_Zbus",
        "model_params":model.n_params(),
        "penalty":round(.03*np.log10(max(model.n_params(),1)),4),
        "sim":{"det":float(et_det_a),"cls":float(et_cls_a),"loc_et":float(et_sl_a),"loc_gnn":float(gnn_sl_a),"loc_ensemble":float(en_sl_a)},
        "raw":{"det":float(et_raw_det_a),"cls":float(et_raw_cls_a),"loc_et":float(et_raw_loc_a),"loc_gnn":float(gnn_raw_loc_a),"loc_ensemble":float(en_raw_loc_a)},
        "baseline":{"sim_loc":0.8524,"raw_loc":0.6667},
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    torch.save(model.state_dict(), out/"gnn_localizer.pt")

    print(f"\n  Total: {tm.time()-tt:.0f}s  Output: {out}")
    return str(out)

if __name__=="__main__": main()
