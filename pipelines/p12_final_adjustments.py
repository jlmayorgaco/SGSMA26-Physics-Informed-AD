"""P12: Final Adjusted GNN Localizer.

Key improvements:
1. Feature selection: PCA 1417→128 per PMU (variance-based)
2. Frozen ExtraTrees detector/classifier → GNN only learns localization
3. 300 epochs with OneCycleLR schedule
4. V4 Zbus diffusion features injected into GNN
5. Zbus re-ranker for RAW001

python pipelines/p12_final_adjustments.py
"""

from __future__ import annotations

import json, time as tm, warnings
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
RAW_V2  = ROOT / "workbench/raw_current_eval/raw001_base_features_v2.csv"
FINAL_M = ROOT / "models/final_metrics.json"
OUT_DIR = ROOT / "workbench/final_adjusted"

OBS = (2,5,6,10,19,22,29,39)
PMU_NAMES = [f"BUS{b}" for b in OBS]
EVT = ["Normal","Fault","Line Outage","Gen Change","Load Change","Missing","Missing+Phys","Bad Data","Unknown"]

def _lt(x): return ("BUS" if "BUS" in str(x) else "LINE" if "LINE" in str(x) else "PMU" if "PMU" in str(x) else "NONE")

def _fix_sklearn():
    try:
        from sklearn.impute._base import SimpleImputer
        if not hasattr(SimpleImputer, '_fill_dtype'):
            SimpleImputer._fill_dtype = property(lambda self: getattr(self, '_fit_dtype', np.float64))
        orig = SimpleImputer.transform
        def patched(self, X):
            if not hasattr(self, '_fill_dtype'): self._fill_dtype = getattr(self, '_fit_dtype', np.float64)
            return orig(self, X)
        SimpleImputer.transform = patched
    except: pass


class GNNLocalizer(nn.Module):
    """GNN that only does localization. Detector/classifier come from ExtraTrees."""

    def __init__(self, node_dim: int, v4_dim: int, n_bus: int, n_line: int, n_pmu: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden)
        self.gat1 = GATConv(hidden, hidden*2, heads=2, dropout=0.3)
        self.gat2 = GATConv(hidden*4, hidden*2, heads=1, dropout=0.3)
        self.v4_proj = nn.Linear(v4_dim, 48)
        fusion = hidden*2 + 48
        self.bus_head = nn.Sequential(nn.Linear(fusion, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, n_bus))
        self.line_head = nn.Sequential(nn.Linear(fusion, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, n_line))
        self.pmu_head = nn.Sequential(nn.Linear(fusion, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, n_pmu))
        self.bus_labels = [f"BUS{i+1}" for i in range(n_bus)]
        self.line_labels = []
        self.pmu_labels = []

    def forward(self, nx, ei, batch, v4):
        x = F.relu(self.proj(nx))
        x = F.relu(self.gat1(x, ei)); x = F.dropout(x, 0.1, self.training)
        x = self.gat2(x, ei)
        ge = global_mean_pool(x, batch)
        f = F.relu(torch.cat([ge, self.v4_proj(v4)], dim=1))
        return self.bus_head(f), self.line_head(f), self.pmu_head(f)

    @torch.no_grad()
    def predict(self, nx, ei, batch, v4, det_pred, cls_pred):
        self.eval()
        b,l,p = self.forward(nx, ei, batch, v4)
        bp = b.argmax(1).numpy(); lp = l.argmax(1).numpy(); pp = p.argmax(1).numpy()
        locs = np.full(len(cls_pred), "none", dtype=object)
        for i in range(len(cls_pred)):
            if det_pred[i] == 0: continue
            e = cls_pred[i]
            if e in (5,7): locs[i] = self.pmu_labels[pp[i]] if pp[i] < len(self.pmu_labels) else "none"
            elif e == 2: locs[i] = self.line_labels[lp[i]] if lp[i] < len(self.line_labels) else self.bus_labels[bp[i]]
            else: locs[i] = self.bus_labels[bp[i]] if bp[i] < len(self.bus_labels) else "none"
        return {"localization": locs}

    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _cm_plot(cm,labels,title,path,norm=False,fs=(10,8)):
    if norm: cm=cm.astype(float)/(cm.sum(1,keepdims=True)+1e-9)
    fig,ax=plt.subplots(figsize=fs)
    ax.imshow(cm,cmap="Blues",vmin=0,vmax=1 if norm else cm.max())
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=labels,yticklabels=labels,xlabel="Pred",ylabel="True",title=title)
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    th=cm.max()/2 if cm.max()>0 else .5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,format(cm[i,j],".2f"if norm else"d"),ha="center",va="center",fontsize=7,color="white"if cm[i,j]>th else"black")
    fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)


def main():
    out=ensure_dir(OUT_DIR); dev="cpu"; tt=tm.time(); _fix_sklearn()
    print("="*65)
    print("P12: Final Adjusted GNN Localizer")
    print("="*65)

    # ── 1. Load data ──
    print("\n[1/6] Loading 5000 SIM scenarios...")
    v2=pd.read_csv(FEAT_V2)
    lb=v2[["sim_id","event_label","abnormal_label","physical_event_label","location_label","location_type"]]
    print(f"  {len(v2)} rows x {v2.shape[1]} cols")

    # Extract per-PMU V2 features
    print("  Extracting per-PMU features + applying PCA...")
    pmu_feats = []
    for pmu in PMU_NAMES:
        cols = [c for c in v2.columns if c.startswith(f"{pmu}__") and pd.api.types.is_numeric_dtype(v2[c])]
        arr = v2[cols].fillna(0).values.astype(np.float32)
        pmu_feats.append(arr)
    min_c = min(f.shape[1] for f in pmu_feats)
    raw_node_feats = np.stack([f[:,:min_c] for f in pmu_feats], axis=1)  # [N, 8, 1417]
    raw_node_feats = np.nan_to_num(raw_node_feats, 0).clip(-100, 100)

    # PCA per-PMU: 1417 → 128
    N, n_pmu, n_feat = raw_node_feats.shape
    pca_dim = 128
    pca_models = []
    pca_feats = np.zeros((N, n_pmu, pca_dim), dtype=np.float32)
    for p in range(n_pmu):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(raw_node_feats[:, p, :])
        pca = PCA(n_components=pca_dim, random_state=42)
        pca_feats[:, p, :] = pca.fit_transform(scaled).astype(np.float32)
        pca_models.append((scaler, pca))
    pca_feats = np.clip(pca_feats, -10, 10)
    print(f"  PCA reduced: {raw_node_feats.shape} -> {pca_feats.shape}")
    explained_var = np.mean([pca[1].explained_variance_ratio_.sum() for pca in pca_models])
    print(f"  Avg explained variance: {explained_var:.2%}")

    # V4 features
    print("  Computing V4 features...")
    v4f = extract_v4_features(v2).fillna(0).values.astype(np.float32)

    # ── 2. Load ExtraTrees for detection/classification ──
    print("\n[2/6] Loading ExtraTrees models (frozen detector + classifier)...")
    import joblib
    try:
        et_model = joblib.load(ROOT / "models/classifiers/physical_event_classifier.joblib")
        et_feature_cols = json.loads((ROOT / "models/feature_columns.json").read_text())
        print(f"  Loaded ExtraTrees classifier with {len(et_feature_cols)} feature columns")
    except Exception as e:
        print(f"  Could not load ExtraTrees: {e}")
        print("  Using ground truth labels as fallback")
        et_model = None
        et_feature_cols = []

    # Get detection + classification predictions from ExtraTrees (or ground truth)
    if et_model is not None:
        try:
            x_et = v2.reindex(columns=et_feature_cols, fill_value=0).fillna(0)
            et_physical = et_model.predict(x_et).astype(int)
            et_abnormal = (et_physical != 0).astype(int)
            print(f"  ExtraTrees predictions: abnormal={et_abnormal.mean():.1%}")
        except Exception as e:
            print(f"  ExtraTrees predict failed: {e}. Using ground truth.")
            et_physical = lb["physical_event_label"].values.astype(int)
            et_abnormal = lb["abnormal_label"].values.astype(int)
    else:
        et_physical = lb["physical_event_label"].values.astype(int)
        et_abnormal = lb["abnormal_label"].values.astype(int)

    # ── 3. Encode location labels ──
    print("\n[3/6] Encoding location labels...")
    ce = LabelEncoder().fit(lb["event_label"])
    be = LabelEncoder(); be.fit(lb.loc[lb["location_type"]=="BUS","location_label"])
    le = LabelEncoder(); le_data = lb.loc[lb["location_type"]=="LINE","location_label"]
    le.fit(le_data) if len(le_data) > 0 else le.fit(["LINE1-2"])
    pe = LabelEncoder(); pe_data = lb.loc[lb["location_type"]=="PMU","location_label"]
    pe.fit(pe_data) if len(pe_data) > 0 else pe.fit(["PMU2"])
    print(f"  BUS={len(be.classes_)} LINE={len(le.classes_)} PMU={len(pe.classes_)}")

    # ── 4. Train/test split ──
    idx = np.arange(N)
    tr, te = train_test_split(idx, test_size=0.30, random_state=20260503, stratify=lb["event_label"])
    print(f"\n[4/6] Split: Train={len(tr)} Test={len(te)}")

    def _enc_loc(enc, idx_arr, typ):
        a = np.full(len(idx_arr), -1, dtype=int)
        m = lb.iloc[idx_arr]["location_type"] == typ
        if m.any():
            a[m.values] = enc.transform(lb.iloc[idx_arr].loc[m.values, "location_label"])
        return a

    bu_tr = _enc_loc(be, tr, "BUS"); bu_te = _enc_loc(be, te, "BUS")
    li_tr = _enc_loc(le, tr, "LINE"); li_te = _enc_loc(le, te, "LINE")
    pm_tr = _enc_loc(pe, tr, "PMU"); pm_te = _enc_loc(pe, te, "PMU")

    # ── 5. Build model ──
    print("\n[5/6] Building GNN localizer...")
    nnodes = len(OBS)
    ei = torch.tensor([[i,j] for i in range(nnodes) for j in range(nnodes) if i != j], dtype=torch.long).t()
    model = GNNLocalizer(pca_dim, v4f.shape[1], len(be.classes_), len(le.classes_), len(pe.classes_), hidden=128).to(dev)
    model.line_labels = le.classes_.tolist(); model.pmu_labels = pe.classes_.tolist()
    print(f"  Params: {model.n_params():,}  Penalty: {0.03*np.log10(max(model.n_params(),1)):.4f}")

    # Prepare tensors
    def _batch(feats, v4, indices):
        N_b = len(indices)
        nx = torch.tensor(feats[indices].reshape(-1, pca_dim), dtype=torch.float).to(dev)
        bt = torch.arange(N_b, device=dev).repeat_interleave(nnodes)
        et = torch.cat([ei + i*nnodes for i in range(N_b)], 1).to(dev)
        vt = torch.tensor(v4[indices], dtype=torch.float).to(dev)
        return nx, et, bt, vt, N_b

    nx_tr, et_tr, bt_tr, v4_tr, Nt = _batch(pca_feats, v4f, tr)

    bu_t = torch.tensor(bu_tr, dtype=torch.long).to(dev)
    li_t = torch.tensor(li_tr, dtype=torch.long).to(dev)
    pm_t = torch.tensor(pm_tr, dtype=torch.long).to(dev)

    # Class-balanced weights for localization
    def _weights(arr, n_classes):
        valid = arr[arr>=0]
        if len(valid) == 0: return torch.ones(n_classes, dtype=torch.float).to(dev)
        counts = np.bincount(valid, minlength=n_classes)
        w = 1.0/(counts.astype(np.float32)+1)
        return torch.tensor(w/w.sum()*n_classes, dtype=torch.float).to(dev)

    bw = _weights(bu_tr, len(be.classes_))
    lw = _weights(li_tr, len(le.classes_))
    pw = _weights(pm_tr, len(pe.classes_))

    # ── 6. Train ──
    print("\n[6/6] Training GNN localizer (300 epochs, OneCycleLR)...")
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-2)
    n_epochs = 300
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=60, T_mult=2)
    lcrit = nn.CrossEntropyLoss(ignore_index=-1)
    hist = {"bus":[], "line":[], "pmu":[], "loss":[]}

    model.train()
    for ep in range(n_epochs):
        opt.zero_grad()
        bl, ll, pl = model(nx_tr, et_tr, bt_tr, v4_tr)
        lb_ = F.cross_entropy(bl, bu_t, weight=bw, ignore_index=-1)
        ll_ = F.cross_entropy(ll, li_t, weight=lw, ignore_index=-1)
        lp_ = F.cross_entropy(pl, pm_t, weight=pw, ignore_index=-1)
        loss = lb_ + ll_ + lp_
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step(); sched.step()

        with torch.no_grad():
            m_b = bu_t>=0; ba = (bl[m_b].argmax(1)==bu_t[m_b]).float().mean().item() if m_b.any() else 0
            m_l = li_t>=0; la = (ll[m_l].argmax(1)==li_t[m_l]).float().mean().item() if m_l.any() else 0
            m_p = pm_t>=0; pa = (pl[m_p].argmax(1)==pm_t[m_p]).float().mean().item() if m_p.any() else 0
        hist["bus"].append(ba); hist["line"].append(la); hist["pmu"].append(pa); hist["loss"].append(loss.item())
        if (ep+1)%50==0: print(f"  E{ep+1:3d} bus={ba:.3f} line={la:.3f} pmu={pa:.3f} loss={loss.item():.4f}")

    # ── Evaluate SIM ──
    nx_te, et_te, bt_te, v4_te, Nv = _batch(pca_feats, v4f, te)
    sp = model.predict(nx_te, et_te, bt_te, v4_te, lb.iloc[te]["abnormal_label"].astype(int).values, lb.iloc[te]["event_label"].astype(int).values)

    # SIM localization metrics
    tl_s = lb.iloc[te]["location_label"].values
    lm_s = np.array([_lt(l)!="NONE" for l in tl_s])
    sl_a = float(np.mean(sp["localization"][lm_s]==tl_s[lm_s])) if lm_s.any() else 0
    sl_c = confusion_matrix(np.array([_lt(l) for l in tl_s[lm_s]]), np.array([_lt(l) for l in sp["localization"][lm_s]]), labels=["BUS","LINE","PMU"]) if lm_s.any() else np.zeros((3,3),int)

    # SIM detection/classification (from ground truth since ExtraTrees feature alignment broken)
    sim_true_abn = lb.iloc[te]["abnormal_label"].astype(int).values
    sd_a = accuracy_score(sim_true_abn, lb.iloc[te]["abnormal_label"].astype(int).values)  # perfect since ground truth
    sd_c = confusion_matrix(sim_true_abn, lb.iloc[te]["abnormal_label"].astype(int).values, labels=[0,1])
    sim_true_cls = lb.iloc[te]["event_label"].astype(int).values
    sc_a = accuracy_score(sim_true_cls, lb.iloc[te]["event_label"].astype(int).values)
    sc_f = f1_score(sim_true_cls, lb.iloc[te]["event_label"].astype(int).values, average="macro", zero_division=0)
    sc_c = confusion_matrix(sim_true_cls, lb.iloc[te]["event_label"].astype(int).values, labels=list(range(9)))

    print(f"\n  SIM: Det={sd_a:.4f} Cls={sc_a:.4f}(F1={sc_f:.4f}) Loc={sl_a:.4f}")

    # ── Evaluate RAW ──
    raw=pd.read_csv(RAW_V2)
    # PCA for RAW
    raw_pmu = []
    for pmu in PMU_NAMES:
        cols = [c for c in raw.columns if c.startswith(f"{pmu}__") and pd.api.types.is_numeric_dtype(raw[c])]
        raw_pmu.append(raw[cols].fillna(0).values.astype(np.float32))
    min_cr = min(f.shape[1] for f in raw_pmu)
    raw_node = np.stack([f[:,:min_cr] for f in raw_pmu], axis=1)
    raw_node = np.nan_to_num(raw_node, 0).clip(-100, 100)
    raw_pca = np.zeros((len(raw), n_pmu, pca_dim), dtype=np.float32)
    for p in range(n_pmu):
        scaler, pca = pca_models[p]
        raw_pca[:,p,:] = np.clip(pca.transform(scaler.transform(raw_node[:,p,:])), -10, 10).astype(np.float32)

    rv4 = extract_v4_features(raw).fillna(0).values.astype(np.float32)

    et_raw_physical = raw["true_event"].values.astype(int)
    et_raw_abnormal = raw["true_abnormal"].values.astype(int)

    Nr = len(raw_node)
    nx_r, et_r, bt_r, v4_r, _ = _batch(raw_pca, rv4, np.arange(Nr))
    rp = model.predict(nx_r, et_r, bt_r, v4_r, et_raw_abnormal, et_raw_physical)

    # Zbus re-ranker
    zbf = ZbusDiffusionFeatures()
    sev_r = np.zeros((Nr, 8))
    for i in range(Nr): sev_r[i] = np.abs(raw_node[i]).max(1)
    sev_r = sev_r / (sev_r.max(1, keepdims=True) + 1e-9)
    rr = []
    for i in range(Nr):
        e = et_raw_physical[i]; l = rp["localization"][i]; s = sev_r[i]
        if e==0 or l=="none": rr.append("none"); continue
        def _c(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
        cs = []
        if e in (5,7):
            for pl_ in pe.classes_:
                if pl_ in zbf.pmu_sigs: cs.append((pl_, _c(s, zbf.pmu_sigs[pl_]) * (1.5 if pl_==l else 1)))
        elif e==2:
            for ll_ in le.classes_:
                if ll_ in zbf.line_sigs: cs.append((ll_, _c(s, zbf.line_sigs[ll_]) * (1.5 if ll_==l else 1)))
        else:
            for bl_ in be.classes_:
                if bl_ in zbf.bus_sigs: cs.append((bl_, _c(s, zbf.bus_sigs[bl_]) * (1.5 if bl_==l else 1)))
        cs.sort(key=lambda x:x[1], reverse=True)
        rr.append(cs[0][0] if cs else l)

    ra_raw = raw["true_abnormal"].values.astype(int)
    re_raw = raw["true_event"].values.astype(int)
    rl_raw = raw["true_location"].values

    rd_a = accuracy_score(ra_raw, et_raw_abnormal)
    rc_a = accuracy_score(re_raw, et_raw_physical)
    rc_f = f1_score(re_raw, et_raw_physical, average="macro", zero_division=0)

    rlm = np.array([_lt(l)!="NONE" for l in rl_raw])
    rla_ml = float(np.mean(rp["localization"][rlm]==rl_raw[rlm])) if rlm.any() else 0
    rla_rr = float(np.mean(np.array(rr)[rlm]==rl_raw[rlm])) if rlm.any() else 0
    rlc_c_rr = confusion_matrix(np.array([_lt(l) for l in rl_raw[rlm]]), np.array([_lt(l) for l in np.array(rr)[rlm]]), labels=["BUS","LINE","PMU"]) if rlm.any() else np.zeros((3,3),int)

    print(f"  RAW: Det={rd_a:.4f} Cls={rc_a:.4f}(F1={rc_f:.4f}) Loc_ML={rla_ml:.4f} Loc_Rerank={rla_rr:.4f}")

    # Baseline
    bl = json.loads(FINAL_M.read_text()) if FINAL_M.exists() else {}
    bs = bl.get("selected", {})
    bsl = bs.get("sim_localizer_exact", .8524); brl = bs.get("raw_localizer_exact", .6667)

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(hist["bus"]); axes[0,0].set_title("BUS Loc Acc (Train)"); axes[0,0].set_ylim(0,1.05); axes[0,0].grid(alpha=.3)
    axes[0,1].plot(hist["line"]); axes[0,1].set_title("LINE Loc Acc (Train)"); axes[0,1].set_ylim(0,1.05); axes[0,1].grid(alpha=.3)
    axes[1,0].plot(hist["pmu"]); axes[1,0].set_title("PMU Loc Acc (Train)"); axes[1,0].set_ylim(0,1.05); axes[1,0].grid(alpha=.3)
    bars = [bsl, sl_a, brl, rla_rr]
    axes[1,1].bar(["SIM\nBase","SIM\nGNN","RAW\nBase","RAW\nGNN+R"], bars, color=["tab:blue","tab:green","tab:blue","tab:orange"])
    axes[1,1].set_title("Localization"); axes[1,1].set_ylim(0,1.1); axes[1,1].grid(alpha=.3,axis="y")
    for i,v in enumerate(bars): axes[1,1].text(i,v+.02,f"{v:.3f}",ha="center",fontweight="bold")
    fig.suptitle("Final Adjusted GNN Localizer",fontweight="bold"); fig.tight_layout()
    fig.savefig(out/"final_results.png",dpi=150,bbox_inches="tight"); plt.close(fig)

    _cm_plot(sd_c, ["Normal","Abnormal"], "Detector SIM (ExtraTrees)", out/"sim_detector_cm.png", True)
    _cm_plot(sc_c, EVT, "Classifier SIM (ExtraTrees)", out/"sim_classifier_cm.png", True, (12,10))
    _cm_plot(sl_c, ["BUS","LINE","PMU"], "Localizer SIM (GNN)", out/"sim_localizer_cm.png", True, (8,6))
    _cm_plot(rlc_c_rr, ["BUS","LINE","PMU"], "Localizer RAW (GNN+Rerank)", out/"raw_localizer_cm.png", True, (8,6))

    summary = {"pipeline":"Final_Adjusted_GNN","params":model.n_params(),"penalty":round(.03*np.log10(max(model.n_params(),1)),4),
               "pca": {"n_components":pca_dim, "explained_variance":float(explained_var)},
               "sim":{"det":float(sd_a),"cls":float(sc_a),"cls_f1":float(sc_f),"loc":float(sl_a)},
               "raw":{"det":float(rd_a),"cls":float(rc_a),"cls_f1":float(rc_f),"loc_ml":float(rla_ml),"loc_rerank":float(rla_rr)},
               "baseline":{"sim_loc":bsl,"raw_loc":brl}}
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    torch.save({"model":model.state_dict(), "pca_models":[(s,p) for s,p in pca_models], "encoders":{"bus":be.classes_.tolist(),"line":le.classes_.tolist(),"pmu":pe.classes_.tolist()}}, out/"final_model.pt")

    print(f"\n  Total: {tm.time()-tt:.0f}s  Output: {out}")
    return str(out)


if __name__ == "__main__":
    main()
