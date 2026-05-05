"""P14: Agnostic Bus Localizer — Fresh ExtraTrees + V4 Zbus Diffusion + Re-ranker.

Trains NEW ExtraTrees models from scratch (no sklearn compat issues).
Target: 85%+ localization on SIM, maximize RAW performance.

Features: V2 per-PMU stats + V4 Zbus diffusion signatures + event-shape scores.
"""

from __future__ import annotations

import json, time as tm, warnings, re
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.data_factory.feature_extractor_v4 import extract_v4_features, ZbusDiffusionFeatures, extract_severity_vector
from src.helpers.paths import ROOT, ensure_dir

warnings.filterwarnings("ignore")

# Paths
FEAT_V2 = ROOT / "workbench/features/sgsma_generated/factory_features_v2.csv"
RAW_V2  = ROOT / "workbench/raw_current_eval/raw001_base_features_v2.csv"
FINAL_M = ROOT / "models/final_metrics.json"
OUT_DIR = ROOT / "workbench/agnostic_bus_loc"

OBS = (2,5,6,10,19,22,29,39)
PMU_NAMES = [f"BUS{b}" for b in OBS]
EVT = ["Normal","Fault","Line Outage","Gen Change","Load Change","Missing","Missing+Phys","Bad Data","Unknown"]

def _lt(x: str) -> str:
    s = str(x)
    if "BUS" in s: return "BUS"
    if "LINE" in s: return "LINE"
    if "PMU" in s: return "PMU"
    return "NONE"


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def prepare_features(v2_df: pd.DataFrame, add_v4: bool = True) -> pd.DataFrame:
    """Use original feature columns + add V4 diffusion features."""
    # Use the SAME feature columns as the original trained models
    original_fcols = json.loads((ROOT / "models/feature_columns.json").read_text())
    available = [c for c in original_fcols if c in v2_df.columns]

    # Add any V3/META columns that exist in the dataframe
    extra_cols = [c for c in v2_df.columns
                  if c not in available
                  and pd.api.types.is_numeric_dtype(v2_df[c])
                  and not c.startswith(("chunk_name", "sim_id", "window_start", "window_end"))
                  and "__label_" not in c
                  and not c.endswith("__Event__min")
                  and not c.endswith("__Event__max")]
    all_cols = available + sorted(extra_cols)

    X = v2_df[all_cols].fillna(v2_df[all_cols].median()).infer_objects(copy=False)
    X = X.replace([np.inf, -np.inf], 0)

    # Remove unstable features (same as original pipeline)
    UNSTABLE = ("freq_angle_mismatch",)
    for col in list(X.columns):
        if any(tok in col for tok in UNSTABLE):
            del X[col]

    # Handle remaining NaN
    X = X.fillna(X.median()).fillna(0)

    if add_v4:
        print("    Computing V4 extension features...")
        v4 = extract_v4_features(v2_df).fillna(0)
        X = pd.concat([X.reset_index(drop=True), v4.reset_index(drop=True)], axis=1)

    return X


# ═══════════════════════════════════════════════════════════════════════════
# LOCALIZER TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_localizers(X_train: pd.DataFrame, labels_train: pd.DataFrame, seed: int = 42) -> dict:
    """Train per-type and per-event typed localizers."""
    models = {}
    n_estimators_base = 650
    n_estimators_typed = 750

    for loc_type in ("BUS", "LINE", "PMU"):
        mask = labels_train["location_type"] == loc_type
        if mask.sum() < 5:
            continue
        y = labels_train.loc[mask, "location_label"]
        if y.nunique() < 2:
            continue
        pipe = ExtraTreesClassifier(
            n_estimators=n_estimators_base, max_features="sqrt",
            class_weight="balanced", random_state=seed + {"BUS":101,"LINE":202,"PMU":303}[loc_type],
            n_jobs=4, max_depth=None, min_samples_leaf=2)
        pipe.fit(X_train.loc[mask], y)
        models[loc_type] = pipe
        print(f"    {loc_type}: {mask.sum()} samples, {y.nunique()} classes")

    # Per-event typed localizers
    for phys_label in sorted(labels_train["physical_event_label"].unique()):
        if phys_label == 0:
            continue
        loc_type = "LINE" if phys_label == 2 else "BUS"
        mask = (labels_train["location_type"] == loc_type) & (labels_train["physical_event_label"] == phys_label)
        if mask.sum() < 5:
            continue
        y = labels_train.loc[mask, "location_label"]
        if y.nunique() < 2:
            continue
        key = f"{loc_type}:event{phys_label}"
        pipe = ExtraTreesClassifier(
            n_estimators=n_estimators_typed, max_features="sqrt",
            class_weight="balanced", random_state=seed + 2000 + phys_label,
            n_jobs=4, max_depth=None, min_samples_leaf=2)
        pipe.fit(X_train.loc[mask], y)
        models[key] = pipe
        print(f"    {key}: {mask.sum()} samples, {y.nunique()} classes")

    return models


def predict_localizer(X: pd.DataFrame, pred_event: np.ndarray, pred_physical: np.ndarray, models: dict) -> np.ndarray:
    """Predict locations using typed + per-event localizers."""
    out = np.full(len(X), "none", dtype=object)

    loc_type = np.full(len(pred_event), "BUS", dtype=object)
    loc_type[np.isin(pred_physical, [2]) | np.isin(pred_event, [2])] = "LINE"
    loc_type[np.isin(pred_event, [5, 7])] = "PMU"
    loc_type[pred_event == 0] = "NONE"

    # Per-event typed (highest priority)
    for phys_label in sorted(set(int(v) for v in pred_physical)):
        if phys_label == 0:
            continue
        current_type = "LINE" if phys_label == 2 else "BUS"
        key = f"{current_type}:event{phys_label}"
        model = models.get(key)
        if model is None:
            continue
        mask = (loc_type == current_type) & (pred_physical == phys_label)
        if mask.any():
            out[mask] = model.predict(X.loc[mask].fillna(0).values)

    # Generic typed (fallback)
    for current_type in ("BUS", "LINE", "PMU"):
        model = models.get(current_type)
        if model is None:
            continue
        mask = (loc_type == current_type) & (out == "none")
        if mask.any():
            out[mask] = model.predict(X.loc[mask].fillna(0).values)

    return out


def predict_localizer(X: pd.DataFrame, pred_event: np.ndarray, pred_physical: np.ndarray, models: dict) -> np.ndarray:
    """Predict locations using typed + per-event localizers."""
    out = np.full(len(X), "none", dtype=object)

    loc_type = np.full(len(pred_event), "BUS", dtype=object)
    loc_type[np.isin(pred_physical, [2]) | np.isin(pred_event, [2])] = "LINE"
    loc_type[np.isin(pred_event, [5, 7])] = "PMU"
    loc_type[pred_event == 0] = "NONE"

    # Per-event typed (highest priority)
    for phys_label in sorted(set(int(v) for v in pred_physical)):
        if phys_label == 0:
            continue
        current_type = "LINE" if phys_label == 2 else "BUS"
        key = f"{current_type}:event{phys_label}"
        model = models.get(key)
        if model is None:
            continue
        mask = (loc_type == current_type) & (pred_physical == phys_label)
        if mask.any():
            out[mask] = model.predict(X.loc[mask].fillna(0).values)

    # Generic typed (fallback)
    for current_type in ("BUS", "LINE", "PMU"):
        model = models.get(current_type)
        if model is None:
            continue


# ═══════════════════════════════════════════════════════════════════════════
# ZBUS RE-RANKER
# ═══════════════════════════════════════════════════════════════════════════

class ZbusReranker:
    def __init__(self):
        self.zbf = ZbusDiffusionFeatures()

    def rerank(self, ml_loc: str, event_type: int, severity: np.ndarray, valid_locs: dict) -> str:
        if event_type == 0 or ml_loc == "none":
            return "none"
        sev_n = severity / (severity.max() + 1e-9)

        def _c(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        candidates = []
        loc_type = _lt(ml_loc) if ml_loc != "none" else "BUS"

        candidate_list = valid_locs.get(loc_type, [])
        sig_dict = {"BUS": self.zbf.bus_sigs, "LINE": self.zbf.line_sigs, "PMU": self.zbf.pmu_sigs}.get(loc_type, {})

        for lbl in candidate_list:
            if lbl in sig_dict:
                cos = _c(sev_n, sig_dict[lbl])
                boost = 1.5 if lbl == ml_loc else 1.0
                candidates.append((lbl, cos * boost))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else ml_loc


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def _cm(cm, labels, title, path, norm=False, fs=(10,8)):
    if norm: cm = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=fs)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if norm else cm.max())
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels, xlabel="Pred", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    th = cm.max() / 2 if cm.max() > 0 else 0.5
    fmt = ".2f" if norm else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > th else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    out = ensure_dir(OUT_DIR)
    tt = tm.time()
    print("=" * 65)
    print("P14: Agnostic Bus Localizer — ExtraTrees + V4 Zbus + Re-ranker")
    print("=" * 65)

    # ── 1. Load data ──
    print("\n[1/5] Loading 5000 SIM scenarios...")
    t0 = tm.time()
    v2 = pd.read_csv(FEAT_V2)
    v3 = pd.read_csv(ROOT / "workbench/features/sgsma_generated/dynamic_features_v3.csv")
    # Merge V2+V3 on sim_id
    if "sim_id" in v3.columns:
        v2 = v2.merge(v3, on="sim_id", how="left", suffixes=("", "_v3dup"))
    print(f"    Loaded {len(v2)} rows x {len(v2.columns)} cols ({tm.time()-t0:.0f}s)")

    # Labels
    labels = v2[["sim_id", "event_label", "abnormal_label", "physical_event_label", "location_label", "location_type"]].copy()

    print("  Preparing features (V2 + V4)...")
    t0 = tm.time()
    X = prepare_features(v2)
    print(f"    {X.shape[1]} features ({tm.time()-t0:.0f}s)")

    # ── 2. Train/test split ──
    print("\n[2/5] Train/test split (70/30 stratified)...")
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.30, random_state=20260503, stratify=labels["event_label"])
    print(f"    Train: {len(tr)}  Test: {len(te)}")

    X_tr, X_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    lb_tr, lb_te = labels.iloc[tr].reset_index(drop=True), labels.iloc[te].reset_index(drop=True)

    # ── 3. Train localizers ──
    print("\n[3/5] Training typed localizers...")
    t0 = tm.time()
    models = train_localizers(X_tr, lb_tr, seed=20260503)
    print(f"    Trained {len(models)} localizers ({tm.time()-t0:.0f}s)")

    # ── 4. Evaluate SIM ──
    print("\n[4/5] Evaluating on SIM test set...")
    # Use ground truth event types for routing (evaluating localizer only)
    # In production, you'd use the detector+classifier predictions
    # For this evaluation, we use ground truth to isolate localizer performance
    pred_event_sim = lb_te["event_label"].values.astype(int)
    pred_physical_sim = lb_te["physical_event_label"].values.astype(int)

    et_loc = predict_localizer(X_te, pred_event_sim, pred_physical_sim, models)

    # Zbus re-ranking with improved severity
    severity_sim = extract_severity_vector(X_te)
    reranker = ZbusReranker()
    bus_labels = sorted(lb_tr.loc[lb_tr["location_type"]=="BUS", "location_label"].unique())
    line_labels = sorted(lb_tr.loc[lb_tr["location_type"]=="LINE", "location_label"].unique())
    pmu_labels = sorted(lb_tr.loc[lb_tr["location_type"]=="PMU", "location_label"].unique())

    reranked_sim = []
    for i in range(len(te)):
        reranked_sim.append(reranker.rerank(
            et_loc[i], pred_event_sim[i], severity_sim[i],
            {"BUS": bus_labels, "LINE": line_labels, "PMU": pmu_labels}))

    # SIM metrics
    true_loc = lb_te["location_label"].values
    loc_mask = np.array([_lt(l) != "NONE" for l in true_loc])

    sim_loc_et = accuracy_score(true_loc[loc_mask], et_loc[loc_mask]) if loc_mask.any() else 0
    sim_loc_rerank = accuracy_score(true_loc[loc_mask], np.array(reranked_sim)[loc_mask]) if loc_mask.any() else 0

    true_lt = np.array([_lt(l) for l in true_loc[loc_mask]])
    pred_lt_rerank = np.array([_lt(l) for l in np.array(reranked_sim)[loc_mask]])
    sim_loc_cm = confusion_matrix(true_lt, pred_lt_rerank, labels=["BUS", "LINE", "PMU"]) if loc_mask.any() else np.zeros((3,3), int)

    # Per-event breakdown
    per_event = {}
    for ev in sorted(set(pred_event_sim[loc_mask])):
        em = loc_mask & (pred_event_sim == ev)
        if em.any():
            per_event[EVT[ev]] = float(np.mean(np.array(reranked_sim)[em] == true_loc[em]))

    print(f"    Localizer (ML):     Acc = {sim_loc_et:.4f}")
    print(f"    Localizer (Rerank): Acc = {sim_loc_rerank:.4f}")
    for ev_name, acc in per_event.items():
        print(f"      {ev_name}: {acc:.4f}")

    # ── 5. Evaluate RAW ──
    print("\n[5/5] Evaluating on RAW001...")
    raw = pd.read_csv(RAW_V2)
    raw_v3 = pd.read_csv(ROOT / "workbench/raw_current_eval/raw001_dynamic_features_v3.csv")
    if "chunk_name" in raw_v3.columns:
        raw_v3 = raw_v3.rename(columns={"chunk_name": "sim_id"})
    # Merge on sim_id-like column
    if "sim_id" in raw_v3.columns and "chunk_name" not in raw.columns:
        raw = raw.merge(raw_v3, on=None, how="left") if False else raw  # skip merge for now
    raw_labels = raw[["true_event", "true_abnormal", "true_location"]].copy()
    raw_labels.columns = ["event_label", "abnormal_label", "location_label"]
    raw_labels["location_type"] = raw_labels["location_label"].apply(_lt)
    raw_labels["physical_event_label"] = raw_labels["event_label"]

    X_raw = prepare_features(raw, add_v4=True)
    # Ensure same columns as training
    missing_cols = [c for c in X.columns if c not in X_raw.columns]
    for c in missing_cols:
        X_raw[c] = 0.0
    X_raw = X_raw[X.columns]

    pred_event_raw = raw_labels["event_label"].values.astype(int)
    pred_physical_raw = raw_labels["physical_event_label"].values.astype(int)

    et_loc_raw = predict_localizer(X_raw, pred_event_raw, pred_physical_raw, models)

    severity_raw = extract_severity_vector(X_raw)
    reranked_raw = []
    for i in range(len(raw)):
        reranked_raw.append(reranker.rerank(
            et_loc_raw[i], pred_event_raw[i], severity_raw[i],
            {"BUS": bus_labels, "LINE": line_labels, "PMU": pmu_labels}))

    true_loc_raw = raw_labels["location_label"].values
    rmask = np.array([_lt(l) != "NONE" for l in true_loc_raw])

    raw_loc_et = accuracy_score(true_loc_raw[rmask], et_loc_raw[rmask]) if rmask.any() else 0
    raw_loc_rerank = accuracy_score(true_loc_raw[rmask], np.array(reranked_raw)[rmask]) if rmask.any() else 0

    raw_lt_true = np.array([_lt(l) for l in true_loc_raw[rmask]])
    raw_lt_pred = np.array([_lt(l) for l in np.array(reranked_raw)[rmask]])
    raw_loc_cm = confusion_matrix(raw_lt_true, raw_lt_pred, labels=["BUS", "LINE", "PMU"]) if rmask.any() else np.zeros((3,3), int)

    # Per-sample predictions
    raw_per_event = {}
    for ev in sorted(set(pred_event_raw[rmask])):
        em = rmask & (pred_event_raw == ev)
        if em.any():
            raw_per_event[EVT[ev]] = float(np.mean(np.array(reranked_raw)[em] == true_loc_raw[em]))

    print(f"    Localizer (ML):     Acc = {raw_loc_et:.4f}")
    print(f"    Localizer (Rerank): Acc = {raw_loc_rerank:.4f}")
    for ev_name, acc in raw_per_event.items():
        print(f"      {ev_name}: {acc:.4f}")

    # Baseline
    bl = json.loads(FINAL_M.read_text()) if FINAL_M.exists() else {}
    bsl = bl.get("selected", {}).get("sim_localizer_exact", 0.8524)
    brl = bl.get("selected", {}).get("raw_localizer_exact", 0.6667)

    # ── Plots ──
    print("\n  Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Localization comparison
    x_labels = ["SIM\nBaseline", "SIM\nNew ML", "SIM\nRerank", "RAW\nBaseline", "RAW\nNew ML", "RAW\nRerank"]
    vals = [bsl, sim_loc_et, sim_loc_rerank, brl, raw_loc_et, raw_loc_rerank]
    colors = ["tab:blue", "tab:cyan", "tab:green", "tab:blue", "tab:cyan", "tab:green"]
    axes[0,0].bar(x_labels, vals, color=colors)
    axes[0,0].set_title("Localization Accuracy"); axes[0,0].set_ylim(0, 1.1); axes[0,0].grid(alpha=.3, axis="y")
    for i, v in enumerate(vals): axes[0,0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=8)

    # SIM per-event
    if per_event:
        items = list(per_event.items())
        axes[0,1].bar([k[:8] for k,v in items], [v for k,v in items], color="tab:green")
        axes[0,1].set_title("SIM Loc per Event (Rerank)"); axes[0,1].set_ylim(0, 1.1); axes[0,1].grid(alpha=.3, axis="y")
        plt.setp(axes[0,1].get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # RAW per-event
    if raw_per_event:
        items = list(raw_per_event.items())
        axes[0,2].bar([k[:8] for k,v in items], [v for k,v in items], color="tab:orange")
        axes[0,2].set_title("RAW Loc per Event (Rerank)"); axes[0,2].set_ylim(0, 1.1); axes[0,2].grid(alpha=.3, axis="y")
        plt.setp(axes[0,2].get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # Feature importance (top BUS localizer)
    if "BUS" in models:
        imp = models["BUS"].named_steps["model"].feature_importances_
        top_n = 20
        top_idx = np.argsort(imp)[-top_n:]
        axes[1,0].barh(range(top_n), imp[top_idx][::-1])
        axes[1,0].set_title(f"Top {top_n} Features (BUS Localizer)"); axes[1,0].set_xlabel("Importance")

    # Confusion matrices
    _cm(sim_loc_cm, ["BUS","LINE","PMU"], "Localizer CM — SIM (Rerank)", out/"sim_localizer_cm.png", True, (8,6))
    _cm(raw_loc_cm, ["BUS","LINE","PMU"], "Localizer CM — RAW (Rerank)", out/"raw_localizer_cm.png", True, (8,6))

    # V4 feature importance
    v4_cols = [c for c in X.columns if c.startswith(("DIFF__", "GSP_", "SHAPE__", "VIC__"))]
    if v4_cols and "BUS" in models:
        all_cols = list(X.columns)
        imp_all = models["BUS"].named_steps["model"].feature_importances_
        v4_imp = [(c, imp_all[all_cols.index(c)]) for c in v4_cols if c in all_cols]
        v4_imp.sort(key=lambda x: x[1], reverse=True)
        top_v4 = v4_imp[:15]
        axes[1,1].barh(range(len(top_v4)), [v for _,v in top_v4][::-1])
        axes[1,1].set_yticks(range(len(top_v4)))
        axes[1,1].set_yticklabels([k[:35] for k,_ in top_v4][::-1], fontsize=6)
        axes[1,1].set_title("Top V4 Feature Importance")

    # Efficiency
    n_trees = sum(m.named_steps["model"].n_estimators for m in models.values() if hasattr(m, "named_steps"))
    penalty = 0.03 * np.log10(max(n_trees * 20, 1))  # ~20 splits per tree
    axes[1,2].bar(["Old\nExtraTrees", "New\n+V4"], [0.21, penalty], color=["tab:red", "tab:green"])
    axes[1,2].set_title("Efficiency Penalty"); axes[1,2].grid(alpha=.3, axis="y")
    for i, v in enumerate([0.21, penalty]):
        axes[1,2].text(i, v+0.002, f"{v:.4f}", ha="center", fontweight="bold")

    fig.suptitle("P14: Agnostic Bus Localizer Results", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out/"agnostic_loc_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save models
    import joblib
    joblib.dump(models, out / "agnostic_localizers.joblib")
    joblib.dump({"bus_labels": bus_labels, "line_labels": line_labels, "pmu_labels": pmu_labels}, out / "label_encodings.joblib")

    # Summary
    summary = {
        "pipeline": "P14_Agnostic_Bus_Localizer",
        "features": {"n_total": X.shape[1], "v4_features": len(v4_cols)},
        "models": {"n_localizers": len(models), "types": list(models.keys())},
        "sim_results": {"localizer_ml": float(sim_loc_et), "localizer_rerank": float(sim_loc_rerank), "per_event": per_event},
        "raw_results": {"localizer_ml": float(raw_loc_et), "localizer_rerank": float(raw_loc_rerank), "per_event": raw_per_event},
        "baseline": {"sim_loc": bsl, "raw_loc": brl},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n  Total time: {tm.time()-tt:.0f}s")
    print(f"  Output: {out}")
    return str(out)


if __name__ == "__main__":
    main()
