"""P11: V4 Localizer — Zbus Diffusion Signatures on 5000 scenarios.

Quick run: python pipelines/p11_v4_localizer.py
"""

from __future__ import annotations

import json, time as tm, warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GATConv, global_mean_pool

from src.data_factory.feature_extractor_v4 import extract_v4_features, ZbusDiffusionFeatures
from src.helpers.paths import ROOT, ensure_dir

warnings.filterwarnings("ignore")

FEAT_V2 = ROOT / "workbench/features/sgsma_generated/factory_features_v2.csv"
RAW_V2  = ROOT / "workbench/raw_current_eval/raw001_base_features_v2.csv"
FINAL_M = ROOT / "models/final_metrics.json"
OUT_DIR = ROOT / "workbench/v4_localizer"

OBS = (2,5,6,10,19,22,29,39)
PMU_NAMES = [f"BUS{b}" for b in OBS]
EVT = ["Normal","Fault","Line Outage","Gen Change","Load Change","Missing","Missing+Phys","Bad Data","Unknown"]

def _lt(x): return ("BUS" if "BUS" in str(x) else "LINE" if "LINE" in str(x) else "PMU" if "PMU" in str(x) else "NONE")

def _node_feats(df, pmus):
    feats=[df[[c for c in df.columns if c.startswith(f"{p}__") and pd.api.types.is_numeric_dtype(df[c])]].fillna(0).values for p in pmus]
    mc=min(f.shape[1] for f in feats)
    return np.nan_to_num(np.stack([f[:,:mc] for f in feats],1).astype(np.float32),0).clip(-100,100)

class GNN(nn.Module):
    def __init__(self, nd, v4d, nb,nl,np,h=96):
        super().__init__()
        self.proj=nn.Linear(nd,h)
        self.gat1=GATConv(h,h*2,heads=2)
        self.gat2=GATConv(h*4,128,heads=1)
        self.v4p=nn.Linear(v4d,32)
        f=128+32
        self.det=nn.Sequential(nn.Linear(f,64),nn.ReLU(),nn.Linear(64,1))
        self.cls=nn.Sequential(nn.Linear(f,64),nn.ReLU(),nn.Linear(64,9))
        self.bus=nn.Sequential(nn.Linear(f,64),nn.ReLU(),nn.Linear(64,nb))
        self.line=nn.Sequential(nn.Linear(f,64),nn.ReLU(),nn.Linear(64,nl))
        self.pmu=nn.Sequential(nn.Linear(f,64),nn.ReLU(),nn.Linear(64,np))
        self.bl=[f"BUS{i+1}" for i in range(nb)]; self.ll=[]; self.pl=[]

    def forward(self,nx,ei,batch,v4):
        x=F.relu(self.proj(nx))
        x=F.relu(self.gat1(x,ei)); x=F.dropout(x,0.1,self.training)
        x=self.gat2(x,ei)
        ge=global_mean_pool(x,batch)
        f=F.relu(torch.cat([ge,self.v4p(v4)],1))
        return self.det(f).squeeze(-1),self.cls(f),self.bus(f),self.line(f),self.pmu(f)

    @torch.no_grad()
    def predict(self,nx,ei,batch,v4):
        self.eval()
        d,c,b,l,p=self.forward(nx,ei,batch,v4)
        dp=(torch.sigmoid(d)>.5).long(); cp=c.argmax(1)
        loc=np.full(len(cp),"none",dtype=object)
        for i in range(len(cp)):
            if dp[i]==0: continue
            if cp[i] in(5,7): loc[i]=self.pl[p.argmax(1)[i].item()] if p.argmax(1)[i]<len(self.pl) else "none"
            elif cp[i]==2: loc[i]=self.ll[l.argmax(1)[i].item()] if l.argmax(1)[i]<len(self.ll) else self.bl[b.argmax(1)[i].item()]
            else: loc[i]=self.bl[b.argmax(1)[i].item()]
        return {"detection":dp.numpy(),"classification":cp.numpy(),"localization":loc}

    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _cm_plot(cm,labels,title,path,norm=False,fs=(10,8)):
    if norm: cm=cm.astype(float)/(cm.sum(1,keepdims=True)+1e-9)
    fig,ax=plt.subplots(figsize=fs)
    ax.imshow(cm,cmap="Blues",vmin=0,vmax=1 if norm else cm.max())
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=labels,yticklabels=labels,xlabel="Predicted",ylabel="True",title=title)
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    th=cm.max()/2 if cm.max()>0 else .5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,format(cm[i,j],".2f"if norm else"d"),ha="center",va="center",fontsize=7,color="white"if cm[i,j]>th else"black")
    fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)


def main():
    out=ensure_dir(OUT_DIR); dev="cpu"; tt=tm.time()
    print("="*60)
    print("V4 Localizer — Zbus Diffusion Signatures on 5000 scenarios")
    print("="*60)

    # Data
    print("\n[1/4] Loading 5000 SIM scenarios..."); t0=tm.time()
    v2=pd.read_csv(FEAT_V2)
    print(f"  {len(v2)} rows, {v2.shape[1]} cols ({tm.time()-t0:.0f}s)")
    lb=v2[["sim_id","event_label","abnormal_label","physical_event_label","location_label","location_type"]]
    nf=_node_feats(v2,PMU_NAMES)
    print(f"  Node features: {nf.shape}")

    print("  Computing V4 features..."); t0=tm.time()
    v4f=extract_v4_features(v2).fillna(0).values.astype(np.float32)
    print(f"  V4: {v4f.shape} ({tm.time()-t0:.0f}s)")

    # Encode
    ce=LabelEncoder().fit(lb["event_label"])
    be=LabelEncoder(); be.fit(lb.loc[lb["location_type"]=="BUS","location_label"])
    le=LabelEncoder(); le.fit(lb.loc[lb["location_type"]=="LINE","location_label"]) if (lb["location_type"]=="LINE").any() else le.fit(["LINE1-2"])
    pe=LabelEncoder(); pe.fit(lb.loc[lb["location_type"]=="PMU","location_label"]) if (lb["location_type"]=="PMU").any() else pe.fit(["PMU2"])

    # Split
    idx=np.arange(len(nf))
    tr,te=train_test_split(idx,test_size=.30,random_state=20260503,stratify=lb["event_label"])
    print(f"  Train:{len(tr)} Test:{len(te)}")

    def _enc(enc,idx_arr,typ):
        a=np.full(len(idx_arr),-1,dtype=int)
        m=lb.iloc[idx_arr]["location_type"]==typ
        a[m.values]=enc.transform(lb.iloc[idx_arr].loc[m.values,"location_label"])
        return a

    # Model
    print("\n[2/4] Building GNN + V4 fusion...")
    n_pmu=len(OBS)
    ei=torch.tensor([[i,j]for i in range(n_pmu)for j in range(n_pmu)if i!=j],dtype=torch.long).t().contiguous()
    
    model=GNN(nf.shape[2],v4f.shape[1],len(be.classes_),len(le.classes_),len(pe.classes_)).to(dev)
    model.ll=le.classes_.tolist(); model.pl=pe.classes_.tolist()
    print(f"  Params: {model.n_params():,}  Penalty: {0.03*np.log10(max(model.n_params(),1)):.4f}")

    # Prepare tensors
    Nt,Nv=len(tr),len(te)
    nnodes=len(OBS)
    nt=torch.tensor(nf[tr].reshape(-1,nf.shape[2]),dtype=torch.float).to(dev)
    bt=torch.arange(Nt,device=dev).repeat_interleave(nnodes)
    et=torch.cat([ei+i*nnodes for i in range(Nt)],1).to(dev)
    v4t=torch.tensor(v4f[tr],dtype=torch.float).to(dev)

    nv=torch.tensor(nf[te].reshape(-1,nf.shape[2]),dtype=torch.float).to(dev)
    bv=torch.arange(Nv,device=dev).repeat_interleave(nnodes)
    ev=torch.cat([ei+i*nnodes for i in range(Nv)],1).to(dev)
    v4v=torch.tensor(v4f[te],dtype=torch.float).to(dev)

    dt_tr=torch.tensor(lb.iloc[tr]["abnormal_label"].values.astype(np.float32)).to(dev)
    ct_tr=torch.tensor(ce.transform(lb.iloc[tr]["event_label"]),dtype=torch.long).to(dev)
    bu_tr=torch.tensor(_enc(be,tr,"BUS"),dtype=torch.long).to(dev)
    li_tr=torch.tensor(_enc(le,tr,"LINE"),dtype=torch.long).to(dev)
    pm_tr=torch.tensor(_enc(pe,tr,"PMU"),dtype=torch.long).to(dev)

    # Train
    print("\n[3/4] Training...")
    opt=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=120)
    lcrit=nn.CrossEntropyLoss(ignore_index=-1)
    hist={"det":[],"cls":[],"loc":[],"loss":[]}
    model.train()
    for ep in range(120):
        opt.zero_grad()
        dl,cl,bl,ll,pl=model(nt,et,bt,v4t)
        ld=F.binary_cross_entropy_with_logits(dl,dt_tr)
        lc=F.cross_entropy(cl,ct_tr)
        lb_=lcrit(bl,bu_tr); ll_=lcrit(ll,li_tr); lp_=lcrit(pl,pm_tr)
        loss=ld+.5*lc+lb_+ll_+lp_
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step(); sch.step()
        with torch.no_grad():
            da=((torch.sigmoid(dl)>=.5).long()==dt_tr.long()).float().mean().item()
            ca=(cl.argmax(1)==ct_tr).float().mean().item()
            m=(bu_tr>=0); la=((bl[m].argmax(1)==bu_tr[m]).float().mean().item() if m.any() else 0+\
                ((ll[li_tr>=0].argmax(1)==li_tr[li_tr>=0]).float().mean().item() if (li_tr>=0).any() else 0)+\
                ((pl[pm_tr>=0].argmax(1)==pm_tr[pm_tr>=0]).float().mean().item() if (pm_tr>=0).any() else 0))/3
        hist["det"].append(da); hist["cls"].append(ca); hist["loc"].append(la); hist["loss"].append(loss.item())
        if (ep+1)%30==0: print(f"  E{ep+1:3d} det={da:.3f} cls={ca:.3f} loc={la:.3f}")

    # Evaluate SIM
    print("\n[4/4] Evaluating...")
    sp=model.predict(nv,ev,bv,v4v)
    sd_a=accuracy_score(lb.iloc[te]["abnormal_label"].astype(int),sp["detection"])
    sd_c=confusion_matrix(lb.iloc[te]["abnormal_label"].astype(int),sp["detection"],labels=[0,1])
    sc_a=accuracy_score(ce.transform(lb.iloc[te]["event_label"]),sp["classification"])
    sc_f=f1_score(ce.transform(lb.iloc[te]["event_label"]),sp["classification"],average="macro",zero_division=0)
    sc_c=confusion_matrix(ce.transform(lb.iloc[te]["event_label"]),sp["classification"],labels=list(range(9)))

    tl=lb.iloc[te]["location_label"].values
    lm=np.array([_lt(l)!="NONE" for l in tl])
    sl_a=float(np.mean(sp["localization"][lm]==tl[lm])) if lm.any() else 0
    sl_c=confusion_matrix(np.array([_lt(l)for l in tl[lm]]),np.array([_lt(l)for l in sp["localization"][lm]]),labels=["BUS","LINE","PMU"]) if lm.any() else np.zeros((3,3),int)
    print(f"  SIM: Det={sd_a:.4f} Cls={sc_a:.4f}(F1={sc_f:.4f}) Loc={sl_a:.4f}")

    # RAW
    raw=pd.read_csv(RAW_V2)
    rnf=_node_feats(raw,PMU_NAMES)
    rv4=extract_v4_features(raw).fillna(0).values.astype(np.float32)
    Nr=len(rnf)
    nr=torch.tensor(rnf.reshape(-1,rnf.shape[2]),dtype=torch.float).to(dev)
    br_r=torch.arange(Nr,device=dev).repeat_interleave(nnodes)
    er=torch.cat([ei+i*nnodes for i in range(Nr)],1).to(dev)
    v4r=torch.tensor(rv4,dtype=torch.float).to(dev)
    rp=model.predict(nr,er,br_r,v4r)

    # Re-rank
    zbf=ZbusDiffusionFeatures()
    sev=np.zeros((Nr,8)); 
    for i in range(Nr): sev[i]=np.abs(rnf[i]).max(1)
    sev=sev/(sev.max(1,keepdims=True)+1e-9)
    rr=[]
    for i in range(Nr):
        e=rp["classification"][i]; l=rp["localization"][i]; s=sev[i]
        if e==0 or l=="none": rr.append("none"); continue
        def _c(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
        cs=[]
        if e in(5,7):
            for pl_ in pe.classes_:
                if pl_ in zbf.pmu_sigs: cs.append((pl_,_c(sev[i],zbf.pmu_sigs[pl_])*(.7 if pl_==l else 1)))
        elif e==2:
            for ll_ in le.classes_:
                if ll_ in zbf.line_sigs: cs.append((ll_,_c(sev[i],zbf.line_sigs[ll_])*(.7 if ll_==l else 1)))
        else:
            for bl_ in be.classes_:
                if bl_ in zbf.bus_sigs: cs.append((bl_,_c(sev[i],zbf.bus_sigs[bl_])*(.5+.5 if bl_==l else 1)))
        cs.sort(key=lambda x:x[1],reverse=True)
        rr.append(cs[0][0] if cs else l)
    rp["loc_rerank"]=np.array(rr)

    ra=raw["true_abnormal"].values.astype(int) if "true_abnormal" in raw.columns else np.ones(Nr,int)
    re=raw["true_event"].values.astype(int)
    rl_=raw["true_location"].values
    rd_a=accuracy_score(ra,rp["detection"]); rd_c=confusion_matrix(ra,rp["detection"],labels=[0,1])
    rc_a=accuracy_score(re,rp["classification"]); rc_f=f1_score(re,rp["classification"],average="macro",zero_division=0)
    rc_c=confusion_matrix(re,rp["classification"],labels=list(range(9)))
    rlm=np.array([_lt(l)!="NONE" for l in rl_])
    rla_ml=float(np.mean(rp["localization"][rlm]==rl_[rlm])) if rlm.any() else 0
    rla_rr=float(np.mean(rp["loc_rerank"][rlm]==rl_[rlm])) if rlm.any() else 0
    rlc_c_ml=confusion_matrix(np.array([_lt(l)for l in rl_[rlm]]),np.array([_lt(l)for l in rp["localization"][rlm]]),labels=["BUS","LINE","PMU"]) if rlm.any() else np.zeros((3,3),int)
    rlc_c_rr=confusion_matrix(np.array([_lt(l)for l in rl_[rlm]]),np.array([_lt(l)for l in rp["loc_rerank"][rlm]]),labels=["BUS","LINE","PMU"]) if rlm.any() else np.zeros((3,3),int)
    print(f"  RAW: Det={rd_a:.4f} Cls={rc_a:.4f}(F1={rc_f:.4f}) Loc_ML={rla_ml:.4f} Loc_Rerank={rla_rr:.4f}")

    # Baseline
    bl=json.loads(FINAL_M.read_text()) if FINAL_M.exists() else {}
    bs=bl.get("selected",{})
    bsl=bs.get("sim_localizer_exact",.8524); brl=bs.get("raw_localizer_exact",.6667)

    # Plots
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    axes[0,0].plot(hist["det"]); axes[0,0].set_title("Detection Acc"); axes[0,0].set_ylim(0,1.05); axes[0,0].grid(alpha=.3)
    axes[0,1].plot(hist["cls"]); axes[0,1].set_title("Classification Acc"); axes[0,1].set_ylim(0,1.05); axes[0,1].grid(alpha=.3)
    axes[1,0].plot(hist["loc"]); axes[1,0].set_title("Localization Acc (Train)"); axes[1,0].set_ylim(0,1.05); axes[1,0].grid(alpha=.3)
    bars=[bsl,sl_a,brl,rla_rr]
    axes[1,1].bar(["SIM\nBase","SIM\nV4","RAW\nBase","RAW\nV4+R"],bars,color=["tab:blue","tab:green","tab:blue","tab:orange"])
    axes[1,1].set_title("Localization"); axes[1,1].set_ylim(0,1.1); axes[1,1].grid(alpha=.3,axis="y")
    for i,v in enumerate(bars): axes[1,1].text(i,v+.02,f"{v:.3f}",ha="center",fontweight="bold")
    fig.suptitle("V4 Localizer Results",fontweight="bold"); fig.tight_layout()
    fig.savefig(out/"v4_results.png",dpi=150,bbox_inches="tight"); plt.close(fig)

    _cm_plot(sd_c,["Normal","Abnormal"],"Detector SIM",out/"sim_detector_cm.png",True)
    _cm_plot(sc_c,EVT,"Classifier SIM",out/"sim_classifier_cm.png",True,(12,10))
    _cm_plot(sl_c,["BUS","LINE","PMU"],"Localizer SIM",out/"sim_localizer_cm.png",True,(8,6))
    _cm_plot(rd_c,["Normal","Abnormal"],"Detector RAW",out/"raw_detector_cm.png",True)
    _cm_plot(rc_c,EVT,"Classifier RAW",out/"raw_classifier_cm.png",True,(12,10))
    _cm_plot(rlc_c_rr,["BUS","LINE","PMU"],"Localizer RAW (Rerank)",out/"raw_localizer_cm.png",True,(8,6))

    summary={"pipeline":"V4_Zbus_Diffusion","params":model.n_params(),"penalty":round(.03*np.log10(max(model.n_params(),1)),4),
             "sim":{"det":float(sd_a),"cls":float(sc_a),"cls_f1":float(sc_f),"loc":float(sl_a)},
             "raw":{"det":float(rd_a),"cls":float(rc_a),"cls_f1":float(rc_f),"loc_ml":float(rla_ml),"loc_rerank":float(rla_rr)},
             "baseline":{"sim_loc":bsl,"raw_loc":brl}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    torch.save(model.state_dict(),out/"v4_model.pt")

    print(f"\n  Total: {tm.time()-tt:.0f}s")
    print(f"  Output: {out}")
    return str(out)

if __name__=="__main__": main()
