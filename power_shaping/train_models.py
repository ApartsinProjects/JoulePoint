# -*- coding: utf-8 -*-
"""
Train and PERSIST the power-shaping prediction models as reusable artifacts (models/*.joblib).

Two deployable models:

  1) elasticity_response  -- predict normalized throughput q(u) from a workload's relative
     power u and coarse workload descriptors. Trained on the pooled REAL power-cap / DVFS
     elasticity data (emerald LLM, Zeus vision/reco/speech/NLP, grid5000 CPU-HPC, and our own
     AWS A10G inference sweep if data/raw/own_aws_sweep.csv is present). This is the model a
     scheduler queries to decide how much power to shed from a workload.

  2) gpu_power            -- predict a workload's power draw (W) from GPU spec features +
     workload + batch/precision, trained on our own 7-GPU Modal measurements. Used for the
     unseen-hardware (S3) generalization and for flexibility/UFR bounds.

Each is saved with a metadata sidecar (features, training sources, CV/leave-one-out MAE, date
is intentionally omitted -- stamp externally). Re-run after the AWS sweep to fold in that data.
"""
from __future__ import annotations
import os, json, re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import joblib

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
MODELS = os.path.join(HERE, "models")
RESULTS = os.path.join(HERE, "results")
os.makedirs(MODELS, exist_ok=True)


# ---------- pooled elasticity table (relative power -> normalized throughput) ----------
def _norm(df_pt):
    """df_pt: columns [source, workload, family, power, thr]; return with relp + qnorm per workload."""
    out = []
    for (src, wl), g in df_pt.groupby(["source", "workload"]):
        g = g.sort_values("power")
        p = g["power"].to_numpy(float)
        thr = np.maximum.accumulate(g["thr"].to_numpy(float))
        if p.max() <= p.min() or thr.max() <= 0:
            continue
        relp = (p - p.min()) / (p.max() - p.min())
        q = thr / thr.max()
        for i in range(len(g)):
            out.append(dict(source=src, workload=wl, family=g["family"].iloc[0],
                            relp=float(relp[i]), q=float(q[i])))
    return pd.DataFrame(out)


def load_elasticity():
    rows = []
    # emerald (LLM)
    p = os.path.join(RAW, "dvfs_sweep.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(dict(source="emerald", workload=r["Workload"], family="llm",
                             power=r["power per GPU"], thr=r["normalized throughput"]))
    # zeus (vision/reco/speech/nlp)
    p = os.path.join(RAW, "zeus_summary_power_v100.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        fam = {"resnet50": "vision", "shufflenetv2": "vision", "bert_base_uncased": "nlp",
               "deepspeech2": "speech", "ncf": "recommendation"}
        for (net, ds), g in d.groupby(["network", "dataset"]):
            b = g["batch_size"].max(); g = g[g["batch_size"] == b]
            g = g.groupby("power_limit", as_index=False).agg(p=("average_power", "mean"), t=("time_per_epoch", "mean"))
            for _, r in g.iterrows():
                rows.append(dict(source="zeus", workload=f"{net}/{ds}", family=fam.get(net, "other"),
                                 power=r["p"], thr=1.0 / r["t"]))
    # grid5000 (CPU HPC)
    p = os.path.join(RAW, "grid5000_dvfs.csv")
    if os.path.exists(p):
        d = pd.read_csv(p, sep=r"\s+")
        d = d[(d["cluster"] == "chifflot") & (~d["bench"].str.startswith("Idle"))]
        for bench, g in d.groupby("bench"):
            gg = g.groupby("fmax").agg(power=("mean_power", "mean"), time=("time", "mean")).reset_index()
            for _, r in gg.iterrows():
                rows.append(dict(source="grid5000", workload=bench, family="hpc",
                                 power=r["power"], thr=1.0 / r["time"]))
    # own AWS A10G inference sweep (if present)
    p = os.path.join(RAW, "own_aws_sweep.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        d = d.groupby(["workload", "cap_w"], as_index=False).agg(power=("power_w", "mean"), thr=("throughput", "mean"))
        fammap = {"gemm_fp16": "compute", "gemm_fp32": "compute", "membw": "memory",
                  "embed_gather": "memory", "resnet50": "vision", "vit_b16": "vision",
                  "decode_like": "llm_decode"}
        for _, r in d.iterrows():
            rows.append(dict(source="aws_a10g", workload=r["workload"],
                             family=fammap.get(r["workload"], "other"), power=r["power"], thr=r["thr"]))
    return _norm(pd.DataFrame(rows))


def train_elasticity():
    tab = load_elasticity()
    tab = pd.concat([tab, pd.get_dummies(tab["family"], prefix="fam"),
                     pd.get_dummies(tab["source"], prefix="src")], axis=1)
    feats = ["relp"] + [c for c in tab.columns if c.startswith(("fam_", "src_"))]
    X, y = tab[feats].astype(float), tab["q"].astype(float)
    # leave-one-workload-out CV (the meaningful test)
    wls = tab["workload"].unique(); maes = []
    for wl in wls:
        tr = tab["workload"] != wl
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=5)
        m.fit(X[tr], y[tr])
        maes.append(float(np.mean(np.abs(m.predict(X[~tr]) - y[~tr]))))
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=400, learning_rate=0.05, min_samples_leaf=5)
    model.fit(X, y)
    joblib.dump({"model": model, "features": feats}, os.path.join(MODELS, "elasticity_response.joblib"))
    meta = {"model": "elasticity_response", "target": "normalized_throughput",
            "features": feats, "n_samples": int(len(tab)),
            "sources": sorted(tab["source"].unique().tolist()),
            "n_workloads": int(len(wls)),
            "leave_one_workload_out_MAE": float(np.mean(maes))}
    json.dump(meta, open(os.path.join(MODELS, "elasticity_response.meta.json"), "w"), indent=2)
    return meta


# ---------- GPU power model (own 7-GPU Modal data) ----------
SPEC_MAP = {"Tesla T4": "NVIDIA T4", "NVIDIA L4": "NVIDIA L4", "NVIDIA A10": "NVIDIA A10",
            "NVIDIA L40S": "NVIDIA L40S", "NVIDIA A100-SXM4-40GB": "NVIDIA A100-SXM4-40GB",
            "NVIDIA H100 80GB HBM3": "NVIDIA H100-SXM-80GB", "NVIDIA H200": "NVIDIA H200-SXM-141GB"}
FALLBACK_TDP = {"Tesla T4": 70, "NVIDIA L4": 72, "NVIDIA A10": 150, "NVIDIA L40S": 350,
                "NVIDIA A100-SXM4-40GB": 400, "NVIDIA H100 80GB HBM3": 700, "NVIDIA H200": 700}


def train_gpu_power():
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
    try:
        from accelerator_specs import SPECS
    except Exception:
        SPECS = {}
    rows = []
    for f in ["c1_bridge.json", "c4_training_grid.json"]:
        p = os.path.join(HERE, "..", "experiments", "results", f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for r in (d.get("rows", []) if isinstance(d, dict) else d):
            if r.get("status") == "ok" and r.get("mean_power_w"):
                rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    def feat(dev, key):
        s = SPECS.get(SPEC_MAP.get(dev, ""), {})
        return {"tdp_w": s.get("tdp_w") or FALLBACK_TDP.get(dev, 300), "mem_gb": s.get("mem_gb") or 24,
                "bw_gbs": s.get("bw_gbs") or 500, "year": s.get("year") or 2021}[key]
    for k in ["tdp_w", "mem_gb", "bw_gbs", "year"]:
        df[k] = df["device"].map(lambda x: feat(x, k))
    df = pd.concat([df, pd.get_dummies(df["load"], prefix="load"),
                    pd.get_dummies(df["precision"], prefix="prec")], axis=1)
    feats = ["batch", "tdp_w", "mem_gb", "bw_gbs", "year"] + \
            [c for c in df.columns if c.startswith(("load_", "prec_"))]
    df["logp"] = np.log(df["mean_power_w"].astype(float))
    # leave-one-GPU-out MAE
    maes = []
    for dev in df["device"].unique():
        tr = df["device"] != dev
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=5)
        m.fit(df[tr][feats].astype(float), df[tr]["logp"])
        pred = np.exp(m.predict(df[~tr][feats].astype(float)))
        maes.append(float(np.mean(np.abs(pred - df[~tr]["mean_power_w"].astype(float)))))
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=400, learning_rate=0.05, min_samples_leaf=5)
    model.fit(df[feats].astype(float), df["logp"])
    joblib.dump({"model": model, "features": feats, "target_transform": "log"},
                os.path.join(MODELS, "gpu_power.joblib"))
    meta = {"model": "gpu_power", "target": "mean_power_w (log-space)", "features": feats,
            "n_samples": int(len(df)), "n_gpus": int(df["device"].nunique()),
            "gpus": sorted(df["device"].unique().tolist()),
            "leave_one_gpu_out_MAE_w": float(np.mean(maes))}
    json.dump(meta, open(os.path.join(MODELS, "gpu_power.meta.json"), "w"), indent=2)
    return meta


def main():
    e = train_elasticity()
    g = train_gpu_power()
    print("== saved models -> power_shaping/models/ ==")
    print(f"  elasticity_response.joblib : {e['n_samples']} samples from {e['sources']}, "
          f"{e['n_workloads']} workloads, leave-one-workload-out MAE {e['leave_one_workload_out_MAE']:.4f}")
    if g:
        print(f"  gpu_power.joblib           : {g['n_samples']} runs, {g['n_gpus']} GPUs, "
              f"leave-one-GPU-out MAE {g['leave_one_gpu_out_MAE_w']:.1f} W")
    json.dump({"elasticity_response": e, "gpu_power": g},
              open(os.path.join(RESULTS, "train_models.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
