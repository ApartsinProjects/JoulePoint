# -*- coding: utf-8 -*-
"""
Reuse the OWN past Modal measurements (7 real GPUs) for the power-shaping project.

The past runs (experiments/results/c1_bridge.json inference, c4_training_grid.json training)
measured mean_power_w and throughput_sps for 5 workloads x 7 GPUs (T4, L4, A10, L40S,
A100-40GB, H100, H200) x batch {8,32,128} x precision {fp32,fp16} -- at each GPU's DEFAULT
power limit (no power-cap sweep). We reuse them two ways:

1) BATCHING as a power-shaping actuator (spec A5): lowering batch lowers power at a
   throughput cost. Per (GPU, workload, precision) the batch points form a real
   power->throughput curve; the kill-test decomposition then measures how much a
   batch-aware controller beats uniform, on OWN hardware.

2) UNSEEN-HARDWARE (S3) generalization: predict a workload's power/throughput on a GPU
   held out of training (leave-one-GPU-out), using GPU spec features. This is the
   cross-hardware test the public data could not provide.

INVARIANTS
  INV-OWN1  batching pool oracle <= uniform at every curtailment
  INV-OWN2  S3 model beats a global-mean baseline on held-out GPUs
"""
from __future__ import annotations
import os, json, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import poca_killtest as K

HERE = os.path.dirname(__file__)
EXP_RESULTS = os.path.join(HERE, "..", "experiments", "results")
RESULTS = K.RESULTS
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
try:
    from accelerator_specs import SPECS
except Exception:
    SPECS = {}

# map measured device name -> accelerator_specs key (nearest spec for features)
SPEC_MAP = {
    "Tesla T4": "NVIDIA T4", "NVIDIA L4": "NVIDIA L4", "NVIDIA A10": "NVIDIA A10",
    "NVIDIA L40S": "NVIDIA L40S", "NVIDIA A100-SXM4-40GB": "NVIDIA A100-SXM4-40GB",
    "NVIDIA H100 80GB HBM3": "NVIDIA H100-SXM-80GB", "NVIDIA H200": "NVIDIA H200-SXM-141GB",
}
# fallback TDPs if a spec row is missing
FALLBACK_TDP = {"Tesla T4": 70, "NVIDIA L4": 72, "NVIDIA A10": 150, "NVIDIA L40S": 350,
                "NVIDIA A100-SXM4-40GB": 400, "NVIDIA H100 80GB HBM3": 700, "NVIDIA H200": 700}


def load_rows():
    rows = []
    for f in ["c1_bridge.json", "c4_training_grid.json"]:
        p = os.path.join(EXP_RESULTS, f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for r in (d.get("rows", []) if isinstance(d, dict) else d):
            if r.get("status") == "ok" and r.get("mean_power_w") and r.get("throughput_sps"):
                rows.append({"device": r["device"], "load": r["load"], "precision": r["precision"],
                             "batch": r["batch"], "power": float(r["mean_power_w"]),
                             "thr": float(r["throughput_sps"]), "src": f[:2]})
    return pd.DataFrame(rows)


def gpu_features(dev):
    s = SPECS.get(SPEC_MAP.get(dev, ""), {})
    return {"tdp_w": s.get("tdp_w") or FALLBACK_TDP.get(dev, 300),
            "mem_gb": s.get("mem_gb") or 24, "bw_gbs": s.get("bw_gbs") or 500,
            "year": s.get("year") or 2021}


# ---------- 1) batching-elasticity kill test -------------------------------
def batching_pool(df, replicate=2):
    """Each (device, load, precision) -> a power->throughput curve over batch (Pareto/monotone
    upper envelope: best throughput achievable at or below each power level)."""
    pool = []
    for (dev, load, prec), g in df.groupby(["device", "load", "precision"]):
        g = g.sort_values("power")
        if g["power"].nunique() < 2:
            continue
        p = g["power"].to_numpy(float)
        thr = np.maximum.accumulate(g["thr"].to_numpy(float))     # best-so-far as power rises
        thr = thr / thr.max()
        for k in range(replicate):
            xg = np.linspace(p.min(), p.max(), K.NGRID)
            qg = np.interp(xg, p, thr)
            pool.append(dict(name=f"{dev}/{load}/{prec}", cls=dev, w=1.0, k=k, xg=xg, qg=qg,
                             cost_g=1.0 * (1 - qg), pmin=float(xg[0]), pmax=float(xg[-1])))
    return pool


def decompose(pool, fracs=(0.90, 0.85, 0.80, 0.75, 0.70)):
    tb = sum(w["pmax"] for w in pool)
    rows = []; ok = True
    for frac in fracs:
        C = frac * tb
        uni = K.pool_cost_power(pool, K.b1_uniform(pool, C))[0]
        ora = K.pool_cost_power(pool, K.b6_oracle(pool, C))[0]
        shared = K.pool_cost_power(pool, K.b5g_shared_elasticity(pool, C))[0]
        if ora > uni + 1e-6:
            ok = False
        den = uni - ora
        rows.append(dict(curt=round((1 - frac) * 100),
                         oracle_gain_vs_uniform_pct=100 * (uni - ora) / uni if uni > 1e-9 else 0.0,
                         per_workload_elasticity_gain_pct=100 * (shared - ora) / shared if shared > 1e-9 else 0.0))
    return rows, ok


# ---------- 2) unseen-hardware (S3) split ----------------------------------
def s3_unseen_gpu(df):
    d = df.copy()
    for k in ["tdp_w", "mem_gb", "bw_gbs", "year"]:
        d[k] = d["device"].map(lambda x: gpu_features(x)[k])
    d = pd.concat([d, pd.get_dummies(d["load"], prefix="load"),
                   pd.get_dummies(d["precision"], prefix="prec")], axis=1)
    feats = ["batch", "tdp_w", "mem_gb", "bw_gbs", "year"] + \
            [c for c in d.columns if c.startswith(("load_", "prec_"))]
    # predict log-power (spans 67..520 W) on a held-out GPU
    d["logp"] = np.log(d["power"])
    maes, base_maes = [], []
    for dev in d["device"].unique():
        tr, te = d[d["device"] != dev], d[d["device"] == dev]
        gb = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=5)
        gb.fit(tr[feats].astype(float), tr["logp"])
        pred = np.exp(gb.predict(te[feats].astype(float)))
        maes.append(float(np.mean(np.abs(pred - te["power"]))))
        base_maes.append(float(np.mean(np.abs(np.exp(tr["logp"].mean()) - te["power"]))))
    return {"mean_abs_err_w": float(np.mean(maes)),
            "global_mean_baseline_w": float(np.mean(base_maes)),
            "per_gpu_mae_w": {dev: round(m, 1) for dev, m in zip(d["device"].unique(), maes)}}


def main():
    df = load_rows()
    n_gpu = df["device"].nunique()
    pool = batching_pool(df)
    rows, inv1 = decompose(pool)
    s3 = s3_unseen_gpu(df)
    inv2 = s3["mean_abs_err_w"] < s3["global_mean_baseline_w"]

    out = {"n_measurements": len(df), "n_gpus": n_gpu,
           "gpus": sorted(df["device"].unique().tolist()),
           "batching_elasticity": {"n_jobs": len(pool) // 2, "rows": rows},
           "unseen_gpu_S3": s3,
           "invariants": {"batching_oracle_le_uniform": bool(inv1),
                          "S3_beats_global_mean": bool(inv2)}}
    with open(os.path.join(RESULTS, "own_measured.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"== OWN measured reuse ({len(df)} runs, {n_gpu} real GPUs incl H100/H200) ==")
    if not (inv1 and inv2):
        print("  invariants:", out["invariants"]); raise SystemExit("INVARIANT FAILURE")
    print("  invariants: pass\n")
    print("  1) batching as a power-shaping actuator (own hardware):")
    print(f"     {'curt%':>5} {'oracle↑vs uniform%':>18} {'per-workload elast%':>19}")
    for r in rows:
        print(f"     {r['curt']:>5} {r['oracle_gain_vs_uniform_pct']:>18.1f} {r['per_workload_elasticity_gain_pct']:>19.1f}")
    print(f"\n  2) unseen-hardware (S3) power prediction: MAE {s3['mean_abs_err_w']:.1f} W "
          f"vs global-mean baseline {s3['global_mean_baseline_w']:.1f} W")
    print("  written -> results/own_measured.json")


if __name__ == "__main__":
    main()
