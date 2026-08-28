# -*- coding: utf-8 -*-
"""
Validate the model against the REAL emerald field experiment (attacks the power-model
realism caveat). During the SRP grid-response experiment the 256-GPU cluster reduced total
power ~30% and measured a weighted service performance of ~0.91 (SRP_exp_performance.csv),
with per-workload performance recorded. We check that our allocator, asked to reduce the
SAME workload pool's power by the SAME amount, predicts a matching per-workload and mean
performance -- a direct model-vs-field-measurement test, not a fit.
"""
from __future__ import annotations
import os, json, re
import numpy as np
import pandas as pd
import poca_killtest as K

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
RESULTS = os.path.join(HERE, "results")


def real_experiment():
    perf = pd.read_csv(os.path.join(RAW, "SRP_exp_performance.csv"))
    perf = perf[perf["Model"].notna() & ~perf["Model"].str.contains("Weighted", na=False)].copy()
    perf["wl"] = perf["Model"].str.strip().str.replace(r"-[a-z0-9]+\s*$", "", regex=True)
    perf["perf"] = pd.to_numeric(perf["Performance"], errors="coerce")
    # map to the dvfs_sweep workload names (strip the run-id suffix)
    perf["base"] = perf["wl"].str.replace(r"\s+$", "", regex=True)
    pw = pd.read_csv(os.path.join(RAW, "SRP_total_power.csv"))
    p = pw["total"].to_numpy(float)
    base_kw = p[:10].mean() / 1000
    # Sustained dip = the LONGEST CONTIGUOUS window below 90% of baseline, median over that window.
    # (The earlier estimator took the 120 lowest samples ANYWHERE in the trace, which are not
    # contiguous and overstate the reduction; the sustained level must be an actual sustained window.)
    below = p < 0.9 * base_kw * 1000
    best_len = cur = best_end = 0
    for i, b in enumerate(below):
        cur = cur + 1 if b else 0
        if cur > best_len:
            best_len, best_end = cur, i + 1
    dip = p[best_end - best_len:best_end]
    sustained_kw = float(np.median(dip)) / 1000
    reduction = 1 - sustained_kw / base_kw
    return perf, base_kw, sustained_kw, reduction


def model_predict(pool_names, reduction):
    """Allocate power to the named workloads to achieve `reduction` of baseline cluster power
    (profiled greedy) and predict each workload's normalized throughput. Uses PRIORITY weights
    (inference > fine-tune > pretrain, from classify()), matching the operational priorities of the
    field experiment; so the test is whether the model reproduces the MAGNITUDE of per-workload
    flexing under those priorities, not that it rediscovers the priorities themselves."""
    meas = {m["name"]: m for m in K._load_measured()}
    pool = []
    for nm in pool_names:
        if nm not in meas:
            continue
        b = meas[nm]
        w = b["w"]                      # priority weight from classify(): infer>ft>pt
        xg = np.linspace(b["p_meas"].min(), b["p_meas"].max(), K.NGRID)
        qg = np.interp(xg, b["p_meas"], b["q_meas"])
        pool.append(dict(name=nm, cls=b["cls"], w=w, k=0, xg=xg, qg=qg,
                         cost_g=w * (1 - qg), pmin=float(xg[0]), pmax=float(xg[-1])))
    total = sum(p["pmax"] for p in pool)
    C = (1 - reduction) * total
    xs = K.b5_profiled(pool, C)
    pred = {p["name"]: K.q_at(p, xs[i]) for i, p in enumerate(pool)}
    return pred, pool


def main():
    perf, base_kw, sustained_kw, reduction = real_experiment()
    names = perf["base"].tolist()
    pred, pool = model_predict(names, reduction)

    rows = []
    for _, r in perf.iterrows():
        nm = r["base"]
        # match to a measured workload (some SRP names have instance suffixes like _32g/_48g)
        cand = [k for k in pred if k == nm] or [k for k in pred if nm.startswith(k) or k.startswith(nm)]
        pv = pred[cand[0]] if cand else np.nan
        rows.append(dict(workload=nm, real_perf=float(r["perf"]), model_perf=pv))
    df = pd.DataFrame(rows).dropna()
    real_mean = df["real_perf"].mean(); model_mean = df["model_perf"].mean()
    mae = float(np.abs(df["real_perf"] - df["model_perf"]).mean())

    out = {"base_kw": base_kw, "sustained_kw": sustained_kw, "cluster_reduction_pct": 100 * reduction,
           "real_mean_perf": real_mean, "model_mean_perf": model_mean,
           "per_workload_MAE": mae, "rows": rows}
    with open(os.path.join(RESULTS, "validate_real.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("== Validation vs REAL SRP field experiment (emerald 256-GPU) ==")
    print(f"real cluster power {base_kw:.0f} kW -> {sustained_kw:.0f} kW ({100*reduction:.0f}% reduction)")
    print(f"{'workload':>22} {'real perf':>10} {'model perf':>11}")
    for _, r in df.iterrows():
        print(f"{r['workload']:>22} {r['real_perf']:>10.3f} {r['model_perf']:>11.3f}")
    print(f"{'MEAN':>22} {real_mean:>10.3f} {model_mean:>11.3f}   (per-workload MAE {mae:.3f})")
    print("written -> results/validate_real.json")


if __name__ == "__main__":
    main()
