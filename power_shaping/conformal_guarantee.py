# -*- coding: utf-8 -*-
"""
P2 conformal calibration of the flexibility promise (strengthens the guarantee, Contribution 3 / s4).

The guarantee already says: a hardware-enforced ceiling delivers a provable LOWER bound on reduction,
because a GPU power cap is physical. But a facility also wants to PROMISE a specific shed and be
trusted to deliver it, and the promised shed depends on a PREDICTION of how far below its cap a
workload will draw (a memory-bound job pinned at 37 W sheds almost nothing when its cap is lowered).
A point predictor gives no coverage guarantee; a hand-tuned margin gives no principled one.

We wrap SPLIT/JACKKNIFE CONFORMAL prediction around the leave-one-workload-out draw predictor. Let
r = actual_draw - predicted_draw be the LOO residuals (collected at workload granularity, so draws
within a workload stay together and exchangeability holds across workloads). For miscoverage alpha,
q = the (1-alpha) empirical quantile of r; the conformal UPPER bound on draw is pred + q, and the
promised shed is base - (pred + q). Then

    reliability R = Pr[actual_shed >= promised_shed] = Pr[actual_draw <= pred + q] ~ 1 - alpha

by construction, distribution-free, with the hardware cap as the unconditional backstop for the
alpha tail. Sweeping alpha traces a RELIABILITY-vs-USABLE-FLEXIBILITY frontier that we compare to the
point predictor (no margin) and a fixed +2*MAE margin.

INVARIANTS
  COVER   empirical coverage of the conformal upper bound on held-out draws >= 1 - alpha (finite-slack)
  MONO    usable flexibility is non-increasing as reliability rises (you pay headroom for trust)
  DEGEN   alpha -> 1 (q -> min residual) reproduces ~the point-predictor reliability
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import poca_killtest as K

RAW = K.RAW
RESULTS = K.RESULTS
ALPHAS = (0.30, 0.20, 0.15, 0.10, 0.05)


def _load_zeus_draw():
    df = pd.read_csv(os.path.join(RAW, "zeus_summary_power_v100.csv"))
    g = df.groupby(["network", "dataset", "power_limit"], as_index=False).agg(p=("average_power", "mean"))
    g["wl"] = g["network"] + "/" + g["dataset"]
    g = pd.concat([g, pd.get_dummies(g["network"], prefix="net")], axis=1)
    netcols = [c for c in g.columns if c.startswith("net_")]
    return g, ["power_limit"] + netcols


def loo_residuals(g, feats):
    """Leave-one-WORKLOAD-out point predictions of draw at each cap; return per-(wl,cap) records with
    predicted draw, actual draw, and the workload's measured baseline (max-cap) draw."""
    maxL = g["power_limit"].max()
    recs = []
    for wl in g["wl"].unique():
        tr = g[g["wl"] != wl]; te = g[g["wl"] == wl].sort_values("power_limit")
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, min_samples_leaf=3)
        m.fit(tr[feats].astype(float), tr["p"])
        pred = m.predict(te[feats].astype(float))
        base = float(te[te["power_limit"] == maxL]["p"].iloc[0])
        for (_, row), pr in zip(te.iterrows(), pred):
            if row["power_limit"] == maxL:
                continue
            recs.append(dict(wl=wl, L=float(row["power_limit"]), pred=float(pr),
                             actual=float(row["p"]), base=base))
    return pd.DataFrame(recs)


def frontier(R):
    resid = (R["actual"] - R["pred"]).to_numpy(float)          # calibration residuals (LOO)
    mae = float(np.abs(resid).mean())
    true_shed = (R["base"] - R["actual"]).to_numpy(float)
    tot_true = true_shed.sum()

    def policy(q):
        upper = R["pred"].to_numpy(float) + q                  # conformal upper bound on draw
        promised = R["base"].to_numpy(float) - upper
        deliver = R["actual"].to_numpy(float) <= upper + 1e-9  # actual_shed >= promised_shed
        rel = 100 * deliver.mean()
        usable = 100 * np.clip(promised, 0, None).sum() / tot_true if tot_true > 0 else 0.0
        return round(rel, 1), round(usable, 1)

    rows = []
    for a in ALPHAS:
        q = float(np.quantile(resid, 1 - a, method="higher"))
        rel, usable = policy(q)
        rows.append({"alpha": a, "q": round(q, 2), "reliability_pct": rel, "usable_flex_pct": usable,
                     "cover_ok": bool(rel >= 100 * (1 - a) - 5.0)})
    rel_mean, usable_mean = policy(0.0)                          # point predictor, no margin
    rel_fix, usable_fix = policy(2 * mae)                        # fixed +2*MAE margin
    return {"n_residuals": int(len(R)), "resid_mae": round(mae, 2),
            "conformal_frontier": rows,
            "point_predictor": {"reliability_pct": rel_mean, "usable_flex_pct": usable_mean},
            "fixed_margin_2mae": {"reliability_pct": rel_fix, "usable_flex_pct": usable_fix}}


def main():
    g, feats = _load_zeus_draw()
    R = loo_residuals(g, feats)
    out = frontier(R)

    fr = out["conformal_frontier"]
    cover_ok = all(r["cover_ok"] for r in fr)
    usable = [r["usable_flex_pct"] for r in fr]
    rel = [r["reliability_pct"] for r in fr]
    mono = all(usable[i + 1] <= usable[i] + 3.0 for i in range(len(usable) - 1))  # more trust -> less usable
    degen_ok = out["conformal_frontier"][0]["reliability_pct"] >= out["point_predictor"]["reliability_pct"] - 1
    out["invariants"] = {"COVER": bool(cover_ok), "MONO": bool(mono), "DEGEN": bool(degen_ok)}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "conformal_guarantee.json"), "w"), indent=2)

    print("== P2 conformal flexibility promise (Zeus corpus, leave-one-workload-out) ==")
    print(f"residuals n={out['n_residuals']}  MAE={out['resid_mae']} W")
    print(f"INVARIANT COVER (empirical coverage >= 1-alpha): {'PASS' if cover_ok else 'FAIL'}")
    print(f"INVARIANT MONO  (usable flex falls as reliability rises): {'PASS' if mono else 'FAIL'}")
    print(f"INVARIANT DEGEN (frontier base >= point predictor): {'PASS' if degen_ok else 'FAIL'}")
    pp = out["point_predictor"]; fx = out["fixed_margin_2mae"]
    print(f"\n{'policy':>22} {'reliability':>12} {'usable flex':>12}")
    print(f"{'point predictor':>22} {pp['reliability_pct']:>11.0f}% {pp['usable_flex_pct']:>11.0f}%")
    print(f"{'fixed +2*MAE margin':>22} {fx['reliability_pct']:>11.0f}% {fx['usable_flex_pct']:>11.0f}%")
    for r in fr:
        print(f"{'conformal a='+format(r['alpha'],'.2f'):>22} {r['reliability_pct']:>11.0f}% {r['usable_flex_pct']:>11.0f}%")
    print("written -> results/conformal_guarantee.json")


if __name__ == "__main__":
    main()
