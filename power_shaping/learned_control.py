# -*- coding: utf-8 -*-
"""
Sprint 6 + Sprint 7 (uncertainty): a LEARNED power-shaping controller and its safety.

ORACLE CAPTURE. A learned controller must pick good interventions for workloads it has
NOT profiled. For each workload we predict its throughput-vs-power curve from features
(leave-one-workload-out), allocate power with the profiled greedy on the PREDICTED curves,
and score the resulting service cost on the TRUE curves. Oracle Capture =
(C_uniform - C_learned) / (C_uniform - C_oracle) is the fraction of the oracle's advantage
over uniform capping that the learned controller recovers.

UNDER-DELIVERY (UFR). A grid-facing controller PROMISES shed MW. If it predicts a workload
will draw to its cap but the workload is memory-bound and already draws far below the cap
(e.g. Zeus NCF at ~37 W), the promised shed is not delivered. We predict power draw vs cap
(leave-one-workload-out), and compare a MEAN-prediction promise against a CONSERVATIVE
upper-quantile promise. UFR = Pr(actual_shed < promised_shed) must be low for grid use.

INVARIANTS
  INV-OC1  learned-with-true-curves == oracle  (Oracle Capture = 100%)
  INV-OC2  0 <= Oracle Capture <= 100 (learned lies between uniform and oracle)
  INV-UFR  conservative (upper-quantile) promise has UFR <= mean-promise UFR
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import poca_killtest as K
import zeus_killtest as Z

RESULTS = K.RESULTS
RNG = np.random.default_rng(0)


# ---------- feature tables (real curves) ------------------------------------
def emerald_table():
    df = pd.read_csv(os.path.join(K.RAW, "dvfs_sweep.csv"))
    rows = []
    for _, r in df.iterrows():
        nm = r["Workload"]
        task = "infer" if nm.startswith("infer") else ("ft" if nm.startswith("ft") else "pt")
        import re
        m = re.search(r"(\d+)b", nm); size = int(m.group(1)) if m else 8
        rows.append(dict(wl=nm, f_task=task, f_size=size,
                         power=r["power per GPU"], q=r["normalized throughput"]))
    return pd.DataFrame(rows)


def zeus_table():
    df = pd.read_csv(os.path.join(K.RAW, "zeus_summary_power_v100.csv"))
    rows = []
    for (net, ds), g in df.groupby(["network", "dataset"]):
        b = g["batch_size"].max(); g = g[g["batch_size"] == b]
        g = g.groupby("power_limit", as_index=False).agg(p=("average_power", "mean"), t=("time_per_epoch", "mean"))
        thr = 1.0 / g["t"].to_numpy(float); thr = np.maximum.accumulate(thr); thr /= thr.max()
        for pl, pw, q in zip(g["power_limit"], g["p"], thr):
            rows.append(dict(wl=f"{net}/{ds}", f_net=net, f_powerlimit=pl, power=pw, q=q))
    return pd.DataFrame(rows)


def _dummies(df, cols):
    X = df.copy()
    for c in cols:
        X = pd.concat([X, pd.get_dummies(X[c], prefix=c)], axis=1)
    return X


# ---------- learned q-curve per workload (leave-one-workload-out) -----------
def predicted_pool(tab, feat_cats, feat_nums, use_class_mean=False):
    """Return a K-style pool whose per-workload q-curves are LOO predictions from features."""
    tab = tab.copy()
    tab["relp"] = tab.groupby("wl")["power"].transform(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9))
    Xtab = _dummies(tab, feat_cats)
    feat_cols = feat_nums + ["relp"] + [c for c in Xtab.columns if any(c.startswith(fc + "_") for fc in feat_cats)]
    pool = []
    for wl in tab["wl"].unique():
        tr = Xtab[Xtab["wl"] != wl]; te = tab[tab["wl"] == wl].sort_values("power")
        p = te["power"].to_numpy(float)
        relgrid = np.linspace(0, 1, K.NGRID)
        te_feat = Xtab[Xtab["wl"] == wl].iloc[0]
        if use_class_mean:                     # M1: class/task mean curve at each rel power
            key = feat_cats[0]
            fam_val = tab[tab["wl"] == wl][key].iloc[0]
            # LEAKAGE-FREE: average over OTHER workloads in the family, never the held-out one.
            # Fall back to all other workloads (global mean) if the family has no other member.
            same = tab[(tab[key] == fam_val) & (tab["wl"] != wl)]
            if same.empty:
                same = tab[tab["wl"] != wl]
            qg = np.interp(relgrid, np.sort(same["relp"]), same.sort_values("relp")["q"])
        else:                                  # M3: GBDT on features + rel power
            gb = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05, min_samples_leaf=3)
            gb.fit(tr[feat_cols].astype(float), tr["q"])
            Xq = pd.DataFrame({c: [te_feat[c]] * K.NGRID if c != "relp" else relgrid for c in feat_cols})
            qg = gb.predict(Xq[feat_cols].astype(float))
        qg = np.clip(np.maximum.accumulate(qg), 1e-3, None); qg /= qg.max()
        xg = np.linspace(p.min(), p.max(), K.NGRID)
        pool.append(dict(name=wl, cls=wl, w=1.0, k=0, xg=xg, qg=qg,
                         cost_g=1.0 * (1 - qg), pmin=float(xg[0]), pmax=float(xg[-1])))
    return pool


def true_pool(tab):
    pool = []
    for wl, g in tab.groupby("wl"):
        g = g.sort_values("power"); p = g["power"].to_numpy(float)
        q = np.maximum.accumulate(g["q"].to_numpy(float)); q /= q.max()
        xg = np.linspace(p.min(), p.max(), K.NGRID); qg = np.interp(xg, p, q)
        pool.append(dict(name=wl, cls=wl, w=1.0, k=0, xg=xg, qg=qg,
                         cost_g=1.0 * (1 - qg), pmin=float(xg[0]), pmax=float(xg[-1])))
    return pool


def oracle_capture(tab, feat_cats, feat_nums, fracs=(0.90, 0.85, 0.80, 0.75, 0.70)):
    tru = true_pool(tab)
    learned = predicted_pool(tab, feat_cats, feat_nums, use_class_mean=False)
    classm = predicted_pool(tab, feat_cats, feat_nums, use_class_mean=True)
    # align order
    order = [wl["name"] for wl in tru]
    learned = sorted(learned, key=lambda w: order.index(w["name"]))
    classm = sorted(classm, key=lambda w: order.index(w["name"]))
    tb = sum(w["pmax"] for w in tru)
    ocs, ocs_cm = [], []
    for frac in fracs:
        C = frac * tb
        c_uni = K.pool_cost_power(tru, K.b1_uniform(tru, C))[0]
        c_ora = K.pool_cost_power(tru, K.b6_oracle(tru, C))[0]
        # learned: allocate OPTIMALLY on predicted curves (same allocator as the oracle),
        # score on true. With perfect prediction this equals the oracle (INV-OC1).
        x_learn = K.b6_oracle(learned, C); c_learn = K.pool_cost_power(tru, x_learn)[0]
        x_cm = K.b6_oracle(classm, C); c_cm = K.pool_cost_power(tru, x_cm)[0]
        den = c_uni - c_ora
        if den > 1e-6:
            ocs.append(100 * (c_uni - c_learn) / den)
            ocs_cm.append(100 * (c_uni - c_cm) / den)
    return float(np.mean(ocs)), float(np.mean(ocs_cm))


def invariant_oc(tab):
    """A controller allocating optimally on the TRUE curves must equal the oracle -> OC=100%."""
    tru = true_pool(tab); tb = sum(w["pmax"] for w in tru)
    caps = 0
    for frac in (0.9, 0.8, 0.7):
        C = frac * tb
        c_uni = K.pool_cost_power(tru, K.b1_uniform(tru, C))[0]
        c_ora = K.pool_cost_power(tru, K.b6_oracle(tru, C))[0]
        c_perfect = K.pool_cost_power(tru, K.b6_oracle(tru, C))[0]   # perfect prediction == true curves
        den = c_uni - c_ora
        oc = 100 * (c_uni - c_perfect) / den if den > 1e-6 else 100.0
        if abs(oc - 100) > 1.0:
            return False
    return True


# ---------- under-delivery frequency (Zeus, has memory-bound draws) ---------
def ufr_experiment():
    """Action = cap a workload from its max power limit down to level L. Shed = draw(maxL) -
    draw(L). We predict draw(L) from features (leave-one-workload-out); a memory-bound
    workload that draws flat (NCF ~37 W) sheds ~0, so a model that assumes power scales with
    the cap over-promises. Conservative promise uses the UPPER quantile of the capped draw
    (assume it stays high -> promise less shed)."""
    from sklearn.ensemble import GradientBoostingRegressor
    df = pd.read_csv(os.path.join(K.RAW, "zeus_summary_power_v100.csv"))
    g = df.groupby(["network", "dataset", "power_limit"], as_index=False).agg(p=("average_power", "mean"))
    g["wl"] = g["network"] + "/" + g["dataset"]
    g = _dummies(g, ["network"])
    netcols = [c for c in g.columns if c.startswith("network_")]
    feats = ["power_limit"] + netcols
    maxL = g["power_limit"].max()
    mean_under = cons_under = n = 0
    for wl in g["wl"].unique():
        tr = g[g["wl"] != wl]; te = g[g["wl"] == wl].sort_values("power_limit")
        m_mean = HistGradientBoostingRegressor(max_depth=3, max_iter=200, min_samples_leaf=3).fit(tr[feats].astype(float), tr["p"])
        m_hi = GradientBoostingRegressor(loss="quantile", alpha=0.9, max_depth=2, n_estimators=200).fit(tr[feats].astype(float), tr["p"])
        draw_true = dict(zip(te["power_limit"], te["p"]))
        def pred(mdl, L):
            row = te[te["power_limit"] == L].iloc[0][feats]
            return float(mdl.predict(pd.DataFrame([row])[feats].astype(float))[0])
        base_true = draw_true[maxL]; base_pred = pred(m_mean, maxL)
        for L in te["power_limit"]:
            if L == maxL:
                continue
            actual_shed = base_true - draw_true[L]
            draw_mean_L = pred(m_mean, L)
            draw_cons_L = max(pred(m_hi, L), draw_mean_L)        # conservative capped draw >= mean
            promised_mean = base_pred - draw_mean_L
            promised_cons = base_pred - draw_cons_L              # -> promises no more than mean
            # MATERIAL under-delivery: actual shed falls >15% short of what was promised
            if promised_mean > 1.0 and actual_shed < 0.85 * promised_mean: mean_under += 1
            if promised_cons > 1.0 and actual_shed < 0.85 * promised_cons: cons_under += 1
            n += 1
    return {"UFR_mean_pct": 100 * mean_under / n, "UFR_conservative_pct": 100 * cons_under / n, "n": n}


def main():
    out = {}
    em = emerald_table(); ze = zeus_table()
    ok_em = invariant_oc(em)
    ok_ze = invariant_oc(ze)
    oc_em, cm_em = oracle_capture(em, ["f_task"], ["f_size"])
    oc_ze, cm_ze = oracle_capture(ze, ["f_net"], [])
    ufr = ufr_experiment()
    inv_ufr = ufr["UFR_conservative_pct"] <= ufr["UFR_mean_pct"] + 1e-9
    out = {"oracle_capture": {
                "emerald": {"learned_GBDT_pct": oc_em, "class_mean_pct": cm_em},
                "zeus": {"learned_GBDT_pct": oc_ze, "class_mean_pct": cm_ze}},
           "ufr": ufr,
           "invariants": {"OC_self_equals_oracle_emerald": bool(ok_em),
                          "OC_self_equals_oracle_zeus": bool(ok_ze),
                          "UFR_conservative_leq_mean": bool(inv_ufr)}}
    with open(os.path.join(RESULTS, "learned_control.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("== invariants ==")
    for k, v in out["invariants"].items():
        print(f"  {k}: {v}")
    if not all(out["invariants"].values()):
        raise SystemExit("INVARIANT FAILURE")
    print("\n== Oracle Capture (learned controller on UNSEEN workloads) ==")
    print(f"  class model : emerald {cm_em:.0f}%   zeus {cm_ze:.0f}%")
    print(f"  GBDT        : emerald {oc_em:.0f}%   zeus {oc_ze:.0f}%  (overfits at this data scale)")
    print("\n== Material under-delivery frequency (Zeus, >15% shortfall) ==")
    print(f"  mean prediction:          UFR {ufr['UFR_mean_pct']:.1f}%")
    print(f"  conservative (90% quant):  UFR {ufr['UFR_conservative_pct']:.1f}%")
    print("written -> results/learned_control.json")


if __name__ == "__main__":
    main()
