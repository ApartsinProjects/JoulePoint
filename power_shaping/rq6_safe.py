# -*- coding: utf-8 -*-
"""
RQ6 resolved: safe power-flexibility promises via an observation-grounded bound.

Investigation (see below / diagnose in learned_control) showed material under-delivery is
NOT random noise: it is entirely the workloads that draw far below the cap (NCF 0 W range,
shufflenet 52 W, deepspeech 81 W). A feature-only model trained on power-scaling workloads
over-promises for these, and a wider statistical interval on the SAME model does not help
because the failure is out-of-distribution, not in-distribution variance.

The fix is physical, not statistical: a workload can shed at most (its current power draw -
the cap). The scheduler already observes current power (one NVML reading -- not a profiling
sweep), so the safe promised shed is bounded by  max(0, observed_draw_full - cap_L). We
compare four promise policies and confirm the observation-bounded policy drives under-delivery
to ~0 while still promising most of the real flexibility (not trivially zero).

Policies:
  feat_mean   : promise = pred_draw(maxL) - pred_draw(L)                 (feature-only mean)
  feat_quant  : promise = pred_draw(maxL) - pred_draw_hi(L)              (feature-only 90% quantile)
  obs_bound   : promise = margin * max(0, observed_draw_full - L)        (observation-grounded)
  hybrid      : promise = min(feat_mean, max(0, observed_draw_full - L)) (model, capped by physics)

Metrics: material under-delivery frequency (UFR, actual shed < 0.85*promised) and usable
flexibility retained (sum promised / sum true achievable shed).

INVARIANTS
  INV1  obs_bound UFR <= feat_mean UFR       (the fix cannot make under-delivery worse)
  INV2  obs_bound usable-flexibility > 0.5   (not trivially promising ~zero)
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
import poca_killtest as K

RESULTS = K.RESULTS
MARGIN = 0.90          # 10% safety margin on the observed physical ceiling


def load_zeus_draw():
    d = pd.read_csv(os.path.join(K.RAW, "zeus_summary_power_v100.csv"))
    g = d.groupby(["network", "dataset", "power_limit"], as_index=False).agg(p=("average_power", "mean"))
    g["wl"] = g["network"] + "/" + g["dataset"]
    g = pd.concat([g, pd.get_dummies(g["network"], prefix="net")], axis=1)
    return g


def run():
    g = load_zeus_draw()
    feats = ["power_limit"] + [c for c in g.columns if c.startswith("net_")]
    maxL = g["power_limit"].max()
    stats = {p: {"under": 0, "n": 0, "promised": 0.0, "achievable": 0.0}
             for p in ["feat_mean", "feat_quant", "obs_bound", "hybrid"]}
    per_wl = {}

    for wl in g["wl"].unique():
        tr = g[g["wl"] != wl]; te = g[g["wl"] == wl].sort_values("power_limit")
        m_mean = HistGradientBoostingRegressor(max_depth=3, max_iter=200, min_samples_leaf=3).fit(
            tr[feats].astype(float), tr["p"])
        m_hi = GradientBoostingRegressor(loss="quantile", alpha=0.9, max_depth=2, n_estimators=200).fit(
            tr[feats].astype(float), tr["p"])
        draw = dict(zip(te["power_limit"], te["p"]))
        obs_full = draw[maxL]                       # the ONE runtime observation (current draw)
        pred_full = float(m_mean.predict(te[te["power_limit"] == maxL][feats].astype(float))[0])
        def pm(L): return float(m_mean.predict(te[te["power_limit"] == L][feats].astype(float))[0])
        def ph(L): return max(float(m_hi.predict(te[te["power_limit"] == L][feats].astype(float))[0]), pm(L))
        wl_under = {p: 0 for p in stats}
        for L in te["power_limit"]:
            if L == maxL:
                continue
            actual = draw[maxL] - draw[L]           # true achievable shed
            promises = {
                "feat_mean":  pred_full - pm(L),
                "feat_quant": pred_full - ph(L),
                "obs_bound":  MARGIN * max(0.0, obs_full - L),
                "hybrid":     min(pred_full - pm(L), max(0.0, obs_full - L)),
            }
            for p, pr in promises.items():
                pr = max(0.0, pr)
                stats[p]["n"] += 1
                stats[p]["promised"] += pr
                stats[p]["achievable"] += max(0.0, actual)
                if pr > 1.0 and actual < 0.85 * pr:
                    stats[p]["under"] += 1; wl_under[p] += 1
        per_wl[wl] = {"range_w": float(draw[maxL] - draw[min(draw)]), "under": wl_under}

    out = {"policies": {}, "per_workload": per_wl, "margin": MARGIN}
    for p, s in stats.items():
        out["policies"][p] = {
            "UFR_pct": 100 * s["under"] / s["n"],
            "usable_flexibility_frac": s["promised"] / max(1e-9, s["achievable"]),
        }
    # margin sweep for the calibration frontier (Figure 10): compliance vs usable flexibility
    frontier = []
    for mg in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
        under = n = prom = ach = 0
        for wl in g["wl"].unique():
            tr = g[g["wl"] != wl]; te = g[g["wl"] == wl].sort_values("power_limit")
            draw = dict(zip(te["power_limit"], te["p"])); obs_full = draw[maxL]
            for L in te["power_limit"]:
                if L == maxL:
                    continue
                actual = draw[maxL] - draw[L]; pr = max(0.0, mg * max(0.0, obs_full - L))
                n += 1; prom += pr; ach += max(0.0, actual)
                if pr > 1.0 and actual < 0.85 * pr:
                    under += 1
        frontier.append({"margin": mg, "compliance_pct": 100 * (1 - under / n),
                         "usable_flex_frac": prom / max(1e-9, ach)})
    out["frontier_obs"] = frontier
    out["feat_mean_point"] = {"compliance_pct": 100 - out["policies"]["feat_mean"]["UFR_pct"],
                              "usable_flex_frac": out["policies"]["feat_mean"]["usable_flexibility_frac"]}
    inv1 = out["policies"]["obs_bound"]["UFR_pct"] <= out["policies"]["feat_mean"]["UFR_pct"] + 1e-9
    inv2 = out["policies"]["obs_bound"]["usable_flexibility_frac"] > 0.5
    out["invariants"] = {"obs_UFR_le_feat": bool(inv1), "obs_usable_gt_half": bool(inv2)}
    with open(os.path.join(RESULTS, "rq6_safe.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("== RQ6: safe flexibility promises (Zeus, has the OOD memory-bound case) ==")
    print(f"{'policy':>12} {'UFR%':>7} {'usable flex':>12}")
    for p in ["feat_mean", "feat_quant", "obs_bound", "hybrid"]:
        pp = out["policies"][p]
        print(f"{p:>12} {pp['UFR_pct']:>7.1f} {pp['usable_flexibility_frac']*100:>11.0f}%")
    print(f"\ninvariants: obs_UFR<=feat {inv1}, obs_usable>50% {inv2}")
    if not (inv1 and inv2):
        raise SystemExit("INVARIANT FAILURE")
    print("written -> results/rq6_safe.json")
    return out


if __name__ == "__main__":
    run()
