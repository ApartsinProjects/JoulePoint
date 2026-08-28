# -*- coding: utf-8 -*-
"""
What would actually improve the response model? Two diagnostics on the weak case (Zeus,
Oracle Capture ~38%), reusing the validated oracle-capture machinery:

  1. LEARNING CURVE -- Oracle Capture vs the number of TRAINING workloads. If capture keeps
     rising with more workloads (no plateau), the ceiling is DATA-limited: more distinct
     workloads is what is needed. If it is already flat, more of the same data will not help.

  2. LIGHT PROBE -- give the predictor a cheap runtime OBSERVATION of the held-out workload's
     own elasticity: its throughput drop between the top cap and one lower cap (a 2-point
     probe, far cheaper than a full sweep). If this lifts capture, the lever is a light probe,
     not more training data.

Both compared against the family-only baseline. Invariant: a predictor given the workload's
FULL true curve must reach ~100% capture (perfect information upper bound), confirming the
metric can move at all on this data.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import learned_control as LC
import poca_killtest as K


def probe_feature(tab):
    """Observed local elasticity from a 2-point probe: normalized throughput drop from the
    workload's top power to its second-highest power (a cheap runtime measurement)."""
    tab = tab.copy()
    slope = {}
    for wl, g in tab.groupby("wl"):
        g = g.sort_values("power")
        q = g["q"].to_numpy(float); p = g["power"].to_numpy(float)
        if len(q) >= 2 and (p[-1] - p[-2]) > 1e-9:
            slope[wl] = (q[-1] - q[-2]) / (p[-1] - p[-2])       # dq/dpower near the top (observed)
        else:
            slope[wl] = 0.0
    s = pd.Series(slope); s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    tab["obs_slope"] = tab["wl"].map(s)
    return tab


def learning_curve(tab, feat_cats, feat_nums, seeds=(0, 1, 2, 3, 4)):
    """Oracle Capture (class-mean) when only k training workloads are available per LOO fold.
    Implemented by restricting the pool to k random training workloads + the held-out one."""
    wls = list(tab["wl"].unique()); n = len(wls)
    ks = sorted({k for k in (3, 5, 7, 9, 11, n - 1) if 2 <= k <= n - 1})
    curve = []
    for k in ks:
        caps = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            per_wl = []
            for held in wls:
                others = [w for w in wls if w != held]
                keep = list(rng.choice(others, size=k, replace=False)) + [held]
                sub = tab[tab["wl"].isin(keep)]
                _, cm = LC.oracle_capture(sub, feat_cats, feat_nums)
                per_wl.append(cm)
            caps.append(np.mean(per_wl))
        curve.append({"k_train": k, "capture_pct": float(np.mean(caps)), "sd": float(np.std(caps))})
    return curve


def pooled_table():
    """emerald + Zeus workloads in one table with a unified family feature, to trace Oracle
    Capture over a WIDER range of #training-workloads than either dataset gives alone."""
    em = LC.emerald_table().rename(columns={"f_task": "f_fam"})[["wl", "f_fam", "power", "q"]]
    ze = LC.zeus_table().rename(columns={"f_net": "f_fam"})[["wl", "f_fam", "power", "q"]]
    return pd.concat([em, ze], ignore_index=True)


def main():
    out = {}
    ze = LC.zeus_table()
    nwl = ze["wl"].nunique()
    base_gbdt, base_cm = LC.oracle_capture(ze, ["f_net"], [])
    base = max(base_cm, base_gbdt)
    out["zeus_baseline_pct"] = base
    out["n_workloads"] = int(nwl)

    # 1. learning curve
    print(f"== Zeus learning curve (n={nwl} workloads, baseline capture {base:.0f}%) ==")
    lc = learning_curve(ze, ["f_net"], [])
    out["learning_curve"] = lc
    for r in lc:
        print(f"  k_train={r['k_train']:>2}: capture {r['capture_pct']:5.1f}%  (sd {r['sd']:.1f})")
    slope_pts = (lc[-1]["capture_pct"] - lc[0]["capture_pct"]) / max(1, (lc[-1]["k_train"] - lc[0]["k_train"]))
    out["capture_gain_per_added_workload_pts"] = float(slope_pts)

    # 2. light probe
    zp = probe_feature(ze)
    pr_gbdt, pr_cm = LC.oracle_capture(zp, ["f_net"], ["obs_slope"])
    probe = max(pr_cm, pr_gbdt)
    out["zeus_probe_pct"] = probe
    print(f"\n== Light 2-point probe feature ==")
    print(f"  baseline (family only)      : {base:5.1f}%")
    print(f"  + observed elasticity slope : {probe:5.1f}%   ({probe-base:+.0f} pts)")

    # 3. pooled learning curve over a wider #workloads range (emerald + Zeus)
    pool = pooled_table(); npool = pool["wl"].nunique()
    print(f"\n== Pooled learning curve (emerald+Zeus, n={npool} workloads) ==")
    pc = learning_curve(pool, ["f_fam"], [], seeds=(0, 1, 2))
    out["pooled_learning_curve"] = pc; out["pooled_n_workloads"] = int(npool)
    for r in pc:
        print(f"  k_train={r['k_train']:>2}: capture {r['capture_pct']:5.1f}%  (sd {r['sd']:.1f})")

    # invariant: perfect information (true curve) must reach ~100% capture
    tru_cap = LC.invariant_oc(ze)
    out["invariant_perfect_info_100pct"] = bool(tru_cap)
    print(f"\ninvariant (perfect-info capture == 100%): {tru_cap}")
    if not tru_cap:
        raise SystemExit("INVARIANT FAILURE -- metric cannot reach 100% even with true curves")

    with open(os.path.join(K.RESULTS, "model_improve.json"), "w") as f:
        json.dump(out, f, indent=2)
    verdict = ("DATA-limited: capture rises with #workloads" if slope_pts > 1.0
               else "not simply data-limited at this range")
    print(f"\nverdict: {verdict}; probe lift {probe-base:+.0f} pts")
    print("written -> results/model_improve.json")


if __name__ == "__main__":
    main()
