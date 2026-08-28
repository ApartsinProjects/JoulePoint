# -*- coding: utf-8 -*-
"""
RQ4 strengthened: raise Oracle Capture on heterogeneous unseen workloads by giving the
predictor an OBSERVED compute-intensity signal (the workload's power-draw fraction: memory-
bound workloads draw low and keep throughput when capped; compute-bound draw high and lose
throughput). This is a cheap runtime observation (mirrors the RQ6 fix), not a profiling
sweep. Reuses learned_control's validated Oracle-Capture machinery and only augments the
feature table, so the numbers are directly comparable to the paper's RQ4 result.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import learned_control as LC
import poca_killtest as K


def add_obs_feature(tab):
    tab = tab.copy()
    tab["obs_draw_frac"] = tab.groupby("wl")["power"].transform("max") / tab["power"].max()
    return tab


def aws_table():
    d = pd.read_csv(os.path.join(K.RAW, "own_aws_sweep.csv"))
    g = d.groupby(["workload", "cap_w"], as_index=False).agg(p=("power_w", "mean"), t=("throughput", "mean"))
    rows = []
    for wl, x in g.groupby("workload"):
        for _, r in x.iterrows():
            rows.append(dict(wl=wl, f_fam=wl, power=r["p"], q=r["t"]))
    df = pd.DataFrame(rows)
    df["q"] = df.groupby("wl")["q"].transform(lambda s: s / s.max())
    return df


def main():
    out = {}
    print("== RQ4: Oracle Capture on unseen workloads (reusing validated machinery) ==")
    print(f"{'dataset':>8} {'baseline (family)':>18} {'+ observed compute-intensity':>28}")

    # Zeus (the weak case): family = network
    ze = add_obs_feature(LC.zeus_table())
    base_gbdt, base_cm = LC.oracle_capture(ze, ["f_net"], [])
    obs_gbdt, obs_cm = LC.oracle_capture(ze, ["f_net"], ["obs_draw_frac"])
    zeus_base = max(base_cm, base_gbdt); zeus_obs = max(obs_cm, obs_gbdt)
    out["zeus"] = {"baseline_best": zeus_base, "obs_best": zeus_obs,
                   "baseline_classmean": base_cm, "obs_gbdt": obs_gbdt}
    print(f"{'zeus':>8} {zeus_base:>17.0f}% {zeus_obs:>27.0f}%")

    # emerald (already strong) -- confirm the feature doesn't hurt
    em = add_obs_feature(LC.emerald_table())
    eb_gbdt, eb_cm = LC.oracle_capture(em, ["f_task"], ["f_size"])
    eo_gbdt, eo_cm = LC.oracle_capture(em, ["f_task"], ["f_size", "obs_draw_frac"])
    out["emerald"] = {"baseline_best": max(eb_cm, eb_gbdt), "obs_best": max(eo_cm, eo_gbdt)}
    print(f"{'emerald':>8} {max(eb_cm,eb_gbdt):>17.0f}% {max(eo_cm,eo_gbdt):>27.0f}%")

    # AWS real inference (family = workload id, so no same-family neighbour -> obs feature is the lever)
    aw = add_obs_feature(aws_table())
    ab_gbdt, ab_cm = LC.oracle_capture(aw, ["f_fam"], [])
    ao_gbdt, ao_cm = LC.oracle_capture(aw, ["f_fam"], ["obs_draw_frac"])
    out["aws"] = {"baseline_best": max(ab_cm, ab_gbdt), "obs_best": max(ao_cm, ao_gbdt)}
    print(f"{'aws':>8} {max(ab_cm,ab_gbdt):>17.0f}% {max(ao_cm,ao_gbdt):>27.0f}%")

    with open(os.path.join(K.RESULTS, "rq4_strengthen.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nZeus Oracle Capture: {zeus_base:.0f}% -> {zeus_obs:.0f}% "
          f"({'+' if zeus_obs>=zeus_base else ''}{zeus_obs-zeus_base:.0f} pts) with the observed compute-intensity feature")
    print("written -> results/rq4_strengthen.json")


if __name__ == "__main__":
    main()
