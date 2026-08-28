# -*- coding: utf-8 -*-
"""
PoC-B track B2 -- mixed AI cluster with REAL priority labels (closes the synthetic-SLO
caveat for the mixed-cluster track).

Alibaba spot-gpu trace (cluster-trace-v2026-spot-gpu, 466,868 jobs): each job has a REAL
priority `job_type` in {HP (guaranteed), Spot (opportunistic)}, a gpu_request, worker_num,
submit_time and duration. Job power = gpu_request * worker_num * per-GPU-TDP(gpu_model).

We build an event-based power timeline for HP and Spot separately, pick the peak-occupancy
window, impose a curtailment envelope, and compare two controllers:
  uniform  : throttle ALL jobs proportionally (HP power is cut -> HP degraded)
  priority : shed Spot first (pause opportunistic jobs), protect HP; only touch HP if the
             cut exceeds all Spot -- uses the REAL HP/Spot labels
The firm-capacity view: how deep a cap can the cluster meet while fully protecting HP?

INVARIANT: at every curtailment, priority delivers >= HP power than uniform (Spot-first
never hurts HP more than class-blind throttling).
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
RESULTS = os.path.join(HERE, "results")

# per-GPU TDP (W) by model; unknown -> 300 W
TDP = {"A10": 150, "A100-SXM4-80GB": 400, "A800-SXM4-80GB": 400, "H800": 350,
       "GPU-series-1": 300, "GPU-series-2": 300}


def load_timeline(step_s=3600):
    d = pd.read_csv(os.path.join(RAW, "alibaba_spot_jobs.csv"))
    d = d[(d["duration"] > 0) & (d["gpu_request"] > 0)]
    d["tdp"] = d["gpu_model"].map(TDP).fillna(300.0)
    d["job_power"] = d["gpu_request"] * d["worker_num"] * d["tdp"]
    d["end"] = d["submit_time"] + d["duration"]
    T = int(np.ceil(d["end"].max() / step_s)) + 1
    hp = np.zeros(T); sp = np.zeros(T)
    for typ, arr in [("HP", hp), ("Spot", sp)]:
        g = d[d["job_type"] == typ]
        s = (g["submit_time"].to_numpy() / step_s).astype(int)
        e = np.clip((g["end"].to_numpy() / step_s).astype(int), 0, T - 1)
        pw = g["job_power"].to_numpy()
        np.add.at(arr, s, pw)          # +power at start
        np.add.at(arr, e, -pw)         # -power at end
    return np.cumsum(hp), np.cumsum(sp), step_s


def peak_window(hp, sp, win_hours=48):
    tot = hp + sp
    if len(tot) <= win_hours:
        return 0, len(tot)
    c = np.convolve(tot, np.ones(win_hours), "valid")
    s = int(np.argmax(c))
    return s, s + win_hours


def curtail(hp_t, sp_t, firm_frac):
    """Envelope = firm_frac * peak total. Return HP power delivered under each controller."""
    peak = (hp_t + sp_t).max()
    C = firm_frac * peak
    res = {}
    tot = hp_t + sp_t
    # priority: remove Spot first
    over = np.maximum(0.0, tot - C)
    spot_shed = np.minimum(over, sp_t)
    hp_cut_pri = np.maximum(0.0, over - sp_t)           # HP only touched if Spot insufficient
    hp_deliv_pri = hp_t - hp_cut_pri
    # uniform: scale everyone by C/tot when over
    scale = np.where(tot > C, C / np.maximum(tot, 1e-9), 1.0)
    hp_deliv_uni = hp_t * scale
    res["priority"] = {"hp_delivered_frac": float(hp_deliv_pri.sum() / hp_t.sum()),
                       "spot_delivered_frac": float((sp_t - spot_shed).sum() / max(sp_t.sum(), 1e-9)),
                       "hp_fully_protected": bool(hp_cut_pri.sum() < 1e-6)}
    res["uniform"] = {"hp_delivered_frac": float(hp_deliv_uni.sum() / hp_t.sum()),
                      "spot_delivered_frac": float((sp_t * scale).sum() / max(sp_t.sum(), 1e-9))}
    return res


def main():
    hp, sp, step = load_timeline()
    s, e = peak_window(hp, sp)
    hp_w, sp_w = hp[s:e], sp[s:e]
    peak = (hp_w + sp_w).max()
    spot_frac = sp_w.sum() / (hp_w + sp_w).sum()
    # firm capacity to fully protect HP = HP peak (Spot fully sheddable)
    firm_for_hp = hp_w.max() / peak
    ccm_real = 1.0 / firm_for_hp

    rows = []
    ok = True
    for f in [0.95, 0.92, 0.90, 0.88, 0.86]:
        r = curtail(hp_w, sp_w, f)
        r["firm_frac"] = f
        if r["priority"]["hp_delivered_frac"] + 1e-9 < r["uniform"]["hp_delivered_frac"]:
            ok = False
        rows.append(r)

    out = {"n_hours": int(e - s), "spot_power_frac": float(spot_frac),
           "firm_frac_to_protect_HP": float(firm_for_hp), "ccm_real_priorities": float(ccm_real),
           "curtailment": rows, "invariant_priority_ge_uniform_HP": bool(ok)}
    with open(os.path.join(RESULTS, "pocb_alibaba.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("== PoC-B track B2: real HP/Spot priorities (Alibaba spot trace) ==")
    print(f"peak-window Spot power fraction: {spot_frac*100:.1f}%  "
          f"-> firm to protect all HP: {firm_for_hp*100:.1f}% of peak, CCM(real priorities) {ccm_real:.2f}x")
    print(f"invariant (priority HP >= uniform HP): {ok}\n")
    print(f"{'firm%':>6} {'uniform HP kept':>16} {'priority HP kept':>17} {'priority Spot kept':>18}")
    for r in rows:
        print(f"{r['firm_frac']*100:>6.0f} {r['uniform']['hp_delivered_frac']*100:>15.1f}% "
              f"{r['priority']['hp_delivered_frac']*100:>16.1f}% {r['priority']['spot_delivered_frac']*100:>17.1f}%")
    print("\nwritten -> results/pocb_alibaba.json")


if __name__ == "__main__":
    main()
