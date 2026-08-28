# -*- coding: utf-8 -*-
"""Stage-3 policy-space experiment: energy saving vs latency-utility loss, under three utility models.
The energy-optimal cap is only "optimal" under a specific notion of what latency is worth. We define three
latency-utility shapes and, for each, trace the achievable energy saving as a function of tolerated utility
loss, per job, on the A100 and A10G. Establishes the policy space: how much energy the operator can bank for
a given latency-utility budget, and how strongly the answer depends on the SLA shape.

  throughput  U = R / R_uncapped                 (every millisecond is worth the same; linear in speed)
  soft        U ramps 1 -> 0 as latency goes T -> 2T, T = 1.25 x uncapped latency  (a soft deadline)
  hard        U = 1 if latency <= 1.25 x uncapped latency else 0                    (a hard SLO)
"""
import os, json
import numpy as np
import pandas as pd
from predict_power import RAW, saturating_set

TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}


def _card(df):
    c = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    return c.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()


def curves(card):
    f = "aws_sweep_a100_batch.csv" if card == "A100" else "aws_sweep_batch.csv"
    d = pd.read_csv(os.path.join(RAW, f)); d["card"] = _card(d); d = saturating_set(d); d = d[d.card == card]
    # average the 3 reps per operating point before tracing utility vs energy (a single noisy rep would
    # otherwise win the min-energy-subject-to-utility search and overstate the saving).
    d = d.groupby(["workload", "cap_w"], as_index=False).agg(
        power_w=("power_w", "mean"), throughput=("throughput", "mean"))
    out = {}
    for wl, g in d.groupby("workload"):
        g = g.sort_values("cap_w")
        R = g.throughput.values; L = 1000.0 / R; E = g.power_w.values / R
        out[wl] = dict(cap=100 * g.cap_w.values / TDP[card], R=R, L=L, E=E)
    return out


def utilities(L):
    Lu = L[-1]                                     # uncapped latency (fastest) is the reference
    T = 1.25 * Lu
    u_thr = (1.0 / L) / (1.0 / L[-1])              # = R/R_uncapped
    u_soft = np.clip((2 * T - L) / (2 * T - T), 0, 1)
    u_hard = (L <= T).astype(float)
    return dict(throughput=u_thr, soft=u_soft, hard=u_hard)


def main():
    os.makedirs("results", exist_ok=True)
    res = {}
    TOL = 0.05                                      # tolerated utility loss
    for card in ["A100", "A10G"]:
        cur = curves(card); res[card] = {}
        print(f"\n== {card}: median energy saving at <= {int(100*TOL)}% latency-utility loss ==")
        print(f"  {'utility model':14} {'energy saved':>12} {'at cap %TDP':>12}  (median over models)")
        for shape in ["throughput", "soft", "hard"]:
            saves, caps = [], []
            for wl, c in cur.items():
                U = utilities(c["L"])[shape]
                Eunc = c["E"][-1]                   # energy at uncapped (max-utility reference)
                ok = U >= (1 - TOL)                 # operating points that keep enough utility
                if not ok.any():
                    continue
                Emin = c["E"][ok].min(); k = np.where(ok)[0][np.argmin(c["E"][ok])]
                saves.append(1 - Emin / Eunc); caps.append(c["cap"][k])
            res[card][shape] = dict(energy_saved=float(np.median(saves)), cap=float(np.median(caps)))
            print(f"  {shape:14} {100*np.median(saves):>11.0f}% {np.median(caps):>11.0f}%")
    # policy frontier: energy saved vs tolerated utility loss (throughput utility, A100)
    print("\n== policy frontier (A100, throughput utility): energy saved vs tolerated utility loss ==")
    cur = curves("A100")
    print(f"  {'util loss':>9} {'energy saved (median)':>22}")
    front = []
    for tol in [0.0, 0.02, 0.05, 0.10, 0.20, 0.35]:
        saves = []
        for wl, c in cur.items():
            U = utilities(c["L"])["throughput"]; ok = U >= (1 - tol)
            if ok.any():
                saves.append(1 - c["E"][ok].min() / c["E"][-1])
        front.append(dict(util_loss=tol, energy_saved=float(np.median(saves))))
        print(f"  {int(100*tol):>8}% {100*np.median(saves):>21.0f}%")
    res["frontier_A100_throughput"] = front
    json.dump(res, open("results/utility_model.json", "w"), indent=1)
    print("\nsaved -> results/utility_model.json")


if __name__ == "__main__":
    main()
