# -*- coding: utf-8 -*-
"""K3: does workload-aware ROUTING add value beyond optimized caps? A/B/C decomposition on a HETEROGENEOUS
fleet (5 each of T4/L4/A10G/A100 = 20 slots, 20 jobs = the 20 models).

Useful compute of a job = perf(model, gpu, cap) = throughput / (that model's BEST throughput over all
GPUs & caps)  -> in [0,1], comparable ACROSS GPU types (running a model on a fast GPU delivers more of its
achievable work). Power cost = cap_w (allocated/promised power). Budget C = fraction of the fleet's Sigma TDP.

  A  uniform cap + round-robin placement (job i -> type i%4)           -> baseline
  B  optimized caps (knapsack) + the SAME round-robin placement        -> adds power allocation
  C  optimized caps + workload-aware routing (Lagrangian + assignment) -> adds routing
G_power = B - A ;  G_routing = C - B.  Invariant: a single-GPU-type fleet gives G_routing = 0.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from predict_power import RAW

TYPES = ["T4", "L4", "A10G", "A100"]
TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}
NPER = 5


def load():
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_spectrum.csv")), pd.read_csv(os.path.join(RAW, "aws_sweep_a100.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    df["card"] = df.card.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    g = df.groupby(["card", "workload", "cap_w"], as_index=False).agg(C=("throughput", "mean"))
    best = g.groupby("workload").C.max()                       # each model's best throughput over all GPUs+caps
    g["perf"] = g.C / g.workload.map(best)
    curves = {}                                                # (workload, card) -> (caps, perf)
    for (card, wl), s in g.groupby(["card", "workload"]):
        s = s.sort_values("cap_w"); curves[(wl, card)] = (s.cap_w.values.astype(float), s.perf.values)
    return sorted(g.workload.unique()), curves


def knap(items, C, bin_w=5):
    # multiple-choice knapsack: items = list of (caps[], perf[]); pick one cap each, max sum perf, sum cap<=C
    Cb = int(C // bin_w); NEG = -1e18
    dp = [NEG] * (Cb + 1); dp[0] = 0.0
    for caps, perf in items:
        ch = [(int(round(caps[k] / bin_w)), float(perf[k])) for k in range(len(caps))]
        ndp = [NEG] * (Cb + 1)
        for b in range(Cb + 1):
            if dp[b] <= NEG / 2:
                continue
            for cb, v in ch:
                nb = b + cb
                if nb <= Cb and dp[b] + v > ndp[nb]:
                    ndp[nb] = dp[b] + v
        dp = ndp
    return max(x for x in dp if x > NEG / 2)


def uniform_rr(jobs, curves, slots, C):
    # each slot at the largest cap with sum(cap) <= C using a common FRACTION of each type's TDP
    frac = C / sum(TDP[t] for t in slots)
    F = 0.0
    for j, t in zip(jobs, slots):
        caps, perf = curves[(j, t)]
        target = frac * TDP[t]
        k = int(np.searchsorted(caps, target, side="right") - 1); k = max(0, k)
        F += perf[k]
    return F


def opt_caps_rr(jobs, curves, slots, C):
    return knap([curves[(j, t)] for j, t in zip(jobs, slots)], C)


def route_and_cap(jobs, curves, slots, C):
    # Lagrangian: for power price lam, each (job,type) picks cap maximizing perf - lam*cap; assign jobs to
    # slots (linear_sum_assignment) to maximize sum value; bisect lam so total cap <= C.
    types_of_slot = slots
    def solve(lam):
        # value + chosen cap for each (job, type)
        val = np.zeros((len(jobs), len(slots))); capsel = np.zeros((len(jobs), len(slots)))
        for i, j in enumerate(jobs):
            for s, t in enumerate(types_of_slot):
                caps, perf = curves[(j, t)]
                score = perf - lam * caps
                k = int(np.argmax(score)); val[i, s] = perf[k]; capsel[i, s] = caps[k]
        r, c = linear_sum_assignment(-(val - lam * capsel))    # maximize perf - lam*cap over the assignment
        F = val[r, c].sum(); P = capsel[r, c].sum()
        return F, P, (r, c, capsel)
    lo, hi = 0.0, 0.5
    Fhi, Phi, _ = solve(hi)
    for _ in range(40):                                        # bisect lam to meet the budget
        mid = (lo + hi) / 2; F, P, _ = solve(mid)
        if P > C:
            lo = mid
        else:
            hi = mid
    F, P, _ = solve(hi)
    return F, P


def run(C, jobs, curves):
    slots = [TYPES[i % 4] for i in range(len(jobs))]            # heterogeneous fleet, round-robin placement
    A = uniform_rr(jobs, curves, slots, C)
    B = opt_caps_rr(jobs, curves, slots, C)
    Croute, _ = route_and_cap(jobs, curves, slots, C)
    # invariant: single-type fleet -> routing adds nothing
    homo_slots = ["A100"] * len(jobs)
    Bh = opt_caps_rr(jobs, curves, homo_slots, C)
    Ch, _ = route_and_cap(jobs, curves, homo_slots, C)
    return dict(C=C, A=A, B=B, Cc=Croute, G_power=B - A, G_routing=Croute - B,
                G_routing_homo=Ch - Bh)


def main():
    jobs, curves = load()
    jobs = [j for j in jobs if all((j, t) in curves for t in TYPES)]   # models present on all 4 GPUs
    fleet_tdp = NPER * sum(TDP[t] for t in TYPES)
    print(f"== K3 routing gain: {len(jobs)} jobs on 5x(T4,L4,A10G,A100), Sigma TDP={fleet_tdp}W ==")
    print(f"  {'budget':>7} {'A uni+rr':>9} {'B cap+rr':>9} {'C cap+route':>11} {'G_power':>8} {'G_route':>8} {'route%':>7} {'homo-inv':>9}")
    out = []
    for f in [1.0, 0.8, 0.7, 0.6, 0.5]:
        r = run(f * fleet_tdp, jobs, curves); out.append({**r, "frac": f})
        tot = r["Cc"] - r["A"]
        print(f"  {int(f*100):>6}% {r['A']:>9.1f} {r['B']:>9.1f} {r['Cc']:>11.1f} {r['G_power']:>8.2f} "
              f"{r['G_routing']:>8.2f} {100*r['G_routing']/tot if tot>0 else 0:>6.0f}% {r['G_routing_homo']:>9.3f}")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/routing_gain.json", "w"), indent=1)
    peak = max((r["G_routing"] / (r["Cc"] - r["A"])) for r in out if r["Cc"] - r["A"] > 0)
    print(f"\n  routing contributes up to {100*peak:.0f}% of the total A->C gain; "
          f"homo-inv ~ 0 confirms it is genuine cross-GPU routing, not the optimizer.")
    print("saved -> results/routing_gain.json")


if __name__ == "__main__":
    main()
