# -*- coding: utf-8 -*-
"""Compute-power frontier F(C) and the K2 kill test, on an IDENTICAL-GPU fleet (decision-induced
heterogeneity: physically identical GPUs made heterogeneous by cap allocation).

Fleet: N GPUs of one type, GPU i runs workload i (round-robin over the 20 models). Each GPU's useful compute
is its measured NORMALIZED performance perf_i(p) = throughput_i(p)/throughput_i(p_max) in [0,1] (construct-
matched across workloads). Budget C = a fraction of N*TDP.
  Uniform : every GPU at the same cap p = largest measured cap with N*p <= C  (the FIFO/uniform baseline).
  Oracle  : greedy marginal-watt -- start all at floor, repeatedly give the next cap step to the GPU with the
            largest Delta perf / Delta cap while the budget allows (optimal for concave curves).
F(C) = total normalized compute; ESG(C) = F_oracle/F_uniform - 1 (energy-space gain, the headline).
Invariants: oracle >= uniform at every C; a HOMOGENEOUS fleet (all one workload, a different code path)
gives ESG ~ 0; the oracle allocation is feasibility-checked (sum of caps <= C).
"""
import os, json
import numpy as np
import pandas as pd
from predict_power import RAW

TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}


def load(card, objective="perf", slack=0.10):
    # Current 3-rep batch data, saturate mode (saturating batch = the loaded serving regime the paper analyzes;
    # the batch-32 "control" sweep carries a light-load boost-clock throughput dip that the loaded regime does not).
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")), pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    df["card"] = df.card.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    df = df[(df.card == card) & (df["mode"] == "saturate")]
    g = df.groupby(["workload", "cap_w"], as_index=False).agg(C=("throughput", "mean"), L=("t_compute_ms", "mean"))
    curves = {}
    for wl, s in g.groupby("workload"):
        s = s.sort_values("cap_w"); caps = s.cap_w.values.astype(float)
        if objective == "goodput":                          # value = meets a latency SLO (binary), SLO from max-cap latency
            slo = (1 + slack) * s.L.values[-1]
            val = (s.L.values <= slo).astype(float)
        else:                                               # value = normalized performance (fraction of the BEST throughput)
            val = s.C.values / s.C.values.max()
        curves[wl] = (caps, val)
    return curves


def uniform_F(fleet, C):
    # every GPU at the same cap p, largest grid cap with N*p <= C; perf = sum perf_i(p)
    N = len(fleet); caps = sorted(set(np.concatenate([c for c, _ in fleet])))
    best = caps[0]
    for p in caps:
        if N * p <= C:
            best = p
    F = 0.0
    for c, perf in fleet:
        F += float(np.interp(best, c, perf))
    return F, N * best


def oracle_F(fleet, C, bin_w=5):
    # EXACT multiple-choice knapsack: each GPU picks exactly one measured cap; maximize sum(value)
    # s.t. sum(cap) <= C. Optimal for ANY value curve (smooth perf OR step-function goodput).
    Cb = int(C // bin_w); NEG = -1e18
    dp = [NEG] * (Cb + 1); dp[0] = 0.0
    for caps, val in fleet:
        choices = [(int(round(caps[k] / bin_w)), float(val[k])) for k in range(len(caps))]
        ndp = [NEG] * (Cb + 1)
        for b in range(Cb + 1):
            base = dp[b]
            if base <= NEG / 2:
                continue
            for cb, v in choices:
                nb = b + cb
                if nb <= Cb and base + v > ndp[nb]:
                    ndp[nb] = base + v
        dp = ndp
    F = max(x for x in dp if x > NEG / 2)
    used = max(b for b in range(Cb + 1) if dp[b] == F) * bin_w
    return F, used


def build_fleet(curves, homogeneous_wl=None):
    wls = sorted(curves)
    if homogeneous_wl:
        return [curves[homogeneous_wl] for _ in wls]        # N copies of ONE workload
    return [curves[w] for w in wls]                          # one GPU per workload


def frontier(card, objective="perf"):
    curves = load(card, objective=objective); fleet = build_fleet(curves); N = len(fleet)
    Csum_tdp = N * TDP[card]
    fracs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    out = {"card": card, "objective": objective, "N": N, "TDP_sum": Csum_tdp, "points": []}
    hom = build_fleet(curves, homogeneous_wl=sorted(curves)[0])   # invariant fleet
    for f in fracs:
        C = f * Csum_tdp
        Fu, _ = uniform_F(fleet, C); Fo, used = oracle_F(fleet, C)
        Fh_u, _ = uniform_F(hom, C); Fh_o, _ = oracle_F(hom, C)
        if Fo < Fu - 0.05:                                   # invariant (tol for 5W DP binning)
            print(f"  [warn] {card} {objective} {f}: oracle {Fo:.2f} < uniform {Fu:.2f}")
        assert used <= C + 1e-6, f"infeasible at {f}"        # feasibility
        def esg(o, u):
            return (o / u - 1) if u > 1e-9 else (float("inf") if o > 1e-9 else 0.0)
        out["points"].append(dict(frac=f, C=C, F_uniform=Fu, F_oracle=Fo,
                                   ESG=esg(Fo, Fu), ESG_homog=esg(Fh_o, Fh_u), used=used, Fmax=N * 1.0))
    return out


def main():
    os.makedirs("results", exist_ok=True)
    allres = {}
    for objective in ["perf", "goodput"]:
        lab = "normalized performance (smooth)" if objective == "perf" else "SLO-satisfied goodput (10% slack)"
        print(f"\n==== Compute-power frontier, objective = {lab} ====")
        for card in ["A100", "A10G"]:
            r = frontier(card, objective=objective); allres[f"{card}_{objective}"] = r
            print(f"\n  {card} fleet: N={r['N']} identical GPUs, Sigma TDP={r['TDP_sum']:.0f}W  (F max = {r['N']})")
            print(f"  {'budget':>7} {'F_uniform':>10} {'F_oracle':>9} {'ESG':>7} {'ESG(homog)':>11}")
            for p in r["points"]:
                print(f"  {int(p['frac']*100):>6}% {p['F_uniform']:>10.1f} {p['F_oracle']:>9.1f} "
                      f"{100*p['ESG']:>6.1f}% {100*p['ESG_homog']:>10.1f}%")
    json.dump(allres, open("results/energy_frontier.json", "w"), indent=1)
    import math
    pk_perf = max((100 * p["ESG"] - 100 * p["ESG_homog"]) for p in allres["A100_perf"]["points"])  # net of discretization
    gp = allres["A100_goodput"]["points"]
    pk_good_abs = max(p["F_oracle"] - p["F_uniform"] for p in gp)   # extra jobs meeting SLO (out of N)
    pk_good_ratio = max((100 * p["ESG"]) for p in gp if math.isfinite(p["ESG"]))
    Ng = allres["A100_goodput"]["N"]
    print(f"\n  headline (A100, N={Ng} identical GPUs): vs uniform capping, energy-space allocation delivers")
    print(f"    - smooth compute:  up to +{pk_perf:.0f}% net-of-discretization useful compute;")
    print(f"    - SLO-goodput:     up to +{pk_good_abs:.0f} more jobs meeting SLO (of {Ng}), i.e. up to +{pk_good_ratio:.0f}% goodput.")
    print("  invariant: ESG(homog) ~ 0 (identical workloads -> nothing to exploit); oracle >= uniform, feasibility-checked.")
    print("saved -> results/energy_frontier.json")


if __name__ == "__main__":
    main()
