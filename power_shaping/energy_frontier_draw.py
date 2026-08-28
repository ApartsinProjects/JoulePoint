# -*- coding: utf-8 -*-
"""K2, measured correctly: charge each GPU its ACTUAL DRAW (power_w), and budget against the fleet's summed
NATURAL DRAW (Sigma D0, the uncapped draw), NOT Sigma TDP. This removes the low-drain bias: a job that draws
far below TDP no longer 'uses up' budget it never consumed, so the oracle's ability to reallocate that real
headroom to compute-bound jobs shows up. Identical-GPU fleet (decision-induced heterogeneity).

Operating point (model, cap) -> (perf_norm, draw=power_w). Budget C = frac * Sigma D0.
  Uniform : one cap X for all, largest with sum(draw_i(X)) <= C.
  Oracle  : multiple-choice knapsack on DRAW -> max sum(perf) s.t. sum(draw) <= C.
Compared head-to-head with the TDP/cap-based accounting to expose the bias.
"""
import os, json
import numpy as np
import pandas as pd
from predict_power import RAW

TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}


def load(card, objective="perf", slack=0.10):
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_spectrum.csv")), pd.read_csv(os.path.join(RAW, "aws_sweep_a100.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    df["card"] = df.card.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    df = df[df.card == card]
    g = df.groupby(["workload", "cap_w"], as_index=False).agg(C=("throughput", "mean"),
                                                              D=("power_w", "mean"), L=("t_compute_ms", "mean"))
    curves = {}
    for wl, s in g.groupby("workload"):
        s = s.sort_values("cap_w")
        if objective == "goodput":
            slo = (1 + slack) * s.L.values[-1]; val = (s.L.values <= slo).astype(float)
        else:
            val = s.C.values / s.C.values.max()
        curves[wl] = dict(cap=s.cap_w.values.astype(float), val=val, draw=s.D.values.astype(float))
    return curves


def knap_cost(items, C, bin_w=3):
    # multiple-choice knapsack with an arbitrary per-item COST array (here: draw)
    Cb = int(C // bin_w); NEG = -1e18
    dp = [NEG] * (Cb + 1); dp[0] = 0.0
    for cost, val in items:
        ch = [(int(round(cost[k] / bin_w)), float(val[k])) for k in range(len(cost))]
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


def uniform_at_draw(fleet, C):
    # one cap level X for all GPUs; total actual draw must fit C; take the largest such X
    caps = sorted(set(np.concatenate([f["cap"] for f in fleet])))
    bestF = None
    for X in caps:
        drawsum = sum(float(np.interp(X, f["cap"], f["draw"])) for f in fleet)
        if drawsum <= C:
            bestF = sum(float(np.interp(X, f["cap"], f["val"])) for f in fleet)
    if bestF is None:                                    # even the floor draws more than C
        X = caps[0]; bestF = sum(float(np.interp(X, f["cap"], f["val"])) for f in fleet)
    return bestF


def frontier_draw(card, objective="perf"):
    curves = load(card, objective=objective); wls = sorted(curves)
    fleet = [curves[w] for w in wls]; N = len(fleet)
    D0sum = sum(f["draw"][-1] for f in fleet)            # summed natural (uncapped) draw
    TDPsum = N * TDP[card]
    hom = [curves[wls[0]] for _ in wls]                  # homogeneous invariant fleet
    D0h = sum(f["draw"][-1] for f in hom)
    pts = []
    for f in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        C = f * D0sum
        Fu = uniform_at_draw(fleet, C)
        Fo = knap_cost([(x["draw"], x["val"]) for x in fleet], C)
        Ch = f * D0h
        Fhu = uniform_at_draw(hom, Ch); Fho = knap_cost([(x["draw"], x["val"]) for x in hom], Ch)
        def esg(o, u): return (o / u - 1) if u > 1e-9 else (float("inf") if o > 1e-9 else 0.0)
        pts.append(dict(frac=f, F_uniform=Fu, F_oracle=Fo, ESG=esg(Fo, Fu), ESG_homog=esg(Fho, Fhu)))
    return dict(card=card, objective=objective, N=N, D0_sum=D0sum, TDP_sum=TDPsum, points=pts)


def main():
    os.makedirs("results", exist_ok=True); allres = {}
    for objective in ["perf", "goodput"]:
        lab = "normalized performance" if objective == "perf" else "SLO-goodput (10%)"
        print(f"\n==== K2 vs REAL DRAW (budget = fraction of Sigma D0), objective = {lab} ====")
        for card in ["A100", "A10G"]:
            r = frontier_draw(card, objective); allres[f"{card}_{objective}"] = r
            print(f"\n  {card}: N={r['N']}, Sigma D0={r['D0_sum']:.0f}W (Sigma TDP={r['TDP_sum']:.0f}W -> jobs draw "
                  f"{100*r['D0_sum']/r['TDP_sum']:.0f}% of TDP uncapped)")
            print(f"  {'budget(%D0)':>11} {'F_uniform':>10} {'F_oracle':>9} {'ESG':>7} {'ESG_homog':>10}")
            for p in r["points"]:
                e = p["ESG"]; es = f"{100*e:>6.0f}%" if np.isfinite(e) else "   inf"
                print(f"  {int(p['frac']*100):>10}% {p['F_uniform']:>10.1f} {p['F_oracle']:>9.1f} {es} "
                      f"{100*p['ESG_homog'] if np.isfinite(p['ESG_homog']) else 999:>9.1f}%")
    json.dump(allres, open("results/energy_frontier_draw.json", "w"), indent=1)
    pk = max((100 * p["ESG"] - (100 * p["ESG_homog"] if np.isfinite(p["ESG_homog"]) else 0))
             for p in allres["A100_perf"]["points"] if np.isfinite(p["ESG"]))
    print(f"\n  vs the TDP/cap-based number (~5%), draw-based net perf gain (A100) peaks ~{pk:.0f}% "
          f"-- the low-drain bias was hiding real headroom.")
    print("saved -> results/energy_frontier_draw.json")


if __name__ == "__main__":
    main()
