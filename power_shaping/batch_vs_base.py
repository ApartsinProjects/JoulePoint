# -*- coding: utf-8 -*-
"""Does driving jobs to saturation (auto-calibrated batch, near-TDP uncapped draw) change the energy-space
story vs the fixed bs=32 spectrum? Head-to-head on the SAME 20 models x 3 g-cards.

Two questions:
  (1) SATURATION -- did the batch calibration actually lift uncapped draw toward TDP? (per card)
  (2) ELASTICITY -- does the cap bite harder at saturation? CPE = dln(throughput)/dln(draw), measured
      construct-matched (same models, same cards, actual DRAW as the power axis) on both datasets.
  (3) K2 vs REAL DRAW -- re-run the draw-based frontier (budget = fraction of Sigma D0, cost = actual draw)
      on the saturating data and compare ESG to the bs=32 number. Tests the user's hypothesis that higher
      intensity lifts the energy-space gain.
"""
import os, json
import numpy as np
import pandas as pd

TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}
RAW = os.path.join(os.path.dirname(__file__), "data", "raw")


def _card(df):
    c = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    return c.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()


def load(path, mode=None):
    df = pd.read_csv(path)
    if mode is not None and "mode" in df.columns:
        df = df[df["mode"] == mode]
    df["card"] = _card(df)
    g = df.groupby(["card", "workload", "cap_w"], as_index=False).agg(
        C=("throughput", "mean"), P=("power_w", "mean"), L=("t_compute_ms", "mean"))
    return g


def cpe_global(s):
    # s sorted by cap_w; elasticity of compute to ACTUAL draw across the full swept range
    s = s.sort_values("cap_w")
    C, P = s.C.values, s.P.values
    if len(C) < 2 or C[0] <= 0 or C[-1] <= 0 or P[0] <= 0 or P[-1] <= 0:
        return np.nan
    dlnC = np.log(C[-1]) - np.log(C[0]); dlnP = np.log(P[-1]) - np.log(P[0])
    return dlnC / dlnP if abs(dlnP) > 1e-6 else np.nan


def load_pair():
    # confound-free: control (batch=32) and saturate (calibrated batch) come from the SAME mode-labeled files,
    # measured back-to-back in one hot-GPU session. Models where b*=32 have no saturate rows -> fall back to
    # their control curve (saturate coincides with control there).
    gfiles = [os.path.join(RAW, "aws_sweep_batch.csv"), os.path.join(RAW, "aws_sweep_a100_batch.csv")]
    ctrl = pd.concat([load(p, mode="control") for p in gfiles], ignore_index=True)
    sat = pd.concat([load(p, mode="saturate") for p in gfiles], ignore_index=True)
    have = set(zip(sat.card, sat.workload))
    miss = ctrl[~ctrl.apply(lambda r: (r.card, r.workload) in have, axis=1)]
    sat = pd.concat([sat, miss], ignore_index=True)
    return ctrl, sat


def saturation_and_elasticity():
    base, sat = load_pair()
    rows = []
    for card in ["T4", "L4", "A10G", "A100"]:
        for name, g in [("control", base), ("saturate", sat)]:
            gc = g[g.card == card]
            draws, cpes = [], []
            for wl, s in gc.groupby("workload"):
                s = s.sort_values("cap_w")
                draws.append(s.P.values[-1] / TDP[card])       # uncapped draw / TDP
                cpes.append(cpe_global(s))
            rows.append(dict(card=card, dataset=name,
                             draw_frac_med=float(np.median(draws)),
                             draw_frac_p90=float(np.percentile(draws, 90)),
                             cpe_med=float(np.nanmedian(cpes)),
                             cpe_p90=float(np.nanpercentile(cpes, 90)),
                             n=len(draws)))
    return rows


# ---- draw-based K2 frontier (same construction as energy_frontier_draw, applied to either dataset) ----
def knap_cost(items, C, bin_w=3):
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
    feasible = [x for x in dp if x > NEG / 2]
    if feasible:
        return max(feasible)
    # budget cannot fit even the cheapest (min-draw) option for every item: forced floor, same as uniform
    return sum(min(zip(cost, val), key=lambda t: t[0])[1] for cost, val in items)


def uniform_at_draw(fleet, C):
    caps = sorted(set(np.concatenate([f["cap"] for f in fleet])))
    bestF = None
    for X in caps:
        drawsum = sum(float(np.interp(X, f["cap"], f["draw"])) for f in fleet)
        if drawsum <= C:
            bestF = sum(float(np.interp(X, f["cap"], f["val"])) for f in fleet)
    if bestF is None:
        X = caps[0]; bestF = sum(float(np.interp(X, f["cap"], f["val"])) for f in fleet)
    return bestF


def frontier_draw(g, card, objective="perf", slack=0.10):
    gc = g[g.card == card]
    curves = {}
    for wl, s in gc.groupby("workload"):
        s = s.sort_values("cap_w")
        if objective == "goodput":
            slo = (1 + slack) * s.L.values[-1]; val = (s.L.values <= slo).astype(float)
        else:
            val = s.C.values / s.C.values.max()
        curves[wl] = dict(cap=s.cap_w.values.astype(float), val=val, draw=s.P.values.astype(float))
    fleet = [curves[w] for w in sorted(curves)]
    D0sum = sum(f["draw"][-1] for f in fleet)
    pts = []
    for fr in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        C = fr * D0sum
        Fu = uniform_at_draw(fleet, C); Fo = knap_cost([(x["draw"], x["val"]) for x in fleet], C)
        esg = (Fo / Fu - 1) if Fu > 1e-9 else 0.0
        pts.append(dict(frac=fr, F_uniform=Fu, F_oracle=Fo, ESG=esg))
    return dict(card=card, N=len(fleet), D0_sum=D0sum, TDP_sum=len(fleet) * TDP[card], points=pts)


def main():
    os.makedirs("results", exist_ok=True)
    print("==== (1)+(2) SATURATION and ELASTICITY: bs=32 vs auto-calibrated saturating batch ====")
    print(f"  {'card':5} {'dataset':9} {'draw/TDP med':>12} {'draw/TDP p90':>12} {'CPE med':>8} {'CPE p90':>8}")
    sat_rows = saturation_and_elasticity()
    for r in sat_rows:
        print(f"  {r['card']:5} {r['dataset']:9} {100*r['draw_frac_med']:>11.0f}% {100*r['draw_frac_p90']:>11.0f}%"
              f" {r['cpe_med']:>8.3f} {r['cpe_p90']:>8.3f}")

    base, sat = load_pair()
    k2 = {}
    for objective in ["perf", "goodput"]:
        lab = "normalized perf" if objective == "perf" else "SLO-goodput(10%)"
        print(f"\n==== (3) K2 vs REAL DRAW, objective={lab}: peak ESG per card, control(bs32) vs saturate ====")
        print(f"  {'card':5} {'bs32 peakESG':>12} {'saturate peakESG':>16}")
        for card in ["T4", "L4", "A10G", "A100"]:
            rb = frontier_draw(base, card, objective); rs = frontier_draw(sat, card, objective)
            pb = max(p["ESG"] for p in rb["points"]); ps = max(p["ESG"] for p in rs["points"])
            k2[f"{card}_{objective}"] = dict(bs32=rb, saturate=rs, peak_bs32=pb, peak_sat=ps)
            print(f"  {card:5} {100*pb:>11.0f}% {100*ps:>15.0f}%")
    json.dump({"saturation_elasticity": sat_rows, "k2": {k: {"peak_bs32": v["peak_bs32"],
              "peak_sat": v["peak_sat"]} for k, v in k2.items()}}, open("results/batch_vs_base.json", "w"), indent=1)
    print("\nsaved -> results/batch_vs_base.json")


if __name__ == "__main__":
    main()
