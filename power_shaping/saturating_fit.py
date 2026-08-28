# -*- coding: utf-8 -*-
"""(1) Which (card, model) pairs NEVER saturate -- best-batch uncapped draw < 0.9*TDP?
(2) Refit P(R)=P0+aR^beta on SATURATE-only data; and on saturate-only restricted to saturating models.
Compares fit quality to see whether dropping the memory-bound (non-saturating) models cleans up the fit.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from predict_power import RAW

TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}
SAT_THR = 0.90


def _card(df):
    c = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    return c.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()


def load_sat():
    fr = []
    for f in ["aws_sweep_batch.csv", "aws_sweep_a100_batch.csv"]:
        p = os.path.join(RAW, f)
        if os.path.exists(p):
            fr.append(pd.read_csv(p))
    df = pd.concat(fr, ignore_index=True)
    df["card"] = _card(df)
    return df


def r2(y, yh):
    ss = np.sum((y - yh) ** 2); st = np.sum((y - y.mean()) ** 2)
    return 1 - ss / st if st > 0 else np.nan


def fit_one(R, P):
    Rn = R / R.max()
    def mdl(x, p0, a, b): return p0 + a * np.power(x, b)
    try:
        popt, _ = curve_fit(mdl, Rn, P, p0=[P.min(), P.max() - P.min(), 2.0],
                            bounds=([0, 0, 0.2], [P.min() * 1.3 + 1, 5 * (P.max() - P.min() + 1), 8]), maxfev=20000)
        return popt[2], r2(P, mdl(Rn, *popt))
    except Exception:
        return np.nan, np.nan


def main():
    df = load_sat()
    # best-batch uncapped draw per (card, workload): take the max-cap row of the saturate sweep (fallback control)
    sat = df[df["mode"] == "saturate"]
    ctl = df[df["mode"] == "control"]
    print(f"== (1) NON-SATURATING (card,model): best-batch uncapped draw < {int(100*SAT_THR)}% TDP ==")
    flags = {}
    per_card = {c: [] for c in TDP}
    for card in ["T4", "L4", "A10G", "A100"]:
        for wl in sorted(df.workload.unique()):
            s = sat[(sat.card == card) & (sat.workload == wl)]
            if len(s) == 0:
                s = ctl[(ctl.card == card) & (ctl.workload == wl)]
            if len(s) == 0:
                continue
            draw = s.sort_values("cap_w").power_w.values[-1]
            frac = draw / TDP[card]
            sat_ok = frac >= SAT_THR
            flags[(card, wl)] = sat_ok
            if not sat_ok:
                per_card[card].append((wl, frac))
    for card in ["T4", "L4", "A10G", "A100"]:
        ns = per_card[card]
        tot = sum(1 for k in flags if k[0] == card)
        print(f"  {card:5} ({TDP[card]}W): {len(ns)}/{tot} never reach {int(100*SAT_THR)}% TDP")
        for wl, fr in sorted(ns, key=lambda x: x[1]):
            print(f"        {wl:22} peaks at {100*fr:3.0f}% TDP")
    # models non-saturating on EVERY card they appear on
    allcards = {}
    for (card, wl), ok in flags.items():
        allcards.setdefault(wl, []).append(ok)
    never = [wl for wl, v in allcards.items() if not any(v)]
    always = [wl for wl, v in allcards.items() if all(v)]
    print(f"\n  never saturate on ANY card: {sorted(never)}")
    print(f"  saturate on EVERY card:     {sorted(always)}")

    # (2) refit on saturate-only, all vs saturating-only
    print(f"\n== (2) refit P(R)=P0+aR^beta, SATURATE-only cap sweeps ==")
    for label, keep_ns in [("all models", True), ("saturating models only", False)]:
        r2s, betas = [], []
        for (card, wl), g in sat.groupby(["card", "workload"]):
            if not keep_ns and not flags.get((card, wl), False):
                continue
            gg = g.groupby("cap_w", as_index=False).agg(R=("throughput", "mean"), P=("power_w", "mean")).sort_values("R")
            if len(gg) < 4:
                continue
            b, rr = fit_one(gg.R.values, gg.P.values)
            if np.isfinite(rr):
                r2s.append(rr); betas.append(b)
        r2s = np.array(r2s)
        print(f"  {label:24}: {len(r2s):3} fits  median R^2={np.median(r2s):.3f}  "
              f">=0.9: {int((r2s>=0.9).sum())}/{len(r2s)}  median beta={np.nanmedian(betas):.2f}")


if __name__ == "__main__":
    main()
