# -*- coding: utf-8 -*-
"""Fit the power-rate model  P(R) = P0 + a*R^beta  to our measured power-cap sweeps, per (GPU, workload).
R = achieved rate (throughput), P = measured board power. This is the same nonlinear form that fit the V100
DVFS dataset; here we test it on the CAP knob across 4 GPUs. Then the energy model E/F = P0/R + a*R^(beta-1)
gives an energy-optimal rate R* = (P0 / (a*(beta-1)))^(1/beta) when beta>1. Reports R^2 vs a linear (beta=1)
fit, fitted P0 (should approximate the ~1/2-TDP fixed floor), beta, and R*."""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from predict_power import RAW

ORDER = ["T4", "L4", "A10G", "A100"]
TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}


def load():
    # pool every measured (rate, power) operating point: old bs=32 sweeps + the new control+saturate batch
    # sweeps. The batch runs vary rate by BOTH cap and batch size, so pooling widens the R range each P(R)
    # curve is fit over. Distinct operating points are keyed by (card, workload, mode, batch, cap_w).
    files = ["aws_sweep_spectrum.csv", "aws_sweep_a100.csv", "aws_sweep_batch.csv", "aws_sweep_a100_batch.csv"]
    frames = []
    for f in files:
        p = os.path.join(RAW, f)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if "mode" not in d.columns:
            d["mode"] = "bs32"
        if "batch" not in d.columns:
            d["batch"] = 32
        frames.append(d[["gpu", "workload", "mode", "batch", "cap_w", "power_w", "throughput"]])
    df = pd.concat(frames, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    df["card"] = df.card.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    return df.groupby(["card", "workload", "mode", "batch", "cap_w"], as_index=False).agg(
        R=("throughput", "mean"), P=("power_w", "mean"))


def r2(y, yhat):
    ss = np.sum((y - yhat) ** 2); st = np.sum((y - y.mean()) ** 2)
    return 1 - ss / st if st > 0 else np.nan


def fit_one(R, P):
    Rn = R / R.max()                                   # normalize rate for numerics
    def mdl(x, p0, a, b): return p0 + a * np.power(x, b)
    try:
        popt, _ = curve_fit(mdl, Rn, P, p0=[P.min(), P.max() - P.min(), 2.0],
                            bounds=([0, 0, 0.2], [P.min() * 1.3 + 1, 5 * (P.max() - P.min() + 1), 8]),
                            maxfev=20000)
        p0, a, b = popt
        r2n = r2(P, mdl(Rn, *popt))
        # energy-optimal (normalized) rate; b>1 required for an interior optimum
        Rstar = (p0 / (a * (b - 1))) ** (1 / b) if b > 1 and a > 0 else np.nan
        Rstar_frac = min(max(Rstar, 0), 1) if np.isfinite(Rstar) else np.nan   # as fraction of max rate
        return p0, b, r2n, Rstar_frac
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def fit_linear(R, P):
    Rn = R / R.max()
    A = np.vstack([np.ones_like(Rn), Rn]).T
    coef, *_ = np.linalg.lstsq(A, P, rcond=None)
    return r2(P, A @ coef)


def main():
    df = load()
    rows = []
    # fit P(R) along each CAP SWEEP at a fixed (card, workload, batch-config); the model is a within-config
    # law, not a cross-batch one (pooling batches destroys it: same R reachable at different P).
    for (card, wl, mode, batch), g in df.groupby(["card", "workload", "mode", "batch"]):
        g = g.sort_values("R")
        if len(g) < 4:
            continue
        R, P = g.R.values, g.P.values
        p0, b, r2nl, rstar = fit_one(R, P)
        r2lin = fit_linear(R, P)
        rows.append(dict(card=card, workload=wl, mode=mode, batch=batch, P0=p0, beta=b,
                         r2_nl=r2nl, r2_lin=r2lin, Rstar_frac=rstar, Pmin=P.min()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RAW, "energy_model_fit.csv"), index=False)
    print("== P(R)=P0 + a*R^beta fit to our power-cap sweeps ==")
    print(f"  fits: {len(t)} (GPU x workload)   median R^2 (nonlinear) = {t.r2_nl.median():.3f}   "
          f"median R^2 (linear) = {t.r2_lin.median():.3f}")
    for thr in (0.90, 0.95, 0.99):
        print(f"  R^2 >= {thr}: {int((t.r2_nl >= thr).sum())}/{len(t)} (nonlinear)  vs  "
              f"{int((t.r2_lin >= thr).sum())}/{len(t)} (linear)")
    print(f"  beta: median {t.beta.median():.2f}  (>1 => power rises superlinearly with rate)  "
          f"[{t.beta.quantile(.1):.2f}-{t.beta.quantile(.9):.2f}]")
    print("\n== fitted P0 (fixed power) vs measured floor, per GPU (mean over workloads) ==")
    for card in [c for c in ORDER if c in set(t.card)]:
        s = t[t.card == card]
        floor = df[df.card == card].P.min()
        print(f"  {card:5} fitted P0 ~ {s.P0.mean():5.0f}W   measured floor {floor:5.0f}W   "
              f"TDP {TDP.get(card,'?')}W   (P0/TDP {s.P0.mean()/TDP.get(card,1):.2f})")
    print("\n== representative fits (A10G, widest cap range): linear vs nonlinear R^2, P0, energy-opt rate ==")
    a = t[t.card == "A10G"].sort_values("beta")
    print(f"{'workload':16} {'R2_lin':>7} {'R2_nl':>7} {'P0(W)':>7} {'beta':>5} {'R*/Rmax':>8}")
    for _, r in a.iterrows():
        print(f"{r.workload:16} {r.r2_lin:>7.3f} {r.r2_nl:>7.3f} {r.P0:>7.0f} {r.beta:>5.2f} "
              f"{r.Rstar_frac:>8.2f}")


if __name__ == "__main__":
    main()
