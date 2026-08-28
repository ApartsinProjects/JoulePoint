# -*- coding: utf-8 -*-
"""Per-model efficiency table for the two interesting cards (A100, A10G). For each model compare:
  MAX  = full-performance operating point (uncapped, highest cap): throughput, latency, energy/step
  EFF  = energy-efficient operating point (minimum energy/step)   : throughput, latency, energy/step
and the deltas of all three at EFF as % of MAX. Batch=32 (control mode). Saves data/raw/efficiency_table.csv.
  energy/step (J) = power_w / throughput ;  latency/step (ms) = 1000 / throughput
"""
import os
import numpy as np
import pandas as pd
from predict_power import RAW, saturating_set

TDP = {"A10G": 300, "A100": 400}


def load():
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")),
          pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    df["card"] = df.card.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    df = saturating_set(df)                                   # loaded (production-relevant) regime
    # Average the repetitions at each operating point BEFORE any argmin. The three reps are consecutive
    # 2 s windows after the cap is set; collapsing them to one point per (card, workload, cap) keeps the
    # min-energy point and the uncapped point on the same footing (a construct-matched comparison).
    df = df.groupby(["card", "workload", "cap_w"], as_index=False).agg(
        power_w=("power_w", "mean"), throughput=("throughput", "mean"))
    df["energy_J"] = df.power_w / df.throughput
    df["lat_ms"] = 1000.0 / df.throughput
    return df


def main():
    df = load()
    rows = []
    for card in ["A100", "A10G"]:
        for wl, g in df[df.card == card].groupby("workload"):
            g = g.sort_values("cap_w")
            mx = g.iloc[-1]                                   # uncapped = max performance
            ef = g.loc[g.energy_J.idxmin()]                  # min energy/step
            d = lambda a, b: 100.0 * (a - b) / b
            rows.append(dict(
                card=card, model=wl,
                max_thr=mx.throughput, max_lat=mx.lat_ms, max_E=mx.energy_J, max_W=mx.power_w,
                eff_thr=ef.throughput, eff_lat=ef.lat_ms, eff_E=ef.energy_J, eff_W=ef.power_w,
                eff_capTDP=100 * ef.cap_w / TDP[card],
                dThr=d(ef.throughput, mx.throughput), dLat=d(ef.lat_ms, mx.lat_ms), dE=d(ef.energy_J, mx.energy_J)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RAW, "efficiency_table.csv"), index=False)
    for card in ["A100", "A10G"]:
        s = t[t.card == card].sort_values("dE")
        print(f"\n===== {card} ({TDP[card]}W): full-performance (uncapped) vs energy-efficient (min energy) =====")
        print(f"  {'model':20} | {'MAX thr/lat/E':>22} | {'EFF thr/lat/E @cap':>26} | {'Δthr%':>6} {'Δlat%':>6} {'ΔE%':>6}")
        for _, r in s.iterrows():
            print(f"  {r.model:20} | {r.max_thr:6.1f}/s {r.max_lat:5.1f}ms {r.max_E:5.2f}J | "
                  f"{r.eff_thr:6.1f}/s {r.eff_lat:5.1f}ms {r.eff_E:5.2f}J @{r.eff_capTDP:3.0f}%TDP | "
                  f"{r.dThr:>+6.0f} {r.dLat:>+6.0f} {r.dE:>+6.0f}")
        print(f"  ---- {card} medians: Δthroughput {s.dThr.median():+.0f}%   Δlatency {s.dLat.median():+.0f}%   "
              f"Δenergy {s.dE.median():+.0f}%   (energy-optimal cap median {s.eff_capTDP.median():.0f}% TDP)")
    print("\nsaved -> data/raw/efficiency_table.csv")


if __name__ == "__main__":
    main()
