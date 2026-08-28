# -*- coding: utf-8 -*-
"""
For a memory-bound job at a MATCHED power budget: is it faster on the strong card CAPPED, or the weak card
UNCAPPED? We take each memory-bound kernel's L4 (weak) uncapped operating point (power_L4, thr_L4), then read
the A10G (strong) throughput CAPPED to that same power, and compare. If A10G-capped > L4-uncapped at equal
power, "strong card + cap" wins for memory-bound work.
"""
import os
import numpy as np
import pandas as pd
from predict_power import RAW, KOP

MEM = [k for k, v in KOP.items() if v in ("reduction", "datamove")]


def main():
    fr = [pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")).assign(gpu="NVIDIA A10G"),
          pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    df = df.groupby(["card", "workload", "cap_frac"], as_index=False).agg(power=("power_w", "mean"), thr=("throughput", "mean"))

    print("== Memory-bound job at matched power: strong (A10G) capped vs weak (L4) uncapped ==\n")
    print(f"{'kernel':14} {'L4 power':>8} {'L4 thr':>10} | {'A10G thr @ that power':>22} {'A10G/L4':>8}")
    print("-" * 70)
    wins = []
    for k in MEM:
        a = df[(df.card == "A10G") & (df.workload == k)].sort_values("power")
        l = df[(df.card == "L4") & (df.workload == k)]
        if a.empty or l.empty:
            continue
        lu = l.loc[l.cap_frac.idxmax()]            # L4 uncapped operating point
        pL, tL = lu.power, lu.thr
        # A10G throughput capped to power pL (interpolate its power->throughput curve; clamp to its range)
        tA = float(np.interp(pL, a.power.values, a.thr.values))
        ratio = tA / tL
        wins.append(ratio)
        note = "" if a.power.min() <= pL <= a.power.max() else "  (extrapolated: L4 power outside A10G cap range)"
        print(f"{k:14} {pL:>7.0f}W {tL:>10.1f} | {tA:>22.1f} {ratio:>7.2f}x{note}")
    if wins:
        print(f"\nmean A10G-capped / L4-uncapped throughput at matched power = {np.mean(wins):.2f}x  "
              f"({'strong+cap wins' if np.mean(wins) > 1 else 'weak+uncapped wins'})")


if __name__ == "__main__":
    main()
