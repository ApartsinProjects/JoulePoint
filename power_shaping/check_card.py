# -*- coding: utf-8 -*-
"""
Does a low-drain (memory-bound) job slow down on a weaker card? For each kernel we take the UNCAPPED
throughput on each card and the throughput RELATIVE to A10G, and split kernels into memory-bound vs
compute-bound. If low-drain memory-bound kernels slow down on the lower-BANDWIDTH cards (L4 300, T4 320
vs A10G 600 GB/s), then low drain does NOT imply no slowdown on a weaker card.
"""
import os
import numpy as np
import pandas as pd
from predict_power import RAW, KOP, KAI

BW = {"A10G": 600, "L4": 300, "T4": 320}
FP16 = {"A10G": 70, "L4": 121, "T4": 65}


def main():
    fr = [pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")).assign(gpu="NVIDIA A10G"),
          pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    # uncapped = max cap_frac per (card, workload); take that throughput
    df = df.groupby(["card", "workload", "cap_frac"], as_index=False).agg(power=("power_w", "mean"), thr=("throughput", "mean"))
    top = df.loc[df.groupby(["card", "workload"])["cap_frac"].idxmax()]
    piv = top.pivot_table(index="workload", columns="card", values="thr")
    pw = top.pivot_table(index="workload", columns="card", values="power")
    piv = piv[piv["A10G"].notna() & piv["L4"].notna()].copy()   # L4-vs-A10G pair (L4 has 21 kernels)
    piv["bound"] = [KOP.get(w, "?") for w in piv.index]
    piv["memory_bound"] = piv["bound"].isin(["reduction", "datamove"])
    piv["L4_rel"] = piv["L4"] / piv["A10G"]
    a10_draw = pw["A10G"] / pw["A10G"].max()

    print("== Uncapped throughput on L4 relative to A10G (L4 has HALF the bandwidth: 300 vs 600 GB/s) ==")
    print("   (if a low-drain memory-bound job slows down here, low drain does NOT imply no slowdown)\n")
    for grp, sub in piv.groupby("memory_bound"):
        lab = "MEMORY-bound (low-drain)" if grp else "COMPUTE-bound"
        print(f"{lab}: n={len(sub)}   L4/A10G throughput = {sub.L4_rel.mean():.2f}x   (median {sub.L4_rel.median():.2f}x)")
    print("\nper-kernel:")
    for w in piv.sort_values(["memory_bound", "L4_rel"], ascending=[False, True]).index:
        r = piv.loc[w]
        print(f"  {w:18} {'MEM' if r.memory_bound else 'CMP'}  A10G_draw~{a10_draw.get(w, float('nan')):.2f}  L4 {r.L4_rel:.2f}x")


if __name__ == "__main__":
    main()
