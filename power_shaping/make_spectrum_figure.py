# -*- coding: utf-8 -*-
"""
Do all jobs have the same shape? On one card (A10G, all 26 kernels), plot each kernel's slowdown vs the cap
expressed RELATIVE to that kernel's natural draw D0 (so the knee lines up at x=1 for every job). The hinge is
universal (flat at x>=1, bending below), but the SLOPE below the knee fans out across a spectrum: memory-bound
kernels stay flat, compute-bound kernels bend steeply. Colored by arithmetic intensity (compute-vs-memory).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from predict_power import RAW, KAI

INK = "#111418"


def main():
    fr = [pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")).assign(gpu="NVIDIA A10G"),
          pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    df = df[df.card == "A10G"].groupby(["workload", "cap_frac"], as_index=False).agg(power=("power_w", "mean"), thr=("throughput", "mean"))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ai = np.array([KAI.get(w, 1) for w in df.workload.unique()])
    norm = plt.Normalize(np.log10(0.3), np.log10(300))
    for w in df.workload.unique():
        c = df[df.workload == w].sort_values("power")
        D0 = c.power.max()                                   # natural draw = uncapped
        x = c.power.values / D0                              # cap relative to natural draw (knee at 1)
        slow = 1 - c.thr.values / c.thr.values[-1]           # slowdown vs uncapped
        col = cm.coolwarm(norm(np.log10(KAI.get(w, 1))))
        ax.plot(x, slow, "-", color=col, lw=1.6, alpha=0.85)
    ax.axvline(1.0, color="#888", ls=(0, (3, 3)), lw=1)
    ax.text(1.01, 0.02, "knee = natural draw\n(cap above here does nothing)", fontsize=7.2, color="#555", va="bottom")
    ax.set_xlabel("power cap  ÷  job's natural draw"); ax.set_ylabel("slowdown (fractional)")
    ax.set_title("Every job has the hinge; the slope below the knee is a spectrum", fontsize=10, loc="left", color=INK)
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02); cb.set_label("arithmetic intensity (memory-bound → compute-bound)", fontsize=8)
    ax.text(0.35, 0.9, "compute-bound:\nsteep", fontsize=8, color="#b2182b", transform=ax.transAxes)
    ax.text(0.05, 0.12, "memory-bound: flat", fontsize=8, color="#2166ac", transform=ax.transAxes)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "figures", "fig_spectrum.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_spectrum.png")


if __name__ == "__main__":
    main()
