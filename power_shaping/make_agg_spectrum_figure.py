# -*- coding: utf-8 -*-
"""
Do AGGREGATED real jobs (DNN training, LLM inference/training) follow the same hinge as the microbenchmarks?
Panel A: Zeus training jobs (A40 card). Panel B: emerald LLM inference/training jobs. For each job, slowdown
vs draw relative to its natural draw D0 (knee at 1), colored by draw-fraction (D0/TDP), a compute-vs-memory
proxy. If real jobs also sit on the hinge spectrum, the model generalizes beyond kernels.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from predict_power import RAW

INK = "#111418"


def panel_zeus(ax):
    d = pd.read_csv(os.path.join(RAW, "zeus_summary_power_a40.csv"))
    d = d.groupby(["network", "dataset", "batch_size", "optimizer", "power_limit"], as_index=False).agg(
        t=("time_per_epoch", "mean"), p=("average_power", "mean"))
    d = d[d.t > 0].copy(); d["rate"] = 1.0 / d.t
    d["job"] = d.groupby(["network", "dataset", "batch_size", "optimizer"]).ngroup()
    norm = plt.Normalize(0.4, 1.0)
    for _, g in d.groupby("job"):
        g = g.sort_values("power_limit"); D0 = g.p.max()
        x = g.p.values / D0; slow = 1 - g.rate.values / g.rate.values[-1]
        ax.plot(x, slow, "-", lw=1.4, alpha=0.8, color=cm.coolwarm(norm(D0 / 300.0)))
    ax.set_title("(A) DNN training jobs (Zeus, A40)", fontsize=9.5, loc="left", color=INK)


def panel_emerald(ax):
    d = pd.read_csv(os.path.join(RAW, "dvfs_sweep.csv")).rename(
        columns={"Workload": "w", "throughput": "thr", "power per GPU": "p"})
    d = d[(d.thr > 0) & (d.p > 0)].copy()
    norm = plt.Normalize(0.4, 1.0); tdp = d.p.max()
    for w, g in d.groupby("w"):
        g = g.sort_values("p"); D0 = g.p.max()
        x = g.p.values / D0; slow = 1 - g.thr.values / g.thr.values[-1]
        lab = w if len(w) < 18 else w[:16]
        ax.plot(x, slow, "-o", lw=1.6, ms=3, alpha=0.85, color=cm.coolwarm(norm(D0 / tdp)))
    ax.set_title("(B) LLM inference/training jobs (emerald)", fontsize=9.5, loc="left", color=INK)


def main():
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    panel_zeus(ax[0]); panel_emerald(ax[1])
    for a in ax:
        a.axvline(1.0, color="#888", ls=(0, (3, 3)), lw=1)
        a.set_xlabel("draw ÷ job's natural draw"); a.grid(True, alpha=0.25)
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    ax[0].set_ylabel("slowdown (fractional)")
    fig.suptitle("Aggregated real jobs follow the same hinge (knee at 1; slope = compute-vs-memory)",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(__file__), "figures", "fig_agg_spectrum.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_agg_spectrum.png")


if __name__ == "__main__":
    main()
