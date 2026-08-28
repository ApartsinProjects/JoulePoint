# -*- coding: utf-8 -*-
"""Energy per inference vs power cap, for four jobs on the A100 (widest cap range, 100-400W). Energy =
power_w * t_compute (mJ/inference). Where beta>1, the product of falling power and rising time has an
interior minimum -> the energy-optimal cap R*, marked. Memory-bound jobs are monotone (cap only helps)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict_power import RAW

INK = "#111418"
JOBS = [("llm_decode", "#2166ac", "LLM decode"), ("resnet152", "#4d9221", "ResNet-152"),
        ("sd_unet", "#d6604d", "SD UNet"), ("vit_b_16", "#762a83", "ViT-B/16")]


def main():
    d = pd.read_csv(os.path.join(RAW, "aws_sweep_a100.csv"))
    d["card"] = "A100"
    g = d.groupby(["workload", "cap_w"], as_index=False).agg(
        P=("power_w", "mean"), tc=("t_compute_ms", "mean"))
    g["E"] = g.P * g.tc                       # mJ per inference
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for wl, col, lab in JOBS:
        c = g[g.workload == wl].sort_values("cap_w")
        En = c.E.values / c.E.max()           # normalize each job to its own max-energy point
        ax.plot(c.cap_w, En, "-o", color=col, lw=2, ms=5, mfc="white", label=lab)
        imin = int(np.argmin(En))
        ax.plot(c.cap_w.values[imin], En[imin], "*", color=col, ms=16, mec="black", mew=.4, zorder=5)
        ax.annotate(f"{c.cap_w.values[imin]:.0f}W", (c.cap_w.values[imin], En[imin]),
                    textcoords="offset points", xytext=(4, -12), fontsize=8, color=col)
    ax.set_xlabel("power cap (W)"); ax.set_ylabel("energy per inference (relative to each job's max)")
    ax.set_title("Energy per inference vs cap on the A100: each job has an energy-optimal power (★)",
                 fontsize=10.5, loc="left", color=INK)
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, title="job", title_fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "figures", "fig_energy_optimal.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_energy_optimal.png")
    for wl, _, lab in JOBS:
        c = g[g.workload == wl].sort_values("cap_w")
        i = int(np.argmin(c.E.values))
        print(f"  {lab:12} energy-optimal cap {c.cap_w.values[i]:.0f}W (of {c.cap_w.max():.0f}W)  "
              f"saves {100*(1-c.E.values[i]/c.E.values[-1]):.0f}% energy vs uncapped")


if __name__ == "__main__":
    main()
