# -*- coding: utf-8 -*-
"""Does the energy U-shape appear on SMALL cards for lighter models? Two panels, same models:
L4 (72W, small) vs A100 (400W, big). x = actual draw (W), y = energy per step (J). Filled dot = energy min.
Shows the U is deep on the big card and shallow/absent on the small card, except light models on the L4.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict_power import RAW

INK = "#111418"
SHOW = ["shufflenet_v2_x1_0", "mobilenet_v3_large", "efficientnet_b0", "resnet50", "vit_b_16", "llm_decode"]
COL = {"shufflenet_v2_x1_0": "#b2182b", "mobilenet_v3_large": "#ef8a62", "efficientnet_b0": "#fddbc7",
       "resnet50": "#67a9cf", "vit_b_16": "#2166ac", "llm_decode": "#14385c"}
LBL = {"shufflenet_v2_x1_0": "shufflenet (tiny)", "mobilenet_v3_large": "mobilenet (tiny)",
       "efficientnet_b0": "efficientnet_b0 (light)", "resnet50": "resnet50 (mid)",
       "vit_b_16": "vit_b_16 (heavy)", "llm_decode": "llm_decode"}


def load(card_file, card_name):
    d = pd.read_csv(os.path.join(RAW, card_file))
    d = d[d["mode"] == "control"]
    d["card"] = d.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    d["card"] = d.card.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    d = d[d.card == card_name]
    d["energy_J"] = d.power_w / d.throughput
    return d


def panel(ax, d, title):
    for wl in SHOW:
        s = d[d.workload == wl].sort_values("power_w")
        if len(s) == 0:
            continue
        E = s.energy_J.values / s.energy_J.values.min()          # normalize each to its own minimum
        ax.plot(s.power_w, E, "-o", color=COL[wl], lw=1.6, ms=3.5, mfc="white", label=LBL[wl], zorder=3)
        k = int(np.argmin(s.energy_J.values))
        ax.scatter([s.power_w.values[k]], [E[k]], s=70, facecolor=COL[wl], edgecolor="white", lw=1.1, zorder=5)
    ax.axhline(1.0, color="#888", lw=0.8, ls=":", zorder=1)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_xlabel("actual board draw (W)", fontsize=9.5)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def main():
    l4 = load("aws_sweep_batch.csv", "L4")
    a100 = load("aws_sweep_a100_batch.csv", "A100")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    panel(ax[0], l4, "L4 (72W, small card): U is shallow, little energy to reclaim by capping")
    panel(ax[1], a100, "A100 (400W, big card): deep U, every model wastes energy uncapped")
    ax[0].set_ylabel("energy per step  /  its own minimum   (1.0 = optimal)", fontsize=9.5)
    ax[1].legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    fig.suptitle("Does the energy U-shape appear on small cards? Same models, small vs big GPU",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(__file__), "figures", "fig_ushape_small_vs_big.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_ushape_small_vs_big.png")
    # numbers
    for name, d in [("L4", l4), ("A100", a100)]:
        print(f"\n  {name}: uncapped energy penalty vs each model's minimum")
        for wl in SHOW:
            s = d[d.workload == wl].sort_values("power_w")
            if len(s) == 0:
                continue
            E = s.energy_J.values; k = int(np.argmin(E))
            print(f"    {wl:20} min@{s.power_w.values[k]:4.0f}W  uncapped +{100*(E[-1]-E[k])/E[k]:4.0f}%")


if __name__ == "__main__":
    main()
