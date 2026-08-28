# -*- coding: utf-8 -*-
"""
Same job, same axes: latency vs ACTUAL DRAW (solid) and latency vs the CAP SETTING (dashed), both in watts,
with each card on its own band. Every measured operating point contributes one latency at two x-positions:
its actual drain (power_w) and the cap that produced it (cap_w). The two curves reveal the hinge directly:
 - where cap BINDS (cap below the job's natural draw D0), the dashed cap curve sits ON the solid drain curve
   (the card draws right up to the cap), and latency climbs as you push the cap down;
 - where cap is INERT (above D0), the drain curve stops at D0 while the dashed cap curve runs flat to the
   RIGHT of it -- raising the cap moves the setting but not the draw and not the latency.
The horizontal gap dashed-minus-solid at a fixed latency is the WASTED cap headroom (watts you "allowed"
that were never drawn). Built from the measured AWS sweeps; latency normalized to the fastest point per job.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict_power import RAW

CARDCOL = {"A10G": "#14385c", "L4": "#e08214", "T4": "#b2182b",
           "L40S": "#1b7837", "V100": "#762a83"}
INK = "#111418"
ORDER = ["A10G", "L4", "T4", "L40S", "V100"]


def load():
    fr = [pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")).assign(gpu="NVIDIA A10G"),
          pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    return df.groupby(["card", "workload", "cap_frac"], as_index=False).agg(
        drain=("power_w", "mean"), cap=("cap_w", "mean"), thr=("throughput", "mean"))


def panel(ax, df, kernel, title):
    sub = df[df.workload == kernel]
    if sub.empty:
        ax.set_visible(False); return None
    tmax = sub.thr.max()                                    # fastest anywhere -> latency = tmax/thr (>=1)
    cards = [c for c in ORDER if c in set(sub.card)]
    widest = None                                           # card with the biggest inert band, for the callout
    for card in cards:
        c = sub[sub.card == card].sort_values("cap")
        if c.empty:
            continue
        col = CARDCOL[card]
        lat = tmax / c.thr.values
        # solid: latency vs ACTUAL DRAIN ; dashed: latency vs CAP SETTING (same latencies, shifted x)
        ax.plot(c.drain.values, lat, "-o", color=col, lw=2, ms=4.5, mfc="white", zorder=3, label=f"{card}")
        ax.plot(c.cap.values, lat, "--", color=col, lw=1.5, alpha=0.9, zorder=2)
        # shade the wasted-headroom gap (cap minus drain) at each operating point
        for xi_d, xi_c, yi in zip(c.drain.values, c.cap.values, lat):
            if xi_c - xi_d > 3:
                ax.plot([xi_d, xi_c], [yi, yi], color=col, lw=0.6, alpha=0.35, zorder=1)
        # mark natural draw D0 (uncapped operating point = largest cap)
        p0 = c.drain.values[-1]
        ax.annotate(f"{card}  D0", (p0, lat[-1]), textcoords="offset points", xytext=(3, -1),
                    fontsize=7, color=col, va="top")
        gap = float(c.cap.max() - c.drain.max())
        if widest is None or gap > widest[0]:
            widest = (gap, card, col, c.drain.max(), c.cap.max(), lat[-1])
    # callout on the widest inert band: raising the cap across it changes nothing
    if widest and widest[0] > 20:
        g, card, col, d0, capmax, y0 = widest
        ax.annotate("", xy=(capmax, y0), xytext=(d0, y0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        ax.annotate(f"cap raised {g:.0f} W past D0\n-> drain & latency unchanged",
                    ((d0 + capmax) / 2, y0), textcoords="offset points", xytext=(0, 8),
                    fontsize=7.2, color=col, ha="center", va="bottom")
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_xlabel("watts  (solid = actual drain,  dashed = cap setting)")
    ax.set_ylabel("latency (relative to fastest)")
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
    return widest


def build(df, specA, specB, out, suptitle):
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    panel(ax[0], df, specA[0], specA[1])
    panel(ax[1], df, specB[0], specB[1])
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#555", lw=2, marker="o", mfc="white", label="latency vs actual drain"),
               Line2D([0], [0], color="#555", lw=1.5, ls="--", label="latency vs cap setting")]
    fig.legend(handles=handles, fontsize=8, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure ->", os.path.basename(out))
    for k in [specA[0], specB[0]]:
        s = df[df.workload == k]
        for card in [c for c in ORDER if c in set(s.card)]:
            c = s[s.card == card]
            gap = (c.cap - c.drain)
            print(f"  {k:16} {card:5} drain {c.drain.min():.0f}-{c.drain.max():.0f}W  "
                  f"cap {c.cap.min():.0f}-{c.cap.max():.0f}W  max wasted headroom {gap.max():.0f}W")


def main():
    df = load()
    figdir = os.path.join(os.path.dirname(__file__), "figures")
    # (1) microbenchmark kernels: the clean physics endpoints
    build(df,
          ("gemm_fp16", "(A) compute-bound kernel (gemm fp16): cap binds, dashed on solid"),
          ("reduction", "(B) memory-bound kernel (reduction): wide inert band, cap >> drain"),
          os.path.join(figdir, "fig_cap_vs_drain.png"),
          "Microbenchmark kernels: actual drain (solid) vs cap setting (dashed). Gap = wasted cap headroom.")
    # (2) high-level AWS tasks: whole-model inference passes
    build(df,
          ("attention_sdpa", "(A) attention (transformer): saturates the card, cap binds near TDP"),
          ("resnet152", "(B) ResNet-152 inference: ~90 W inert band on A10G, cap >> drain"),
          os.path.join(figdir, "fig_cap_vs_drain_tasks.png"),
          "High-level AWS tasks (whole-model passes): actual drain (solid) vs cap setting (dashed).")


if __name__ == "__main__":
    main()
