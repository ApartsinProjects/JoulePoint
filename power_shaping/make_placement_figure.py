# -*- coding: utf-8 -*-
"""
For a SINGLE job: duration (y) vs power draw (x), as the cap sweeps each card and the job is moved between
cards. Each card occupies its own power BAND on the x axis (they are disjoint), so the plot shows both
levers at once: capping moves you left/right WITHIN a card's band (trading power for duration), routing
JUMPS to another card's band. The lower envelope is the best duration achievable at each power budget.
Built from the measured AWS sweeps (A10G/L4/T4). Duration is normalized to the fastest point for the job.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict_power import RAW

CARDCOL = {"A10G": "#14385c", "L4": "#e08214", "T4": "#b2182b"}
INK = "#111418"


def load():
    fr = [pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")).assign(gpu="NVIDIA A10G"),
          pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    return df.groupby(["card", "workload", "cap_frac"], as_index=False).agg(power=("power_w", "mean"), thr=("throughput", "mean"))


def panel(ax, df, kernel, title):
    sub = df[df.workload == kernel]
    tmax = sub.thr.max()                                   # fastest anywhere -> duration_rel = tmax/thr (>=1)
    for card in ["A10G", "L4", "T4"]:
        c = sub[sub.card == card].sort_values("power")
        if c.empty:
            continue
        dur = tmax / c.thr.values
        ax.plot(c.power.values, dur, "-o", color=CARDCOL[card], lw=2, ms=5, mfc="white", label=card)
        # D0 = natural draw = highest-power (uncapped) point of this card
        p0, d0 = c.power.values[-1], dur[-1]
        ax.annotate(f"{card}\nuncapped", (p0, d0), textcoords="offset points", xytext=(4, -2),
                    fontsize=7, color=CARDCOL[card], va="top")
        # capping arrow: from uncapped toward the capped (lower-power, higher-duration) end
        if len(c) > 1:
            ax.annotate("", xy=(c.power.values[0], dur[0]), xytext=(p0, d0),
                        arrowprops=dict(arrowstyle="->", color=CARDCOL[card], lw=1, alpha=0.5))
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_xlabel("power draw (W)"); ax.set_ylabel("duration (relative to fastest)")
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")


def main():
    df = load()
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    panel(ax[0], df, "gemm_fp16", "(A) compute-bound job (gemm fp16)")
    panel(ax[1], df, "reduction", "(B) memory-bound job (reduction)")
    fig.suptitle("One job, duration vs power draw: capping moves within a card's band, routing jumps between bands",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(__file__), "figures", "fig_placement.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"figure -> figures/fig_placement.png")
    for k in ["gemm_fp16", "reduction"]:
        s = df[df.workload == k]
        for card in ["A10G", "L4", "T4"]:
            c = s[s.card == card]
            if not c.empty:
                print(f"  {k:12} {card:5} power {c.power.min():.0f}-{c.power.max():.0f}W  thr {c.thr.min():.1f}-{c.thr.max():.1f}")


if __name__ == "__main__":
    main()
