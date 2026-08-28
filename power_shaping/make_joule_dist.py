# -*- coding: utf-8 -*-
"""Three per-card panels (A100, A10G, L4) of the Joule-point distribution across the 20 workloads (loaded
regime). Each panel is on its own % of TDP range, showing that within a card the energy-optimal operating point
is tightly peaked: one fixed point per card serves every job. Measured argmin of energy per inference; the x
axis is board draw (% TDP) at the minimum, so it is knob-agnostic (large cards via power cap, L4 via clock). The
T4 is excluded from the analysis (see paper limitations)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"
COL = {"T4": "#b2182b", "L4": "#ef8a62", "A10G": "#2166ac", "A100": "#14385c"}
TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}


def main():
    M = pd.read_csv(os.path.join(HERE, "data", "processed", "elf_master.csv"))
    # per-card control-range knob: large cards via power cap, L4 via clock (draw-based, knob-agnostic x axis)
    KNOB = {"A100": "loaded_primary", "A10G": "loaded_primary", "L4": "clock"}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    rng = np.random.RandomState(0)
    for ax, card in zip(axes.ravel(), ["A100", "A10G", "L4"]):
        if KNOB[card] == "clock":
            sub = M[(M.card == card) & (M.actuator == "clock")]
        else:
            sub = M[(M.card == card) & (M.loaded_primary)]
        jc = []
        for wl, g in sub.groupby("workload"):
            g = g.assign(E=g.power_w / g.throughput).sort_values("power_w")
            E = g.E.values
            jc.append(100 * g.power_w.values[int(np.argmin(E))] / TDP[card])  # draw %TDP at energy min
        jc = np.array(jc); med = np.median(jc); sd = jc.std()
        # strip of workloads (jittered) + median + +/-1 sd band
        ax.axvspan(med - sd, med + sd, color=COL[card], alpha=0.12)
        ax.axvline(med, color=COL[card], lw=2.0)
        ax.scatter(jc, 0.5 + (rng.rand(len(jc)) - 0.5) * 0.5, s=44, color=COL[card], alpha=0.8,
                   edgecolor="white", linewidth=0.5)
        lo = min(jc.min() - 6, med - 10); hi = max(jc.max() + 6, med + 10)
        ax.set_xlim(lo, hi); ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"{card}: median {med:.0f}% TDP, sd {sd:.1f} pts", fontsize=10, loc="left",
                     color=COL[card], fontweight="bold")
        ax.set_xlabel("energy-optimal draw  (Joule point, % of TDP)", fontsize=9)
        ax.grid(True, axis="x", alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.annotate(f"n={len(jc)} workloads", (0.97, 0.9), xycoords="axes fraction", ha="right",
                    fontsize=8, color="#5a626c")
    fig.suptitle("The Joule point is a per-card constant: its distribution across 20 workloads is tightly peaked on each GPU\n"
                 "(each panel on its own % of TDP range; shaded band = +/-1 standard deviation; line = per-card median)",
                 fontsize=10, y=1.02, color=INK)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_joule_dist.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_joule_dist.png")


if __name__ == "__main__":
    main()
