# -*- coding: utf-8 -*-
"""Power-constrained goodput: behind a fixed power budget, the Joule point is the INTERIOR optimum, not a corner.
For a fixed budget W, installing N cards forces each to the cap with per-card power W/N; served throughput is
then N * (per-card rate) = W * (rate/power) = W / e(theta), so maximizing served work is exactly minimizing
energy per inference -- the Joule point. Too few cards run hot past the Joule point (the superlinear tail wastes
the budget); too many cards drown the budget in the idle floor N*P0. Computed from the measured A100 ViT-B/32
loaded sweep, W = 1 MW."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"
W = 1e6  # 1 MW budget


def main():
    M = pd.read_csv(os.path.join(HERE, "data", "processed", "elf_master.csv"))
    g = M[(M.card == "A100") & (M.loaded_primary) & (M.workload == "vit_b_32")].sort_values("power_w")
    P = g.power_w.values; batch = int(g.batch.iloc[0]); R = g.throughput.values * batch
    N = W / P                                   # cards that fit the budget at this operating point
    served = W * R / P                          # total inferences/s served
    draw = 100 * P / 400.0
    j = int(np.argmax(served)); u = int(np.argmax(P))
    sn = served / served[u]                      # relative to uncapped

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(N, sn, "-o", color="#14385c", lw=2.6, ms=6, mfc="white", zorder=4)
    ax.axhline(1.0, color="#999", lw=0.8, ls=":", zorder=1)
    ax.scatter([N[u]], [sn[u]], s=130, color="#5a626c", edgecolor="white", linewidth=1.3, zorder=6)
    ax.annotate(f"uncapped\n{N[u]:.0f} cards @ {draw[u]:.0f}% TDP", (N[u], sn[u]), xytext=(10, -4),
                textcoords="offset points", fontsize=8.8, color="#5a626c")
    ax.scatter([N[j]], [sn[j]], s=220, marker="*", color="#b2182b", edgecolor="white", linewidth=1.3, zorder=7)
    ax.annotate(f"Joule point: {N[j]:.0f} cards @ {draw[j]:.0f}% TDP\n+{100*(sn[j]-1):.0f}% inferences per MW",
                (N[j], sn[j]), xytext=(-8, -34), textcoords="offset points", fontsize=9.2, color="#b2182b",
                fontweight="bold", ha="center")
    # branch annotations (rising = few hot cards; falling = too many idle-heavy cards)
    ax.annotate("too few, hot cards:\nsuperlinear tail\nwastes the budget", (3150, 1.15),
                fontsize=8, color="#5a626c", ha="center", style="italic")
    ax.annotate("too many cards:\nidle floor eats\nthe budget", (8600, 1.02),
                fontsize=8, color="#5a626c", ha="center", style="italic")
    ax.set_xlabel("GPUs installed behind a fixed 1 MW feed", fontsize=10)
    ax.set_ylabel("inferences served per MW  (relative to uncapped)", fontsize=10)
    ax.set_title("Behind a fixed power budget, served throughput peaks at the Joule point: a 1 MW site serves the\n"
                 "most inferences with its cards capped there; too few hot cards or too many idle cards serve less",
                 fontsize=9.1, loc="left", color=INK)
    ax.grid(True, alpha=0.18); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_power_budget.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"figure -> figures/fig_power_budget.png  uncapped {N[u]:.0f} cards -> Joule {N[j]:.0f} cards, "
          f"+{100*(sn[j]-1):.0f}% inferences/MW")


if __name__ == "__main__":
    main()
