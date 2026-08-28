# -*- coding: utf-8 -*-
"""Power-constrained goodput, stacked: behind a fixed 1 MW budget, how the megawatt splits between STATIC (idle
floor N*P0) and DYNAMIC (useful compute N*(P-P0)) power as more cards are installed, with served throughput
overlaid. Adding cards (capping harder) trades dynamic budget for static overhead; served throughput peaks at
the Joule point, where the draw is (beta-1)/beta static and only 1/beta dynamic. Measured A100 ViT-B/32."""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings; warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__); INK = "#111418"; W = 1e6


def main():
    M = pd.read_csv(os.path.join(HERE, "data", "processed", "elf_master.csv"))
    g = M[(M.card == "A100") & (M.loaded_primary) & (M.workload == "vit_b_32")].sort_values("power_w")
    P = g.power_w.values; batch = int(g.batch.iloc[0]); R = g.throughput.values * batch
    # empirical floor: the least measured board draw (idle + fixed), so the static/dynamic split stays physical
    P0 = P.min(); b = None
    N = W / P
    static = N * P0 / 1000.0                 # kW
    dynamic = (W - N * P0) / 1000.0          # kW
    served = W * R / P; u = int(np.argmax(P)); sn = served / served[u]
    lat = R[u] / R                          # avg per-request latency, relative to uncapped (the card multiplier)
    j = int(np.argmax(served))

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.fill_between(N, 0, dynamic, color="#2166ac", alpha=0.85, label="dynamic power  (useful compute, N(P-P0))", zorder=2)
    ax.fill_between(N, dynamic, dynamic + static, color="#c9ccd1", alpha=0.9, label="static power  (idle floor, N*P0)", zorder=2)
    ax.set_ylabel("power behind the 1 MW feed  (kW)", fontsize=10)
    ax.set_xlabel("GPUs installed behind a fixed 1 MW feed", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(N, sn, "-", color="#b2182b", lw=2.6, zorder=5, label="served inferences per MW (rel. uncapped)")
    ax2.scatter([N[j]], [sn[j]], s=180, marker="*", color="#b2182b", edgecolor="white", linewidth=1.2, zorder=6)
    ax2.plot(N, lat, "-", color="#e08214", lw=2.2, zorder=5, label="avg per-request latency (rel. uncapped)")
    ax2.scatter([N[j]], [lat[j]], s=70, color="#e08214", edgecolor="white", linewidth=1.1, zorder=6)
    ax2.annotate(f"{lat[j]:.1f}x latency", (N[j], lat[j]), xytext=(8, 2), textcoords="offset points",
                 fontsize=8.2, color="#c05a2b")
    ax2.set_ylabel("relative to uncapped  (served, latency)", fontsize=10, color="#5a626c")
    latvis = lat[N <= 8000].max()
    ax2.tick_params(axis="y", colors="#5a626c"); ax2.set_yscale("log"); ax2.set_ylim(0.75, latvis * 1.12)
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax2.yaxis.set_major_locator(FixedLocator([0.8, 1.0, 1.4, 2.0, 3.0]))
    ax2.yaxis.set_major_formatter(FixedFormatter(["0.8", "1.0", "1.4", "2.0", "3.0"]))
    ax.axvline(N[j], color="#b2182b", lw=1.0, ls=":", zorder=4)
    ax.annotate(f"Joule Point\n{N[j]:.0f} cards, +{100*(sn[j]-1):.0f}% served\n(draw is {100*P0/P[j]:.0f}% static)",
                (N[j], (dynamic[j]) * 0.5), xytext=(12, 0), textcoords="offset points", fontsize=8.8,
                color="#14385c", fontweight="bold")
    XMAX = 8000
    ax.set_xlim(N.min(), XMAX); ax.set_ylim(0, 1050)
    # top axis: power allocated per card (% of TDP) = W/N / TDP; each card count implies a per-card cap
    tdp = 400.0
    sx = ax.secondary_xaxis("top", functions=(lambda n: 100 * (W / np.maximum(n, 1)) / tdp,
                                              lambda d: W / (np.maximum(d, 1) / 100 * tdp)))
    sx.set_xlabel("power allocated per card  (% of TDP)   ← deeper cap", fontsize=9, color="#5a626c")
    sx.tick_params(colors="#5a626c")
    # no title on the image; the description lives in the paper caption
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=8.3, loc="upper left", frameon=True, facecolor="white",
              framealpha=0.92, edgecolor="none")
    ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_power_budget_stacked.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"figure -> figures/fig_power_budget_stacked.png  floor P0={P0:.0f}W, at Joule static={100*P0/P[j]:.0f}% of draw, +{100*(sn[j]-1):.0f}% served")


if __name__ == "__main__":
    main()
