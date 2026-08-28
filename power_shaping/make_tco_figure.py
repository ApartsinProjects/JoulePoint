# -*- coding: utf-8 -*-
"""The TCO curve as ONE tradeoff curve, everything in a single energy currency (MWh per 1M inferences), with
capital converted to an equivalent energy via the electricity price. On the measured A100 ResNet-50 loaded
curve. Each point on the curve is one operating point (a power cap); sweeping the cap traces the whole set of
possibilities.
  x = capital per 1M inferences, as equivalent MWh (card cost attributed to 1M inferences / electricity price)
  y = energy per 1M inferences (opex), MWh
Moving down the sweep from uncapped raises capital (a slower card serves fewer inferences over its life) while
energy first falls, to the Joule point, then rises: so the Joule point sits in the middle of the curve, at its
least-energy dip. Because both axes are the same unit, total cost is x+y and a line of constant TCO is a 45-degree
diagonal; the cheapest operating point is where the curve touches the lowest such line. At today's electricity
price capital is the long axis, so that point sits near uncapped; as electricity gets dearer the capital axis
shrinks and the cheapest point slides toward the Joule point. Price assumptions are in the caption."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"
CARD_USD = 12000.0                      # A100 capital cost
LIFE_S = 4 * 8760 * 3600.0             # 4-year life, seconds
J_PER_MWH = 3.6e9
ELEC_TODAY = 80.0                       # $/MWh (current), used to put capital in energy units


def main():
    M = pd.read_csv(os.path.join(HERE, "data", "processed", "elf_master.csv"))
    s = M[(M.card == "A100") & (M.loaded_primary) & (M.workload == "resnet50")].sort_values("cap_w")
    P = s.power_w.values; batch = int(s.batch.iloc[0]); R_inf = s.throughput.values * batch
    opex = 1e6 * (P / R_inf) / J_PER_MWH                       # MWh per 1M inferences (energy)
    capex = (1e6 * CARD_USD / (R_inf * LIFE_S)) / ELEC_TODAY   # equivalent MWh per 1M inferences (capital)
    order = np.argsort(capex); cx, cy = capex[order], opex[order]
    unc = int(np.argmin(capex))                               # uncapped = least capital (fastest card)
    j = int(np.argmin(opex))                                  # Joule point = least energy
    jo = list(order).index(j)                                 # Joule position in capex-sorted order

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # efficient arc: uncapped (least capital, most energy) -> Joule point (least energy, most cards)
    ax.plot(cx[:jo + 1], cy[:jo + 1], "-o", color="#14385c", lw=2.6, ms=4.5, mfc="white", zorder=4,
            label="efficient tradeoff (uncapped → Joule point)")
    # dominated tail beyond the Joule point: more cards AND more energy
    ax.plot(cx[jo:], cy[jo:], "--o", color="#9aa4b0", lw=1.4, ms=3, mfc="white", zorder=3, alpha=0.7,
            label="dominated (more cards and more energy)")
    ax.scatter([capex[unc]], [opex[unc]], s=120, color="#5a626c", edgecolor="white", linewidth=1.3, zorder=6)
    ax.annotate("uncapped\n(fewest cards, most energy)", (capex[unc], opex[unc]), xytext=(8, -2),
                textcoords="offset points", fontsize=8.8, color="#5a626c")
    ax.scatter([capex[j]], [opex[j]], s=210, marker="*", color="#b2182b", edgecolor="white", linewidth=1.3, zorder=7)
    ax.annotate("Joule point\n(least energy, most cards)", (capex[j], opex[j]), xytext=(10, 6),
                textcoords="offset points", fontsize=9.2, color="#b2182b", fontweight="bold")
    # cheapest total (capital + energy, both MWh) at today's prices, and where it moves as electricity gets dearer
    tot = capex + opex; kmin = int(np.argmin(tot))
    ax.scatter([capex[kmin]], [opex[kmin]], s=110, color="#2a9d8f", edgecolor="white", linewidth=1.2, zorder=6)
    ax.annotate("cheapest total cost\ntoday (capital dominates)", (capex[kmin], opex[kmin]), xytext=(10, -20),
                textcoords="offset points", fontsize=8.6, color="#2a9d8f", fontweight="bold")
    ax.annotate("", xy=(capex[j], opex[j] * 1.05), xytext=(capex[kmin], opex[kmin] * 0.97),
                arrowprops=dict(arrowstyle="->", color="#2a9d8f", lw=1.4, ls="--"), zorder=5)
    ax.text((capex[kmin] + capex[j]) / 2, opex[kmin] * 0.90, "as electricity\ngets dearer",
            fontsize=8, color="#2a9d8f", ha="center", style="italic")

    ax.set_xlabel("capital per 1M inferences  (equivalent MWh at $80/MWh)   →  more cards", fontsize=9.4)
    ax.set_ylabel("energy per 1M inferences  (MWh)   ↓ = greener", fontsize=9.4)
    ax.set_title("The TCO curve: each power cap is one point trading capital for energy. The efficient arc runs from\n"
                 "uncapped to the Joule point; today capital dominates so the cheapest point is near uncapped",
                 fontsize=9.0, loc="left", color=INK)
    ax.set_xlim(capex.min() * 0.85, capex[j] * 1.5); ax.set_ylim(opex[j] * 0.82, opex[unc] * 1.14)
    ax.grid(True, alpha=0.18); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.6, frameon=False, loc="upper center")
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_tco.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"figure -> fig_tco.png  Joule at capex={capex[j]:.2e}/opex={opex[j]:.2e} MWh; "
          f"cheapest-total draw {100*P[kmin]/400:.0f}% vs Joule {100*P[j]/400:.0f}% TDP")


if __name__ == "__main__":
    main()
