# -*- coding: utf-8 -*-
"""Energy-to-capital break-even horizon across NVIDIA data-center GPU generations, 2018-2025.
break_even_years = card_price / (power_kW * electricity_$per_kWh * 8760)
= the years of continuous full-power running at which cumulative electricity spend equals the card's price.
Card prices and TDPs are representative/sourced figures (NVIDIA never published V100/A100 MSRPs; H100+ are
2024-26 street/analyst prices). US industrial electricity from EIA. Two curves: card-only TDP, and full-system
power (~2x TDP: PUE, cooling, host, networking). The shaded band is a typical ~3-5 year depreciation life.
The horizon stays decades above that life throughout, so capital dominates energy in TCO (published split
~9-12% electricity); the one exception is a price spike like the EU 2022 crisis (~5-8x US industrial)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"
# (label, launch year, representative price USD, TDP W, industrial elec that year $/kWh) -- sourced/estimated
CARDS = [
    ("V100", 2018, 10000, 300, 0.069),
    ("A100", 2020, 12000, 400, 0.0667),
    ("H100", 2023, 30000, 700, 0.0804),
    ("H200", 2024, 31000, 700, 0.0813),
    ("B200", 2025, 50000, 1000, 0.086),
]
SYS_MULT = 2.0   # full-system power / GPU TDP (PUE + cooling + host + networking), ~1.5-2.3x in practice


def be(price, w, elec, mult=1.0):
    return price / ((w / 1000.0) * mult * elec * 8760.0)


def main():
    yrs = [c[1] for c in CARDS]; labels = [c[0] for c in CARDS]
    card_only = [be(c[2], c[3], c[4]) for c in CARDS]
    full_sys = [be(c[2], c[3], c[4], SYS_MULT) for c in CARDS]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axhspan(3, 5, color="#9aa4b0", alpha=0.18, zorder=0)
    ax.annotate("typical 3-5 yr card life", (2018.05, 4.0), fontsize=8.5, color="#5a626c", va="center")
    ax.plot(yrs, card_only, "-o", color="#14385c", lw=2.4, ms=7, zorder=4, label="break-even at card TDP")
    ax.plot(yrs, full_sys, "-s", color="#b2182b", lw=2.0, ms=6, zorder=4,
            label=f"break-even at full-system power (~{SYS_MULT:.0f}x TDP)")
    for x, y, l in zip(yrs, card_only, labels):
        ax.annotate(l, (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color="#14385c")
    # EU-2022 what-if: A100 at the European spike price (~5x US industrial)
    eu = be(12000, 400, 0.0667 * 5.0)
    ax.scatter([2022], [eu], s=90, marker="*", color="#e08214", zorder=5)
    ax.annotate("A100 at EU-2022\nspike (~5x price)", (2022, eu), xytext=(-92, -2), textcoords="offset points",
                fontsize=8, color="#e08214", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("GPU generation (launch year)", fontsize=10)
    ax.set_ylabel("energy-to-capital break-even  (years)", fontsize=10)
    ax.set_title("A GPU must run for decades before its electricity bill equals its price: capital dominates\n"
                 "energy in TCO across every generation (break-even = price / (power x electricity))",
                 fontsize=9.2, loc="left", color=INK)
    ax.set_xticks(yrs); ax.set_xlim(2017.6, 2025.7); ax.set_ylim(2, 120)
    ax.grid(True, which="both", alpha=0.18); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.8, frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_breakeven.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_breakeven.png")
    for c, co, fs in zip(CARDS, card_only, full_sys):
        print(f"  {c[0]:5s} {c[1]}  ${c[2]:,}/{c[3]}W @ {100*c[4]:.1f}c/kWh -> card-only {co:.0f} yr, full-system {fs:.0f} yr")


if __name__ == "__main__":
    main()
