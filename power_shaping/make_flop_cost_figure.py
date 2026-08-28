# -*- coding: utf-8 -*-
"""CAPEX:OPEX ratio per FLOP for data-center GPUs, 2016-2025, and an extrapolation of when (if ever) the Joule
point becomes cost-optimal. ratio = (capex per PFLOP, card price amortized over 4 yr) / (opex per PFLOP,
electricity). Because peak FLOPS cancels in the ratio, per-FLOP and per-card give the same number (it equals the
break-even horizon divided by the card life). The Joule point becomes the COST-optimal operating point only when
this ratio falls to ~0.4 (opex must dominate); today it is ~13-17 and RISING, because hardware energy efficiency
(W/FLOPS) has improved faster than price/performance ($/FLOPS). Prices are analyst/reseller estimates; FP16
tensor throughput is dense vendor spec; US industrial electricity is EIA."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"
LIFE_S = 4 * 8760 * 3600.0
ELEC = {2016: 0.0676, 2017: 0.0688, 2018: 0.0692, 2020: 0.0667, 2021: 0.0718,
        2022: 0.0832, 2023: 0.0804, 2024: 0.0813, 2025: 0.086}
# (label, year, price USD, TDP W, FP16-dense TFLOPS) -- prices EST, TFLOPS/TDP vendor spec (dense)
CARDS = [
    ("P100", 2016, 6000, 300, 18.7), ("V100", 2017, 9000, 300, 125), ("T4", 2018, 2200, 70, 65),
    ("A100-40", 2020, 11000, 400, 312), ("A100-80", 2021, 15000, 400, 312), ("A10", 2021, 3000, 150, 125),
    ("A40", 2021, 4500, 300, 149.7), ("L40", 2022, 7500, 300, 181), ("H100-PCIe", 2022, 27000, 350, 756),
    ("H100-SXM", 2022, 32000, 700, 989), ("L4", 2023, 2500, 72, 121), ("L40S", 2023, 11000, 350, 366),
    ("H200", 2024, 35000, 700, 989), ("B100", 2024, 30000, 700, 1750), ("B200", 2025, 35000, 1000, 2250),
]
JOULE_THRESH = 0.4   # ratio below which the TCO-optimum reaches the Joule point (measured A100: ~$3300/MWh)


def main():
    yrs, ratios, labels = [], [], []
    for lab, y, price, w, tf in CARDS:
        flops = tf * 1e12; elec = ELEC[y]
        capex = price * 1e15 / (flops * LIFE_S)
        opex = (1e15 * w / flops) / 3.6e6 * elec
        yrs.append(y); ratios.append(capex / opex); labels.append(lab)
    yrs = np.array(yrs, float); ratios = np.array(ratios)
    m, b = np.polyfit(yrs, ratios, 1)                       # trend
    xf = np.array([2016, 2035])

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.scatter(yrs, ratios, s=46, color="#14385c", zorder=4)
    ax.plot(xf, m * xf + b, "--", color="#14385c", lw=1.6, alpha=0.8, zorder=3,
            label=f"trend +{m:.1f}/yr (extrapolated)")
    ax.axhspan(0.3, 0.5, color="#2a9d8f", alpha=0.18, zorder=0)
    ax.annotate("Joule point becomes cost-optimal below here\n(needs electricity ~40x today's)",
                (2016.1, 0.4), fontsize=8, color="#1d6f63", va="center")
    ax.axhline(1.0, color="#b2182b", lw=1.0, ls=":", zorder=1)
    ax.annotate("capex = opex (parity)", (2016.1, 1.35), fontsize=8, color="#b2182b")
    # a couple of representative labels (avoid clutter)
    for lab, y, p, w, tf in CARDS:
        if lab in ("P100", "H100-SXM", "B200", "L4", "T4"):
            r = (p * 1e15 / (tf * 1e12 * LIFE_S)) / ((1e15 * w / (tf * 1e12)) / 3.6e6 * ELEC[y])
            ax.annotate(lab, (y, r), xytext=(4, 4), textcoords="offset points", fontsize=8, color="#5a626c")
    ax.set_xlabel("GPU launch year", fontsize=10)
    ax.set_ylabel("capex : opex per FLOP  (buy-cost / run-cost, 4-yr)", fontsize=10)
    ax.set_title("Buying a FLOP costs ~13-17x running it, and the ratio is RISING: on hardware trends the Joule\n"
                 "point never becomes cost-optimal; only much dearer electricity or carbon pricing flips it",
                 fontsize=9.1, loc="left", color=INK)
    ax.set_xticks([2016, 2018, 2020, 2022, 2024, 2026, 2028, 2030]); ax.set_xlim(2015.6, 2030.5)
    ax.set_ylim(0, 24)
    ax.grid(True, alpha=0.18); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.6, frameon=False, loc="upper right")
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_flop_cost.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    cross = "never (ratio rising)" if m > 0 else f"~{(JOULE_THRESH - b) / m:.0f}"
    print(f"figure -> figures/fig_flop_cost.png  trend +{m:.2f}/yr; mean ratio {ratios.mean():.1f}; "
          f"crosses Joule-threshold {JOULE_THRESH}: {cross}")


if __name__ == "__main__":
    main()
