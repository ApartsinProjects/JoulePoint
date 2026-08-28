# -*- coding: utf-8 -*-
"""The capping tradeoff under a latency SLO band, on the measured A100 ResNet-50 loaded curve. As the power cap
drops from uncapped, energy per inference (opex) falls to the Joule point, but per-inference latency rises, and
holding the served throughput requires proportionally more cards (capex). The card multiplier equals the
latency ratio exactly (both are uncapped/capped throughput), so a single curve carries both. A latency SLO
tolerance band (e.g. +20%) bounds how hard one may cap. Shows: capex up, latency up, throughput held, opex down."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"
SLO = 0.20                                            # latency tolerance band: +20%


def main():
    M = pd.read_csv(os.path.join(HERE, "data", "processed", "elf_master.csv"))
    s = M[(M.card == "A100") & (M.loaded_primary) & (M.workload == "resnet50")].sort_values("cap_w")
    cap = s.cap_pct_tdp.values; R = s.throughput.values; E = (s.power_w / R).values
    unc = int(np.argmax(cap))
    mult = R[unc] / R                                  # latency ratio == card multiplier
    en = E / E[unc]                                    # energy per inference, relative
    j = int(np.argmin(E))

    # fine interpolation over the useful range for smooth curves + a clean SLO crossing
    xs = np.linspace(cap.min(), 100, 300)
    fi = np.argsort(cap)
    mult_i = np.interp(xs, cap[fi], mult[fi]); en_i = np.interp(xs, cap[fi], en[fi])
    slo_cap = xs[np.where(mult_i <= 1 + SLO)[0][0]]    # deepest cap within the latency band (lowest x that fits)
    en_at_slo = np.interp(slo_cap, xs, en_i); mult_at_slo = 1 + SLO

    fig, axL = plt.subplots(figsize=(7.4, 5.0))
    axR = axL.twinx()
    xlo = 40
    # opex (energy) on left axis
    lE, = axL.plot(xs[xs >= xlo], en_i[xs >= xlo], color="#14385c", lw=2.4, label="energy per inference (opex)")
    axL.scatter([cap[j]], [en[j]], s=150, marker="*", color="#14385c", edgecolor="white", linewidth=1.3, zorder=6)
    # capex == latency multiplier on right axis
    lM, = axR.plot(xs[xs >= xlo], mult_i[xs >= xlo], color="#ef8a62", lw=2.4,
                   label="latency ratio = cards to hold throughput (capex)")
    axR.axhline(1 + SLO, color="#b2182b", ls="--", lw=1.2)
    axR.annotate(f"+{int(100*SLO)}% latency SLO", (xlo + 1, 1 + SLO + 0.01), fontsize=8.5, color="#b2182b")
    # SLO-allowed vs violating regions
    axL.axvspan(slo_cap, 100, color="#2a9d8f", alpha=0.07)
    axL.axvspan(xlo, slo_cap, color="#b2182b", alpha=0.07)
    axL.axvline(slo_cap, color="#333", ls=":", lw=1.2)
    axL.annotate(f"deepest cap within the band\n~{slo_cap:.0f}% TDP", (slo_cap, en_i[xs >= xlo].max()),
                 xytext=(6, -4), textcoords="offset points", fontsize=8.5, color="#333", va="top")
    axL.annotate("Joule point", (cap[j], en[j]), xytext=(-2, 12), textcoords="offset points",
                 fontsize=9, color="#14385c", fontweight="bold", ha="center")
    # outcome box in the empty lower-left region
    box = (f"At the +{int(100*SLO)}% latency limit (cap ~{slo_cap:.0f}% TDP):\n"
           f"  opex energy  -{100*(1-en_at_slo):.0f}%   (down)\n"
           f"  latency      +{int(100*SLO)}%        (up)\n"
           f"  cards/capex  x{mult_at_slo:.2f}      (up, = latency)\n"
           f"  throughput   unchanged   (held by the extra cards)")
    axL.text(0.03, 0.04, box, transform=axL.transAxes, fontsize=8.2, family="monospace", va="bottom",
             ha="left", bbox=dict(boxstyle="round", fc="#f4f5f7", ec="#c7ced6"))
    axL.set_xlabel("power cap (% of TDP)   ← cap harder", fontsize=10)
    axL.set_ylabel("energy per inference  (relative to uncapped)", fontsize=10, color="#14385c")
    axR.set_ylabel("latency ratio = card multiplier  (relative to uncapped)", fontsize=10, color="#b06a45")
    axL.set_title("The capping tradeoff under a latency SLO (A100, ResNet-50, loaded):\n"
                  "cap harder → less energy (opex), but more latency and more cards (capex), throughput held",
                  fontsize=9.6, loc="left", color=INK)
    axL.set_xlim(xlo, 101); axL.invert_xaxis()
    axL.grid(True, alpha=0.2); axL.spines["top"].set_visible(False); axR.spines["top"].set_visible(False)
    axL.legend([lE, lM], [lE.get_label(), lM.get_label()], fontsize=8.6, frameon=False, loc="upper left")
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_slo_tradeoff.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"figure -> figures/fig_slo_tradeoff.png  (SLO {int(100*SLO)}%: cap ~{slo_cap:.0f}%TDP, "
          f"energy -{100*(1-en_at_slo):.0f}%, cards x{mult_at_slo:.2f}; Joule {cap[j]:.0f}%TDP)")


if __name__ == "__main__":
    main()
