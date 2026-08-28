# -*- coding: utf-8 -*-
"""Capping by watts (power cap) vs by clock (graphics-clock lock) trace the SAME energy curve. On the A10G with
convnext_small at a matched batch (512 on both sweeps), the power-cap points and the clock points overlay where
their board-draw ranges overlap, and the clock reaches below the power-cap floor. This shows the energy
operating point is one physical thing; which actuator reaches it is an implementation detail (the power cap
cannot go below its driver floor, the clock can). Energy per inference is normalized to the uncapped value."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"
TDP = 300.0


def curve(g):
    g = g.assign(draw=100 * g.power_w / TDP, E=g.power_w / (g.throughput * g.batch)).sort_values("draw")
    return g.draw.values, g.E.values


def main():
    M = pd.read_csv(os.path.join(HERE, "data", "processed", "elf_master.csv"))
    sub = M[(M.card == "A10G") & (M.workload == "convnext_small")]
    dc, ec = curve(sub[sub.loaded_primary])        # power cap
    dk, ek = curve(sub[sub.actuator == "clock"])   # graphics clock
    ref = ec[-1]                                    # uncapped (highest-draw cap point)
    ec, ek = ec / ref, ek / ref
    capfloor = dc.min()

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.plot(dc, ec, "-o", color="#14385c", lw=2.2, ms=8, mfc="white", zorder=4, label="capping by watts (power cap)")
    ax.plot(dk, ek, "--s", color="#ef8a62", lw=2.0, ms=7, mfc="white", zorder=3, label="capping by clock (graphics clock)")
    ax.axvline(capfloor, color="#b2182b", lw=1.0, ls=":", zorder=1)
    ax.annotate("power-cap floor\n(the cap cannot go lower)", (capfloor + 1, 1.28), fontsize=8.2, color="#b2182b")
    ax.annotate("only the clock\nreaches here", (dk.min() + 1, ek[np.argmin(dk)] + 0.04), fontsize=8.2,
                color="#c05a2b", style="italic")
    ax.axhline(1.0, color="#999", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("board draw  (% of TDP)", fontsize=10)
    ax.set_ylabel("energy per inference  (relative to uncapped)", fontsize=10)
    # no title on the image; the description lives in the paper caption
    ax.grid(True, alpha=0.18); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper center")
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_knob_compare.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    # overlap agreement metric
    lo, hi = max(dc.min(), dk.min()), min(dc.max(), dk.max())
    ic = (dc >= lo) & (dc <= hi)
    ek_at = np.interp(dc[ic], dk, ek)
    print(f"figure -> figures/fig_knob_compare.png  overlap {lo:.0f}-{hi:.0f}%TDP, mean |cap-clock| energy = "
          f"{np.mean(np.abs(ec[ic]-ek_at)):.3f} (relative); clock floor {dk.min():.0f}% vs cap floor {capfloor:.0f}%")


if __name__ == "__main__":
    main()
