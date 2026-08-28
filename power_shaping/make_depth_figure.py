# -*- coding: utf-8 -*-
"""
Curtailment-depth sweep figure from results/pocb_fair.json (all controllers at equal, strict
grid compliance). Two panels vs curtailment depth: (A) critical-class SLO violations, (B)
weighted service cost. Shows the SLO win over uniform is priority-aware deferral and that it
holds across depths, not at a single operating point (addresses the review's "curves not one
point"). priority and elasticity coincide throughout on this homogeneous trace.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
INK = "#111418"; NAVY = "#14385c"
COL = {"uniform": "#b42318", "priority": "#14385c", "elasticity": "#137a4b"}
LAB = {"uniform": "uniform capping", "priority": "priority-aware deferral", "elasticity": "elasticity-aware"}


def main():
    rows = json.load(open(os.path.join(HERE, "results", "pocb_fair.json")))
    rows = sorted(rows, key=lambda r: r["dip_frac"], reverse=True)      # shallow -> deep
    depth = [round((1 - r["dip_frac"]) * 100) for r in rows]            # % reduction from peak

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.25))
    for k in ("uniform", "priority", "elasticity"):
        crit = [r[k]["crit_slo"] for r in rows]
        wc = [max(r[k]["wcost"], 0.5) for r in rows]                    # floor for log axis
        ls = "--" if k == "elasticity" else "-"                        # elasticity dashed (overlaps priority)
        lw = 2.4 if k == "elasticity" else 2.0
        ax[0].plot(depth, crit, ls, color=COL[k], lw=lw, marker="o", ms=4, label=LAB[k])
        ax[1].plot(depth, wc, ls, color=COL[k], lw=lw, marker="o", ms=4, label=LAB[k])

    ax[0].set_ylabel("critical-class SLO violations (%)"); ax[0].set_ylim(-2, 50)
    ax[1].set_yscale("log"); ax[1].set_ylabel("weighted service cost (log)")
    for a in ax:
        a.set_xlabel("curtailment depth (% reduction from peak)")
        a.grid(True, alpha=0.25); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.axvspan(50, max(depth) + 2, color="#9aa0aa", alpha=0.10)     # deep, binding regime
    ax[0].legend(fontsize=8, frameon=False, loc="upper left")
    ax[0].set_title("(A) High-priority SLO", fontsize=9.5, color=INK, loc="left")
    ax[1].set_title("(B) Weighted service cost", fontsize=9.5, color=INK, loc="left")
    ax[0].annotate("uniform starves\ncritical inference", xy=(60, 44), xytext=(40, 33),
                   fontsize=7.4, color=COL["uniform"],
                   arrowprops=dict(arrowstyle="->", color=COL["uniform"], lw=0.8))
    ax[0].text(0.98, 0.06, "priority = elasticity\n(homogeneous trace)", transform=ax[0].transAxes,
               ha="right", fontsize=7.2, color=NAVY, style="italic")

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig_depth_sweep.png"), dpi=140, facecolor="white",
                bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_depth_sweep.png")
    print("depths:", depth)
    print("uniform crit:", [r["uniform"]["crit_slo"] for r in rows])


if __name__ == "__main__":
    main()
