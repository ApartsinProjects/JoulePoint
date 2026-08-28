# -*- coding: utf-8 -*-
"""
Scaling-law figure: per-workload elasticity value is not a fixed number but scales with fleet diversity.
Blending the measured 26-workload fleet continuously from homogeneous (every workload identical, in both
curve and power range) to its true measured diversity, the pure elasticity value (equal-weight oracle gain
over uniform) rises smoothly and monotonically from 0 to its full value. The "elasticity adds nothing"
case is the zero-diversity endpoint of a continuum, not a dead end. Reads results/elasticity_scaling.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"; NAVY = "#14385c"


def main():
    d = json.load(open(os.path.join(HERE, "results", "elasticity_scaling.json")))
    sw = sorted(d["sweep"], key=lambda r: r["lam"], reverse=True)   # lam 1 -> 0
    div = [round(1 - r["lam"], 2) for r in sw]                      # diversity fraction 0 -> 1
    val = [r["elasticity_value_pct"] for r in sw]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(div, val, "-o", color=NAVY, lw=2, ms=5, mfc="white", zorder=3)
    ax.fill_between(div, 0, val, color=NAVY, alpha=0.07)
    ax.annotate("homogeneous fleet\n(elasticity worth 0)", xy=(0, val[0]), xytext=(0.08, 6.5),
                fontsize=7.8, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.annotate(f"full measured diversity\n({val[-1]:.0f}%)", xy=(1, val[-1]), xytext=(0.5, val[-1] - 2),
                fontsize=7.8, color=NAVY, ha="left", va="top")
    ax.set_xlabel("fleet diversity  (blend from homogeneous → measured)")
    ax.set_ylabel("per-workload elasticity value (%)")
    ax.set_title("Elasticity value scales with fleet diversity", fontsize=10.5, loc="left", color=INK)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-1, max(val) + 3)
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_scaling.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"figure -> figures/fig_scaling.png  (0% at homogeneous -> {val[-1]:.0f}% at full diversity, monotone)")


if __name__ == "__main__":
    main()
