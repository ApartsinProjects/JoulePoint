# -*- coding: utf-8 -*-
"""
The hardware-grounded flexibility guarantee (paper Figure). Two panels from results/rq6_safe.json:
 (A) reliability R = Pr[delivered >= promised] vs safety margin: the hardware-enforced ceiling is
     a flat 100% (solid, square markers) across all margins; a feature-only statistical predictor
     sits far below (dashed, open circles). The gap is the "trust gap".
 (B) usable flexibility offered vs safety margin: the hardware ceiling offers up to ~77% of true
     flexibility, ALL of it delivered (100% reliable); the statistical predictor "promises" more
     but delivers it only ~half the time (over-promising).
Encoding (per the figure language): hardware-enforced = solid + squares; statistical = dashed + open circles.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
HW = "#1b7837"; STAT = "#b2182b"; INK = "#111418"


def main():
    d = json.load(open(os.path.join(HERE, "results", "rq6_safe.json")))
    fr = sorted(d["frontier_obs"], key=lambda r: r["margin"])
    m = [r["margin"] for r in fr]
    rel = [r["compliance_pct"] for r in fr]                       # obs-bound reliability (all ~100)
    flex = [r["usable_flex_frac"] * 100 for r in fr]
    feat_rel = d["feat_mean_point"]["compliance_pct"]
    feat_flex = d["feat_mean_point"]["usable_flex_frac"] * 100

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))

    # (A) reliability vs margin
    ax[0].plot(m, rel, "-s", color=HW, lw=2.2, ms=6, label="hardware-enforced ceiling")
    ax[0].plot(m, [feat_rel] * len(m), "--o", color=STAT, lw=1.8, ms=5, mfc="white",
               label="statistical predictor")
    ax[0].fill_between(m, [feat_rel] * len(m), rel, color="#cccccc", alpha=0.35)
    ax[0].text(np.mean(m), (feat_rel + 100) / 2, "trust gap", ha="center", fontsize=8.5,
               color="#555", style="italic")
    ax[0].set_xlabel("safety margin m"); ax[0].set_ylabel("reliability  R = Pr[delivered ≥ promised]  (%)")
    ax[0].set_ylim(0, 108); ax[0].set_title("(A) A guarantee, not a forecast", fontsize=9.5, loc="left", color=INK)
    ax[0].legend(fontsize=7.6, frameon=False, loc="center left")
    ax[0].annotate(f"{feat_rel:.0f}% vs 100%: same promise,\ndifferent physics", xy=(m[len(m)//2], feat_rel),
                   xytext=(m[len(m)//2]-0.02, feat_rel-22), fontsize=7.2, color=STAT,
                   arrowprops=dict(arrowstyle="->", color=STAT, lw=0.8))

    # (B) usable flexibility vs margin
    ax[1].plot(m, flex, "-s", color=HW, lw=2.2, ms=6, label="hardware ceiling (100% reliable)")
    ax[1].axhline(feat_flex, ls="--", color=STAT, lw=1.6)
    ax[1].plot([m[0]], [feat_flex], "o", color=STAT, mfc="white", ms=6)
    ax[1].fill_between(m, flex, color=HW, alpha=0.10)
    ax[1].text(m[len(m)//2], feat_flex + 3, f"statistical: promises {feat_flex:.0f}% but "
               f"delivers only {feat_rel:.0f}% of the time", fontsize=7.0, color=STAT, ha="center")
    ax[1].text(m[len(m)//2], np.mean(flex) - 10, "every point here is\n100% reliable", fontsize=7.4,
               color=HW, ha="center", style="italic")
    ax[1].set_xlabel("safety margin m"); ax[1].set_ylabel("usable flexibility offered (% of true)")
    ax[1].set_ylim(0, max(feat_flex, max(flex)) + 15)
    ax[1].set_title("(B) Full flexibility, no trade-off", fontsize=9.5, loc="left", color=INK)
    ax[1].legend(fontsize=7.6, frameon=False, loc="lower left")

    for a in ax:
        a.grid(True, alpha=0.25); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.invert_xaxis()                                          # tighter margin (more headroom) to the right
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_guarantee.png"), dpi=140, facecolor="white",
                bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_guarantee.png")


if __name__ == "__main__":
    main()
