# -*- coding: utf-8 -*-
"""
P2 figure: conformal calibration of the flexibility promise. Reliability (Pr[delivered >= promised])
versus usable flexibility offered, on the Zeus corpus (leave-one-workload-out). The conformal frontier
(swept over miscoverage alpha) dominates the up-and-to-the-right region: at matched reliability it offers
more usable flexibility than a fixed +2*MAE margin, and unlike the bare point predictor it keeps its
promise. Reads results/conformal_guarantee.json.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"; NAVY = "#14385c"; RED = "#b2182b"; GREY = "#5a626c"


def main():
    cf = json.load(open(os.path.join(HERE, "results", "conformal_guarantee.json")))
    fr = sorted(cf["conformal_frontier"], key=lambda r: r["usable_flex_pct"])
    ux = [r["usable_flex_pct"] for r in fr]; ry = [r["reliability_pct"] for r in fr]
    pt = cf["point_predictor"]; fx = cf["fixed_margin_2mae"]

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    # conformal frontier
    ax.plot(ux, ry, "-o", color=NAVY, lw=2, ms=6, mfc="white", label="conformal promise (sweep α)", zorder=3)
    for r in fr:
        ax.annotate(f"α={r['alpha']:.2f}", (r["usable_flex_pct"], r["reliability_pct"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=6.8, color=NAVY)
    # baselines
    ax.scatter([fx["usable_flex_pct"]], [fx["reliability_pct"]], s=70, color=RED, marker="s", zorder=4,
               label="fixed +2·MAE margin")
    ax.scatter([pt["usable_flex_pct"]], [pt["reliability_pct"]], s=70, color=GREY, marker="D", zorder=4,
               label="point predictor")
    # guide: at the fixed-margin reliability, the conformal frontier offers more usable flexibility
    ax.annotate("2× the usable flexibility\nat matched reliability", xy=(fx["usable_flex_pct"], fx["reliability_pct"]),
                xytext=(fx["usable_flex_pct"] + 20, fx["reliability_pct"] - 18), fontsize=7.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))

    ax.set_xlabel("usable flexibility offered (% of true)")
    ax.set_ylabel("reliability  Pr[delivered ≥ promised] (%)")
    ax.set_title("Conformal promise dominates a fixed margin", fontsize=10, loc="left", color=INK)
    ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_conformal.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    a05 = next(r for r in fr if r["alpha"] == 0.05)
    print(f"figure -> figures/fig_conformal.png  (conformal a=0.05 {a05['reliability_pct']:.0f}%@{a05['usable_flex_pct']:.0f}%, "
          f"fixed {fx['reliability_pct']:.0f}%@{fx['usable_flex_pct']:.0f}%, point {pt['reliability_pct']:.0f}%)")


if __name__ == "__main__":
    main()
