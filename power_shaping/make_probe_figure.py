# -*- coding: utf-8 -*-
"""
P1 figure: elasticity is PROBEABLE, not predictable. (A) Fraction of the oracle's per-workload
elasticity gap (priority-only -> oracle) recovered vs the number of cheap in-situ probes, for linear
interpolation and a knee fit, against the zero-probe feature-only class-mean and a learned curve prior;
one probe already reaches the deployable regime. (B) Why: for a compute-bound (steep) and a memory-bound
(flat) workload, a single reduced-power probe plus linear interpolation tracks the measured curve.
Reads results/probe_predict.json and reconstructs (B) from the measured curves.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import probe_predict as PP
from rq4_strengthen import aws_table

HERE = os.path.dirname(__file__)
INK = "#111418"; BLUE = "#2166ac"; ORANGE = "#e08214"; GREEN = "#1b7837"; GREY = "#8a8f98"; NAVY = "#14385c"


def main():
    pp = json.load(open(os.path.join(HERE, "results", "probe_predict.json")))
    ks = [0, 1, 2, 3]
    def series(m): return [pp["value_vs_k"][m].get(str(k)) for k in ks]
    lin = series("linear"); knee = series("knee"); prior = series("prior")
    cm = pp["classmean_feature_pct"]

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))

    # (A) value vs probes
    def plot_pts(y, col, mk, lab):
        xs = [k for k, v in zip(ks, y) if v is not None]; ys = [v for v in y if v is not None]
        ax[0].plot(xs, ys, "-", marker=mk, color=col, lw=2, ms=6, mfc="white", label=lab)
    ax[0].axhline(100, color=INK, lw=1, ls=(0, (4, 3)))
    ax[0].text(3.02, 100, "oracle", fontsize=7.4, color=INK, va="center")
    ax[0].axhline(cm, color=GREEN, lw=1.2, ls=(0, (2, 2)))
    ax[0].text(3.02, cm, f"class-mean\n(0 probes) {cm:.0f}%", fontsize=7.0, color=GREEN, va="center")
    plot_pts(lin, BLUE, "o", "linear interp")
    plot_pts(knee, ORANGE, "s", "knee fit")
    plot_pts(prior, GREY, "^", "learned prior")
    y1 = pp["value_vs_k"]["linear"]["1"]
    ax[0].annotate(f"1 probe: {y1:.0f}%", xy=(1, y1), xytext=(1.15, y1 - 34), fontsize=8, color=BLUE,
                   arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax[0].set_xticks(ks); ax[0].set_xlabel("number of in-situ probes"); ax[0].set_ylim(-30, 112)
    ax[0].set_ylabel("% of oracle elasticity gap recovered")
    ax[0].set_title("(A) Probing beats predicting", fontsize=9.5, loc="left", color=INK)
    ax[0].legend(fontsize=8, frameon=False, loc="center right")

    # (B) reconstruction for one steep + one flat workload
    tab = aws_table(); _, _, Q = PP.true_curves(tab)
    NG = len(next(iter(Q.values()))); rg = np.linspace(0, 1, NG)
    pidx = int(np.argmin(np.abs(rg - PP.PROBES[1][0])))
    show = [("gemm_fp16", NAVY, "GEMM fp16 (compute-bound)"), ("membw", "#b2182b", "membw (memory-bound)")]
    for wl, col, lab in show:
        if wl not in Q:
            continue
        true = Q[wl]; recon = PP.linear_interp(Q, wl, 1, rg)
        ax[1].plot(rg, true, "-", color=col, lw=2, label=lab)
        ax[1].plot(rg, recon, ls=(0, (3, 2)), color=col, lw=1.4)
        ax[1].plot(rg[pidx], true[pidx], "o", color=col, ms=7, mfc="white", mew=1.6)
    ax[1].plot([], [], "o", color=INK, ms=7, mfc="white", mew=1.4, label="probe (relp 0.5)")
    ax[1].plot([], [], ls=(0, (3, 2)), color=INK, lw=1.4, label="1-probe linear")
    ax[1].set_xlabel("relative power (cap fraction)"); ax[1].set_ylabel("normalized throughput")
    ax[1].set_title("(B) One probe pins the curve", fontsize=9.5, loc="left", color=INK)
    ax[1].legend(fontsize=7.2, frameon=False, loc="lower right")

    for a in ax:
        a.grid(True, alpha=0.25); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_probe.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"figure -> figures/fig_probe.png  (1 probe {pp['value_vs_k']['linear']['1']:.0f}%, "
          f"2-probe knee {pp['value_vs_k']['knee']['2']:.0f}%, class-mean {cm:.0f}%)")


if __name__ == "__main__":
    main()
