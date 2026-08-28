# -*- coding: utf-8 -*-
"""Two data-grounded figures for the core principles:
  (A) the response law P(R)=P0+aR^beta fitted to measured (rate, power) points, a few representative curves.
  (B) the reaction-time trace: measured power(t) tracking stepped cap(t), showing ~200 ms settling.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict_power import RAW

INK = "#111418"
COL = {"A100": "#14385c", "A10G": "#2166ac", "L4": "#ef8a62", "T4": "#b2182b"}


def _card(df):
    c = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    return c.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()


def fig_law():
    """Three panels, one per card (A100, A10G, L4): every workload's measured (rate, power) points in the
    loaded/saturated regime, normalized to R/Rmax and P/TDP, collapse onto a single fitted response law
    P(R)=P0+aR^beta per card, showing the law is a card property. One pooled fit per card (over all workloads);
    the panel reports beta and the pooled R^2. The T4 is excluded from the analysis (too little dynamic range to
    pin the law; see paper limitations)."""
    M = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "processed", "elf_master.csv"))
    TDP = {"L4": 72, "A10G": 300, "A100": 400}
    def mdl(x, p0, a, b): return p0 + a * np.power(x, b)
    def _draw(nrows, ncols, figsize, outname):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for ax, card in zip(axes.ravel(), ["A100", "A10G", "L4"]):
            sub = M[(M.card == card) & (M.loaded_primary)]; xs_all, ys_all = [], []
            for wl, g in sub.groupby("workload"):
                R = g.throughput.values; P = g.power_w.values
                if len(R) < 4:
                    continue
                Rn = R / R.max(); Pt = P / TDP[card]
                ax.scatter(Rn, Pt, s=15, color=COL[card], alpha=0.30, edgecolor="none", zorder=2)
                xs_all += list(Rn); ys_all += list(Pt)
            xs_all, ys_all = np.array(xs_all), np.array(ys_all)
            popt, _ = curve_fit(mdl, xs_all, ys_all, p0=[ys_all.min(), ys_all.max() - ys_all.min(), 2.0],
                                bounds=([0, 0, 0.5], [ys_all.min() * 1.3 + 0.05, 5, 12]), maxfev=20000)
            gx = np.linspace(xs_all.min(), 1, 100)
            ax.plot(gx, mdl(gx, *popt), color=COL[card], lw=2.4, zorder=4)
            ax.set_title(card, fontsize=11, loc="left", color=COL[card], fontweight="bold")
            ax.set_xlabel("rate  (fraction of workload's own max)", fontsize=9)
            ax.set_ylabel("board power  (fraction of TDP)", fontsize=9)
            ax.grid(True, alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "figures", outname)
        fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
        print("figure ->", outname)
    # horizontal (side-by-side): web page + single-column paper, where full width suits it
    _draw(1, 3, (11.4, 3.7), "fig_law_fit.png")
    # vertical (stacked): two-column paper only, so the figure flows single-column with no full-width float gap
    _draw(3, 1, (5.6, 11.4), "fig_law_fit_stacked.png")


def fig_reaction():
    d = pd.read_csv(os.path.join(RAW, "reaction_time.csv"))
    g = d[d.workload == "resnet50"].sort_values("t_ms")
    t = g.t_ms.values / 1000.0
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(t, g.power_w.values, color="#14385c", lw=1.4, label="measured power draw", zorder=3)
    ax.step(t, g.cap_set_w.values, color="#b2182b", lw=1.3, ls="--", where="post", label="power cap (set)", zorder=2)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_ylabel("watts", fontsize=10)
    ax.set_title("Actuation: power tracks a stepped cap, settling in about 0.2 s (A10G, resnet50)",
                 fontsize=10, loc="left", color=INK)
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "figures", "fig_reaction.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_reaction.png")


if __name__ == "__main__":
    fig_law(); fig_reaction()
