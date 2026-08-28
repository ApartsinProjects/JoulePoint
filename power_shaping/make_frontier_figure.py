# -*- coding: utf-8 -*-
"""Compute-power frontier figure from results/energy_frontier.json: uniform vs energy-space (oracle) on an
identical-A100 fleet, under a smooth-compute objective (left) and an SLO-goodput objective (right)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#111418"


def main():
    r = json.load(open("results/energy_frontier.json"))
    PANELS = [
        ("A100_perf", "useful compute (of 18)", "(A) smooth compute: heterogeneity gain is modest"),
        ("A100_goodput", "jobs meeting SLO (of 18)", "(B) SLO-goodput: non-uniform is decisive")]

    def _draw(nrows, ncols, figsize, outname):
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize); axf = ax.ravel()
        for k, (obj, ylab, title) in enumerate(PANELS):
            a = axf[k]; pts = r[obj]["points"]
            C = [p["frac"] * 100 for p in pts]
            a.plot(C, [p["F_uniform"] for p in pts], "-o", color="#b2182b", lw=2, ms=5, mfc="white", label="uniform cap")
            a.plot(C, [p["F_oracle"] for p in pts], "-o", color="#14385c", lw=2, ms=5, mfc="white", label="energy-space (oracle)")
            a.fill_between(C, [p["F_uniform"] for p in pts], [p["F_oracle"] for p in pts], color="#14385c", alpha=0.10)
            a.set_title(title, fontsize=9.5, loc="left", color=INK)
            a.set_xlabel("fleet power budget (% of Σ TDP)"); a.set_ylabel(ylab)
            a.invert_xaxis(); a.grid(True, alpha=0.25)
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
            a.legend(fontsize=8.5, frameon=False, loc="lower left")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(os.path.dirname(__file__), "figures", outname)
        fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
        print("figure ->", outname)
    # horizontal (side-by-side): web page + single-column paper
    _draw(1, 2, (9.6, 4.2), "fig_frontier.png")
    # vertical (stacked): two-column paper only, so it flows single-column with no full-width float gap
    _draw(2, 1, (5.4, 8.4), "fig_frontier_stacked.png")


if __name__ == "__main__":
    main()
