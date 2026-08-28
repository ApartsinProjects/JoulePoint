# -*- coding: utf-8 -*-
"""
Inference-cost surface from the full-spectrum cap sweep (data/raw/aws_sweep_spectrum.csv), for the ~20-model
zoo. For each (card, model), as the power cap sweeps from the driver floor to default we get two costs per
served inference:
  * latency per inference (ms) = t_compute (+ t_h2d + t_d2h ; model-load is amortized over the serving life)
  * energy  per inference (mJ) = power_w x t_compute       (W x ms = mJ)
A cap trades energy DOWN for latency UP; the steepness is the hinge, in cost units.

Figures (per CARD, all models as lines colored by draw-fraction D0/TDP = compute-vs-memory proxy read from
the data, so no external features are needed):
  fig_infer_cost_frontier.png -- energy/inf vs latency/inf, one line per model, one panel per card.
  fig_infer_cost_vs_cap.png   -- latency slowdown vs cap fraction, one line per model, one panel per card.
  fig_infer_phase_split.png   -- mean phase share per card + cap-invariance check (only compute moves w/ cap).
Prints the measured driver floor per card and a min-vs-max-cap cost table.
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from predict_power import RAW

INK = "#111418"
ORDER = ["A10G", "L4", "T4", "L40S", "V100"]


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    for c in ["cap_frac", "cap_w", "power_w", "t_load_ms", "t_h2d_ms", "t_compute_ms", "t_d2h_ms", "throughput"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    g = df.groupby(["card", "workload", "cap_w"], as_index=False).agg(
        cap_frac=("cap_frac", "mean"), power=("power_w", "mean"),
        t_load=("t_load_ms", "mean"), t_h2d=("t_h2d_ms", "mean"),
        t_compute=("t_compute_ms", "mean"), t_d2h=("t_d2h_ms", "mean"), thr=("throughput", "mean"))
    g["lat_ms"] = g.t_compute + g.t_h2d + g.t_d2h
    g["energy_mJ"] = g.power * g.t_compute
    # draw-fraction D0/TDP per (card, model): uncapped draw over default cap -> compute(1) vs memory(low)
    top = g.sort_values("cap_w").groupby(["card", "workload"]).last().reset_index()
    top["draw_frac"] = top.power / top.cap_w
    g = g.merge(top[["card", "workload", "draw_frac"]], on=["card", "workload"], how="left")
    return g


def cards_in(g):
    return [c for c in ORDER if c in set(g.card)]


def _col(df_row_frac):
    return cm.coolwarm(plt.Normalize(0.4, 1.0)(df_row_frac))


def fig_frontier(g, figdir):
    cards = cards_in(g); n = len(cards)
    fig, ax = plt.subplots(1, n, figsize=(3.6 * n, 3.9), squeeze=False)
    for i, card in enumerate(cards):
        a = ax[0][i]; sub = g[g.card == card]
        for wl, c in sub.groupby("workload"):
            c = c.sort_values("cap_w")
            a.plot(c.lat_ms, c.energy_mJ, "-", lw=1.3, alpha=0.8, color=_col(c.draw_frac.iloc[0]))
        a.set_title(card, fontsize=10, loc="left", color=INK)
        a.set_xlabel("latency / inference (ms)"); a.grid(True, alpha=0.25)
        a.set_xscale("log"); a.set_yscale("log")
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        if i == 0:
            a.set_ylabel("energy / inference (mJ)")
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=plt.Normalize(0.4, 1.0)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax.ravel().tolist(), pad=0.01, fraction=0.03)
    cb.set_label("draw fraction D0/TDP  (memory-bound -> compute-bound)", fontsize=8)
    fig.suptitle("Inference-cost frontier per card: each line a model, capping walks it (energy down, latency up)",
                 fontsize=10.5, color=INK, x=0.02, ha="left")
    out = os.path.join(figdir, "fig_infer_cost_frontier.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure ->", os.path.basename(out))


def fig_vs_cap(g, figdir):
    cards = cards_in(g); n = len(cards)
    fig, ax = plt.subplots(1, n, figsize=(3.6 * n, 3.9), squeeze=False)
    for i, card in enumerate(cards):
        a = ax[0][i]; sub = g[g.card == card]
        for wl, c in sub.groupby("workload"):
            c = c.sort_values("cap_frac"); base = c.lat_ms.iloc[-1]
            a.plot(c.cap_frac, c.lat_ms / base, "-", lw=1.3, alpha=0.8, color=_col(c.draw_frac.iloc[0]))
        a.set_title(card, fontsize=10, loc="left", color=INK)
        a.set_xlabel("power cap (fraction of default)"); a.grid(True, alpha=0.25)
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        if i == 0:
            a.set_ylabel("latency slowdown (x uncapped)")
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=plt.Normalize(0.4, 1.0)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax.ravel().tolist(), pad=0.01, fraction=0.03)
    cb.set_label("draw fraction D0/TDP", fontsize=8)
    fig.suptitle("Latency cost of the cap per card: memory-bound models (blue) barely move, compute-bound (red) bend",
                 fontsize=10.5, color=INK, x=0.02, ha="left")
    out = os.path.join(figdir, "fig_infer_cost_vs_cap.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure ->", os.path.basename(out))


def fig_phase(g, figdir):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
    cards = cards_in(g)
    top = g.sort_values("cap_w").groupby(["card", "workload"]).last().reset_index()
    comp = [top[top.card == c].t_compute.mean() for c in cards]
    h2d = [top[top.card == c].t_h2d.mean() for c in cards]
    d2h = [top[top.card == c].t_d2h.mean() for c in cards]
    xp = np.arange(len(cards))
    ax[0].bar(xp, comp, color="#14385c", label="compute")
    ax[0].bar(xp, h2d, bottom=comp, color="#e08214", label="h2d")
    ax[0].bar(xp, d2h, bottom=np.array(comp) + np.array(h2d), color="#b2182b", label="d2h")
    ax[0].set_xticks(xp); ax[0].set_xticklabels(cards); ax[0].set_ylabel("mean ms / inference (uncapped)")
    ax[0].set_title("(A) mean phase breakdown per card", fontsize=9.5, loc="left", color=INK)
    ax[0].legend(fontsize=8, frameon=False); ax[0].spines["top"].set_visible(False); ax[0].spines["right"].set_visible(False)

    def cov(x):
        x = x.values; m = np.nanmean(x); return np.nanstd(x) / m if m else np.nan
    rows = g.groupby(["card", "workload"]).agg(cc=("t_compute", cov), ch=("t_h2d", cov), cl=("t_load", cov)).reset_index()
    ax[1].scatter(rows.ch, rows.cc, s=20, color="#14385c", label="h2d vs compute")
    ax[1].scatter(rows.cl, rows.cc, s=20, marker="^", color="#762a83", label="load vs compute")
    ax[1].axline((0, 0), slope=1, color="#888", ls=(0, (3, 3)), lw=1)
    ax[1].set_xlabel("CoV of transfer/load across caps (~0 = cap-invariant)")
    ax[1].set_ylabel("CoV of compute across caps (moves with cap)")
    ax[1].set_title("(B) only compute moves with the cap", fontsize=9.5, loc="left", color=INK)
    ax[1].legend(fontsize=8, frameon=False); ax[1].grid(True, alpha=0.25)
    ax[1].spines["top"].set_visible(False); ax[1].spines["right"].set_visible(False)
    fig.suptitle("Load and transfer are cap-invariant overhead; the cap acts on compute alone",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(figdir, "fig_infer_phase_split.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure ->", os.path.basename(out))


def summary(g):
    print("\n== measured driver floor per card ==")
    for card in cards_in(g):
        c = g[g.card == card]
        print(f"  {card:5} floor {c.cap_w.min():.0f}W .. default {c.cap_w.max():.0f}W  "
              f"({100*c.cap_w.min()/c.cap_w.max():.0f}% of default)")
    print("\n== per model on A10G: min-vs-max cap (energy saved for latency paid), ranked by inert band ==")
    a = g[g.card == "A10G"]
    if a.empty:
        return
    rows = []
    for wl, c in a.groupby("workload"):
        c = c.sort_values("cap_w"); hi, lo = c.iloc[-1], c.iloc[0]
        rows.append((wl, c.draw_frac.iloc[0], hi.lat_ms, lo.lat_ms, hi.energy_mJ, lo.energy_mJ,
                     lo.lat_ms / hi.lat_ms, 1 - lo.energy_mJ / hi.energy_mJ))
    rows.sort(key=lambda r: r[1])       # by draw_frac: memory-bound first
    print(f"{'model':20} {'D0/TDP':>7} {'lat@max':>8} {'lat@min':>8} {'latx':>6} {'Esave':>7}")
    for wl, dfrac, lhi, llo, ehi, elo, latx, esav in rows:
        print(f"{wl:20} {dfrac:>7.2f} {lhi:>7.2f}m {llo:>7.2f}m {latx:>5.2f}x {100*esav:>5.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(RAW, "aws_sweep_spectrum.csv"))
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        print(f"[wait] {args.csv} not present yet -- run the spectrum sweep first."); return
    g = load(args.csv)
    figdir = os.path.join(os.path.dirname(__file__), "figures"); os.makedirs(figdir, exist_ok=True)
    fig_frontier(g, figdir); fig_vs_cap(g, figdir); fig_phase(g, figdir)
    summary(g)


if __name__ == "__main__":
    main()
