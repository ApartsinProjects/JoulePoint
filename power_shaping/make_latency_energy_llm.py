# -*- coding: utf-8 -*-
"""Actual drain (W) vs latency for the LLM decode workload, across GPUs, traced by the power cap.
Each point is one cap level; sweeping the cap moves along the draw-latency curve.
  x = power_w  (actual measured board draw, W)
  y = latency per step (ms) = 1000 / throughput
Shows how much real power each GPU pulls to hit a given decode latency, and the diminishing return:
past the knee, a lot more watts buys very little latency.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict_power import RAW

INK = "#111418"; ACC = "#14385c"
CARDS = {"T4": "#b2182b", "L4": "#ef8a62", "A10G": "#67a9cf", "A100": "#14385c"}
TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}


def load_llm():
    fr = []
    for f in ["aws_sweep_batch.csv", "aws_sweep_a100_batch.csv"]:
        p = os.path.join(RAW, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    df["card"] = df.card.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    df = df[(df.workload == "llm_decode") & (df["mode"] == "control")]      # one clean config (batch=32)
    g = df.groupby(["card", "cap_w"], as_index=False).agg(
        R=("throughput", "mean"), P=("power_w", "mean"))
    g["lat_ms"] = 1000.0 / g.R
    g["energy_J"] = g.P / g.R
    return g


def main():
    g = load_llm()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for card in ["T4", "L4", "A10G", "A100"]:
        s = g[g.card == card].sort_values("power_w" if "power_w" in g.columns else "P")
        s = g[g.card == card].sort_values("P")
        if len(s) == 0:
            continue
        ax.plot(s.P, s.lat_ms, "-o", color=CARDS[card], lw=1.8, ms=4, mfc="white", label=card, zorder=3)
        # mark the "knee": last point before latency gains flatten (min-energy cap, where P/R is smallest)
        k = int(np.argmin((s.P / s.R).values))
        ax.scatter([s.P.values[k]], [s.lat_ms.values[k]], s=95, facecolor=CARDS[card],
                   edgecolor="white", linewidth=1.3, zorder=5)
        # annotate the min-draw (most-capped) and max-draw (uncapped) ends with cap as % TDP
        lo, hi = s.iloc[0], s.iloc[-1]
        ax.annotate(f"{lo.P:.0f}W ({100*lo.cap_w/TDP[card]:.0f}%TDP)", (lo.P, lo.lat_ms), fontsize=7,
                    color=CARDS[card], xytext=(3, 4), textcoords="offset points")
        ax.annotate(f"{hi.P:.0f}W", (hi.P, hi.lat_ms), fontsize=7,
                    color=CARDS[card], xytext=(4, -2), textcoords="offset points")
    ax.set_xlabel("actual board draw  (Watts)  →  more power right", fontsize=10)
    ax.set_ylabel("latency per decode step  (ms)  →  faster down", fontsize=10)
    ax.set_title("LLM decode: actual power draw vs latency, traced by the power cap\n"
                 "(filled dot = energy-optimal draw; past it, many more watts buy little latency)",
                 fontsize=9.5, loc="left", color=INK)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(title="GPU", fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "figures", "fig_llm_draw_latency.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_llm_draw_latency.png")

    # ---- companion figure: actual draw (W) vs THROUGHPUT (steps/s) ----
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.6))
    for card in ["T4", "L4", "A10G", "A100"]:
        s = g[g.card == card].sort_values("P")
        if len(s) == 0:
            continue
        ax2.plot(s.P, s.R, "-o", color=CARDS[card], lw=1.8, ms=4, mfc="white", label=card, zorder=3)
        k = int(np.argmax((s.R / s.P).values))                 # peak efficiency (throughput per watt)
        ax2.scatter([s.P.values[k]], [s.R.values[k]], s=95, facecolor=CARDS[card],
                    edgecolor="white", linewidth=1.3, zorder=5)
        lo, hi = s.iloc[0], s.iloc[-1]
        ax2.annotate(f"{lo.P:.0f}W ({100*lo.cap_w/TDP[card]:.0f}%TDP)", (lo.P, lo.R), fontsize=7,
                     color=CARDS[card], xytext=(3, -9), textcoords="offset points")
        ax2.annotate(f"{hi.P:.0f}W", (hi.P, hi.R), fontsize=7,
                     color=CARDS[card], xytext=(4, 2), textcoords="offset points")
    ax2.set_xlabel("actual board draw  (Watts)  →  more power right", fontsize=10)
    ax2.set_ylabel("throughput  (decode steps / s)  →  faster up", fontsize=10)
    ax2.set_title("LLM decode: actual power draw vs throughput, traced by the power cap\n"
                  "(filled dot = peak efficiency, most steps per watt; past it throughput flattens)",
                  fontsize=9.5, loc="left", color=INK)
    ax2.grid(True, which="both", alpha=0.2)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.legend(title="GPU", fontsize=9, frameon=False, loc="upper left")
    fig2.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), "figures", "fig_llm_draw_throughput.png")
    fig2.savefig(out2, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig2)
    print("figure -> figures/fig_llm_draw_throughput.png")

    # ---- companion figure: RELATIVE draw (% TDP) vs RELATIVE energy per decode step, all cards overlaid ----
    # Normalizing collapses the four very different absolute scales onto one frame so the U-shapes are directly
    # comparable. Each curve carries its scaling (TDP, min energy in J) in the legend; the filled dot is the
    # measured minimum-energy operating point.
    fig3, ax3 = plt.subplots(figsize=(6.6, 4.8))
    for card in ["T4", "L4", "A10G", "A100"]:
        s = g[g.card == card].sort_values("P")
        if len(s) == 0:
            continue
        tdp = TDP[card]; Emin = s.energy_J.min()
        xr = 100.0 * s.P.values / tdp; yr = s.energy_J.values / Emin
        ax3.plot(xr, yr, "-o", color=CARDS[card], lw=1.8, ms=4, mfc="white", zorder=3,
                 label=f"{card}  ({tdp}W TDP, min {Emin:.2g} J)")
        k = int(np.argmin(s.energy_J.values))                  # measured minimum energy
        ax3.scatter([xr[k]], [yr[k]], s=95, facecolor=CARDS[card], edgecolor="white", linewidth=1.3, zorder=5)
    ax3.scatter([], [], s=95, facecolor="#444", edgecolor="white", label="measured Joule point")
    ax3.set_xlabel("board draw  (% of TDP)  →  more power right", fontsize=10)
    ax3.set_ylabel("energy per decode step  (relative to each card's own minimum)", fontsize=10)
    ax3.set_title("LLM decode: the energy U-shape, normalized across GPUs\n"
                  "(filled dot = measured Joule point; big GPUs dip deep, small cards sit near the ceiling)",
                  fontsize=9.5, loc="left", color=INK)
    ax3.grid(True, which="both", alpha=0.2)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    ax3.legend(fontsize=8.2, frameon=False, loc="upper center", ncol=2)
    fig3.tight_layout()
    out3 = os.path.join(os.path.dirname(__file__), "figures", "fig_llm_draw_energy.png")
    fig3.savefig(out3, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig3)
    print("figure -> figures/fig_llm_draw_energy.png")

    print("\n  per-GPU LLM decode (control, batch=32): draw range and the diminishing-return knee")
    for card in ["T4", "L4", "A10G", "A100"]:
        s = g[g.card == card].sort_values("P")
        if len(s) == 0:
            continue
        k = int(np.argmin((s.P / s.R).values)); knee = s.iloc[k]; lo = s.iloc[0]; hi = s.iloc[-1]
        print(f"  {card:5} draw {lo.P:4.0f}->{hi.P:4.0f}W ({100*lo.cap_w/TDP[card]:.0f}->{100*hi.cap_w/TDP[card]:.0f}%TDP)"
              f"  latency {lo.lat_ms:5.1f}->{hi.lat_ms:5.1f}ms"
              f"   knee@{knee.P:4.0f}W({knee.lat_ms:5.1f}ms): {hi.P-knee.P:4.0f}W more buys only "
              f"{100*(1-hi.lat_ms/knee.lat_ms):.0f}% less latency")


if __name__ == "__main__":
    main()
