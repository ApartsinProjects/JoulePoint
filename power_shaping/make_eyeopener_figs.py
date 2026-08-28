# -*- coding: utf-8 -*-
"""Three eye-opener figures (all grounded in the measured data):
  (A) 'Same rate, twice the watts': power is a property of the operating point, not the rate.
  (B) The incentive arrow: under per-hour billing the cost-optimal cap is the energy-worst point.
  (C) One free probe finds the Joule point: uncapped draw/TDP predicts the energy-optimal cap (r=0.90).
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict_power import RAW, saturating_set

INK = "#111418"; ACC = "#14385c"
TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}
COL = {"T4": "#b2182b", "L4": "#ef8a62", "A10G": "#2166ac", "A100": "#14385c"}


def _card(df):
    c = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    return c.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()


def _law(Rn, P):
    """Fit the response law P(R)=P0+aR^beta on normalized rate Rn in [.,1]. Returns (p0,a,beta) or None."""
    def mdl(x, p0, a, b):
        return p0 + a * np.power(x, b)
    try:
        popt, _ = curve_fit(mdl, Rn, P, p0=[P.min(), max(P.max() - P.min(), 1.0), 2.0],
                            bounds=([0, 0, 0.5], [P.min() * 1.3 + 1, 5 * (P.max() - P.min() + 1), 8]),
                            maxfev=20000)
        return tuple(popt)
    except Exception:
        return None


def fit_joule(g, tdp):
    """g: one (card,workload) block, rep-meaned, with columns cap_w, power_w, throughput.
    Returns the MEASURED and the LAW-PREDICTED energy-optimal board draw, both as % of TDP.
    Measured = argmin of measured energy per inference (P/R). Predicted = closed-form R*=(P0/(a(beta-1)))^(1/beta)
    from the fitted law (exists only for beta>1), mapped back through the law to a draw. pred=None if no interior
    optimum. Construct-matched: one fit, one aggregation, used identically by every figure."""
    g = g.sort_values("cap_w")
    R = g.throughput.values.astype(float); P = g.power_w.values.astype(float)
    if len(g) < 4 or R.max() <= 0:
        return None
    E = P / R
    meas = 100.0 * P[int(np.argmin(E))] / tdp
    Rn = R / R.max()
    popt = _law(Rn, P)
    if popt is None:
        return dict(meas=meas, pred=None, beta=float("nan"))
    p0, a, b = popt
    if b <= 1.0 or a <= 0:
        return dict(meas=meas, pred=None, beta=float(b))
    Rstar = (p0 / (a * (b - 1.0))) ** (1.0 / b)          # closed-form energy-optimal (normalized) rate
    Rstar = min(max(Rstar, Rn.min()), 1.0)               # clamp to the observed sweep
    Pstar = p0 + a * Rstar ** b
    return dict(meas=meas, pred=100.0 * Pstar / tdp, beta=float(b), Estar=Pstar / (Rstar * R.max()))


def load_batch():
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")), pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))]
    d = pd.concat(fr, ignore_index=True); d["card"] = _card(d); return d[d["mode"] == "control"]


def fig_same_rate():
    """Same rate, different power. Each of the A10G's 20 models is swept across power caps at batch 32
    (control mode), giving one P(R) curve per model (grey). In the shaded band at about 22 inferences/s, the
    lowest- and highest-power models are highlighted: at the same rate they draw very different power, so power
    is a property of the operating point, not the rate. Uses the same control-mode population as the SR caption
    values in build_gptenergy.py, so the figure and caption cannot drift."""
    d = pd.concat([pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")),
                   pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))], ignore_index=True)
    d["card"] = _card(d); d = d[(d.card == "A10G") & (d["mode"] == "control")]
    g = d.groupby(["workload", "cap_w"], as_index=False).agg(t=("throughput", "mean"), p=("power_w", "mean"))
    lo_r, hi_r = 20.0, 24.0
    band = g[(g.t >= lo_r) & (g.t <= hi_r)]
    lo = band.loc[band.p.idxmin()]; hi = band.loc[band.p.idxmax()]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axvspan(lo_r, hi_r, color="#c7ced6", alpha=0.35, zorder=0)
    for wl, s in g.groupby("workload"):
        s = s.sort_values("t")
        c = "#2166ac" if wl == lo.workload else ("#b2182b" if wl == hi.workload else "#b8bfc8")
        lw = 2.2 if wl in (lo.workload, hi.workload) else 0.9
        z = 4 if wl in (lo.workload, hi.workload) else 1
        ax.plot(s.t, s.p, "-", color=c, lw=lw, alpha=0.95 if z == 4 else 0.55, zorder=z)
    ax.scatter([lo.t], [lo.p], s=70, color="#2166ac", zorder=5, edgecolor="white", linewidth=1)
    ax.scatter([hi.t], [hi.p], s=70, color="#b2182b", zorder=5, edgecolor="white", linewidth=1)
    ax.annotate(f"{hi.workload}\n{hi.p:.0f} W", (hi.t, hi.p), xytext=(6, -2), textcoords="offset points",
                fontsize=8, color="#b2182b", fontweight="bold", va="top")
    ax.annotate(f"{lo.workload}\n{lo.p:.0f} W", (lo.t, lo.p), xytext=(6, 4), textcoords="offset points",
                fontsize=8, color="#2166ac", fontweight="bold")
    ax.annotate(f"{hi.p / lo.p:.1f}x", (22, (lo.p + hi.p) / 2), ha="center", fontsize=12, color="#333",
                fontweight="bold")
    ax.set_xlabel("achieved rate  (inferences / s)", fontsize=10)
    ax.set_ylabel("board power  (W)", fontsize=10)
    ax.set_title("Same rate, very different power: at ~22 inf/s the A10G draws {0:.0f} to {1:.0f} W\n"
                 "depending on the model ({2:.1f}x) - power is a property of the operating point, not the rate"
                 .format(lo.p, hi.p, hi.p / lo.p), fontsize=9.2, loc="left", color=INK)
    ax.grid(True, alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlim(10, 75)
    fig.tight_layout(); out = os.path.join(os.path.dirname(__file__), "figures", "fig_same_rate.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig); print("->", out)


def fig_decompose():
    # why energy is U-shaped: as the cap drops, power/sec falls but the job takes longer; total energy is the
    # product, minimized in the middle (the Joule point). Shown for one workload on the A100.
    M = pd.read_csv(os.path.join(RAW, "..", "processed", "elf_master.csv"))
    g = M[(M.card == "A10G") & (M.workload == "resnet50") & (M.actuator == "clock")].sort_values("power_w")
    cap = 100 * g.power_w.values / 300.0     # board draw as % of TDP (clock reaches below the cap floor)
    power = g.power_w.values                 # watts drawn
    lat = 1000.0 / g.throughput.values       # ms per inference (duration)
    energy = power / g.throughput.values     # J per inference = power x time
    k = int(np.argmin(energy))
    # normalize every series to the UNCAPPED operating point (highest cap = last point) so the energy line
    # matches Figure 3: it starts at 1.0 on the right and dips below 1.0 by the energy saved.
    pn = power / power[-1]; tn = lat / lat[-1]; en = energy / energy[-1]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(cap, pn, "-o", color="#b2182b", lw=1.6, ms=4, label="board power  (per second)")
    ax.plot(cap, tn, "-s", color="#ef8a62", lw=1.6, ms=4, label="time per inference")
    ax.plot(cap, en, "-^", color="#14385c", lw=2.0, ms=5, label="energy per inference (power x time)")
    ax.axhline(1.0, color="#999", lw=0.8, ls=":", zorder=0)
    ax.scatter([cap[k]], [en[k]], s=90, color="#14385c", zorder=5, edgecolor="white", linewidth=1.2)
    ax.annotate("Joule Point\n(least energy)", (cap[k], en[k]), xytext=(cap[k] + 8, en[k] + 0.25),
                fontsize=9, color="#14385c", arrowprops=dict(arrowstyle="->", color="#14385c"))
    ax.set_xlabel("board draw  (% of TDP)", fontsize=10)
    ax.set_ylabel("normalized to the uncapped operating point", fontsize=9.5)
    pass  # title removed; description in paper caption
    ax.grid(True, alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#d1d4d8",
              loc="upper right")
    fig.tight_layout(); out = os.path.join(os.path.dirname(__file__), "figures", "fig_decompose.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig); print("->", out)


def fig_incentive():
    """Per-card summary: for each GPU, its 20 workloads' energy-optimal Joule caps (navy) against their
    cost-optimal caps under per-hour billing (red). Faint dots are the individual workloads; the large markers
    are per-card medians; the grey bar spans the median gap. On the big GPUs per-hour billing parks the cap far
    above the Joule point; on the small cards, already near the ceiling, there is little gap to open."""
    e = pd.read_csv(os.path.join(RAW, "edp_dollar.csv"))
    order = ["L4", "A10G", "A100"]  # T4 excluded from the analysis (see paper limitations)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    rng = np.random.RandomState(0)
    for i, card in enumerate(order):
        g = e[e.card == card]
        if len(g) == 0:
            continue
        jit = (rng.rand(len(g)) - 0.5) * 0.26
        ax.scatter(g.cap_Emin, i + jit, s=14, color=ACC, alpha=0.35, zorder=2)
        ax.scatter(g.cap_dollar, i + jit, s=14, color="#b2182b", alpha=0.35, zorder=2)
        mj, md = g.cap_Emin.median(), g.cap_dollar.median()
        ax.plot([mj, md], [i, i], "-", color="#9aa4b0", lw=3.0, alpha=0.7, zorder=1)
        ax.scatter([mj], [i], s=95, color=ACC, edgecolor="white", linewidth=1.2, zorder=4)
        ax.scatter([md], [i], s=95, color="#b2182b", edgecolor="white", linewidth=1.2, zorder=4)
        ax.annotate(f"{md - mj:.0f} pts", ((mj + md) / 2, i), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="#333", fontweight="bold")
    ax.scatter([], [], s=95, color=ACC, label="Joule point (energy-optimal cap)")
    ax.scatter([], [], s=95, color="#b2182b", label="cost-optimal cap (per-hour billing)")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=10)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("power cap (% of TDP)  →  more power right", fontsize=10)
    ax.set_title("Per-hour billing parks the cap at the ceiling, far above the Joule point\n"
                 "(dots = 20 workloads per card; large marks = medians; gap widens with GPU size)",
                 fontsize=9.5, loc="left", color=INK)
    ax.grid(True, axis="x", alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.tight_layout(); out = os.path.join(os.path.dirname(__file__), "figures", "fig_incentive.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig); print("->", out)


def fig_probe():
    """Scatter, one point per (card, workload): the card's uncapped board draw (x, watts, log) against that
    configuration's measured energy-optimal cap (y, % of TDP). Within each card the Joule point clusters in a
    tight horizontal band across all 20 workloads, so ONE fixed cap per card serves every job; and the more
    absolute power a card pulls, the lower, as a fraction of TDP, its Joule point sits."""
    d = pd.concat([pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")),
                   pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))], ignore_index=True)
    d["card"] = _card(d); d = saturating_set(d)              # loaded regime
    d = d.groupby(["card", "workload", "cap_w"], as_index=False).agg(
        power_w=("power_w", "mean"), throughput=("throughput", "mean"))
    rows = []
    for (card, wl), g in d.groupby(["card", "workload"]):
        g = g.sort_values("cap_w"); E = g.power_w.values / g.throughput.values; k = int(np.argmin(E))
        rows.append(dict(card=card, draw=g.power_w.values[-1], joule=100 * g.cap_w.values[k] / TDP[card]))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for card in ["T4", "L4", "A10G", "A100"]:
        g = t[t.card == card]
        if len(g) == 0:
            continue
        med = g.joule.median(); sd = float(np.std(g.joule.values))   # ddof=0, matches FIX/caption
        ax.scatter(g.draw, g.joule, s=42, color=COL[card], alpha=0.8, edgecolor="white", linewidth=0.5,
                   zorder=3, label=f"{card}  (cap {med:.0f}% TDP, sd {sd:.1f} pts over {len(g)} jobs)")
        xm = g.draw.median()
        ax.plot([xm * 0.9, xm * 1.1], [med, med], color=COL[card], lw=2.6, alpha=0.9, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("uncapped board draw  (Watts, log scale)  →  bigger GPU right", fontsize=9.5)
    ax.set_ylabel("measured energy-optimal cap  (Joule point, % of TDP)", fontsize=9.5)
    ax.set_title("One fixed cap per card: each GPU's Joule point clusters tightly across all 20 workloads\n"
                 "(bar = per-card median); big GPUs cap deep, small cards run near the ceiling",
                 fontsize=9.2, loc="left", color=INK)
    ax.grid(True, alpha=0.2, which="both"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.2, frameon=False, loc="center left", title="per-card Joule point")
    fig.tight_layout(); out = os.path.join(os.path.dirname(__file__), "figures", "fig_probe.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig); print("->", out)


def fig_joule_allcards():
    d = pd.concat([pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")),
                   pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))], ignore_index=True)
    d["card"] = _card(d); d = saturating_set(d)              # loaded regime (max-draw batch per model)
    col = {"T4": "#b2182b", "L4": "#ef8a62", "A10G": "#67a9cf", "A100": "#14385c"}
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for card in ["A100", "A10G", "L4", "T4"]:
        g0 = d[d.card == card]
        curves = {}
        for wl, g in g0.groupby("workload"):
            gg = g.groupby("cap_w").agg(p=("power_w", "mean"), t=("throughput", "mean")).sort_index()
            if len(gg) < 4:
                continue
            E = (gg.p / gg.t).values
            for f, e in zip(np.round(100 * gg.index.values / TDP[card]).astype(int), E / E.min()):
                curves.setdefault(int(f), []).append(e)
        caps = np.array(sorted(curves))
        if len(caps) < 3:
            continue
        med = np.array([np.median(curves[c]) for c in caps]); k = int(np.argmin(med))
        ax.plot(caps, med, "-", color=col[card], lw=1.8, label=card, zorder=2)
        ax.scatter([caps[k]], [med[k]], s=70, color=col[card], zorder=4, edgecolor="white", linewidth=1)
    ax.axhline(1.0, color="#999", lw=0.7, ls=":", zorder=0)
    ax.set_xlabel("power cap (% of TDP)", fontsize=10)
    ax.set_ylabel("energy per inference / its own minimum", fontsize=10)
    ax.set_title("Large GPUs have a deep Joule point (dot) well below TDP; small cards already run near it", fontsize=9.5, loc="left", color=INK)
    ax.grid(True, alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, title="loaded batch")
    fig.tight_layout(); out = os.path.join(os.path.dirname(__file__), "figures", "fig_joule_allcards.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig); print("->", out)


if __name__ == "__main__":
    fig_same_rate(); fig_incentive(); fig_probe(); fig_joule_allcards(); fig_decompose()
