# -*- coding: utf-8 -*-
"""
Headline six-panel systems figure, SIMULATED-ON-MEASURED.

Panels 3-6 are the ACTUAL output of the PoC-B tick simulator (real Azure arrival trace +
measured emerald inference power-cap elasticity) under a dynamic evening power envelope:
  3. facility power (uncontrolled vs controlled vs allowance), MW (100 MW facility)
  4. reduction decomposed into GPU power-cap vs deferral, MW, WITH the required-reduction
     line (max(0, uncontrolled - allowance)) overlaid -- the controller sheds only what the
     envelope demands (delivered tracks required), i.e. it does NOT over-curtail.
  5. per-class p95 latency over time (sliding window), seconds -- critical/interactive held
     near their deadlines while offline latency grows (deferred), the real QoS trade.
  6. deferred backlog (# requests) and the post-dip REBOUND power above the pre-dip baseline
     (only after the window ends), MW -- the recovery spike the backlog-aware controller bounds.
Panels 1-2 are the grid CONTEXT (Israeli evening, Noga-anchored) and the IL-2 allowance scenario.

Provenance is stated once at the figure foot (no per-panel clutter): panels 1-2 are a grid
scenario; panels 3-6 are simulated on the real Azure trace + measured elasticity.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pocb_sim as S

FIG = os.path.join(os.path.dirname(__file__), "figures")
DIP_START_S, DIP_END_S = 1200, 2400          # curtailment window (20-40 min)


def sliding_p95(comp, cls, t_minutes, win_s=60.0):
    """Per-class p95 latency in a trailing window, evaluated on the minute grid."""
    ct = np.array([c[0] for c in comp]); cc = np.array([c[1] for c in comp])
    cl = np.array([c[2] for c in comp])
    out = np.full((len(t_minutes), len(S.CLASSES)), np.nan)
    ts = t_minutes * 60.0
    for k, tt in enumerate(ts):
        m = (ct > tt - win_s) & (ct <= tt)
        for c in range(len(S.CLASSES)):
            v = cl[m & (cc == c)]
            if len(v):
                out[k, c] = np.percentile(v, 95)
    return out


def main():
    reqs = S.load_requests(subsample=1.0)
    T = int(3600 / S.DT)
    ref = S.uncontrolled_peak(reqs, T=T)
    # allowance: flat at the installed peak (100 MW after scaling), dipping to 45% mid-window.
    C = np.full(T, ref)                                    # ceiling == installed capacity (not 1.02)
    C[int(DIP_START_S / S.DT):int(DIP_END_S / S.DT)] = ref * 0.45

    _, ts_u = S.simulate(reqs, np.full(T, 1e12), S.ctrl_none, serve="priority")
    det = {"gate": [], "qlen": [], "util": [], "comp": []}
    _, ts_c = S.simulate(reqs, C, S.ctrl_elasticity, serve="priority", detail=det)
    qage = np.array(det["qage"])                           # per-tick p95 in-queue wait (s), per class
    predP = np.array(det["pred_P"])                        # controller's feedforward power PREDICTION

    Pu = np.array(ts_u["P"]); Pc = np.array(ts_c["P"]); cap = np.array(ts_c["cap"])
    Cs = np.array(ts_c["C"]); util = np.array(det["util"]); qlen = np.array(det["qlen"])
    peakW = Pu.max(); to_mw = 100.0 / peakW                # scale sim watts -> a 100 MW facility
    t = np.arange(T) * S.DT / 60.0                         # minutes
    d0, d1 = DIP_START_S / 60.0, DIP_END_S / 60.0          # window edges in minutes

    # reduction decomposition (MW) and the reduction the envelope actually REQUIRES
    full = S.cluster_power(1.0, util); capped = S.cluster_power(cap, util)
    cap_mw = np.maximum(0.0, (full - capped)) * to_mw
    red_total = np.maximum(0.0, Pu - Pc) * to_mw
    defer_mw = np.maximum(0.0, red_total - cap_mw)
    required_mw = np.maximum(0.0, Pu - Cs) * to_mw          # what the allowance demands

    def sm(a, w=30):
        k = np.ones(w) / w; return np.convolve(a, k, mode="same")

    fig, ax = plt.subplots(6, 1, figsize=(7.8, 12.8), sharex=True,
                           gridspec_kw={"height_ratios": [1, 0.8, 1.2, 1, 1.1, 1]})
    for a in ax:
        a.axvspan(d0, d1, color="#9aa0aa", alpha=0.12)     # neutral curtailment shading
        for x in (d0, d1):
            a.axvline(x, color="#6b7280", lw=0.9, ls=(0, (4, 3)), alpha=0.8)
        a.grid(True, alpha=0.22); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    ax[0].annotate("curtailment window", xy=((d0 + d1) / 2, 0.92), xycoords=("data", "axes fraction"),
                   ha="center", fontsize=8, color="#4a5160")

    # Panel 1 -- grid context (Noga-anchored scenario)
    demand = 9.0 + 6.0 * np.exp(-((t - 30) ** 2) / 120)
    pv = np.clip(4.0 * np.exp(-((t - 2) ** 2) / 200), 0, None)
    ax[0].plot(t, demand, color="#1a1d24", lw=2, label="Israeli demand")
    ax[0].fill_between(t, 0, pv, color="#137a4b", alpha=0.25, label="PV")
    ax[0].set_ylabel("grid (GW)"); ax[0].legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")

    # Panel 2 -- allowance (scenario)
    ax[1].plot(t, Cs * to_mw, color="#b42318", lw=2, drawstyle="steps-mid", label="power allowance")
    ax[1].axhline(peakW * to_mw, color="#888", ls="--", lw=1, label="installed 100 MW")
    ax[1].set_ylabel("allowance (MW)"); ax[1].set_ylim(0, 110)
    ax[1].legend(fontsize=7.5, frameon=False, ncol=2, loc="lower left")

    # Panel 3 -- actual power, with the response MODEL's feedforward prediction the controller acts on
    ax[2].plot(t, sm(Pu) * to_mw, color="#9a3b8c", ls="--", lw=1.6, label="uncontrolled (sim)")
    ax[2].plot(t, sm(Pc) * to_mw, color="#1f6feb", lw=2, label="controlled (sim)")
    ax[2].plot(t, sm(predP) * to_mw, color="#137a4b", lw=1.3, ls=":", label="model-predicted power")
    ax[2].plot(t, Cs * to_mw, color="#b42318", lw=1.1, alpha=.7, drawstyle="steps-mid", label="allowance")
    ax[2].set_ylabel("facility (MW)"); ax[2].legend(fontsize=6.8, frameon=False, ncol=4, loc="lower left")
    # model prediction accuracy (how well the online model tracks the realized sim power)
    _w = (t >= d0) & (t < d1)
    _mae = float(np.nanmedian(np.abs((predP - Pc)[_w]))) * to_mw
    # clarify: the flat allowance is the installed-capacity ceiling (non-binding in normal
    # operation, where the facility runs ~65% utilised); only the dip makes it bind.
    ax[2].annotate("allowance = installed cap\n(non-binding: facility ~65% utilised)",
                   xy=(9, 100), xytext=(2.0, 52), fontsize=6.6, color="#8a3b2e",
                   arrowprops=dict(arrowstyle="->", color="#b42318", lw=0.7, alpha=0.7))
    ax[2].annotate("allowance binds →\ncontroller sheds the gap", xy=(30, 45), xytext=(43, 52),
                   fontsize=6.6, color="#8a3b2e",
                   arrowprops=dict(arrowstyle="->", color="#b42318", lw=0.7, alpha=0.7))

    # Panel 4 -- intervention mix + required-reduction line (no over-shed)
    ax[3].fill_between(t, 0, sm(cap_mw), color="#1f6feb", alpha=0.55, label="delivered: GPU power cap")
    ax[3].fill_between(t, sm(cap_mw), sm(cap_mw) + sm(defer_mw), color="#b0641f", alpha=0.55,
                       label="delivered: deferred low-priority")
    ax[3].plot(t, sm(required_mw), color="#1a1d24", lw=1.8, label="required by allowance")
    ax[3].set_ylabel("reduction (MW)"); ax[3].legend(fontsize=7.2, frameon=False, ncol=1, loc="upper left")

    # Panel 5 -- per-class p95 IN-QUEUE WAIT over time (gap-free QoS, seconds). Critical and
    # interactive are served first so their wait stays at their deadline; elastic then offline
    # accumulate wait DURING the window and relax once the backlog drains -- the real QoS trade.
    cols = ["#137a4b", "#1f6feb", "#b0641f", "#b42318"]
    for c, name in enumerate(S.CLASSES):
        ax[4].plot(t, np.maximum(sm(qage[:, c], 10), 1e-2), color=cols[c], lw=1.5, label=name)
        ax[4].axhline(S.CLASS_DEADLINE_S[name], color=cols[c], lw=0.8, ls=":", alpha=0.45)
    ax[4].set_yscale("log"); ax[4].set_ylabel("p95 in-queue wait (s, log)")
    ax[4].set_ylim(0.05, None)
    ax[4].legend(fontsize=7, frameon=False, ncol=4, loc="upper left", title=None)

    # Panel 6 -- deferred backlog builds during the window, then drains; the drain forces a
    # recovery pulse: controlled power RISES above the pre-event baseline right after the window
    # and relaxes back as the backlog clears. Shown continuously (0 before/during, a pulse after)
    # so the rise-then-fall reads as one bounded rebound, not a bare decaying line.
    # backlog stacked by class: offline (sacrificed first) sits on the bottom, elastic on top,
    # so the split shows the controller absorbs the shortfall mostly on the least-valuable class.
    off_bl = sm(qlen[:, 3]); ela_bl = sm(qlen[:, 2])
    # reference = the pre-event NORMAL PEAK (95th pct), so recovery power isolates the genuine
    # post-event rebound pulse rather than ordinary tick-to-tick load variation.
    baseline_pre = np.percentile((Pc * to_mw)[t < d0], 95)
    rebound = np.maximum(0.0, sm(Pc, 20) * to_mw - baseline_pre)
    ax[5].fill_between(t, 0, off_bl, color="#b42318", alpha=0.30, label="offline backlog (requests)")
    ax[5].fill_between(t, off_bl, off_bl + ela_bl, color="#b0641f", alpha=0.35, label="elastic backlog (requests)")
    ax5b = ax[5].twinx()
    ax5b.plot(t, rebound, color="#137a4b", lw=2.0, label="recovery power above pre-event baseline (MW)")
    ax5b.axhline(0, color="#137a4b", lw=0.6, alpha=0.4)
    ax5b.set_ylabel("recovery power (MW)", color="#137a4b"); ax5b.tick_params(axis="y", labelcolor="#137a4b")
    ax5b.set_ylim(0, None); ax5b.spines["top"].set_visible(False)
    peak_reb_t = t[d1 <= t][np.nanargmax(rebound[d1 <= t])] if np.any(d1 <= t) else d1
    ax5b.annotate("backlog drains\n→ bounded rebound", xy=(peak_reb_t, np.nanmax(rebound[t >= d1])),
                  xytext=(peak_reb_t + 6, np.nanmax(rebound) * 0.9), fontsize=6.8, color="#137a4b",
                  arrowprops=dict(arrowstyle="->", color="#137a4b", lw=0.8))
    ax[5].set_ylabel("backlog (requests)"); ax[5].set_xlabel("minutes into the evening event")
    l1, la1 = ax[5].get_legend_handles_labels(); l2, la2 = ax5b.get_legend_handles_labels()
    ax[5].legend(l1 + l2, la1 + la2, fontsize=7.2, frameon=False, ncol=1, loc="upper left")

    # per-panel provenance tags (scenario / simulated-on-measured / model prediction)
    tags = ["scenario", "scenario",
            "SIM on measured  +  MODEL prediction (green ···)",
            "SIM on measured", "SIM on measured", "SIM on measured"]
    for a, tg in zip(ax, tags):
        a.text(0.995, 1.02, tg, transform=a.transAxes, ha="right", va="bottom",
               fontsize=6.4, color="#8a94a3", style="italic")
    # annotate how the model is USED: predict power for the action -> keep it under the allowance
    ax[2].annotate(f"model predicts each action's power;\ncontroller picks the least-shedding one\n"
                   f"under the allowance (window MAE {_mae:.1f} MW)",
                   xy=(33, np.nanmedian(predP[_w]) * to_mw), xytext=(21, 90),
                   fontsize=6.2, color="#137a4b", ha="left",
                   arrowprops=dict(arrowstyle="->", color="#137a4b", lw=0.7, alpha=0.8))

    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    fig.text(0.5, 0.016, "Provenance: panels 1-2 = grid SCENARIO (Noga-anchored).  Panels 3-6 = SIMULATED "
             "on the real Azure trace + MEASURED emerald elasticity.", ha="center", fontsize=7.0, color="#6b7280")
    fig.text(0.5, 0.006, "Green dotted = the trained response MODEL's online prediction the controller acts on "
             "(measured on hardware only in the separate A10G control figure).",
             ha="center", fontsize=6.6, color="#9aa0aa")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig_six_panel.png"), dpi=130, facecolor="white"); plt.close(fig)

    # numeric check for the over-shed claim: delivered vs required in the binding window
    win = (t >= d0) & (t < d1)
    over = np.median((red_total[win] - required_mw[win]))
    print(f"  in-window medians (MW): Pu={np.median(Pu[win])*to_mw:.1f} "
          f"Pc={np.median(Pc[win])*to_mw:.1f} allowance={np.median(Cs[win])*to_mw:.1f} "
          f"required={np.median(required_mw[win]):.1f} delivered={np.median(red_total[win]):.1f}")
    print(f"figure -> figures/fig_six_panel.png  (peak {peakW*to_mw:.0f} MW, "
          f"max cap-reduction {cap_mw.max():.1f} MW, max defer-reduction {defer_mw.max():.1f} MW)")
    print(f"  in-window median delivered-minus-required reduction = {over:.2f} MW "
          f"(near 0 => sheds only what the allowance demands)")


if __name__ == "__main__":
    main()
