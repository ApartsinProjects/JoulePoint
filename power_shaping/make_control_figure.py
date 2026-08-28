# -*- coding: utf-8 -*-
"""
MEASURED closed-loop control figure from data/raw/aws_control.csv (real A10G).
Three aligned panels over time, with the constrained window shaded:
  A. measured GPU power (uncontrolled vs controlled) vs the power target
     -> controlled tracks the target (sheds only what's required); uncontrolled violates.
  B. critical-stream p95 latency (uncontrolled vs controlled) -> critical protected.
  C. deferrable throughput (uncontrolled vs controlled) -> low-priority work deferred.
All three panels are MEASURED on real hardware (not simulated).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")


def main():
    d = pd.read_csv(os.path.join(HERE, "data", "raw", "aws_control.csv"))
    for c in ["t", "target_w", "power_w", "crit_p95_ms", "defer_thru"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    base = d[d["run"] == "base"].sort_values("t"); ctrl = d[d["run"] == "ctrl"].sort_values("t")
    win = (20, 55)

    fig, ax = plt.subplots(3, 1, figsize=(6.8, 8.0), sharex=True,
                           gridspec_kw={"height_ratios": [1.2, 1, 1]})
    for a in ax:
        a.axvspan(*win, color="#9a6700", alpha=0.10); a.grid(True, alpha=0.25)
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)

    ax[0].plot(ctrl["t"], ctrl["target_w"], color="#b42318", lw=2, drawstyle="steps-post", label="power target")
    ax[0].plot(base["t"], base["power_w"], color="#9a3b8c", lw=1.6, ls="--", label="uncontrolled power")
    ax[0].plot(ctrl["t"], ctrl["power_w"], color="#1f6feb", lw=2, label="controlled power")
    ax[0].set_ylabel("GPU power (W)"); ax[0].legend(fontsize=8, frameon=False, ncol=3, loc="lower left")
    ax[0].set_title("Measured closed-loop control on a real A10G (constrained window shaded)")

    ax[1].plot(base["t"], base["crit_p95_ms"], color="#9a3b8c", lw=1.6, ls="--", label="uncontrolled")
    ax[1].plot(ctrl["t"], ctrl["crit_p95_ms"], color="#137a4b", lw=2, label="controlled")
    ax[1].set_ylabel("critical p95 latency (ms)"); ax[1].legend(fontsize=8, frameon=False, ncol=2)

    ax[2].plot(base["t"], base["defer_thru"], color="#9a3b8c", lw=1.6, ls="--", label="uncontrolled")
    ax[2].plot(ctrl["t"], ctrl["defer_thru"], color="#b0641f", lw=2, label="controlled")
    ax[2].set_ylabel("deferrable throughput (it/s)"); ax[2].set_xlabel("time (s)")
    ax[2].legend(fontsize=8, frameon=False, ncol=2)

    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig_measured_control.png"), dpi=130, facecolor="white"); plt.close(fig)

    # summary metrics
    def inwin(x): return x[(x["t"] >= win[0]) & (x["t"] < win[1])]
    bw, cw = inwin(base), inwin(ctrl)
    tgt = cw["target_w"].median()
    out = {"platform": "AWS g5.xlarge A10G (measured)",
           "window_target_w": float(tgt),
           "uncontrolled_power_w_median_inwindow": float(bw["power_w"].median()),
           "controlled_power_w_median_inwindow": float(cw["power_w"].median()),
           "uncontrolled_violation_w": float(max(0, bw["power_w"].median() - tgt)),
           "controlled_violation_w": float(max(0, cw["power_w"].median() - tgt)),
           "crit_p95_ms_uncontrolled_inwindow": float(bw["crit_p95_ms"].median()),
           "crit_p95_ms_controlled_inwindow": float(cw["crit_p95_ms"].median()),
           "defer_thru_uncontrolled_inwindow": float(bw["defer_thru"].median()),
           "defer_thru_controlled_inwindow": float(cw["defer_thru"].median())}
    json.dump(out, open(os.path.join(RES, "aws_control.json"), "w"), indent=2)
    print("== MEASURED control (A10G, in constrained window) ==")
    print(f"  target {tgt:.0f} W | uncontrolled power {out['uncontrolled_power_w_median_inwindow']:.0f} W "
          f"(violate {out['uncontrolled_violation_w']:.0f} W) -> controlled {out['controlled_power_w_median_inwindow']:.0f} W "
          f"(violate {out['controlled_violation_w']:.0f} W)")
    print(f"  critical p95 latency: uncontrolled {out['crit_p95_ms_uncontrolled_inwindow']:.2f} ms -> "
          f"controlled {out['crit_p95_ms_controlled_inwindow']:.2f} ms")
    print(f"  deferrable throughput: {out['defer_thru_uncontrolled_inwindow']:.1f} -> "
          f"{out['defer_thru_controlled_inwindow']:.1f} it/s (deferred)")
    print("figure -> figures/fig_measured_control.png")


if __name__ == "__main__":
    main()
