# -*- coding: utf-8 -*-
"""
Measured strict-envelope control on real GPUs (paper Figure). 3x3 small multiples: rows = GPU
(A10G/L4/T4), cols = target trajectory (step/staircase/ramp). Each cell: measured GPU power
(teal) tracking the hardware-cap target (black dashed), with the per-cell time-above-target %.
The A10G (wide power-cap range) holds the target strictly (~0% above); the low-TDP L4/T4 saturate
their cap floor at aggressive targets, motivating the software-actuator portfolio for deep
curtailment. Also writes results/aws_control_strict_summary.json.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
TEAL = "#158a8a"; INK = "#111418"
# T4 excluded: its NVML power readings exceed the card's 70 W TDP (a telemetry artifact on the
# g4dn instance), so they are not trustworthy. A10G (wide cap range) + L4 (narrow) tell the story.
GPU_ORDER = ["NVIDIA A10G", "NVIDIA L4"]
GPU_SHORT = {"NVIDIA A10G": "A10G", "NVIDIA L4": "L4"}
TRAJ = ["step", "staircase", "ramp"]


def main():
    d = pd.read_csv(os.path.join(HERE, "data", "raw", "aws_control_strict.csv"))
    d = d[d["gpu"].isin(GPU_ORDER)].copy()
    for c in ["t", "target_w", "power_w", "crit_p95_ms", "defer_thru"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["t", "target_w", "power_w"])

    # telemetry summary (from results/aws_control_strict.json)
    tel = [r for r in json.load(open(os.path.join(HERE, "results", "aws_control_strict.json")))
           if r.get("gpu") in GPU_ORDER]
    def fnum(r, k):
        try: return float(r[k])
        except: return float("nan")
    dflt = {g: max(fnum(r, "target_w") for r in tel if r["gpu"] == g) for g in GPU_ORDER}
    summ = {}
    for g in GPU_ORDER:
        con = [r for r in tel if r["gpu"] == g and fnum(r, "target_w") < dflt[g] - 1]
        summ[GPU_SHORT[g]] = {
            "default_w": dflt[g],
            "time_above_target_pct_mean": round(float(np.nanmean([fnum(r, "frac_above_pct") for r in con])), 2),
            "time_above_target_pct_max": round(float(np.nanmax([fnum(r, "frac_above_pct") for r in con])), 1),
            "actuation_s_median": round(float(np.nanmedian([fnum(r, "actuation_s") for r in con if fnum(r, "actuation_s") >= 0])), 3),
            "crit_p95_ms_median": round(float(np.nanmedian([fnum(r, "crit_p95_ms") for r in con if fnum(r, "crit_p95_ms") >= 0])), 2),
        }
    json.dump(summ, open(os.path.join(HERE, "results", "aws_control_strict_summary.json"), "w"), indent=2)

    fig, ax = plt.subplots(len(GPU_ORDER), 3, figsize=(8.6, 4.4), sharex="col")
    for i, g in enumerate(GPU_ORDER):
        for jx, tj in enumerate(TRAJ):
            a = ax[i][jx]
            sub = d[(d["gpu"] == g) & (d["trajectory"] == tj)].sort_values("t")
            if len(sub):
                a.plot(sub["t"], sub["target_w"], "--", color=INK, lw=1.2, drawstyle="steps-post")
                a.plot(sub["t"], sub["power_w"], "-", color=TEAL, lw=1.5)
                # per-cell time-above-target from telemetry (constrained segments)
                con = [r for r in tel if r["gpu"] == g and r["trajectory"] == tj
                       and fnum(r, "target_w") < dflt[g] - 1]
                fa = np.nanmean([fnum(r, "frac_above_pct") for r in con]) if con else float("nan")
                a.text(0.97, 0.90, f"above: {fa:.1f}%", transform=a.transAxes, ha="right", va="top",
                       fontsize=7.2, color=(TEAL if fa < 3 else "#b2182b"),
                       fontfamily="monospace")
            a.grid(True, alpha=0.22); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
            if i == 0:
                a.set_title(tj, fontsize=9.5, color=INK)
            if jx == 0:
                a.set_ylabel(f"{GPU_SHORT[g]}\npower (W)", fontsize=8.5)
            if i == 2:
                a.set_xlabel("time (s)", fontsize=8.5)
    fig.suptitle("Measured hardware-cap control: power (teal) vs target (dashed) across GPUs and trajectories",
                 fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(HERE, "figures", "fig_strict_control.png"), dpi=140, facecolor="white",
                bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_strict_control.png")
    print("summary:", json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
