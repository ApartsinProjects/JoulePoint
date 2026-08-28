# -*- coding: utf-8 -*-
"""
From a GPU-power primitive to a facility-meter commitment (a sensitivity MODEL, not a measurement).

The hardware cap bounds GPU power. A grid contract, however, is written against the FACILITY meter,
whose power is GPU power plus non-GPU IT load (CPU, memory, NIC, storage) and cooling/overhead. This
model makes the dilution explicit and quantifies (a) how much of a GPU-power reduction survives to the
facility meter, and (b) the safety margin a facility-meter commitment needs, as a function of two
plausible-range parameters:

  g_it  = GPU share of IT power (GPU / (GPU + non-GPU server load)). GPU nodes are GPU-dominated but the
          host CPU/mem/NIC/storage are non-trivial; we sweep g_it in [0.55, 0.85].
  PUE   = facility power / IT power (cooling + distribution overhead). Modern facilities sweep [1.1, 1.3].

Facility power P_fac = (P_gpu + P_nongpu_server) * PUE, so the GPU share of FACILITY power is
  s = g_it / PUE.
A GPU-power reduction of fraction r (of GPU power) delivers, at the meter, a facility-power reduction of
  r_fac = r * s      (the non-GPU load is assumed to hold; if it also drops, r_fac only improves).
The hardware LOWER-BOUND property survives at the meter: since GPU draw <= L is physical, the facility
meter is guaranteed to fall by at least r*s relative to the committed baseline, PROVIDED non-GPU load is
separately bounded. The margin the facility must hold back to keep the meter under a ceiling C_fac is the
non-GPU + overhead headroom, (1 - s) of committed facility power.

This is a model over stated ranges; we do not measure a real facility meter here.
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"; NAVY = "#14385c"

G_IT = [0.55, 0.65, 0.75, 0.85]         # GPU share of IT power
PUE = [1.1, 1.2, 1.3]                    # facility / IT power
GPU_REDUCTIONS = [0.10, 0.20, 0.30, 0.40, 0.50]   # fraction of GPU power removed by the cap


def main():
    rows = []
    for g in G_IT:
        for pue in PUE:
            s = g / pue                                  # GPU share of facility power
            for r in GPU_REDUCTIONS:
                rows.append({"g_it": g, "pue": pue, "gpu_share_facility": round(s, 3),
                             "gpu_reduction_pct": round(100 * r), "facility_reduction_pct": round(100 * r * s, 1)})
    # headline points
    def fac(g, pue, r): return 100 * r * (g / pue)
    mid = fac(0.75, 1.2, 0.30)                            # a 30% GPU cut at g=0.75, PUE=1.2
    lo = fac(0.55, 1.3, 0.30); hi = fac(0.85, 1.1, 0.30)
    out = {"params": {"g_it_range": [G_IT[0], G_IT[-1]], "pue_range": [PUE[0], PUE[-1]]},
           "rows": rows,
           "gpu30_to_facility_pct": {"low": round(lo, 1), "mid": round(mid, 1), "high": round(hi, 1)},
           "gpu_share_facility_range": [round(0.55 / 1.3, 2), round(0.85 / 1.1, 2)]}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "facility_power.json"), "w"), indent=2)

    # figure: facility-meter reduction vs GPU-power reduction, band over (g_it, PUE)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    rr = np.array(GPU_REDUCTIONS) * 100
    best = np.array([fac(0.85, 1.1, r) for r in GPU_REDUCTIONS])
    worst = np.array([fac(0.55, 1.3, r) for r in GPU_REDUCTIONS])
    midl = np.array([fac(0.75, 1.2, r) for r in GPU_REDUCTIONS])
    ax.fill_between(rr, worst, best, color=NAVY, alpha=0.12, label="range over g$_{IT}$∈[.55,.85], PUE∈[1.1,1.3]")
    ax.plot(rr, midl, "-o", color=NAVY, lw=2, ms=5, mfc="white", label="g$_{IT}$=0.75, PUE=1.2")
    ax.plot(rr, rr, ls=(0, (3, 2)), color="#888", lw=1.2, label="if GPU were the whole meter (y=x)")
    ax.annotate(f"a 30% GPU cut delivers\n~{midl[2]:.0f}% at the facility meter", xy=(30, midl[2]),
                xytext=(31, midl[2] - 9), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.set_xlabel("GPU-power reduction (%)"); ax.set_ylabel("facility-meter reduction (%)")
    ax.set_title("GPU cap to facility meter: the dilution", fontsize=10.5, loc="left", color=INK)
    ax.legend(fontsize=7.4, frameon=False, loc="upper left")
    ax.grid(True, alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_facility.png"), dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print("== GPU-power primitive -> facility-meter reduction (model) ==")
    print(f"GPU share of facility power s = g_it/PUE in [{out['gpu_share_facility_range'][0]}, {out['gpu_share_facility_range'][1]}]")
    print(f"a 30% GPU-power cut delivers {lo:.0f}-{hi:.0f}% at the facility meter (mid {mid:.0f}% at g=0.75, PUE=1.2)")
    print("figure -> figures/fig_facility.png ; written -> results/facility_power.json")


if __name__ == "__main__":
    main()
