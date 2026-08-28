# -*- coding: utf-8 -*-
"""Energy U-shape per card, one best-illustrating workload per card, normalized to each card's uncapped
(full-power) value so the curve starts at 1.0 and dips below it by the energy saved; x is measured board draw as
a fraction of TDP. One operating-point sweep per card (the actuator is a technical detail and is not shown): the
large GPUs reach the interior minimum within their reachable draw range, the small cards reach a lower draw than
the cap floor allows. A shaded band shows +/-1 sd propagated from per-repetition variance. Dot = Joule point.
NOTE: cap and clock sweeps use per-workload-calibrated batches that differ on T4/L4, so their per-step energies
are not directly splice-able; each card here uses a single consistent sweep to avoid mixing batches."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INK = "#111418"
COL = {"T4": "#b2182b", "L4": "#ef8a62", "A10G": "#2166ac", "A100": "#14385c"}
TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}
# consistent display names for workloads across all figures (legends/captions)
PRETTY = {"vit_b_32": "ViT-B/32", "vit_b_16": "ViT-B/16", "swin_t": "Swin-T", "resnet50": "ResNet-50",
          "resnet152": "ResNet-152", "wide_resnet50_2": "Wide-ResNet-50-2", "sd_unet": "SD-UNet",
          "llm_decode": "LLM-decode", "densenet121": "DenseNet-121", "convnext_small": "ConvNeXt-S"}
# (card, workload, subset): big GPUs via the loaded cap sweep, small card via the clock sweep (single sweep
# each). The T4 is excluded from the analysis (see paper limitations).
PICK = [("A100", "vit_b_32", "loaded_primary"), ("A10G", "resnet50", "clock"),
        ("L4", "wide_resnet50_2", "clock")]


def main():
    M = pd.read_csv(os.path.join(HERE, "data", "processed", "elf_master.csv"))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for card, wl, subset in PICK:
        if subset == "clock":
            g = M[(M.card == card) & (M.workload == wl) & (M.actuator == "clock")]
        else:
            g = M[(M.card == card) & (M.workload == wl) & (M.loaded_primary)]
        g = g.assign(draw=100 * g.power_w / TDP[card], E=g.power_w / g.throughput)
        sP = (g.power_w_sd.fillna(0) / g.power_w); sR = (g.throughput_sd.fillna(0) / g.throughput)
        g = g.assign(Esd=g.E * np.sqrt(sP**2 + sR**2)).sort_values("draw")
        if len(g) < 4:
            print(f"skip {card} {wl}: {len(g)} pts"); continue
        Eunc = g.E.values[-1]
        En = g.E.values / Eunc; band = g.Esd.values / Eunc
        k = int(np.argmin(En))
        ax.fill_between(g.draw.values, En - band, En + band, color=COL[card], alpha=0.15, zorder=1)
        ax.plot(g.draw.values, En, "-o", color=COL[card], lw=2.0, ms=3.5, mfc="white", zorder=3,
                label=f"{card}  {PRETTY.get(wl, wl)}")
        ax.scatter([g.draw.values[k]], [En[k]], s=95, color=COL[card], edgecolor="white", linewidth=1.2, zorder=5)
    ax.axhline(1.0, color="#999", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("board draw  (% of TDP)   →  more power", fontsize=10)
    ax.set_ylabel("energy per inference  (relative to uncapped)   ↓ = savings", fontsize=10)
    pass  # title removed; description in paper caption
    ax.grid(True, alpha=0.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    leg = ax.legend(fontsize=9, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#d1d4d8",
                    loc="upper center", ncol=1, title="card    workload", title_fontsize=8.5)
    leg._legend_box.align = "left"
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "fig_energy_u.png")
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_energy_u.png")


if __name__ == "__main__":
    main()
