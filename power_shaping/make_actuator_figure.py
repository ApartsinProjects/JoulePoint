# -*- coding: utf-8 -*-
"""
Two measured actuators beyond power-cap + deferral (paper Figure), from data/raw/aws_actuator.csv
(one A10G). (A) DVFS-vs-cap crossover: throughput vs mean power for SM-clock scaling (squares) and
the power cap (circles); at matched power the clock delivers more throughput / lower energy-per-op.
(B) Model-cascade power-vs-quality frontier: measured energy-per-inference (log) vs published
ImageNet top-1 accuracy across a model ladder, the quality axis that power-cap (latency) and
deferral (delay) do not reach.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CLK = "#e08214"; CAP = "#2166ac"; INK = "#111418"
# published torchvision ImageNet-1k top-1 accuracy (documentation values, cited in the paper)
ACC = {"mobilenet_v3_small": 67.7, "resnet50": 76.1, "densenet121": 74.4, "vit_b_16": 81.1, "vit_l_16": 79.7}
LAB = {"mobilenet_v3_small": "MobileNetV3-S", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
       "vit_b_16": "ViT-B/16", "vit_l_16": "ViT-L/16"}


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "raw", "aws_actuator.csv"))))
    def sel(k): return [r for r in rows if r["kind"] == k]
    cl = sorted(sel("clock"), key=lambda r: float(r["power_w"]))
    cp = sorted(sel("cap"), key=lambda r: float(r["power_w"]))
    md = sel("model")

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))

    # (A) DVFS vs cap crossover
    for grp, col, mk, lab in [(cl, CLK, "s", "SM-clock scaling"), (cp, CAP, "o", "power cap")]:
        x = [float(r["power_w"]) for r in grp]; y = [float(r["throughput"]) for r in grp]
        ax[0].plot(x, y, "-", marker=mk, color=col, lw=2, ms=6, mfc="white" if mk == "o" else col, label=lab)
    ax[0].set_xlabel("mean GPU power (W)"); ax[0].set_ylabel("throughput (ops/s)")
    ax[0].set_title("(A) Clock scaling beats the cap", fontsize=9.5, loc="left", color=INK)
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    # annotate matched-power advantage
    jclk = min(float(r["energy_per_op_j"]) for r in cl); jcap = min(float(r["energy_per_op_j"]) for r in cp)
    ax[0].text(0.03, 0.93, f"min energy/op: clock {jclk:.2f} J < cap {jcap:.2f} J", transform=ax[0].transAxes,
               fontsize=7.4, color=INK, va="top")

    # (B) model-cascade power-vs-quality frontier
    E = [float(r["energy_per_op_j"]) for r in md]; A = [ACC[r["setting"]] for r in md]; N = [r["setting"] for r in md]
    o = np.argsort(E)
    ax[1].plot([E[i] for i in o], [A[i] for i in o], "-", color="#1b7837", lw=1.4, alpha=0.6, zorder=1)
    ax[1].scatter(E, A, s=42, color="#1b7837", zorder=2)
    for e, a, n in zip(E, A, N):
        ax[1].annotate(LAB[n], (e, a), textcoords="offset points", xytext=(6, -3), fontsize=7.0, color=INK)
    ax[1].set_xscale("log"); ax[1].set_xlabel("energy per inference (J, log) — measured")
    ax[1].set_ylabel("ImageNet top-1 accuracy (%) — published")
    ax[1].set_title("(B) A power-vs-quality frontier", fontsize=9.5, loc="left", color=INK)
    ax[1].text(0.03, 0.06, f"{max(E)/min(E):.0f}× energy range", transform=ax[1].transAxes, fontsize=7.6,
               color="#1b7837", style="italic")

    for a in ax:
        a.grid(True, alpha=0.25); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig_actuators.png"), dpi=140, facecolor="white",
                bbox_inches="tight"); plt.close(fig)
    print(f"figure -> figures/fig_actuators.png  (energy range {max(E)/min(E):.0f}x; DVFS min J/op {jclk:.2f} vs cap {jcap:.2f})")


if __name__ == "__main__":
    main()
