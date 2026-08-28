# -*- coding: utf-8 -*-
"""Four representative jobs spanning the arithmetic-intensity spectrum: latency vs power cap, across all
four GPUs (T4/L4/A10G/A100). Latency = t_compute_ms (the cap-sensitive phase). Shows both levers at once:
capping moves left along a card's curve; each card occupies its own cap band (60W T4 -> 400W A100)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict_power import RAW

CARDCOL = {"A100": "#762a83", "A10G": "#14385c", "L4": "#e08214", "T4": "#b2182b"}
ORDER = ["T4", "L4", "A10G", "A100"]
INK = "#111418"
JOBS = [("llm_decode", "LLM decode (memory-bound)"),
        ("resnet152", "ResNet-152 (CNN inference)"),
        ("sd_unet", "Stable Diffusion UNet (compute-bound)"),
        ("vit_b_16", "ViT-B/16 (compute-bound)")]


def load():
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_spectrum.csv")),
          pd.read_csv(os.path.join(RAW, "aws_sweep_a100.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    df["card"] = df.card.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    return df.groupby(["card", "workload", "cap_w"], as_index=False).agg(
        lat=("t_compute_ms", "mean"), power=("power_w", "mean"))


def main():
    df = load()
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 7.2))
    for k, (wl, title) in enumerate(JOBS):
        a = ax[k // 2][k % 2]; sub = df[df.workload == wl]
        for card in [c for c in ORDER if c in set(sub.card)]:
            c = sub[sub.card == card].sort_values("cap_w")
            a.plot(c.cap_w, c.lat, "-o", color=CARDCOL[card], lw=2, ms=4.5, mfc="white", label=card)
        a.set_title(title, fontsize=10, loc="left", color=INK)
        a.set_xlabel("power cap (W)"); a.set_ylabel("latency per inference (ms)")
        a.grid(True, alpha=0.25); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.legend(fontsize=8.5, frameon=False, title="GPU", title_fontsize=8)
    fig.suptitle("Latency vs power cap, four jobs x four GPUs (60W T4 -> 400W A100)",
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(__file__), "figures", "fig_latency_vs_cap.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_latency_vs_cap.png")
    for wl, _ in JOBS:
        s = df[df.workload == wl]
        for card in [c for c in ORDER if c in set(s.card)]:
            c = s[s.card == card]
            print(f"  {wl:12} {card:5} cap {c.cap_w.min():.0f}-{c.cap_w.max():.0f}W  lat {c.lat.min():.1f}-{c.lat.max():.1f}ms")


if __name__ == "__main__":
    main()
