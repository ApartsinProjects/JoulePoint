# -*- coding: utf-8 -*-
"""
Placement view for AGGREGATED training jobs across multiple cards: duration vs power draw, with each card
(V100, A40) occupying its power band on the x-axis, for a single training job as the cap sweeps each card.
Same idea as the AWS placement figure, but for real DNN training and the Zeus GPUs. We plot the training
jobs that are present on BOTH cards; duration normalized to the fastest point per job.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict_power import RAW

CARDCOL = {"V100": "#1b7837", "A40": "#762a83"}
INK = "#111418"


def load():
    fr = []
    for card, f in [("V100", "zeus_summary_power_v100.csv"), ("A40", "zeus_summary_power_a40.csv")]:
        d = pd.read_csv(os.path.join(RAW, f)); d["card"] = card; fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df = df.groupby(["card", "network", "dataset", "batch_size", "optimizer", "power_limit"], as_index=False).agg(
        t=("time_per_epoch", "mean"), p=("average_power", "mean"))
    df = df[df.t > 0].copy(); df["rate"] = 1.0 / df.t
    df["jobkey"] = df.network + "/" + df.dataset + "/bs" + df.batch_size.astype(str)
    return df


def main():
    df = load()
    # jobs present on BOTH cards with several cap points each
    both = (df.groupby(["jobkey", "card"]).size().unstack(fill_value=0))
    common = both[(both["V100"] >= 3) & (both["A40"] >= 3)].index.tolist()
    # pick up to 3 representative jobs
    picks = common[:3] if len(common) >= 3 else common
    fig, ax = plt.subplots(1, max(1, len(picks)), figsize=(3.3 * max(1, len(picks)), 3.8), squeeze=False)
    for i, jk in enumerate(picks):
        a = ax[0][i]; sub = df[df.jobkey == jk]
        tmax = sub.rate.max()
        for card in ["V100", "A40"]:
            c = sub[sub.card == card].sort_values("p")
            if c.empty:
                continue
            dur = tmax / c.rate.values
            a.plot(c.p.values, dur, "-o", color=CARDCOL[card], lw=2, ms=5, mfc="white", label=card)
            a.annotate(card, (c.p.values[-1], dur[-1]), textcoords="offset points", xytext=(3, 2), fontsize=7.5, color=CARDCOL[card])
            if len(c) > 1:
                a.annotate("", xy=(c.p.values[0], dur[0]), xytext=(c.p.values[-1], dur[-1]),
                           arrowprops=dict(arrowstyle="->", color=CARDCOL[card], lw=1, alpha=0.5))
        a.set_title(jk, fontsize=8.5, loc="left", color=INK)
        a.set_xlabel("power draw (W)"); a.grid(True, alpha=0.25)
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        if i == 0:
            a.set_ylabel("duration (relative to fastest)"); a.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Training jobs across cards (Zeus V100 & A40): duration vs power draw",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(os.path.dirname(__file__), "figures", "fig_placement_zeus.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_placement_zeus.png ; jobs:", picks)
    for card in ["V100", "A40"]:
        s = df[df.card == card]
        print(f"  {card} power range across jobs: {s.p.min():.0f}-{s.p.max():.0f}W")


if __name__ == "__main__":
    main()
