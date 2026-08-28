# -*- coding: utf-8 -*-
"""
Control-point portfolio + supervisor schematic (paper Figure). Left: the actuator taxonomy, a
grid of time-scale (ms -> minutes, log) x mechanism class (shift energy in time / reduce energy
per work / reduce the work), with each actuator placed at its native time scale; the GPU power
cap carries a bold border = the one hardware-enforced, guarantee-bearing actuator. Right: the
supervisor loop, an ML layer shapes demand with software actuators and the hardware cap enforces
the ceiling, with clip-frequency closing the loop. Pure design; no data.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#111418"; MUT = "#5a626c"
SHIFT = "#2166ac"; ENERGY = "#e08214"; WORK = "#b2182b"           # mechanism-class colors
# row index -> (label, color); index matches ACT row field (0 bottom .. 2 top)
ROWS = [("shift energy\nin time", SHIFT), ("reduce energy\nper work", ENERGY), ("reduce\nthe work", WORK)]
# actuator: (name, log10(seconds), row_index, class_color, hardware_enforced)
ACT = [
    ("decode preempt.", 0.0, 0, SHIFT, False),
    ("request deferral", 1.0, 0, SHIFT, False),
    ("checkpoint+pause", 1.9, 0, SHIFT, False),
    ("power cap", -3.0, 1, ENERGY, True),
    ("clock lock (DVFS)", -1.85, 1, ENERGY, False),
    ("batch size", -0.35, 1, ENERGY, False),
    ("hetero-GPU routing", 0.9, 1, ENERGY, False),
    ("precision/quant.", 1.95, 1, ENERGY, False),
    ("model cascade", -0.5, 2, WORK, False),
    ("output-length limit", 0.6, 2, WORK, False),
    ("admission / drop", 1.6, 2, WORK, False),
]


def chip(ax, x, y, text, color, bold=False):
    w = 0.045 + 0.0125 * len(text)
    ax.add_patch(FancyBboxPatch((x - w/2, y - 0.15), w, 0.30, boxstyle="round,pad=0.008,rounding_size=0.04",
                 linewidth=2.4 if bold else 1.0, edgecolor=INK if bold else color,
                 facecolor=color, alpha=0.95 if bold else 0.20, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.8, color="white" if bold else INK,
            fontweight="bold" if bold else "normal", zorder=4)


def box(ax, x, y, w, h, text, ec, fc="#ffffff", fs=8.0, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                 linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", zorder=4, wrap=True)


def main():
    fig = plt.figure(figsize=(11.4, 4.5))
    gsL = fig.add_axes([0.04, 0.10, 0.56, 0.82]); gsR = fig.add_axes([0.64, 0.10, 0.33, 0.82])

    # ---- left: taxonomy grid ----
    gsL.set_xlim(-4.7, 2.7); gsL.set_ylim(-0.7, 2.9); gsL.axis("off")
    for i, (label, col) in enumerate(ROWS):
        gsL.axhspan(i - 0.42, i + 0.42, color=col, alpha=0.06)
        gsL.text(-4.6, i, label, ha="left", va="center", fontsize=8.0, color=col, fontweight="bold")
    ticks = [(-3, "ms"), (-1, "100 ms"), (0, "1 s"), (1, "10 s"), (1.78, "1 min")]
    for tx, tl in ticks:
        gsL.axvline(tx, color="#e2e6ee", lw=1, zorder=0)
        gsL.text(tx, -0.62, tl, ha="center", va="center", fontsize=7.6, color=MUT)
    gsL.text(-0.6, 2.78, "actuation time scale  →", ha="center", fontsize=8.2, color=MUT, style="italic")
    for name, lx, row, col, hw in ACT:
        chip(gsL, lx, row, name, col, bold=hw)
    gsL.annotate("hardware-enforced:\nthe only guarantee", xy=(-3.0, 1.2), xytext=(-2.55, 2.35),
                 fontsize=7.2, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    gsL.set_title("A control-point portfolio", fontsize=10, color=INK, loc="left", x=0.12)

    # ---- right: supervisor loop ----
    gsR.set_xlim(0, 1); gsR.set_ylim(0, 1); gsR.axis("off")
    box(gsR, 0.30, 0.86, 0.40, 0.11, "grid allowance C(t)", "#111418")
    box(gsR, 0.16, 0.55, 0.68, 0.20, "ML supervisor\n(response models + constrained optimizer)", SHIFT, "#eef3fa", 8.0)
    box(gsR, 0.06, 0.24, 0.42, 0.16, "software actuators\n(shape demand)", ENERGY, "#fdf1e3", 7.8)
    box(gsR, 0.56, 0.24, 0.38, 0.16, "hardware cap\n(enforces ceiling)", INK, "#f2f2f2", 7.8)
    box(gsR, 0.28, 0.02, 0.44, 0.11, "GPU fleet", "#137a4b", "#e6f3ec", 8.2)
    A = lambda a, b, **k: gsR.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                       lw=1.6, color=k.get("c", MUT), connectionstyle=k.get("cs", "arc3,rad=0"), zorder=2))
    A((0.5, 0.86), (0.5, 0.755))
    A((0.42, 0.55), (0.3, 0.405), c=ENERGY)
    A((0.62, 0.55), (0.72, 0.405), c=INK)
    A((0.27, 0.24), (0.42, 0.13), c=ENERGY)
    A((0.75, 0.24), (0.6, 0.13), c=INK)
    A((0.72, 0.075), (0.86, 0.55), c=WORK, cs="arc3,rad=-0.35")     # feedback
    gsR.text(0.99, 0.33, "clip frequency\n(feedback)", ha="right", fontsize=7.0, color=WORK, style="italic")
    gsR.text(0.5, 0.475, "allocate headroom", ha="center", fontsize=6.9, color=MUT, style="italic")
    gsR.set_title("Software shapes, hardware guarantees", fontsize=10, color=INK, loc="left")

    fig.savefig(os.path.join(os.path.dirname(__file__), "figures", "fig_portfolio.png"),
                dpi=140, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_portfolio.png")


if __name__ == "__main__":
    main()
