# -*- coding: utf-8 -*-
"""
Dataflow diagram of the power-shaping response MODEL: what feeds it, what it predicts,
how the controller uses it, and what signals it emits. Rendered with matplotlib so every
box/arrow is explicit and traceable. Saved to figures/fig_model_dataflow.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(__file__), "figures")

INK = "#1a1d24"; MUT = "#5b6472"
C_MEAS = "#137a4b"; C_MEAS_S = "#e2f5ea"
C_MODEL = "#0b4fc4"; C_MODEL_S = "#e7eefb"
C_CTRL = "#7a3fb0"; C_CTRL_S = "#f0e8f8"
C_SIG = "#b0641f"; C_SIG_S = "#fbf0e2"
C_OUT = "#0f6f66"; C_OUT_S = "#e0f2f0"


def box(ax, x, y, w, h, title, lines, ec, fc, tsize=10.5, lsize=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.014",
                                linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 0.033, title, ha="center", va="top", fontsize=tsize,
            fontweight="bold", color=ec, zorder=3)
    ax.text(x + 0.014, y + h - 0.086, "\n".join(lines), ha="left", va="top", fontsize=lsize,
            color=INK, zorder=3, linespacing=1.35)


def arrow(ax, xy0, xy1, color=MUT, style="-|>", lw=1.9, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=15, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 linestyle=ls, zorder=1, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(12.4, 6.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- column 1: inputs ----
    box(ax, 0.012, 0.560, 0.210, 0.300, "1a. Measured elasticity",
        ["OFFLINE, once:", "• emerald DVFS  (8 workloads", "   × 6 power caps = 48 rows)",
         "• Zeus sweeps  (~14 workloads)", "• own AWS A10G + Modal", "power-cap / batching sweeps",
         "→ trains the response model"], C_MEAS, C_MEAS_S)
    box(ax, 0.012, 0.140, 0.210, 0.360, "1b. Live signals / tick (0.5 s)",
        ["ONLINE, every tick:", "• GPU power draw (NVML)", "• per-class queue lengths",
         "• arrival rate  (EMA demand)", "• deferred backlog", "• grid allowance  C(t)",
         "  (evening curtailment)"], C_MEAS, C_MEAS_S)

    # ---- column 2: model ----
    box(ax, 0.268, 0.410, 0.226, 0.320, "2. Response model",
        ["predicts, for any ACTION:", "  f(cap, gates) →", "     (facility power,",
         "      serving throughput)", "", "SIMPLE by design:", "class-mean / physical bound",
         "beats GBDT on unseen", "(only ~14 workloads →", "a deep model overfits)"],
        C_MODEL, C_MODEL_S)

    # ---- column 3: controller ----
    box(ax, 0.540, 0.360, 0.232, 0.420, "3. Controller",
        ["feedforward + feedback:", "• for each shed-ladder action,", "   PREDICT power via the model",
         "• pick the LEAST-shedding action", "   whose prediction ≤ C(t)", "• continuous cap-trim (no",
         "   discrete over-shoot)", "• fractional multi-class admission", "   regulated on MEASURED power",
         "   (most-valuable class first)"], C_CTRL, C_CTRL_S)

    # ---- column 4: signals produced ----
    box(ax, 0.800, 0.560, 0.192, 0.300, "4. Signals produced",
        ["• power-cap fraction ∈ [0,1]", "   → GPUs (nvidia-smi -pl)", "• per-class gates ∈ [0,1]",
         "   → admit / defer", "• predicted power", "   (feedforward estimate)"], C_SIG, C_SIG_S)
    box(ax, 0.800, 0.140, 0.192, 0.360, "5. Outcomes (observed)",
        ["• facility power tracks C(t)", "• critical / interactive", "   SLO protected (0%)",
         "• offline deferred, then", "   drained", "• bounded post-event", "   rebound"],
        C_OUT, C_OUT_S)

    # ---- arrows ----
    arrow(ax, (0.222, 0.700), (0.268, 0.610), C_MEAS)                       # 1a -> model (train)
    ax.text(0.245, 0.660, "train", fontsize=7.6, color=C_MEAS, style="italic", ha="center")
    arrow(ax, (0.494, 0.560), (0.540, 0.520), C_MODEL)                      # model -> controller (used)
    ax.text(0.518, 0.560, "used", fontsize=7.6, color=C_MODEL, style="italic", ha="center")
    arrow(ax, (0.222, 0.320), (0.540, 0.470), C_MEAS, rad=-0.06)            # live signals -> controller
    ax.text(0.36, 0.352, "live signals", fontsize=7.6, color=C_MEAS, style="italic", ha="center")
    arrow(ax, (0.772, 0.610), (0.800, 0.660), C_CTRL)                       # controller -> signals
    ax.text(0.788, 0.648, "emit", fontsize=7.6, color=C_CTRL, style="italic", ha="center")
    arrow(ax, (0.896, 0.560), (0.896, 0.500), C_SIG)                        # signals -> outcomes (apply)
    ax.text(0.930, 0.530, "apply", fontsize=7.6, color=C_SIG, style="italic", ha="left")
    # feedback loop: measured power (outcomes) -> back to controller
    arrow(ax, (0.800, 0.300), (0.656, 0.360), C_OUT, rad=0.18, ls=(0, (5, 3)))
    ax.text(0.726, 0.284, "measured power feedback", fontsize=7.6, color=C_OUT,
            style="italic", ha="center")

    ax.text(0.5, 0.055,
            "The MODEL is the feedforward predictor the controller queries each tick; the loop is closed on "
            "the MEASURED power signal.\nSimulated-on-measured everywhere except the separate A10G run, where "
            "power / latency / throughput are read from real hardware.",
            ha="center", va="center", fontsize=8.2, color=MUT, linespacing=1.5)
    ax.text(0.5, 0.965, "Power-shaping response model: inputs → prediction → control → signals",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK)

    fig.tight_layout(pad=0.4)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig_model_dataflow.png"), dpi=140, facecolor="white",
                bbox_inches="tight"); plt.close(fig)
    print("figure -> figures/fig_model_dataflow.png")


if __name__ == "__main__":
    main()
