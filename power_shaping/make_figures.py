# -*- coding: utf-8 -*-
"""Generate the PoC figures (PNG) from the results JSONs, for the HTML report."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
C = {"uniform": "#b0641f", "priority": "#9a3b8c", "largest": "#777", "profiled": "#1f6feb",
     "oracle": "#137a4b", "elasticity": "#1f6feb"}


def load(n): return json.load(open(os.path.join(RES, n)))


def fig_killtest():
    d = load("poca_killtest.json")["static"]
    x = [r["curtailment_pct"] for r in d]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for key, lab, col in [("B1_uniform", "uniform cap", C["uniform"]),
                          ("B2_priority", "priority-first", C["priority"]),
                          ("B3_largest_power", "largest-power", C["largest"]),
                          ("B5_profiled", "profiled elasticity", C["profiled"]),
                          ("B6_oracle", "oracle", C["oracle"])]:
        ax.plot(x, [r[key]["cost"] for r in d], "o-", label=lab, color=col, lw=2, ms=4)
    ax.set_xlabel("requested curtailment (%)"); ax.set_ylabel("weighted service cost")
    ax.set_title("PoC-A kill figure: service cost vs curtailment\n(real emerald DVFS, 40-job pool)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_A2_killtest.png")); plt.close(fig)


def fig_decomp():
    d = load("poca_ablations.json")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    labels = {"real_weighted": "real LLM (weighted)", "real_equalweight": "real LLM (equal weight)",
              "surrogate_hetero": "heterogeneous surrogate"}
    cols = {"real_weighted": "#1f6feb", "real_equalweight": "#137a4b", "surrogate_hetero": "#b42318"}
    for k, lab in labels.items():
        rows = d[k]["rows"]
        x = [r["curt"] for r in rows]
        ax.plot(x, [r["oracle_gain_vs_uniform_pct"] for r in rows], "o-",
                label=f"{lab} (spread {d[k]['heterogeneity_spread']:.2f})", color=cols[k], lw=2, ms=4)
    ax.set_xlabel("curtailment (%)"); ax.set_ylabel("oracle gain over uniform (%)")
    ax.set_title("Workload-aware value scales with heterogeneity")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_decomp.png")); plt.close(fig)


def fig_b_tracking():
    d = load("pocb_azure.json")
    el = d["controllers"]["elasticity"]["ts"]; un = d["controllers"]["uniform"]["ts"]
    t = np.arange(len(el["P"])) / 60.0    # minutes (ts already 1 s downsampled)
    def smooth(a, w=15):
        a = np.array(a, float); k = np.ones(w) / w
        return np.convolve(a, k, mode="same")
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(t, smooth(un["P"]) / 1000, color=C["uniform"], lw=1.3, alpha=.85, label="uniform power")
    ax.plot(t, smooth(el["P"]) / 1000, color=C["elasticity"], lw=1.5, label="elasticity power")
    ax.plot(t, np.array(el["C"]) / 1000, "k--", lw=1.3, label="envelope C(t)")
    ax.set_ylabel("facility power (kW)"); ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center")
    ax.set_title("PoC-B: dynamic envelope tracking (real Azure trace, dip to 42%)")
    ax2 = axes[1]
    ax2.plot(t, el["cap"], color=C["elasticity"], lw=1.2, label="elasticity GPU cap")
    ax2.plot(t, un["cap"], color=C["uniform"], lw=1, alpha=.8, label="uniform GPU cap")
    ax2.set_ylabel("GPU power cap"); ax2.set_xlabel("time (min)"); ax2.set_ylim(0.3, 1.05)
    ax2.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_B_tracking.png")); plt.close(fig)


def fig_b_sweep():
    d = load("pocb_azure.json")["dip_sweep"]
    x = [r["dip_frac"] * 100 for r in d]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    ax = axes[0]
    ax.plot(x, [r["uniform"]["crit_slo"] for r in d], "o-", color=C["uniform"], label="uniform", lw=2)
    ax.plot(x, [r["elasticity"]["crit_slo"] for r in d], "s-", color=C["elasticity"], label="elasticity", lw=2)
    ax.set_xlabel("firm cap (% of peak)"); ax.set_ylabel("critical-class SLO violation (%)")
    ax.set_title("Critical SLO vs curtailment depth"); ax.legend(fontsize=8, frameon=False)
    ax.invert_xaxis()
    ax = axes[1]
    ax.plot(x, [max(1, r["uniform"]["wcost"]) for r in d], "o-", color=C["uniform"], label="uniform", lw=2)
    ax.plot(x, [max(1, r["elasticity"]["wcost"]) for r in d], "s-", color=C["elasticity"], label="elasticity", lw=2)
    ax.set_yscale("log"); ax.set_xlabel("firm cap (% of peak)"); ax.set_ylabel("weighted service cost (log)")
    ax.set_title("Service cost vs curtailment depth"); ax.legend(fontsize=8, frameon=False); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_B_sweep.png")); plt.close(fig)


def fig_model():
    d = load("learned_model.json")
    models = ["M0_mean", "M1_classlut", "M2_linear", "M3_gbdt"]
    labs = ["M0 mean", "M1 class-lookup", "M2 linear", "M3 GBDT"]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(x - w / 2, [d["S1_random"][m] for m in models], w, label="S1 random", color="#1f6feb")
    ax.bar(x + w / 2, [d["S2_unseen_workload"][m] for m in models], w, label="S2 unseen workload", color="#137a4b")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=15); ax.set_ylabel("MAE (normalized throughput)")
    ax.set_title("Learned response model: unseen-workload generalization")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_model.png")); plt.close(fig)


def fig_ccm():
    d = load("rebound_israel.json")["ccm"]
    mixes = list(d.keys()); x = np.arange(len(mixes)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(x - w / 2, [d[m]["uniform"]["ccm"] for m in mixes], w, label="uniform cap", color=C["uniform"])
    ax.bar(x + w / 2, [d[m]["elasticity"]["ccm"] for m in mixes], w, label="elasticity shaping", color=C["elasticity"])
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_", "\n") for m in mixes]); ax.set_ylabel("CCM = installed / firm MW")
    ax.set_title("Compute Capacity Multiplier by workload mix")
    for i, m in enumerate(mixes):
        ax.text(i + w / 2, d[m]["elasticity"]["ccm"] + 0.05, f"{d[m]['elasticity']['ccm']:.1f}x",
                ha="center", fontsize=8, color=C["elasticity"])
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ccm.png")); plt.close(fig)


def fig_validate():
    import pandas as pd
    d = load("validate_real.json")
    pw = pd.read_csv(os.path.join(HERE, "data", "raw", "SRP_total_power.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    ax = axes[0]
    ax.plot(np.arange(len(pw)), pw["total"] / 1000, color="#1f6feb", lw=1.4)
    ax.axhline(d["base_kw"], color="#777", ls=":", lw=1)
    ax.set_xlabel("minutes"); ax.set_ylabel("measured cluster power (kW)")
    ax.set_title(f"Real SRP grid experiment\n{d['base_kw']:.0f}kW to {d['sustained_kw']:.0f}kW ({d['cluster_reduction_pct']:.0f}% cut)")
    ax = axes[1]
    rows = [r for r in d["rows"] if r["model_perf"] == r["model_perf"]]
    x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w/2, [r["real_perf"] for r in rows], w, label="real (field)", color="#137a4b")
    ax.bar(x + w/2, [r["model_perf"] for r in rows], w, label="model (predicted)", color="#1f6feb")
    ax.set_xticks(x); ax.set_xticklabels([r["workload"].split("_")[0] + "\n" + r["workload"].split("_")[-1] for r in rows], fontsize=7)
    ax.set_ylabel("service performance"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Model vs field per-workload (MAE {d['per_workload_MAE']:.3f})"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_validate.png")); plt.close(fig)


def fig_zeus():
    import pandas as pd
    df = pd.read_csv(os.path.join(HERE, "data", "raw", "zeus_summary_power_v100.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    ax = axes[0]
    palette = {"ncf": "#b42318", "shufflenetv2": "#137a4b", "resnet50": "#1f6feb",
               "bert_base_uncased": "#9a3b8c", "deepspeech2": "#b0641f"}
    for net, g in df.groupby("network"):
        b = g["batch_size"].max(); g = g[g["batch_size"] == b]
        g = g.groupby("power_limit", as_index=False).agg(p=("average_power", "mean"), t=("time_per_epoch", "mean"))
        g = g.sort_values("p"); thr = 1 / g["t"]; thr = np.maximum.accumulate(thr.values); thr /= thr.max()
        ax.plot(g["p"], thr, "o-", color=palette.get(net, "#777"), lw=1.8, ms=3,
                label=net.replace("_base_uncased", ""))
    ax.set_xlabel("measured GPU power (W)"); ax.set_ylabel("normalized throughput")
    ax.set_title("Real heterogeneous elasticity (Zeus)"); ax.legend(fontsize=7.5, frameon=False)
    ax = axes[1]
    rng = df.groupby("network").agg(pmin=("average_power", "min"), pmax=("average_power", "max"))
    nets = rng.index.tolist(); y = np.arange(len(nets))
    ax.barh(y, rng["pmax"] - rng["pmin"], left=rng["pmin"], color=[palette.get(n, "#777") for n in nets])
    ax.set_yticks(y); ax.set_yticklabels([n.replace("_base_uncased", "") for n in nets], fontsize=8)
    ax.set_xlabel("power draw range across the cap sweep (W)")
    ax.set_title("Memory-bound NCF draws ~37 W\nregardless of the cap")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_zeus.png")); plt.close(fig)


if __name__ == "__main__":
    fig_killtest(); fig_decomp(); fig_b_tracking(); fig_b_sweep(); fig_model(); fig_ccm()
    fig_validate(); fig_zeus()
    print("figures ->", sorted(os.listdir(FIG)))
