# -*- coding: utf-8 -*-
"""
Kill caveat #1 (homogeneous workloads) with REAL heterogeneous data.

The Zeus NSDI'23 traces measure GPU power-limit sweeps (100..250 W) for six genuinely
heterogeneous training workloads on a V100: vision (ResNet50/ImageNet, ShuffleNetV2/
CIFAR100), speech (DeepSpeech2/LibriSpeech), recommendation (NCF/MovieLens -- memory-bound),
NLP (BERT on SQuAD and Sentiment140). We build a cluster pool from these REAL curves and
re-run the PoC-A kill-test decomposition: unlike the near-homogeneous emerald LLM data, the
per-workload elasticity value here is large, confirming that the value of per-workload
elasticity prediction scales with real fleet heterogeneity (no surrogate needed).
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import poca_killtest as K

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
RESULTS = os.path.join(HERE, "results")

# coarse priority: recommendation/serving-like high, vision mid, NLP/speech training lower
NET_WEIGHT = {"ncf": 4.0, "shufflenetv2": 2.0, "resnet50": 2.0,
              "deepspeech2": 1.0, "bert_base_uncased": 1.0}


def build_zeus_pool(replicate=5, equal_weight=False):
    d = pd.read_csv(os.path.join(RAW, "zeus_summary_power_v100.csv"))
    pool = []
    for (net, ds), g in d.groupby(["network", "dataset"]):
        # representative config: largest batch, first optimizer; aggregate dup power_limits
        b = g["batch_size"].max()
        g = g[g["batch_size"] == b]
        g = g.groupby("power_limit", as_index=False).agg(
            t=("time_per_epoch", "mean"), p=("average_power", "mean"))
        g = g.sort_values("p")
        thr = 1.0 / g["t"].to_numpy(float)
        thr = np.maximum.accumulate(thr)          # enforce monotone (more power -> >= throughput)
        thr = thr / thr.max()
        p = g["p"].to_numpy(float)
        w = 1.0 if equal_weight else NET_WEIGHT.get(net, 1.0)
        for k in range(replicate):
            xg = np.linspace(p.min(), p.max(), K.NGRID)
            qg = np.interp(xg, p, thr)
            pool.append(dict(name=f"{net}/{ds}", cls=net, w=w, k=k, xg=xg, qg=qg,
                             cost_g=w * (1 - qg), pmin=float(xg[0]), pmax=float(xg[-1])))
    return pool


def het_spread(pool):
    qs = [K.q_at(wl, wl["pmin"] + 0.5 * (wl["pmax"] - wl["pmin"])) for wl in pool]
    return float(max(qs) - min(qs))


def run(pool, fracs=(0.90, 0.80, 0.70)):
    tb = sum(wl["pmax"] for wl in pool)
    rows = []
    for frac in fracs:
        C = frac * tb
        uni = K.pool_cost_power(pool, K.b1_uniform(pool, C))[0]
        ora = K.pool_cost_power(pool, K.b6_oracle(pool, C))[0]
        shared = K.pool_cost_power(pool, K.b5g_shared_elasticity(pool, C))[0]
        denom = uni - ora
        rows.append(dict(curt=round((1 - frac) * 100),
                         oracle_gain_vs_uniform_pct=100 * (uni - ora) / uni if uni > 1e-9 else 0.0,
                         shared_capture_pct=100 * (uni - shared) / denom if denom > 1e-6 else float("nan"),
                         per_workload_elasticity_gain_pct=100 * (shared - ora) / shared if shared > 1e-9 else 0.0))
    return rows


def main():
    out = {}
    for label, eq in [("zeus_weighted", False), ("zeus_equalweight", True)]:
        pool = build_zeus_pool(equal_weight=eq)
        out[label] = {"heterogeneity_spread": het_spread(pool), "n_pool": len(pool),
                      "rows": run(pool)}
    # comparison anchor: emerald numbers from the existing ablation
    emerald = json.load(open(os.path.join(RESULTS, "poca_ablations.json")))
    out["emerald_equalweight_spread"] = emerald["real_equalweight"]["heterogeneity_spread"]
    with open(os.path.join(RESULTS, "zeus_killtest.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("== Kill test on REAL heterogeneous Zeus workloads (vision/reco/speech/NLP) ==")
    print(f"emerald LLM-only spread {out['emerald_equalweight_spread']:.3f}  vs  "
          f"Zeus spread {out['zeus_weighted']['heterogeneity_spread']:.3f}")
    for label in ("zeus_weighted", "zeus_equalweight"):
        print(f"\n{label} (spread {out[label]['heterogeneity_spread']:.3f}):")
        print(f"  {'curt%':>5} {'orac↑vsUni%':>11} {'shared_cap%':>11} {'perWL_elast%':>12}")
        for r in out[label]["rows"]:
            print(f"  {r['curt']:>5} {r['oracle_gain_vs_uniform_pct']:>11.1f} "
                  f"{r['shared_capture_pct']:>11.1f} {r['per_workload_elasticity_gain_pct']:>12.1f}")
    print("\nwritten -> results/zeus_killtest.json")


if __name__ == "__main__":
    main()
