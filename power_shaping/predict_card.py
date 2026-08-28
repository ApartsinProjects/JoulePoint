# -*- coding: utf-8 -*-
"""Symmetric to the job-feature question: which curve parameters scale with CARD (GPU) features? We have four
GPUs, so this is a scaling observation across n=4 cards, not a fitted predictor. Per card we take the median
P0, the median P0/TDP, the median beta, and the geometric-mean rate ceiling Rmax over the 20 workloads, and
report their Pearson correlation with card specs (TDP, memory bandwidth, FP16 TFLOPS, SM count, roofline
ridge point). Frames what a cross-card predictor could exploit with more GPUs.
"""
import os, json
import numpy as np
import pandas as pd
from predict_power import RAW

# card specs: (mem_bw GB/s, fp16_tflops, sm, tdp_w). A100-SXM4-40GB added.
SPEC = {"T4": (320, 65.0, 40, 70), "L4": (300, 121.0, 58, 72), "A10G": (600, 70.0, 80, 300),
        "A100": (1555, 312.0, 108, 400)}


def _card(df):
    c = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    return c.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()


def main():
    # Current 3-rep batch data, control mode (batch 32); was the older fine-grid bs32 spectrum sweeps.
    fit = pd.read_csv(os.path.join(RAW, "energy_model_fit.csv")); fit = fit[fit["mode"] == "control"]
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")), pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))]
    sw = pd.concat(fr, ignore_index=True); sw["card"] = _card(sw)
    sw = sw[sw["mode"] == "control"].groupby(["card", "workload", "cap_w"], as_index=False).agg(throughput=("throughput", "mean"))
    rmax = sw.groupby(["card", "workload"]).apply(lambda g: g.sort_values("cap_w").throughput.values[-1])
    rows = []
    for c in ["L4", "A10G", "A100"]:  # T4 excluded from the analysis (see paper limitations)
        f = fit[fit.card == c]
        rows.append(dict(card=c, P0=f.P0.median(), P0frac=f.P0.median() / SPEC[c][3], beta=f.beta.median(),
                         logRmax=np.log10(rmax[c].values).mean(),
                         mem_bw=SPEC[c][0], fp16=SPEC[c][1], sm=SPEC[c][2], tdp=SPEC[c][3],
                         ridge=SPEC[c][1] * 1000.0 / SPEC[c][0]))
    t = pd.DataFrame(rows)
    print("== per-card curve parameters and specs (n=4) ==")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n== scaling: Pearson r across the 4 cards (descriptive; n=4) ==")
    print(f"  {'spec':8} {'P0':>7} {'P0/TDP':>8} {'beta':>7} {'logRmax':>8}")
    out = {}
    for s in ["tdp", "mem_bw", "fp16", "sm", "ridge"]:
        r = {k: float(np.corrcoef(t[s], t[k])[0, 1]) for k in ["P0", "P0frac", "beta", "logRmax"]}
        out[s] = r
        print(f"  {s:8} {r['P0']:>+7.2f} {r['P0frac']:>+8.2f} {r['beta']:>+7.2f} {r['logRmax']:>+8.2f}")
    json.dump({"per_card": t.to_dict(orient="records"), "corr": out}, open("results/predict_card.json", "w"), indent=1)
    print("\nsaved -> results/predict_card.json")


if __name__ == "__main__":
    main()
