# -*- coding: utf-8 -*-
"""
Predict the NORMALIZED power-cap response (the elasticity curve) from STATIC features, on the two datasets
with rich enough descriptions: Zeus (V100/A40, rich job attributes) and our AWS sweeps (A10G/L4/T4, kernels
described by physics attributes). The normalized response = work-rate at a cap divided by the uncapped
work-rate, per (job, card); slowdown = 1 - that ratio. This is the unit-free, comparable quantity the
controller needs (absolute duration units differ across jobs).

Richer HARDWARE SPECS per card, including the derived ROOFLINE RIDGE POINT (compute / bandwidth = FLOP per
byte): paired with a job's arithmetic intensity it says whether that job on that card is compute- or
memory-bound, which is what sets cap-sensitivity. We report leave-one-JOB-out and leave-one-CARD-out.
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")

# card -> mem_bw(GB/s), fp16_tflops, fp32_tflops, mem_gb, sm, tensor_cores, boost_mhz, l2_mb, bus_bit, tdp_w
RICH = {
    "V100": (900, 112.0, 15.7, 16, 80, 640, 1530, 6, 4096, 300),
    "A40":  (696, 149.7, 37.4, 48, 84, 336, 1740, 6, 384, 300),
    "A10G": (600, 70.0, 31.2, 24, 80, 320, 1710, 6, 384, 300),
    "L4":   (300, 121.0, 30.3, 24, 58, 232, 2040, 48, 192, 72),
    "T4":   (320, 65.0, 8.1, 16, 40, 320, 1590, 4, 256, 70),
}
SPEC_COLS = ["mem_bw", "fp16_tflops", "fp32_tflops", "mem_gb", "sm", "tensor_cores", "boost_mhz", "l2_mb", "bus_bit", "tdp_w", "ridge"]
# job arithmetic intensity (FLOP/byte proxy) for the AWS kernels; compute-bound high, memory-bound low
KAI = {"gemm_fp16": 300, "gemm_fp32": 300, "gemm_bf16": 300, "bmm_fp16": 300, "attention_sdpa": 80, "vit_b16": 80,
       "decode_like": 8, "resnet50": 120, "resnet152": 120, "vgg16": 200, "densenet121": 90, "inception_v3": 100,
       "convnext_tiny": 60, "efficientnet_b0": 40, "mobilenet_v3_l": 30, "fft2d": 20, "cholesky": 30,
       "reduction": 2, "softmax_big": 2, "layernorm_big": 2, "sort_big": 1, "embed_gather": 0.6, "scatter_add": 0.6,
       "memcpy": 0.3, "elementwise_chain": 1.2, "membw": 0.3}
KOP = {**{k: "matmul" for k in ["gemm_fp16", "gemm_fp32", "gemm_bf16", "bmm_fp16"]},
       **{k: "attention" for k in ["attention_sdpa", "vit_b16", "decode_like"]},
       **{k: "conv" for k in ["resnet50", "resnet152", "vgg16", "densenet121", "inception_v3", "convnext_tiny", "efficientnet_b0", "mobilenet_v3_l"]},
       **{k: "linalg" for k in ["fft2d", "cholesky"]},
       **{k: "reduction" for k in ["reduction", "softmax_big", "layernorm_big"]},
       **{k: "datamove" for k in ["sort_big", "embed_gather", "scatter_add", "memcpy", "membw", "elementwise_chain"]}}


def add_specs(df):
    for i, c in enumerate(SPEC_COLS[:-1]):
        df[c] = df.card.map(lambda k: RICH[k][i])
    df["ridge"] = df["fp16_tflops"] * 1000.0 / df["mem_bw"]      # FLOP per byte (machine balance)
    return df


def gbr(): return HistGradientBoostingRegressor(max_depth=4, max_iter=350, learning_rate=0.05, min_samples_leaf=4)
def mdape(t, p): return float(np.median(np.abs((p - t) / t)) * 100)


def cv(df, X, group, target):
    Xv = X.values.astype(float); y = df[target].values; p = np.zeros(len(df))
    for g in df[group].unique():
        te = (df[group] == g).values; tr = ~te
        p[te] = gbr().fit(Xv[tr], y[tr]).predict(Xv[te]) if tr.sum() >= 8 else y[tr].mean()
    return np.clip(p, 1e-3, 1.2)


def design(df, cat, num):
    X = pd.get_dummies(df[cat].astype(str))
    for c in num: X[c] = df[c].values
    return X


def evaluate(name, df, cat, num):
    df = df.reset_index(drop=True)
    X = design(df, cat, num)
    nj = cv(df, X, "job", "norm_rate")
    sd_t = 1 - df.norm_rate.values; sd_p = 1 - nj; m = sd_t > 0.05
    r = {"n_rows": len(df), "n_jobs": int(df.job.nunique()), "n_cards": int(df.card.nunique()),
         "job_norm_MdAPE": round(mdape(df.norm_rate.values, nj), 1),
         "job_slowdown_MAE": round(float(np.mean(np.abs(sd_p - sd_t))), 3),
         "job_slowdown_MAE_gt5pct": round(float(np.mean(np.abs(sd_p[m] - sd_t[m]))), 3) if m.any() else None}
    if df.card.nunique() > 1:
        nc = cv(df, X, "card", "norm_rate")
        r["card_norm_MdAPE"] = round(mdape(df.norm_rate.values, nc), 1)
        r["card_slowdown_MAE"] = round(float(np.mean(np.abs((1 - nc) - sd_t))), 3)
    return r


def load_zeus():
    fr = []
    for card, f in [("V100", "zeus_summary_power_v100.csv"), ("A40", "zeus_summary_power_a40.csv")]:
        d = pd.read_csv(os.path.join(RAW, f)); d["card"] = card; fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df = df.groupby(["card", "dataset", "network", "batch_size", "optimizer", "power_limit"], as_index=False).agg(
        t=("time_per_epoch", "mean"))
    df = df[df.t > 0].copy(); df["rate"] = 1.0 / df.t
    df["cap_frac"] = df.power_limit / df.groupby("card")["power_limit"].transform("max")
    df["log_batch"] = np.log(df.batch_size.clip(lower=1))
    df["job"] = df.groupby(["network", "dataset", "batch_size", "optimizer"]).ngroup()
    df["norm_rate"] = df.rate / df.groupby(["job", "card"])["rate"].transform("max")
    df = add_specs(df)
    return df, ["network", "dataset", "optimizer", "card"], ["log_batch", "cap_frac"] + SPEC_COLS


def load_aws():
    fr = [pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")).assign(gpu="NVIDIA A10G"),
          pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    df = df.groupby(["card", "workload", "cap_frac"], as_index=False).agg(thr=("throughput", "mean"))
    df = df[df.thr > 0].copy(); df["rate"] = df.thr
    df["op_class"] = df.workload.map(KOP); df["arith_intensity"] = df.workload.map(KAI)
    df["job"] = df.groupby(["workload"]).ngroup()
    df["norm_rate"] = df.rate / df.groupby(["job", "card"])["rate"].transform("max")
    df = add_specs(df)
    df["ai_over_ridge"] = df["arith_intensity"] / df["ridge"]      # >1 compute-bound, <1 memory-bound, on this card
    return df, ["op_class", "card"], ["arith_intensity", "ai_over_ridge", "cap_frac"] + SPEC_COLS


def main():
    out = {}
    for name, ld in [("zeus", load_zeus), ("aws", load_aws)]:
        df, cat, num = ld()
        out[name] = evaluate(name, df, cat, num)
    json.dump(out, open(os.path.join(HERE, "results", "predict_response.json"), "w"), indent=2)
    print("== Predict the NORMALIZED cap-response from static features + rich card specs (ridge point) ==")
    hdr = f"{'dataset':>7} {'rows':>5} {'jobs':>5} {'cards':>5} | {'job normMdAPE':>14} {'job slowMAE':>12} {'>5%':>6} | {'card normMdAPE':>15} {'card slowMAE':>12}"
    print(hdr); print("-" * len(hdr))
    for name, r in out.items():
        print(f"{name:>7} {r['n_rows']:>5} {r['n_jobs']:>5} {r['n_cards']:>5} | "
              f"{r['job_norm_MdAPE']:>13.1f}% {r['job_slowdown_MAE']:>12} {str(r['job_slowdown_MAE_gt5pct']):>6} | "
              f"{r.get('card_norm_MdAPE','-'):>14}% {str(r.get('card_slowdown_MAE','-')):>12}")
    print("\nslowMAE = mean abs error in predicted slowdown (fraction); lower is better.")
    print("written -> results/predict_response.json")


if __name__ == "__main__":
    main()
