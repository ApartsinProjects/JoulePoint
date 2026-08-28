# -*- coding: utf-8 -*-
"""
Rescue the collected AWS data (A10G + L4 + T4, power-cap sweeps of 26 kernels) by describing each kernel
with DERIVED PHYSICS ATTRIBUTES instead of an opaque name. The earlier run failed (~90% error) because a
held-out kernel's one-hot name carries no information; but these kernels have clear static attributes:
operation class, precision, compute-vs-memory character, and arithmetic intensity (roofline position),
which is exactly what determines how a job responds to a power cap. With those features, leave-one-kernel-out
becomes a real feature-based prediction, and the THREE cards give a strong cross-card transfer test.

We compare NAME-ONLY features vs PHYSICS features, and report leave-one-KERNEL-out and leave-one-CARD-out.
Target: duration = 1/throughput, energy = power/throughput (per unit work).
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
SPECS = {"A10G": (600.0, 70.0, 24.0, 80.0, 300.0), "L4": (300.0, 121.0, 24.0, 58.0, 72.0), "T4": (320.0, 65.0, 16.0, 40.0, 70.0)}
SPEC_COLS = ["mem_bw_gbs", "fp16_tflops", "mem_gb", "sm", "tdp_w"]

# kernel -> (operation class, precision, bound, arithmetic intensity proxy [FLOP/byte])
KERNEL = {
    "gemm_fp16": ("matmul", "fp16", "compute", 40), "gemm_fp32": ("matmul", "fp32", "compute", 40),
    "gemm_bf16": ("matmul", "bf16", "compute", 40), "bmm_fp16": ("matmul", "fp16", "compute", 40),
    "attention_sdpa": ("attention", "fp16", "compute", 15), "vit_b16": ("attention", "fp16", "compute", 15),
    "decode_like": ("attention", "fp16", "memory", 3),
    "resnet50": ("conv", "fp16", "compute", 25), "resnet152": ("conv", "fp16", "compute", 25),
    "vgg16": ("conv", "fp16", "compute", 30), "densenet121": ("conv", "fp16", "compute", 20),
    "inception_v3": ("conv", "fp16", "compute", 22), "convnext_tiny": ("conv", "fp16", "compute", 12),
    "efficientnet_b0": ("conv", "fp16", "compute", 10), "mobilenet_v3_l": ("conv", "fp16", "compute", 8),
    "fft2d": ("linalg", "fp32", "compute", 8), "cholesky": ("linalg", "fp32", "compute", 8),
    "reduction": ("reduction", "fp16", "memory", 1.5), "softmax_big": ("reduction", "fp16", "memory", 1.5),
    "layernorm_big": ("reduction", "fp16", "memory", 1.5), "sort_big": ("datamove", "fp16", "memory", 0.8),
    "embed_gather": ("datamove", "fp16", "memory", 0.5), "scatter_add": ("datamove", "fp16", "memory", 0.5),
    "memcpy": ("datamove", "fp16", "memory", 0.3), "elementwise_chain": ("elementwise", "fp16", "memory", 1.0),
    "membw": ("datamove", "fp16", "memory", 0.3),
}


def load():
    fr = []
    o = pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")); o["gpu"] = "NVIDIA A10G"; fr.append(o)
    fr.append(pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv")))
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    df = df.groupby(["card", "workload", "cap_frac"], as_index=False).agg(power=("power_w", "mean"), thr=("throughput", "mean"))
    df = df[(df.thr > 0) & (df.power > 0)].reset_index(drop=True)
    df["duration"] = 1.0 / df.thr; df["energy"] = df.power / df.thr
    # NORMALIZED response: throughput relative to the best (uncapped) point, per (workload, card).
    # This is the comparable, unit-free elasticity curve; absolute throughput units differ per kernel.
    df["norm_thr"] = df.thr / df.groupby(["workload", "card"])["thr"].transform("max")
    df["op_class"] = df.workload.map(lambda w: KERNEL[w][0]); df["precision"] = df.workload.map(lambda w: KERNEL[w][1])
    df["bound"] = df.workload.map(lambda w: KERNEL[w][2]); df["arith_intensity"] = df.workload.map(lambda w: KERNEL[w][3])
    for i, c in enumerate(SPEC_COLS):
        df[c] = df.card.map(lambda k: SPECS[k][i])
    return df


def gbr(): return HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06, min_samples_leaf=4)
def mdape(t, p): return float(np.median(np.abs((p - t) / t)) * 100)
def log_r2(t, p):
    lt, lp = np.log(t), np.log(np.clip(p, 1e-9, None)); ss = np.sum((lt-lp)**2); tot = np.sum((lt-lt.mean())**2)
    return float(1 - ss/tot) if tot > 0 else 0.0


def design(df, cat, num):
    X = pd.get_dummies(df[cat].astype(str))
    for c in num: X[c] = df[c].values
    return X


def cv(df, X, group, target):
    Xv = X.values.astype(float); y = np.log(df[target].values); p = np.zeros(len(df))
    for g in df[group].unique():
        te = (df[group] == g).values; tr = ~te
        p[te] = gbr().fit(Xv[tr], y[tr]).predict(Xv[te]) if tr.sum() >= 8 else y[tr].mean()
    return np.exp(p)


def slowdown_mae(df, dpred):
    d = df.assign(dp=dpred); ts, ps = [], []
    for _, g in d.groupby(["workload", "card"]):
        g = g.sort_values("cap_frac"); ts.extend(g.duration.values / g.duration.values[-1] - 1); ps.extend(g.dp.values / g.dp.values[-1] - 1)
    ts, ps = np.array(ts), np.array(ps); return round(float(np.mean(np.abs(ps - ts))), 3)


def main():
    df = load()
    PHYS_CAT = ["op_class", "precision", "bound", "card"]; PHYS_NUM = ["arith_intensity", "cap_frac"] + SPEC_COLS
    NAME_CAT = ["workload", "card"]; NAME_NUM = ["cap_frac"] + SPEC_COLS

    Xp = design(df, PHYS_CAT, PHYS_NUM); Xn = design(df, NAME_CAT, NAME_NUM)
    Dp = cv(df, Xp, "workload", "duration"); Ep = cv(df, Xp, "workload", "energy")
    Dn = cv(df, Xn, "workload", "duration")                       # name-only, for contrast
    Dpc = cv(df, Xp, "card", "duration"); Epc = cv(df, Xp, "card", "energy")
    # the MEANINGFUL target for microbenchmarks: the normalized cap-response (elasticity curve)
    NRp = cv(df, Xp, "workload", "norm_thr")
    sd_true = 1 - df.norm_thr.values; sd_pred = 1 - np.clip(NRp, 1e-3, 1.0); m = sd_true > 0.05
    norm_response = {"norm_thr_MdAPE": round(mdape(df.norm_thr.values, NRp), 1),
                     "slowdown_MAE": round(float(np.mean(np.abs(sd_pred - sd_true))), 3),
                     "slowdown_MAE_where_gt5pct": round(float(np.mean(np.abs(sd_pred[m] - sd_true[m]))), 3) if m.any() else None,
                     "card_out_norm_MdAPE": round(mdape(df.norm_thr.values, cv(df, Xp, "card", "norm_thr")), 1)}

    out = {"n_rows": len(df), "n_kernels": int(df.workload.nunique()), "n_cards": int(df.card.nunique()),
           "leave_one_kernel_out": {
               "physics_duration_MdAPE": round(mdape(df.duration.values, Dp), 1), "physics_duration_logR2": round(log_r2(df.duration.values, Dp), 3),
               "physics_energy_MdAPE": round(mdape(df.energy.values, Ep), 1),
               "physics_slowdown_MAE": slowdown_mae(df, Dp),
               "nameonly_duration_MdAPE": round(mdape(df.duration.values, Dn), 1)},
           "leave_one_card_out": {"physics_duration_MdAPE": round(mdape(df.duration.values, Dpc), 1),
                                  "physics_energy_MdAPE": round(mdape(df.energy.values, Epc), 1)},
           "normalized_cap_response": norm_response}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "predict_aws_kernels.json"), "w"), indent=2)

    k = out["leave_one_kernel_out"]; c = out["leave_one_card_out"]
    print("== AWS collected data (A10G + L4 + T4, 26 kernels): predict with PHYSICS attributes ==")
    print(f"rows {out['n_rows']}  kernels {out['n_kernels']}  cards {out['n_cards']}")
    print("\nleave-one-KERNEL-out (predict an unseen kernel from its attributes):")
    print(f"  duration : PHYSICS MdAPE {k['physics_duration_MdAPE']}%  logR2 {k['physics_duration_logR2']}   "
          f"vs NAME-ONLY {k['nameonly_duration_MdAPE']}%")
    print(f"  energy   : MdAPE {k['physics_energy_MdAPE']}%")
    print(f"  cap response: slowdown MAE {k['physics_slowdown_MAE']}")
    print("\nleave-one-CARD-out (train two cards, predict the third):")
    print(f"  duration MdAPE {c['physics_duration_MdAPE']}%   energy MdAPE {c['physics_energy_MdAPE']}%")
    nr = out["normalized_cap_response"]
    print("\nNORMALIZED cap-response (the meaningful target: the elasticity curve, unit-free):")
    print(f"  norm-throughput MdAPE {nr['norm_thr_MdAPE']}%   slowdown MAE {nr['slowdown_MAE']} "
          f"(where slowdown>5%: {nr['slowdown_MAE_where_gt5pct']})   card-out {nr['card_out_norm_MdAPE']}%")
    print("written -> results/predict_aws_kernels.json")


if __name__ == "__main__":
    main()
