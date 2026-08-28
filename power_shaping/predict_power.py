# -*- coding: utf-8 -*-
"""
Predict the ENERGY-DRAIN RATE (average power draw, watts) of a job on a card at a given power cap, from
STATIC features. This is the quantity the grid constrains: an assignment fits the target C(t) iff the sum
of predicted draws stays under it. The point of prediction (vs. assuming draw = cap) is that a job's actual
draw is usually BELOW its cap: a memory-bound job leaves headroom, a compute-bound job pins the cap.

Two targets:
  power_w    : absolute average draw (what the grid sums).
  draw_frac  : draw / cap = the unit-free fraction of the allotted cap the job actually uses (<=1); this is
               the comparable "how much headroom" signal and should be predictable from the job's
               compute-vs-memory character (arithmetic intensity vs the card's roofline ridge point).

Datasets with rich descriptions: Zeus (V100/A40) and our AWS sweeps (A10G/L4/T4). Rich card specs incl. the
ridge point. Reported leave-one-JOB-out and leave-one-CARD-out.
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")


def saturating_set(df):
    """The production-relevant LOADED operating set: for each (card, workload) keep the batch mode whose
    uncapped draw is highest, i.e. the most-loaded measured point. On the large GPUs (real power headroom)
    this is the auto-calibrated saturate batch; on the small cards (T4, L4), which already draw near TDP at
    batch 32, control (batch 32) is the loaded point and their lighter 'saturate' calibration is discarded.
    Real inference serving keeps GPUs loaded, so this is the regime to characterize. Requires a `card` column."""
    d = df[df["mode"].isin(["control", "saturate"])]
    unc = (d.sort_values("cap_w").groupby(["card", "workload", "mode"], as_index=False)
           .apply(lambda g: g[g.cap_w == g.cap_w.max()].power_w.mean(), include_groups=False)
           .rename(columns={None: "draw"}))
    unc.columns = ["card", "workload", "mode", "draw"]
    best = unc.sort_values("draw").groupby(["card", "workload"], as_index=False).tail(1)
    keep = set(zip(best.card, best.workload, best["mode"]))
    return d[[(c, w, m) in keep for c, w, m in zip(d.card, d.workload, d["mode"])]].copy()
RICH = {
    "V100": (900, 112.0, 15.7, 16, 80, 640, 1530, 6, 4096, 300), "A40": (696, 149.7, 37.4, 48, 84, 336, 1740, 6, 384, 300),
    "A10G": (600, 70.0, 31.2, 24, 80, 320, 1710, 6, 384, 300), "L4": (300, 121.0, 30.3, 24, 58, 232, 2040, 48, 192, 72),
    "T4": (320, 65.0, 8.1, 16, 40, 320, 1590, 4, 256, 70),
}
SPEC_COLS = ["mem_bw", "fp16_tflops", "fp32_tflops", "mem_gb", "sm", "tensor_cores", "boost_mhz", "l2_mb", "bus_bit", "tdp_w", "ridge"]
KAI = {"gemm_fp16": 300, "gemm_fp32": 300, "gemm_bf16": 300, "bmm_fp16": 300, "attention_sdpa": 80, "vit_b16": 80,
       "decode_like": 8, "resnet50": 120, "resnet152": 120, "vgg16": 200, "densenet121": 90, "inception_v3": 100,
       "convnext_tiny": 60, "efficientnet_b0": 40, "mobilenet_v3_l": 30, "fft2d": 20, "cholesky": 30, "reduction": 2,
       "softmax_big": 2, "layernorm_big": 2, "sort_big": 1, "embed_gather": 0.6, "scatter_add": 0.6, "memcpy": 0.3,
       "elementwise_chain": 1.2, "membw": 0.3}
KOP = {**{k: "matmul" for k in ["gemm_fp16", "gemm_fp32", "gemm_bf16", "bmm_fp16"]},
       **{k: "attention" for k in ["attention_sdpa", "vit_b16", "decode_like"]},
       **{k: "conv" for k in ["resnet50", "resnet152", "vgg16", "densenet121", "inception_v3", "convnext_tiny", "efficientnet_b0", "mobilenet_v3_l"]},
       **{k: "linalg" for k in ["fft2d", "cholesky"]}, **{k: "reduction" for k in ["reduction", "softmax_big", "layernorm_big"]},
       **{k: "datamove" for k in ["sort_big", "embed_gather", "scatter_add", "memcpy", "membw", "elementwise_chain"]}}


def add_specs(df):
    for i, c in enumerate(SPEC_COLS[:-1]):
        df[c] = df.card.map(lambda k: RICH[k][i])
    df["ridge"] = df["fp16_tflops"] * 1000.0 / df["mem_bw"]
    return df


def gbr(): return HistGradientBoostingRegressor(max_depth=4, max_iter=350, learning_rate=0.05, min_samples_leaf=4)
def mdape(t, p): return float(np.median(np.abs((p - t) / t)) * 100)
def r2(t, p):
    ss = np.sum((t - p) ** 2); tot = np.sum((t - t.mean()) ** 2); return float(1 - ss / tot) if tot > 0 else 0.0


def cv(df, X, group, target, logt=False):
    Xv = X.values.astype(float); y = np.log(df[target].values) if logt else df[target].values; p = np.zeros(len(df))
    for g in df[group].unique():
        te = (df[group] == g).values; tr = ~te
        p[te] = gbr().fit(Xv[tr], y[tr]).predict(Xv[te]) if tr.sum() >= 8 else y[tr].mean()
    return np.exp(p) if logt else p


def design(df, cat, num):
    X = pd.get_dummies(df[cat].astype(str))
    for c in num: X[c] = df[c].values
    return X


def evaluate(df, cat, num):
    df = df.reset_index(drop=True); X = design(df, cat, num)
    Pj = cv(df, X, "job", "power_w", logt=True); Fj = cv(df, X, "job", "draw_frac")
    r = {"n_rows": len(df), "n_jobs": int(df.job.nunique()), "n_cards": int(df.card.nunique()),
         "job_power_MdAPE": round(mdape(df.power_w.values, Pj), 1), "job_power_R2": round(r2(df.power_w.values, Pj), 3),
         "job_drawfrac_MdAPE": round(mdape(df.draw_frac.values, Fj), 1),
         "job_drawfrac_MAE": round(float(np.mean(np.abs(Fj - df.draw_frac.values))), 3),
         "assume_draw_eq_cap_MdAPE": round(mdape(df.draw_frac.values, np.ones(len(df))), 1)}  # baseline: assume draw=cap
    if df.card.nunique() > 1:
        Pc = cv(df, X, "card", "power_w", logt=True)
        r["card_power_MdAPE"] = round(mdape(df.power_w.values, Pc), 1)
    return r


def load_zeus():
    fr = []
    for card, f in [("V100", "zeus_summary_power_v100.csv"), ("A40", "zeus_summary_power_a40.csv")]:
        d = pd.read_csv(os.path.join(RAW, f)); d["card"] = card; fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df = df.groupby(["card", "dataset", "network", "batch_size", "optimizer", "power_limit"], as_index=False).agg(
        power_w=("average_power", "mean"))
    df = df[df.power_w > 0].copy()
    df["cap_frac"] = df.power_limit / df.groupby("card")["power_limit"].transform("max")
    df["draw_frac"] = (df.power_w / df.power_limit).clip(upper=1.5)
    df["log_batch"] = np.log(df.batch_size.clip(lower=1))
    df["job"] = df.groupby(["network", "dataset", "batch_size", "optimizer"]).ngroup()
    df = add_specs(df)
    return df, ["network", "dataset", "optimizer", "card"], ["log_batch", "cap_frac", "power_limit"] + SPEC_COLS


def load_aws():
    fr = [pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")).assign(gpu="NVIDIA A10G"),
          pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    df = df.groupby(["card", "workload", "cap_frac", "cap_w"], as_index=False).agg(power_w=("power_w", "mean"))
    df = df[df.power_w > 0].copy()
    df["draw_frac"] = (df.power_w / df.cap_w).clip(upper=1.5)
    df["op_class"] = df.workload.map(KOP); df["arith_intensity"] = df.workload.map(KAI)
    df["job"] = df.groupby(["workload"]).ngroup()
    df = add_specs(df); df["ai_over_ridge"] = df["arith_intensity"] / df["ridge"]
    return df, ["op_class", "card"], ["arith_intensity", "ai_over_ridge", "cap_frac", "cap_w"] + SPEC_COLS


def main():
    out = {}
    for name, ld in [("zeus", load_zeus), ("aws", load_aws)]:
        df, cat, num = ld(); out[name] = evaluate(df, cat, num)
    json.dump(out, open(os.path.join(HERE, "results", "predict_power.json"), "w"), indent=2)
    print("== Predict energy-drain rate (power draw) from static features + rich card specs ==")
    hdr = f"{'dataset':>7} {'rows':>5} {'jobs':>5} {'cards':>5} | {'power MdAPE':>12} {'power R2':>9} | {'drawfrac MdAPE':>15} {'MAE':>6} {'vs assume=cap':>14} | {'card power':>11}"
    print(hdr); print("-" * len(hdr))
    for name, r in out.items():
        print(f"{name:>7} {r['n_rows']:>5} {r['n_jobs']:>5} {r['n_cards']:>5} | "
              f"{r['job_power_MdAPE']:>11.1f}% {r['job_power_R2']:>9} | "
              f"{r['job_drawfrac_MdAPE']:>14.1f}% {r['job_drawfrac_MAE']:>6} {r['assume_draw_eq_cap_MdAPE']:>13.1f}% | "
              f"{r.get('card_power_MdAPE','-'):>10}%")
    print("\ndraw_frac = actual draw / cap (how much of the cap the job uses); 'vs assume=cap' is the error if you")
    print("naively assumed a job draws its whole cap. written -> results/predict_power.json")


if __name__ == "__main__":
    main()
