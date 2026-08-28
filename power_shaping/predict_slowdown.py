# -*- coding: utf-8 -*-
"""
How well does the power CAP predict SLOWDOWN, and how much does the JOB's character add?
Slowdown = 1 - work-rate(cap)/work-rate(uncapped), the fractional throughput loss. We predict it
leave-one-JOB-out under four feature sets, to see where the signal is:

  cap only        : just cap_frac                       -> the cap as a standalone predictor
  cap + card       : cap_frac + card specs (incl. ridge) -> add the hardware
  cap + job        : cap_frac + the job's compute/memory character (arch/kernel, arithmetic intensity)
  full             : cap + card + job

The point: the cap alone is weak, because the SAME cap slows a compute-bound job a lot and a memory-bound
job barely. Slowdown is a cap x job-character INTERACTION; adding the job's character is what makes it
predictable. Reported for Zeus (V100/A40) and AWS (A10G/L4/T4).
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

from predict_power import RICH, SPEC_COLS, KAI, KOP, add_specs, gbr, RAW
HERE = os.path.dirname(__file__)


def design(df, cat, num):
    X = pd.get_dummies(df[cat].astype(str)) if cat else pd.DataFrame(index=df.index)
    for c in num: X[c] = df[c].values
    return X


def cv_slow(df, X):
    Xv = X.values.astype(float); y = df["slowdown"].values; p = np.zeros(len(df))
    for g in df["job"].unique():
        te = (df["job"] == g).values; tr = ~te
        p[te] = gbr().fit(Xv[tr], y[tr]).predict(Xv[te]) if tr.sum() >= 8 else y[tr].mean()
    return np.clip(p, 0, 0.95)


def mae(t, p, thresh=None):
    if thresh is not None:
        m = t > thresh
        return round(float(np.mean(np.abs(p[m] - t[m]))), 3) if m.any() else None
    return round(float(np.mean(np.abs(p - t))), 3)


def load_zeus():
    fr = []
    for card, f in [("V100", "zeus_summary_power_v100.csv"), ("A40", "zeus_summary_power_a40.csv")]:
        d = pd.read_csv(os.path.join(RAW, f)); d["card"] = card; fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df = df.groupby(["card", "dataset", "network", "batch_size", "optimizer", "power_limit"], as_index=False).agg(t=("time_per_epoch", "mean"))
    df = df[df.t > 0].copy(); df["rate"] = 1.0 / df.t
    df["cap_frac"] = df.power_limit / df.groupby("card")["power_limit"].transform("max")
    df["log_batch"] = np.log(df.batch_size.clip(lower=1))
    df["job"] = df.groupby(["network", "dataset", "batch_size", "optimizer"]).ngroup()
    df["slowdown"] = 1 - df.rate / df.groupby(["job", "card"])["rate"].transform("max")
    df = add_specs(df)
    return df, {"cap only": ([], ["cap_frac"]),
                "cap + card": (["card"], ["cap_frac"] + SPEC_COLS),
                "cap + job": (["network", "dataset", "optimizer"], ["cap_frac", "log_batch"]),
                "full": (["network", "dataset", "optimizer", "card"], ["cap_frac", "log_batch"] + SPEC_COLS)}


def load_aws():
    fr = [pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")).assign(gpu="NVIDIA A10G"),
          pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    df = df.groupby(["card", "workload", "cap_frac"], as_index=False).agg(thr=("throughput", "mean"))
    df = df[df.thr > 0].copy(); df["rate"] = df.thr
    df["op_class"] = df.workload.map(KOP); df["arith_intensity"] = df.workload.map(KAI)
    df["job"] = df.groupby(["workload"]).ngroup()
    df["slowdown"] = 1 - df.rate / df.groupby(["job", "card"])["rate"].transform("max")
    df = add_specs(df); df["ai_over_ridge"] = df.arith_intensity / df.ridge
    return df, {"cap only": ([], ["cap_frac"]),
                "cap + card": (["card"], ["cap_frac"] + SPEC_COLS),
                "cap + job": (["op_class"], ["cap_frac", "arith_intensity", "ai_over_ridge"]),
                "full": (["op_class", "card"], ["cap_frac", "arith_intensity", "ai_over_ridge"] + SPEC_COLS)}


def main():
    out = {}
    for name, ld in [("zeus", load_zeus), ("aws", load_aws)]:
        df, sets = ld()
        res = {}
        for label, (cat, num) in sets.items():
            p = cv_slow(df, design(df, cat, num))
            res[label] = {"slowdown_MAE": mae(df.slowdown.values, p),
                          "slowdown_MAE_where_gt5pct": mae(df.slowdown.values, p, 0.05)}
        out[name] = res
    json.dump(out, open(os.path.join(HERE, "results", "predict_slowdown.json"), "w"), indent=2)
    print("== How the CAP predicts SLOWDOWN, and what the job's character adds (leave-one-job-out) ==")
    print("slowdown MAE (fraction; lower is better) -- and in [ ] the MAE only where true slowdown > 5%\n")
    print(f"{'features':>14} | {'ZEUS':>18} | {'AWS':>18}")
    print("-" * 58)
    for label in ["cap only", "cap + card", "cap + job", "full"]:
        z = out["zeus"][label]; a = out["aws"][label]
        print(f"{label:>14} | {z['slowdown_MAE']:>7} [{str(z['slowdown_MAE_where_gt5pct']):>5}] | "
              f"{a['slowdown_MAE']:>7} [{str(a['slowdown_MAE_where_gt5pct']):>5}]")
    print("\nwritten -> results/predict_slowdown.json")


if __name__ == "__main__":
    main()
