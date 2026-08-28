# -*- coding: utf-8 -*-
"""
Empirical check of the physics: a power cap only affects a job once it drops BELOW the job's natural
(uncapped) draw. For each (job, card) we take the natural draw D0 = draw at the highest cap, then split
every cap level into SLACK (cap >= D0, should not bind) vs BINDING (cap < D0). We report mean slowdown and
mean draw-fraction in each regime. Expectation: in the SLACK regime slowdown ~ 0 and draw < cap; in the
BINDING regime slowdown > 0 and draw ~ cap.
"""
import os
import numpy as np
import pandas as pd
from predict_power import RAW


def analyze(name, df, cap_w, draw_w, rate):
    df = df.copy()
    df["D0"] = df.groupby(["job", "card"])[draw_w].transform("max")     # natural draw = draw at top cap
    df["rate_max"] = df.groupby(["job", "card"])[rate].transform("max")
    df["slowdown"] = 1 - df[rate] / df["rate_max"]
    df["draw_frac"] = df[draw_w] / df[cap_w]
    df["slack"] = df[cap_w] >= 0.98 * df["D0"]                          # cap at/above natural draw
    for lab, sub in [("cap ABOVE natural draw (slack)", df[df.slack]), ("cap BELOW natural draw (binding)", df[~df.slack])]:
        if len(sub):
            print(f"  {lab:36} n={len(sub):4}  mean slowdown {sub.slowdown.mean():.3f}   mean draw/cap {sub.draw_frac.mean():.2f}")
    # correlation: does 'is the cap below natural draw' explain slowdown?
    print(f"  corr(cap-is-binding, slowdown) = {np.corrcoef((~df.slack).astype(float), df.slowdown)[0,1]:.2f}")


def main():
    print("== Does the cap only bite below the job's natural draw? ==\n")
    # AWS
    fr = [pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")).assign(gpu="NVIDIA A10G"),
          pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv"))]
    a = pd.concat(fr, ignore_index=True)
    a["card"] = a.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.strip()
    a = a.groupby(["card", "workload", "cap_frac", "cap_w"], as_index=False).agg(power_w=("power_w", "mean"), thr=("throughput", "mean"))
    a["job"] = a.groupby(["workload"]).ngroup()
    print("AWS (A10G/L4/T4):")
    analyze("aws", a, "cap_w", "power_w", "thr")

    # Zeus
    fr = []
    for card, f in [("V100", "zeus_summary_power_v100.csv"), ("A40", "zeus_summary_power_a40.csv")]:
        d = pd.read_csv(os.path.join(RAW, f)); d["card"] = card; fr.append(d)
    z = pd.concat(fr, ignore_index=True)
    z = z.groupby(["card", "network", "dataset", "batch_size", "optimizer", "power_limit"], as_index=False).agg(
        power=("average_power", "mean"), t=("time_per_epoch", "mean"))
    z = z[z.t > 0].copy(); z["rate"] = 1.0 / z.t
    z["job"] = z.groupby(["network", "dataset", "batch_size", "optimizer"]).ngroup()
    print("\nZeus (V100/A40):")
    analyze("zeus", z, "power_limit", "power", "rate")

    # what fraction of all sampled (job,card,cap) points are in the non-binding (slack) regime?
    for nm, df, capc, draww in [("AWS", a, "cap_w", "power_w"), ("Zeus", z, "power_limit", "power")]:
        d = df.copy(); d["D0"] = d.groupby(["job", "card"])[draww].transform("max")
        frac = float((d[capc] >= 0.98 * d["D0"]).mean())
        print(f"\n{nm}: {100*frac:.0f}% of sampled cap points sit AT/ABOVE the job's natural draw (cap does nothing there).")


if __name__ == "__main__":
    main()
