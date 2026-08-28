# -*- coding: utf-8 -*-
"""
Did the $10 data-collection campaign improve the model? Uses the EXPANDED A10G sweep
(own_aws_sweep.csv, 26 diverse workloads) with genuine TYPE families (gemm/conv/attn/linalg/
mem/lat), evaluated LEAKAGE-FREE (the class-mean fix in learned_control excludes the held-out
workload from its family mean). Reports:

  * full-data Oracle Capture on the 26-workload heterogeneous set, and
  * the learning curve (capture vs #training workloads) -- the direct test that capture RISES
    as distinct workloads are added, on a construct where we control the count.

Context it is compared against (all leakage-free): emerald (homogeneous, 8 wl) generalizes at
~77%; Zeus (heterogeneous but only 6 wl, mostly unique families) FAILS (worse than uniform).
The campaign supplies the heterogeneous-AND-populous set that was missing.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import learned_control as LC
import poca_killtest as K
import model_improve as MI
from rq4_strengthen import aws_table, add_obs_feature

FAM = {"gemm_fp16": "gemm", "gemm_fp32": "gemm", "gemm_bf16": "gemm", "bmm_fp16": "gemm",
       "attention_sdpa": "attn", "vit_b16": "attn", "fft2d": "linalg", "cholesky": "linalg",
       "resnet50": "conv", "resnet152": "conv", "vgg16": "conv", "densenet121": "conv",
       "convnext_tiny": "conv", "efficientnet_b0": "conv", "mobilenet_v3_l": "conv", "inception_v3": "conv",
       "membw": "mem", "reduction": "mem", "softmax_big": "mem", "layernorm_big": "mem",
       "embed_gather": "mem", "scatter_add": "mem", "sort_big": "mem", "memcpy": "mem",
       "elementwise_chain": "mem", "decode_like": "lat"}


def main():
    tab = add_obs_feature(aws_table())
    tab["f_type"] = tab["wl"].map(FAM)
    nwl = tab["wl"].nunique()
    sizes = dict(tab.groupby("f_type")["wl"].nunique())
    print(f"== Expanded A10G sweep: {nwl} workloads, {len(sizes)} type families {sizes} ==")

    g_full, cm_full = LC.oracle_capture(tab, ["f_type"], ["obs_draw_frac"])
    print(f"  full-data Oracle Capture (leakage-free): class-mean {cm_full:.0f}%  GBDT {g_full:.0f}%")

    # Cheap learning curve: subsample the POOL to k workloads and run ONE leave-one-out pass
    # (O(1) oracle_capture calls per (k,seed), not O(n)). Capture vs #workloads on our construct.
    wls = list(tab["wl"].unique()); n = len(wls)
    curve = []
    for k in [6, 10, 14, 18, 22, n]:
        if k > n:
            continue
        caps = []
        for seed in range(4):
            rng = np.random.default_rng(seed)
            keep = list(rng.choice(wls, size=k, replace=False))
            sub = tab[tab["wl"].isin(keep)]
            _, cm = LC.oracle_capture(sub, ["f_type"], ["obs_draw_frac"])
            caps.append(cm)
        curve.append({"k_train": k, "capture_pct": float(np.mean(caps)), "sd": float(np.std(caps))})
    print("  learning curve (capture vs #workloads in pool, leave-one-out within):")
    for r in curve:
        print(f"    n={r['k_train']:>2}: {r['capture_pct']:6.1f}%  (sd {r['sd']:.1f})")

    lo, hi = curve[0], curve[-1]
    out = {"n_workloads": int(nwl), "family_sizes": {k: int(v) for k, v in sizes.items()},
           "capture_classmean_pct": float(cm_full), "capture_gbdt_pct": float(g_full),
           "learning_curve": curve,
           "capture_rose_with_workloads": bool(hi["capture_pct"] > lo["capture_pct"] + 2.0),
           "gain_pts_lo_to_hi": float(hi["capture_pct"] - lo["capture_pct"])}
    json.dump(out, open(os.path.join(K.RESULTS, "aws_data_impact.json"), "w"), indent=2)
    print(f"\n  capture {lo['capture_pct']:.0f}% @ {lo['k_train']} wl -> {hi['capture_pct']:.0f}% @ {hi['k_train']} wl "
          f"({'RISES with data' if out['capture_rose_with_workloads'] else 'flat'})")
    print("written -> results/aws_data_impact.json")


if __name__ == "__main__":
    main()
