# -*- coding: utf-8 -*-
"""
Where does per-workload power-cap ELASTICITY carry decision value? For each measured pool,
compute the PURE per-workload elasticity value = equal-weight oracle gain over uniform capping
(equal weights, so the gain is elasticity alone, not priority), plus the elasticity heterogeneity
(std of per-workload sheddable-power fraction). Confirms elasticity value scales with fleet
heterogeneity, so on a homogeneous single-workload-type trace (PoC-B) it is ~0 by construction,
while on a realistic heterogeneous fleet it is large. Saved to results/elasticity_value.json.
"""
import os, json
import numpy as np
import learned_control as LC
import poca_killtest as K
from rq4_strengthen import aws_table


def elasticity_value(tab, fracs=(0.90, 0.85, 0.80, 0.75, 0.70)):
    pool = LC.true_pool(tab)                       # equal weight -> gain is elasticity, no priority
    tb = sum(w["pmax"] for w in pool); gains = []
    for f in fracs:
        C = f * tb
        uni = K.pool_cost_power(pool, K.b1_uniform(pool, C))[0]
        ora = K.pool_cost_power(pool, K.b6_oracle(pool, C))[0]
        if uni > 1e-9:
            gains.append(100 * (uni - ora) / uni)
    return float(np.mean(gains)) if gains else 0.0


def spread(tab):
    g = tab.groupby("wl")["power"].agg(["min", "max"])
    return float(((g["max"] - g["min"]) / g["max"]).std())


def main():
    pools = {"emerald": (LC.emerald_table(), ["f_task"]),
             "zeus": (LC.zeus_table(), ["f_net"]),
             "aws26": (aws_table(), ["f_fam"])}
    # AWS type families for a fair leakage-free class-mean (unique f_fam otherwise falls back)
    FAM = {"gemm_fp16":"gemm","gemm_fp32":"gemm","gemm_bf16":"gemm","bmm_fp16":"gemm","attention_sdpa":"attn",
           "vit_b16":"attn","fft2d":"linalg","cholesky":"linalg","resnet50":"conv","resnet152":"conv",
           "vgg16":"conv","densenet121":"conv","convnext_tiny":"conv","efficientnet_b0":"conv",
           "mobilenet_v3_l":"conv","inception_v3":"conv","membw":"mem","reduction":"mem","softmax_big":"mem",
           "layernorm_big":"mem","embed_gather":"mem","scatter_add":"mem","sort_big":"mem","memcpy":"mem",
           "elementwise_chain":"mem","decode_like":"lat"}
    out = {}
    for name, (tab, cats) in pools.items():
        ev = elasticity_value(tab)
        if name == "aws26":
            tab = tab.copy(); tab["f_type"] = tab["wl"].map(FAM); cats = ["f_type"]
        _, cm = LC.oracle_capture(tab, cats, ["obs_draw_frac"] if "obs_draw_frac" in tab.columns else [])
        out[name] = {"n_workloads": int(tab["wl"].nunique()),
                     "elasticity_spread": round(spread(tab), 3),
                     "elasticity_value_pct": round(ev, 1),
                     "oracle_capture_classmean_pct": round(cm, 0)}
        print(f"{name:8} n={out[name]['n_workloads']:>3} spread={out[name]['elasticity_spread']:.3f} "
              f"elasticity_value={ev:5.1f}%  oracle_capture={cm:4.0f}%")
    json.dump(out, open(os.path.join(K.RESULTS, "elasticity_value.json"), "w"), indent=2)
    print("written -> results/elasticity_value.json")


if __name__ == "__main__":
    main()
