# -*- coding: utf-8 -*-
"""Extract a table of STATIC job design features (parameter count, FLOPs, depth, dominant op type) for the 20
models, and test which are usable predictors of the response-curve parameters (P0, beta, Rmax). FLOPs are
counted with forward hooks on Conv2d and Linear (the dominant ops); depth is the number of such layers; the
op mix is the parameter share in convolution vs linear/attention layers. Saves data/raw/job_features.csv.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from predict_power import RAW
TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}


def vision_feats(nm):
    import torch, torch.nn as nn, torchvision.models as tvm
    m = getattr(tvm, nm)(weights=None).eval()
    macs = {"conv": 0, "linear": 0}; depth = {"conv": 0, "linear": 0}; hooks = []

    def hk(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            o = out.shape; k = mod.kernel_size[0] * mod.kernel_size[1]
            macs["conv"] += (mod.in_channels // mod.groups) * mod.out_channels * o[2] * o[3] * k
            depth["conv"] += 1
        elif isinstance(mod, nn.Linear):
            n = int(np.prod(out.shape[:-1]))
            macs["linear"] += mod.in_features * mod.out_features * n
            depth["linear"] += 1
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            hooks.append(mod.register_forward_hook(hk))
    with torch.no_grad():
        m(torch.randn(1, 3, 224, 224))
    for h in hooks:
        h.remove()
    params = sum(p.numel() for p in m.parameters())
    conv_params = sum(p.numel() for mod in m.modules() if isinstance(mod, torch.nn.Conv2d) for p in mod.parameters())
    tmac = macs["conv"] + macs["linear"]
    return dict(params_M=params / 1e6, gflops=2 * tmac / 1e9, depth=depth["conv"] + depth["linear"],
                conv_frac=conv_params / max(params, 1), op_type="conv" if conv_params > 0.5 * params else "attn/linear")


def main():
    tv = ["resnet50", "resnet152", "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
          "efficientnet_v2_s", "convnext_tiny", "convnext_small", "vit_b_16", "vit_b_32", "swin_t",
          "mnasnet1_0", "shufflenet_v2_x1_0", "regnet_y_1_6gf", "regnet_y_8gf", "wide_resnet50_2",
          "resnext50_32x4d"]
    rows = []
    for nm in tv:
        try:
            rows.append(dict(model=nm, **vision_feats(nm)))
        except Exception as e:
            print("skip", nm, str(e)[:60])
    # analytic entries for the two non-torchvision workloads (batch-32 decode / UNet forward, gflops per step)
    H, nL, L, B = 4096, 6, 2048, 32
    llm_mac = nL * (4 * H * H + 2 * H * L) * B
    rows.append(dict(model="llm_decode", params_M=nL * (3 * H * H + H * H + 8 * H * H) / 1e6,
                     gflops=2 * llm_mac / 1e9, depth=nL, conv_frac=0.0, op_type="attn/linear"))
    rows.append(dict(model="sd_unet", params_M=300.0, gflops=200.0, depth=40, conv_frac=0.85, op_type="conv"))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RAW, "job_features.csv"), index=False)
    print("== Static job-feature table (data/raw/job_features.csv) ==")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # usability: correlate each feature with the curve parameters, pooled and within A100. Current 3-rep batch
    # data, control mode (batch 32); was the older fine-grid bs32 spectrum sweeps.
    fit = pd.read_csv(os.path.join(RAW, "energy_model_fit.csv")); fit = fit[fit["mode"] == "control"]
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")), pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))]
    sw = pd.concat(fr, ignore_index=True)
    sw["card"] = sw.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False).str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    sw = sw[sw["mode"] == "control"].groupby(["card", "workload", "cap_w"], as_index=False).agg(throughput=("throughput", "mean"))
    rmax = sw.groupby(["card", "workload"]).apply(lambda g: g.sort_values("cap_w").throughput.values[-1]).rename("Rmax").reset_index()
    a = fit[fit.card == "A100"].merge(rmax[rmax.card == "A100"], on=["card", "workload"]).merge(t, left_on="workload", right_on="model")
    a["logparams"] = np.log10(a.params_M); a["loggflops"] = np.log10(a.gflops); a["logRmax"] = np.log10(a.Rmax)
    print("\n== usable features: Pearson r with curve params on the A100 (n=%d) ==" % len(a))
    print(f"  {'feature':12} {'P0':>7} {'beta':>7} {'logRmax':>8}")
    corr = {}
    for f in ["logparams", "loggflops", "depth", "conv_frac"]:
        r = {k: float(np.corrcoef(a[f], a[k])[0, 1]) for k in ["P0", "beta", "logRmax"]}
        corr[f] = r
        print(f"  {f:12} {r['P0']:>+7.2f} {r['beta']:>+7.2f} {r['logRmax']:>+8.2f}")
    json.dump({"corr_A100": corr}, open("results/job_features.json", "w"), indent=1)
    print("\nsaved -> data/raw/job_features.csv, results/job_features.json")


if __name__ == "__main__":
    main()
