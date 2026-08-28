# -*- coding: utf-8 -*-
"""
STATIC model features that should predict the cap-response, computed by ANALYZING the model graph (no timing
run). The physics (roofline) says the knee of the hinge -- the job's natural draw D0 -- is set by how
compute-bound the work is, i.e. its ARITHMETIC INTENSITY = FLOPs / bytes-moved. Compute-bound (high AI) ->
draws near TDP, cap has almost no inert band; memory-bound (low AI, e.g. depthwise convs) -> low draw, wide
inert band. We compute per whole-model task: params, FLOPs, weight+activation bytes, arithmetic intensity,
and the conv/linear/other FLOP mix -- all statically -- then check they rank the MEASURED D0 on A10G.

Meta-device forward: torchvision builders run shape propagation with no real compute, so FLOPs and activation
shapes come out fast. MACs: Conv2d = out_elems * (in_ch/groups * kH * kW); Linear = out_elems * in_features.
(SDPA/softmax matmuls inside ViT are functional, not module leaves, so ViT compute is mildly undercounted;
the projection Linears that dominate its bytes are counted.)
"""
import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm
from predict_power import RAW

BS = 32                                   # matches the sweep's vision(bs=32)
BYTES = 2                                 # fp16 in the sweep

MODELS = {
    "resnet50": tvm.resnet50, "resnet152": tvm.resnet152, "vgg16": tvm.vgg16,
    "densenet121": tvm.densenet121, "inception_v3": lambda **k: tvm.inception_v3(aux_logits=False, **k),
    "efficientnet_b0": tvm.efficientnet_b0, "convnext_tiny": tvm.convnext_tiny,
    "mobilenet_v3_l": tvm.mobilenet_v3_large, "vit_b16": tvm.vit_b_16,
}


def analyze(builder):
    acc = {"conv": 0, "linear": 0, "dw_conv": 0, "act_elems": 0}
    hooks = []

    def conv_hook(m, inp, out):
        oe = out.numel()
        macs = oe * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        acc["conv"] += 2 * macs
        if m.groups == m.in_channels and m.groups > 1:      # depthwise
            acc["dw_conv"] += 2 * macs
        acc["act_elems"] += oe

    def lin_hook(m, inp, out):
        acc["linear"] += 2 * out.numel() * m.in_features
        acc["act_elems"] += out.numel()

    def other_hook(m, inp, out):
        if isinstance(out, torch.Tensor):
            acc["act_elems"] += out.numel()

    m = builder(weights=None).eval().to("meta")
    for mod in m.modules():
        if isinstance(mod, torch.nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.ReLU,
                              torch.nn.GELU, torch.nn.Hardswish, torch.nn.SiLU)):
            hooks.append(mod.register_forward_hook(other_hook))
    res = 299 if builder is MODELS["inception_v3"] else 224
    x = torch.randn(BS, 3, res, res, device="meta", dtype=torch.float32)
    with torch.no_grad():
        m(x)
    for h in hooks:
        h.remove()
    params = sum(p.numel() for p in m.parameters())
    flops = acc["conv"] + acc["linear"]
    weight_bytes = params * BYTES
    act_bytes = acc["act_elems"] * BYTES
    ai = flops / (weight_bytes + act_bytes)                 # FLOP per byte moved
    return dict(params_M=params / 1e6, gflops=flops / 1e9, ai=ai,
                dw_frac=acc["dw_conv"] / max(flops, 1),
                linear_frac=acc["linear"] / max(flops, 1),
                weight_MB=weight_bytes / 1e6, act_MB=act_bytes / 1e6)


def measured_d0():
    fr = [pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv")).assign(gpu="NVIDIA A10G"),
          pd.read_csv(os.path.join(RAW, "aws_sweep_multi.csv"))]
    df = pd.concat(fr, ignore_index=True)
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.strip()
    a = df[df.card == "A10G"].groupby("workload", as_index=False).agg(D0=("power_w", "max"))
    return dict(zip(a.workload, a.D0))


def main():
    d0 = measured_d0()
    rows = []
    for name, b in MODELS.items():
        try:
            f = analyze(b)
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}"); continue
        f["model"] = name; f["D0_A10G"] = d0.get(name, np.nan)
        f["inert_band"] = 300 - f["D0_A10G"] if not np.isnan(f["D0_A10G"]) else np.nan
        rows.append(f)
    t = pd.DataFrame(rows).sort_values("ai")
    pd.set_option("display.width", 160, "display.max_columns", 20)
    show = t[["model", "params_M", "gflops", "ai", "dw_frac", "D0_A10G", "inert_band"]].copy()
    for c in ["params_M", "gflops", "ai", "dw_frac", "D0_A10G", "inert_band"]:
        show[c] = show[c].round(2)
    print(show.to_string(index=False))
    v = t.dropna(subset=["D0_A10G"])
    if len(v) >= 3:
        for feat in ["ai", "gflops", "params_M", "dw_frac"]:
            rho = v[feat].corr(v["D0_A10G"], method="spearman")
            print(f"  Spearman( {feat:9} , measured D0 ) = {rho:+.3f}")
    t.to_csv(os.path.join(RAW, "model_static_features.csv"), index=False)
    print("saved -> data/raw/model_static_features.csv")


if __name__ == "__main__":
    main()
