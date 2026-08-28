# -*- coding: utf-8 -*-
"""Which parameters of a job's response curve can be predicted, and from what? The curve is
P(R)=P0+aR^beta over R in [0, Rmax]. A cold-start scheduler would like all of P0, beta, and Rmax without
sweeping. We test each target: how much of its variance is the card vs the model, and whether a
leave-one-model-out predictor from static features (parameter count) plus the card beats the per-card mean.
Batch 32. Establishes the honest division of labor: what the card fixes, what static features give, and what
still needs a measurement.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from predict_power import RAW
TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}


def _card(df):
    c = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    return c.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()


def static_params():
    import torch, torchvision.models as tvm
    tv = ["resnet50", "resnet152", "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
          "efficientnet_v2_s", "convnext_tiny", "convnext_small", "vit_b_16", "vit_b_32", "swin_t",
          "mnasnet1_0", "shufflenet_v2_x1_0", "regnet_y_1_6gf", "regnet_y_8gf", "wide_resnet50_2",
          "resnext50_32x4d"]
    f = {}
    for nm in tv:
        try:
            f[nm] = sum(x.numel() for x in getattr(tvm, nm)(weights=None).parameters())
        except Exception:
            pass
    H, nL = 4096, 6
    f["llm_decode"] = nL * (3 * H * H + H * H + 8 * H * H); f["sd_unet"] = int(3e8)
    return f


def load():
    # Current 3-rep batch data, control mode (batch 32), matching the law figure's regime. (Was the older
    # fine-grid bs32 spectrum sweeps; switched to the released 3-rep batch files for consistency.)
    fit = pd.read_csv(os.path.join(RAW, "energy_model_fit.csv")); fit = fit[(fit["mode"] == "control") & (fit.card != "T4")]
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")), pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))]
    sw = pd.concat(fr, ignore_index=True); sw["card"] = _card(sw); sw = sw[(sw["mode"] == "control") & (sw.card != "T4")]
    sw = sw.groupby(["card", "workload", "cap_w"], as_index=False).agg(throughput=("throughput", "mean"))  # rep-mean
    rmax = sw.groupby(["card", "workload"]).apply(lambda g: g.sort_values("cap_w").throughput.values[-1]).rename("Rmax").reset_index()
    sp = static_params()
    df = fit.merge(rmax, on=["card", "workload"], how="inner")
    df["logparams"] = df.workload.map(lambda w: np.log10(sp[w]) if w in sp else np.nan)
    df["logRmax"] = np.log10(df.Rmax)
    return df.dropna(subset=["logparams"])


def loo(t, feats, target):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = t[feats].values; y = t[target].values; p = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        r = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.06, min_samples_leaf=3)
        r.fit(X[m], y[m]); p[i] = r.predict(X[i:i+1])[0]
    return p


def var_split(t, target):
    g = t[target].values; grand = g.mean(); tot = ((g - grand) ** 2).sum()
    card = sum(len(x) * (x[target].mean() - grand) ** 2 for _, x in t.groupby("card"))
    model = sum(len(x) * (x[target].mean() - grand) ** 2 for _, x in t.groupby("workload"))
    return 100 * card / tot, 100 * model / tot


def main():
    t = load(); t["card_code"] = t.card.map({c: i for i, c in enumerate(TDP)})
    os.makedirs("results", exist_ok=True); out = {}
    print(f"== Predictability of the response curve (bs32, n={len(t)} card x model) ==")
    print(f"  {'target':10} {'var% card':>9} {'var% model':>10} | {'skill: card':>11} {'card+params':>11}")
    for target, lab in [("P0", "P0 (floor)"), ("beta", "beta"), ("logRmax", "log Rmax")]:
        vc, vm = var_split(t, target)
        def skill(feats):
            pr = loo(t, feats, target); mae = np.mean(np.abs(pr - t[target].values))
            base = np.array([t[t.card == c][target].mean() for c in t.card])
            mb = np.mean(np.abs(base - t[target].values))
            return 1 - mae / mb
        s_card = skill(["card_code"]); s_cp = skill(["card_code", "logparams"])
        out[target] = dict(var_card=vc, var_model=vm, skill_card=s_card, skill_cardparams=s_cp)
        print(f"  {lab:10} {vc:>8.0f}% {vm:>9.0f}% | {s_card:>10.0%} {s_cp:>10.0%}")
    # correlations with params
    print("\n  corr(log params, .): " + ", ".join(f"{k}={np.corrcoef(t.logparams, t[k])[0,1]:+.2f}" for k in ["P0", "beta", "logRmax"]))
    out["corr_params"] = {k: float(np.corrcoef(t.logparams, t[k])[0, 1]) for k in ["P0", "beta", "logRmax"]}
    # P0 as fraction of TDP: how card-constant is it?
    t["P0frac"] = t.P0 / t.card.map(TDP)
    print("  P0/TDP per card (mean +/- std): " + ", ".join(f"{c} {g.P0frac.mean():.2f}+/-{g.P0frac.std():.2f}" for c, g in t.groupby("card")))
    out["P0frac_by_card"] = {c: [float(g.P0frac.mean()), float(g.P0frac.std())] for c, g in t.groupby("card")}
    json.dump(out, open("results/predict_curve.json", "w"), indent=1)
    print("saved -> results/predict_curve.json")


if __name__ == "__main__":
    main()
