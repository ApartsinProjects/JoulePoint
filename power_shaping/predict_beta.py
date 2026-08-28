# -*- coding: utf-8 -*-
"""The per-job model question. Fitting P(R)=P0+aR^beta characterizes a job we have already swept. For a
COLD-START job (never run), can we predict the curve's parameters, especially the exponent beta, from static
job design features (parameter count, FLOPs) plus the card? This tests how far a purely static model gets,
and thereby what a scheduler still needs (a cheap runtime probe, or richer features).

Static features per model: parameter count and FLOPs at batch 32 (torchvision for the CNN/ViT; analytic for
llm_decode and sd_unet). Target: beta (and, for reference, the energy-optimal cap) per (card, model), bs32.
Leave-one-model-out CV, skill against the per-card mean baseline.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from predict_power import RAW
TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}


def static_features():
    import torch, torchvision.models as tvm
    feats = {}
    tv = ["resnet50", "resnet152", "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
          "efficientnet_v2_s", "convnext_tiny", "convnext_small", "vit_b_16", "vit_b_32", "swin_t",
          "mnasnet1_0", "shufflenet_v2_x1_0", "regnet_y_1_6gf", "regnet_y_8gf", "wide_resnet50_2",
          "resnext50_32x4d"]
    for nm in tv:
        try:
            m = getattr(tvm, nm)(weights=None)
            p = sum(x.numel() for x in m.parameters())
            feats[nm] = dict(params=p)
        except Exception:
            pass
    # analytic params for the two non-torchvision workloads
    H, nL = 4096, 6
    feats["llm_decode"] = dict(params=nL * (3 * H * H + H * H + 4 * H * H + 4 * H * H))   # qkv+proj+mlp
    feats["sd_unet"] = dict(params=int(3.0e8))                                            # ~UNet-scale
    return feats


def load():
    fit = pd.read_csv(os.path.join(RAW, "energy_model_fit.csv"))
    fit = fit[(fit["mode"] == "control") & (fit.card != "T4")]   # 3-rep batch-32; T4 excluded (see paper limitations)
    eff = pd.read_csv(os.path.join(RAW, "efficiency_table.csv"))
    sf = static_features()
    rows = []
    for _, r in fit.iterrows():
        if r.workload not in sf:
            continue
        rows.append(dict(card=r.card, model=r.workload, beta=r.beta,
                         logparams=np.log10(sf[r.workload]["params"])))
    return pd.DataFrame(rows).dropna()


def loo(t, feats, target):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = t[feats].values; y = t[target].values; pred = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        r = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.06, min_samples_leaf=3)
        r.fit(X[m], y[m]); pred[i] = r.predict(X[i:i+1])[0]
    return pred


def main():
    t = load()
    t["card_code"] = t.card.map({c: i for i, c in enumerate(TDP)})
    os.makedirs("results", exist_ok=True)
    print(f"== Predict beta from static job features (n={len(t)} card x model), leave-one-model-out ==")
    out = {}
    for feats, label in [(["card_code"], "card only"),
                         (["logparams"], "params only"),
                         (["card_code", "logparams"], "card + params")]:
        pred = loo(t, feats, "beta"); mae = np.mean(np.abs(pred - t.beta.values))
        base = np.array([t[t.card == c].beta.mean() for c in t.card]);  # per-card mean baseline
        mae_b = np.mean(np.abs(base - t.beta.values))
        skill = 1 - mae / mae_b
        out[label] = dict(mae=float(mae), mae_baseline=float(mae_b), skill=float(skill))
        print(f"  {label:16}: MAE(beta)={mae:.2f}  baseline(per-card mean)={mae_b:.2f}  skill={skill:+.0%}")
    # how much of beta's variance is card vs model?
    grand = t.beta.mean()
    ss_tot = ((t.beta - grand) ** 2).sum()
    ss_card = sum(len(g) * (g.beta.mean() - grand) ** 2 for _, g in t.groupby("card"))
    ss_model = sum(len(g) * (g.beta.mean() - grand) ** 2 for _, g in t.groupby("model"))
    print(f"\n  variance of beta explained by CARD alone: {ss_card/ss_tot:.0%};  by MODEL alone: {ss_model/ss_tot:.0%}")
    print(f"  corr(log params, beta): {np.corrcoef(t.logparams, t.beta)[0,1]:+.2f}")
    out["variance"] = dict(card=float(ss_card/ss_tot), model=float(ss_model/ss_tot),
                           corr_params_beta=float(np.corrcoef(t.logparams, t.beta)[0,1]))
    json.dump(out, open("results/predict_beta.json", "w"), indent=1)
    print("saved -> results/predict_beta.json")


if __name__ == "__main__":
    main()
