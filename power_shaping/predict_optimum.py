# -*- coding: utf-8 -*-
"""Stage-2 predictor experiment: can a scheduler predict a job's energy-optimal cap and its cap-elasticity
from a CHEAP uncapped probe, instead of a full power sweep? Features are all read from a single uncapped run
(utilization, draw/TDP, throughput, latency); targets are the energy-optimal cap (% TDP) and the
compute-power elasticity CPE = d ln C / d ln p. Leave-one-model-out cross-validation, per the pooled fleet,
compared to a naive baseline (predict the per-card mean). Establishes that the derivatives needed for
allocation are observable without sweeping every model.
"""
import os
import numpy as np
import pandas as pd
from predict_power import RAW, saturating_set

TDP = {"T4": 70, "L4": 72, "A10G": 300, "A100": 400}


def _card(df):
    c = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    return c.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()


def load():
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_batch.csv")),
          pd.read_csv(os.path.join(RAW, "aws_sweep_a100_batch.csv"))]
    df = pd.concat(fr, ignore_index=True); df["card"] = _card(df)
    df = saturating_set(df)                                   # loaded (production-relevant) regime
    # Mean the repetitions per operating point before locating the Joule point (see make_efficiency_table.load).
    df = df.groupby(["card", "workload", "cap_w"], as_index=False).agg(
        power_w=("power_w", "mean"), throughput=("throughput", "mean"), util=("util", "mean"))
    rows = []
    for (card, wl), g in df.groupby(["card", "workload"]):
        g = g.sort_values("cap_w")
        C, P, L = g.throughput.values, g.power_w.values, (1000.0 / g.throughput.values)
        E = P / C; k = int(np.argmin(E))
        # cheap uncapped probe (features)
        unc = g.iloc[-1]
        cpe = (np.log(C[-1]) - np.log(C[0])) / (np.log(P[-1]) - np.log(P[0])) if P[0] > 0 and C[0] > 0 else np.nan
        rows.append(dict(card=card, model=wl,
                         f_util=unc.util, f_drawTDP=100 * unc.power_w / TDP[card],
                         f_logthr=np.log(unc.throughput), f_loglat=np.log(1000.0 / unc.throughput),
                         y_optcap=100 * g.cap_w.values[k] / TDP[card], y_cpe=cpe))
    return pd.DataFrame(rows).dropna()


def loo_predict(t, feats, target):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = t[feats].values; y = t[target].values
    pred = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        r = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.06, min_samples_leaf=3)
        r.fit(X[m], y[m]); pred[i] = r.predict(X[i:i+1])[0]
    return pred


def main():
    t = load()
    os.makedirs("results", exist_ok=True)
    feats = ["card_code", "f_util", "f_drawTDP", "f_logthr", "f_loglat"]
    t["card_code"] = t.card.map({c: i for i, c in enumerate(TDP)})
    out = {}
    print("== Predict from a single uncapped probe (leave-one-model-out) ==")
    print(f"  {'target':18} {'MAE model':>10} {'MAE baseline':>13} {'skill':>7}")
    for target, unit in [("y_optcap", "% TDP"), ("y_cpe", "elasticity")]:
        pred = loo_predict(t, feats, target)
        mae = np.mean(np.abs(pred - t[target].values))
        base = np.zeros(len(t))                                   # baseline: per-card mean
        for c in t.card.unique():
            m = (t.card == c).values; base[m] = t[target].values[m].mean()
        mae_b = np.mean(np.abs(base - t[target].values))
        skill = 1 - mae / mae_b
        out[target] = dict(mae=float(mae), mae_baseline=float(mae_b), skill=float(skill))
        print(f"  {target:18} {mae:>10.2f} {mae_b:>13.2f} {skill:>6.0%}")
    # single-feature correlations (which cheap signal carries the info)
    print("\n  single-feature correlation with targets (Pearson):")
    for target in ["y_optcap", "y_cpe"]:
        cs = {f: np.corrcoef(t[f], t[target])[0, 1] for f in ["f_util", "f_drawTDP", "f_logthr", "f_loglat"]}
        best = max(cs, key=lambda k: abs(cs[k]))
        print(f"    {target:10}: " + ", ".join(f"{f}={cs[f]:+.2f}" for f in cs) + f"   (best: {best})")
        out.setdefault("corr", {})[target] = {k: float(v) for k, v in cs.items()}
    # cheap-probe -> Joule point: Pearson r of uncapped draw/TDP vs the measured energy-optimal cap
    out["probe_optcap_r_pooled"] = round(float(np.corrcoef(t.f_drawTDP, t.y_optcap)[0, 1]), 2)
    pc = {}
    for c in t.card.unique():
        m = (t.card == c).values
        pc[c] = round(float(np.corrcoef(t.f_drawTDP.values[m], t.y_optcap.values[m])[0, 1]), 2)
    out["probe_optcap_r_per_card"] = pc
    print(f"\n  probe->Joule point: pooled r={out['probe_optcap_r_pooled']}, per-card {pc}")
    import json
    json.dump(out, open("results/predict_optimum.json", "w"), indent=1)
    print("\nsaved -> results/predict_optimum.json")


if __name__ == "__main__":
    main()
