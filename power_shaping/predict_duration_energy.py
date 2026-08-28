# -*- coding: utf-8 -*-
"""
Predict job DURATION and ENERGY from STATIC job features + card specs + GPU power cap, on the only
corpus with rich job descriptions: Zeus (V100 + A40), which sweeps the power limit for many
(network, dataset, batch, optimizer) jobs and records per-epoch time and average power. No runtime
counters: pure prediction from the job's description and the card's physical specs.

We report:
  - absolute accuracy of duration and energy (median abs % error, log-R2), leave-one-JOB-out;
  - the DECISION-RELEVANT quantity: the CAP RESPONSE, i.e. the predicted slowdown D(cap)/D(max cap)-1
    versus the true slowdown (this is the hard part; absolute duration is dominated by batch/dataset);
  - leave-one-CARD-out (train V100 predict A40 and vice versa): does describing the card by its specs
    let the model transfer to an unseen card?
  - an ABLATION: duration error with vs without the power-cap feature, to isolate the cap's contribution.

SANITY (pre-registered): predicted duration is (near) non-increasing in the power cap.
"""
from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
SPECS = {"V100": (900.0, 112.0, 16.0, 80.0, 300.0), "A40": (696.0, 150.0, 48.0, 84.0, 300.0)}
SPEC_COLS = ["mem_bw_gbs", "fp16_tflops", "mem_gb", "sm", "tdp_w"]
CAT = ["network", "dataset", "optimizer", "card"]
NUM_FULL = ["log_batch", "power_limit", "power_frac"] + SPEC_COLS
NUM_NOCAP = ["log_batch"] + SPEC_COLS                      # ablation: drop the cap feature


def load():
    fr = []
    for card, f in [("V100", "zeus_summary_power_v100.csv"), ("A40", "zeus_summary_power_a40.csv")]:
        d = pd.read_csv(os.path.join(RAW, f)); d["card"] = card; fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df = df.groupby(["card", "dataset", "network", "batch_size", "optimizer", "power_limit"], as_index=False).agg(
        t=("time_per_epoch", "mean"), p=("average_power", "mean"))
    df = df[(df.t > 0) & (df.p > 0)].reset_index(drop=True)
    df["duration"] = df.t; df["energy"] = df.p * df.t
    df["power_frac"] = df.power_limit / df.groupby("card")["power_limit"].transform("max")
    df["log_batch"] = np.log(df.batch_size.clip(lower=1))
    for i, c in enumerate(SPEC_COLS):
        df[c] = df.card.map(lambda g: SPECS[g][i])
    df["job"] = df.groupby(["network", "dataset", "batch_size", "optimizer"]).ngroup()
    return df


def gbr():
    return HistGradientBoostingRegressor(max_depth=4, max_iter=350, learning_rate=0.05, min_samples_leaf=4)


def design(df, num):
    X = pd.get_dummies(df[CAT].astype(str))
    for c in num:
        X[c] = df[c].values
    return X


def mdape(t, p): return float(np.median(np.abs((p - t) / t)) * 100)
def log_r2(t, p):
    lt, lp = np.log(t), np.log(np.clip(p, 1e-9, None)); ss = np.sum((lt - lp) ** 2); tot = np.sum((lt - lt.mean()) ** 2)
    return float(1 - ss / tot) if tot > 0 else 0.0


def cv(df, X, group, target):
    Xv = X.values.astype(float); y = np.log(df[target].values); p = np.zeros(len(df))
    for g in df[group].unique():
        te = (df[group] == g).values; tr = ~te
        p[te] = gbr().fit(Xv[tr], y[tr]).predict(Xv[te]) if tr.sum() >= 8 else y[tr].mean()
    return np.exp(p)


def slowdown_metrics(df, dpred):
    d = df.assign(dp=dpred); ts, ps = [], []
    for _, g in d.groupby(["job", "card"]):
        g = g.sort_values("power_limit"); dt, dp = g.duration.values, g.dp.values
        ts.extend(dt / dt[-1] - 1); ps.extend(dp / dp[-1] - 1)     # slowdown vs the max-cap (uncapped) point
    ts, ps = np.array(ts), np.array(ps); m = ts > 0.05
    return round(float(np.mean(np.abs(ps - ts))), 3), (round(float(np.median(np.abs((ps[m]-ts[m])/ts[m]))*100), 1) if m.any() else None), int(m.sum())


def mono(df, dpred):
    d = df.assign(dp=dpred); ok = tot = 0
    for _, g in d.groupby(["job", "card"]):
        v = g.sort_values("power_limit").dp.values
        for a, b in zip(v[:-1], v[1:]): tot += 1; ok += (b <= a + 1e-9)
    return round(100 * ok / tot, 1)


def main():
    df = load()
    Xf = design(df, NUM_FULL)
    Dp = cv(df, Xf, "job", "duration"); Ep = cv(df, Xf, "job", "energy")
    sl_mae, sl_mdape, n_sl = slowdown_metrics(df, Dp)
    # leave-one-card-out
    Dc = cv(df, Xf, "card", "duration"); Ec = cv(df, Xf, "card", "energy")
    # ablation: no cap feature
    Xn = design(df, NUM_NOCAP); Dp_nocap = cv(df, Xn, "job", "duration")
    base = np.exp(np.full(len(df), np.log(df.duration).mean()))

    out = {"n_rows": len(df), "n_jobs": int(df.job.nunique()), "n_cards": 2,
           "leave_one_job_out": {
               "duration_MdAPE": round(mdape(df.duration.values, Dp), 1), "duration_logR2": round(log_r2(df.duration.values, Dp), 3),
               "energy_MdAPE": round(mdape(df.energy.values, Ep), 1), "energy_logR2": round(log_r2(df.energy.values, Ep), 3),
               "slowdown_MAE": sl_mae, "slowdown_MdAPE_where_gt5pct": sl_mdape, "n_slowdown_points": n_sl,
               "duration_monotone_pct": mono(df, Dp)},
           "leave_one_card_out": {"duration_MdAPE": round(mdape(df.duration.values, Dc), 1),
                                  "energy_MdAPE": round(mdape(df.energy.values, Ec), 1)},
           "ablation_duration_MdAPE": {"with_cap_feature": round(mdape(df.duration.values, Dp), 1),
                                       "without_cap_feature": round(mdape(df.duration.values, Dp_nocap), 1)},
           "baseline_globalmean_duration_MdAPE": round(mdape(df.duration.values, base), 1)}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "predict_duration_energy.json"), "w"), indent=2)

    j = out["leave_one_job_out"]
    print("== Predict duration & energy from STATIC features -- Zeus (V100 + A40) ==")
    print(f"rows {out['n_rows']}  jobs {out['n_jobs']}  cards 2")
    print("\nleave-one-JOB-out (predict an unseen job's curve):")
    print(f"  duration : MdAPE {j['duration_MdAPE']}%   logR2 {j['duration_logR2']}   (mean baseline {out['baseline_globalmean_duration_MdAPE']}%)")
    print(f"  energy   : MdAPE {j['energy_MdAPE']}%   logR2 {j['energy_logR2']}")
    print(f"  CAP RESPONSE (the hard part): slowdown MAE {j['slowdown_MAE']}  "
          f"(rel err where slowdown>5%: {j['slowdown_MdAPE_where_gt5pct']}%, n={j['n_slowdown_points']})")
    print(f"  SANITY duration monotone in cap: {j['duration_monotone_pct']}%")
    print("\nleave-one-CARD-out (train one GPU, predict the other):")
    print(f"  duration MdAPE {out['leave_one_card_out']['duration_MdAPE']}%   energy MdAPE {out['leave_one_card_out']['energy_MdAPE']}%")
    a = out["ablation_duration_MdAPE"]
    print(f"\nablation: duration MdAPE with cap feature {a['with_cap_feature']}%  vs  without {a['without_cap_feature']}%  "
          f"(the gap is the cap's contribution)")
    print("written -> results/predict_duration_energy.json")


if __name__ == "__main__":
    main()
