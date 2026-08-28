# -*- coding: utf-8 -*-
"""
Sprint 5 -- learned response model on the REAL emerald DVFS data.

Predict a workload's normalized throughput at a given GPU power cap from features a
scheduler actually has (task type, model size, power-cap ratio). The decision-relevant
test is S2: can we predict the response of an UNSEEN workload (leave-one-workload-out)?

Model ladder: M0 global-mean, M1 task-class lookup, M2 linear, M3 gradient-boosted trees.
Splits: S1 random 5-fold, S2 leave-one-workload-out (unseen model). Metric: MAE on
normalized throughput.

Invariant: on S1, M3 (GBDT) MAE <= M0 (global mean) MAE -- a fitted model must not do
worse than predicting the grand mean on seen data.
"""
from __future__ import annotations
import os, json, re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
RESULTS = os.path.join(HERE, "results")


def parse_features(name):
    task = "infer" if name.startswith("infer") else ("ft" if name.startswith("ft") else "pt")
    m = re.search(r"(\d+)b", name)
    size = int(m.group(1)) if m else 8
    return task, size


def load():
    df = pd.read_csv(os.path.join(RAW, "dvfs_sweep.csv"))
    rows = []
    for _, r in df.iterrows():
        task, size = parse_features(r["Workload"])
        rows.append(dict(wl=r["Workload"], task=task, size=size,
                         cap=r["GPU power cap"] / 400.0, y=r["normalized throughput"]))
    d = pd.DataFrame(rows)
    d = pd.concat([d, pd.get_dummies(d["task"], prefix="task")], axis=1)
    return d


FEATS = ["cap", "size"]  # + task dummies added dynamically


def feat_cols(d):
    return FEATS + [c for c in d.columns if c.startswith("task_")]


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def predict_models(tr, te, cols):
    out = {}
    # M0 global mean
    out["M0_mean"] = np.full(len(te), tr["y"].mean())
    # M1 task-class lookup (mean y per (task, cap) -- class curve)
    lut = tr.groupby(["task", "cap"])["y"].mean()
    gmean = tr["y"].mean()
    out["M1_classlut"] = np.array([lut.get((row.task, row.cap), gmean) for row in te.itertuples()])
    # M2 linear
    lr = LinearRegression().fit(tr[cols].astype(float), tr["y"])
    out["M2_linear"] = lr.predict(te[cols].astype(float))
    # M3 GBDT
    gb = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                       min_samples_leaf=3)
    gb.fit(tr[cols].astype(float), tr["y"])
    out["M3_gbdt"] = gb.predict(te[cols].astype(float))
    return out


def run():
    d = load()
    cols = feat_cols(d)
    res = {"S1_random": {}, "S2_unseen_workload": {}}

    # S1: random 5-fold
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    acc = {m: [] for m in ["M0_mean", "M1_classlut", "M2_linear", "M3_gbdt"]}
    for tri, tei in kf.split(d):
        tr, te = d.iloc[tri], d.iloc[tei]
        pred = predict_models(tr, te, cols)
        for m, p in pred.items():
            acc[m].append(mae(te["y"], p))
    res["S1_random"] = {m: float(np.mean(v)) for m, v in acc.items()}

    # S2: leave-one-workload-out
    acc = {m: [] for m in ["M0_mean", "M1_classlut", "M2_linear", "M3_gbdt"]}
    for wl in d["wl"].unique():
        tr = d[d["wl"] != wl]; te = d[d["wl"] == wl]
        pred = predict_models(tr, te, cols)
        for m, p in pred.items():
            acc[m].append(mae(te["y"], p))
    res["S2_unseen_workload"] = {m: float(np.mean(v)) for m, v in acc.items()}

    # invariant: on S1, GBDT should beat the global mean
    ok = res["S1_random"]["M3_gbdt"] <= res["S1_random"]["M0_mean"] + 1e-9
    res["invariant_gbdt_beats_mean_S1"] = ok
    with open(os.path.join(RESULTS, "learned_model.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("== Sprint 5: learned response model (real emerald DVFS, MAE on norm. throughput) ==")
    print(f"{'model':>14} {'S1 random':>10} {'S2 unseen-wl':>13}")
    for m in ["M0_mean", "M1_classlut", "M2_linear", "M3_gbdt"]:
        print(f"{m:>14} {res['S1_random'][m]:>10.4f} {res['S2_unseen_workload'][m]:>13.4f}")
    print(f"\ninvariant (GBDT<=mean on S1): {ok}")
    print("written -> results/learned_model.json")
    return res


if __name__ == "__main__":
    run()
