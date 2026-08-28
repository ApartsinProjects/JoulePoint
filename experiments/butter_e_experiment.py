# -*- coding: utf-8 -*-
"""
Method vs baselines on public data (BUTTER-E).

TASK. For a workload never seen in training, choose the machine class (CPU node or
2xV100 GPU node) that minimises training energy. This is the load-cold-start regime.

WHY THIS DATASET. BUTTER-E measures 13,121 workloads on BOTH machine classes with
node-level watt-meters, so a measured oracle exists for every test workload and
regret is exactly computable.

KEY STRUCTURAL POINT. With two machines, a purely additive model
    log E(load, machine) = mu + b(load) + b(machine)
predicts a machine ranking that is identical for every workload, so it is provably
equivalent to a fixed "always machine X" policy. All decision value therefore lives
in the interaction term. This experiment measures how much that is worth.
"""
import io, json, zipfile, csv, math, sys
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

RNG = np.random.default_rng(0)
Z = "data/butter-e/runs_with_standardized_energy.csv.zip"


def load():
    recs = []
    with zipfile.ZipFile(Z).open("runs_with_standardized_energy.csv") as f:
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace")):
            try:
                e = float(r.get("std_energy") or r.get("energy"))
                size = float(r["size"]); depth = float(r["depth"])
            except (TypeError, ValueError):
                continue
            if e <= 0 or size <= 0:
                continue
            recs.append((r["dataset"], r["shape"], size, depth, r["is_gpu"] == "1", e))
    return recs


def build():
    recs = load()
    agg = defaultdict(list)
    for ds, sh, size, depth, gpu, e in recs:
        agg[(ds, sh, size, depth, gpu)].append(e)
    med = {k: float(np.median(v)) for k, v in agg.items()}
    loads = {}
    for (ds, sh, size, depth, gpu), v in med.items():
        loads.setdefault((ds, sh, size, depth), {})[gpu] = v
    both = {k: v for k, v in loads.items() if True in v and False in v}
    print(f"workloads with both machine classes: {len(both)}")
    keys = sorted(both)
    X_cat = [[k[0], k[1]] for k in keys]
    X_num = np.array([[math.log10(k[2]), k[3], math.log10(k[2]) / max(k[3], 1)] for k in keys])
    e_cpu = np.array([both[k][False] for k in keys])
    e_gpu = np.array([both[k][True] for k in keys])
    groups = np.array([k[0] for k in keys])          # group by dataset: harder, more honest split
    return keys, X_cat, X_num, e_cpu, e_gpu, groups


def features(X_cat, X_num, enc=None, fit=False):
    if fit:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_cat)
    return np.hstack([enc.transform(X_cat), X_num]), enc


def evaluate(name, pick_gpu, e_cpu, e_gpu, res):
    """pick_gpu: boolean array, True = send to GPU node."""
    chosen = np.where(pick_gpu, e_gpu, e_cpu)
    best = np.minimum(e_cpu, e_gpu)
    worst = np.maximum(e_cpu, e_gpu)
    opt = float((chosen == best).mean())
    regret = float((chosen.sum() - best.sum()) / best.sum() * 100)
    # energy relative to the production default (always GPU)
    vs_gpu = float((chosen.sum() - e_gpu.sum()) / e_gpu.sum() * 100)
    res.append((name, opt * 100, regret, vs_gpu))
    return res


def main():
    keys, X_cat, X_num, e_cpu, e_gpu, groups = build()
    n = len(keys)
    gpu_better = e_gpu < e_cpu
    print(f"oracle: GPU optimal for {gpu_better.mean()*100:.1f}% of workloads")
    print(f"headroom: always-GPU costs {100*(e_gpu.sum()-np.minimum(e_cpu,e_gpu).sum())/np.minimum(e_cpu,e_gpu).sum():.1f}% "
          f"more than oracle; always-CPU costs "
          f"{100*(e_cpu.sum()-np.minimum(e_cpu,e_gpu).sum())/np.minimum(e_cpu,e_gpu).sum():.1f}% more\n")

    gkf = GroupKFold(n_splits=5)
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for fold, (tr, te) in enumerate(gkf.split(X_num, groups=groups)):
        Xtr_c = [X_cat[i] for i in tr]; Xte_c = [X_cat[i] for i in te]
        Ftr, enc = features(Xtr_c, X_num[tr], fit=True)
        Fte, _ = features(Xte_c, X_num[te], enc=enc)
        ec_tr, eg_tr = e_cpu[tr], e_gpu[tr]
        ec_te, eg_te = e_cpu[te], e_gpu[te]
        res = []

        # --- baselines ---
        evaluate("B1 always GPU (production default)", np.ones(len(te), bool), ec_te, eg_te, res)
        evaluate("B2 always CPU", np.zeros(len(te), bool), ec_te, eg_te, res)
        maj = (eg_tr < ec_tr).mean() > 0.5
        evaluate("B3 global majority (train-fitted)", np.full(len(te), maj), ec_te, eg_te, res)
        # additive model in log space == constant machine offset == always-one-machine
        delta_tr = np.log(ec_tr) - np.log(eg_tr)
        evaluate("B4 additive / no interaction", np.full(len(te), delta_tr.mean() > 0), ec_te, eg_te, res)
        # size threshold rule fitted on train
        best_thr, best_score = None, -1
        for thr in np.linspace(X_num[tr, 0].min(), X_num[tr, 0].max(), 60):
            p = X_num[tr, 0] >= thr
            s = ((np.where(p, eg_tr, ec_tr)) == np.minimum(ec_tr, eg_tr)).mean()
            if s > best_score:
                best_score, best_thr = s, thr
        evaluate("B5 size-threshold rule", X_num[te, 0] >= best_thr, ec_te, eg_te, res)

        # --- independent per-machine regression (the 'just regress' arm) ---
        m_c = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Ftr, np.log(ec_tr))
        m_g = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Ftr, np.log(eg_tr))
        evaluate("M1 independent regression per machine", m_g.predict(Fte) < m_c.predict(Fte), ec_te, eg_te, res)

        # --- direct interaction model: regress the log-ratio ---
        m_d = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Ftr, np.log(ec_tr) - np.log(eg_tr))
        evaluate("M2 interaction model (log-ratio)", m_d.predict(Fte) > 0, ec_te, eg_te, res)

        # linear version, to show it is not just model capacity
        m_l = Ridge(alpha=1.0).fit(Ftr, np.log(ec_tr) - np.log(eg_tr))
        evaluate("M3 interaction model, linear", m_l.predict(Fte) > 0, ec_te, eg_te, res)

        evaluate("oracle (measured)", gpu_better[te], ec_te, eg_te, res)

        for name, o, r, v in res:
            a = acc[name]
            a[0] += o; a[1] += r; a[2] += v; a[3] += 1

    print(f"{'policy':40} {'optimal %':>10} {'regret %':>10} {'vs always-GPU %':>16}")
    print("-" * 80)
    order = ["B1 always GPU (production default)", "B2 always CPU", "B3 global majority (train-fitted)",
             "B4 additive / no interaction", "B5 size-threshold rule",
             "M1 independent regression per machine", "M2 interaction model (log-ratio)",
             "M3 interaction model, linear", "oracle (measured)"]
    out = {}
    for name in order:
        o, r, v, c = acc[name]
        print(f"{name:40} {o/c:10.1f} {r/c:10.1f} {v/c:16.1f}")
        out[name] = {"optimal_pct": o/c, "regret_pct": r/c, "vs_always_gpu_pct": v/c}
    json.dump(out, io.open("experiments/butter_e_experiment_results.json", "w", encoding="utf-8"), indent=1)
    print("\n5-fold GroupKFold, grouped by dataset (test datasets unseen in training)")


if __name__ == "__main__":
    main()
