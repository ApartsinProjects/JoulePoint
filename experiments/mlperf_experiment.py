# -*- coding: utf-8 -*-
"""
Matrix completion and machine cold start on the public MLPerf Power matrix.

Matrix: rows = MLPerf models (loads), columns = accelerators (machines),
entries = log10(Inference_per_Joule), a measured energy-efficiency figure.

Tasks
  A  pair completion   : hold out random observed cells
  B  machine cold start: hold out an ENTIRE accelerator column
  C  load cold start   : hold out an ENTIRE model row

Caveat carried through the whole analysis: MLPerf submissions are vendor-selected,
so the unobserved cells are not missing at random. Absolute numbers are optimistic;
the comparison BETWEEN methods on the same cells is the meaningful part.
"""
import io, json, sys, math
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
from accelerator_specs import SPECS

RNG = np.random.default_rng(0)


def build(scenario="Offline", confident_only=False):
    recs = json.load(io.open("data/mlperf-power/records.json", encoding="utf-8"))
    cells = defaultdict(list)
    for r in recs:
        if not r.get("has_power") or r.get("Scenario") != scenario:
            continue
        a = r.get("accelerator_model_name")
        m = r.get("MlperfModel")
        ipj = r.get("Inference_per_Joule")
        if not a or a in ("-", "N/A", "") or not m or not ipj:
            continue
        if a not in SPECS:
            continue
        if confident_only and not SPECS[a]["confident"]:
            continue
        try:
            v = float(ipj)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        npn = r.get("accelerators_per_node") or 1
        cells[(m, a)].append((v, float(npn)))
    obs = {k: (float(np.median([x[0] for x in v])), float(np.median([x[1] for x in v])))
           for k, v in cells.items()}
    models = sorted({k[0] for k in obs})
    accs = sorted({k[1] for k in obs})
    Y = np.full((len(models), len(accs)), np.nan)
    NPN = np.full((len(models), len(accs)), np.nan)
    mi = {m: i for i, m in enumerate(models)}
    ai = {a: j for j, a in enumerate(accs)}
    for (m, a), (v, n) in obs.items():
        Y[mi[m], ai[a]] = math.log10(v)
        NPN[mi[m], ai[a]] = n
    return models, accs, Y, NPN


def acc_features(accs):
    vend = sorted({SPECS[a]["vendor"] for a in accs})
    klass = sorted({SPECS[a]["klass"] for a in accs})
    form = sorted({str(SPECS[a]["form"]) for a in accs})
    F = []
    for a in accs:
        s = SPECS[a]
        row = [
            math.log10(s["tdp_w"]) if s["tdp_w"] else np.nan,
            math.log10(s["mem_gb"]) if s["mem_gb"] else np.nan,
            math.log10(s["bw_gbs"]) if s["bw_gbs"] else np.nan,
            (s["year"] - 2018) if s["year"] else np.nan,
        ]
        row += [1.0 if s["vendor"] == v else 0.0 for v in vend]
        row += [1.0 if s["klass"] == k else 0.0 for k in klass]
        row += [1.0 if str(s["form"]) == f else 0.0 for f in form]
        F.append(row)
    return np.array(F, dtype=float)


# ---------------- models ----------------
def fit_additive(Y, mask, iters=200):
    """mu + row + col, fitted by alternating means on observed entries only."""
    mu = np.nanmean(np.where(mask, Y, np.nan))
    r = np.zeros(Y.shape[0]); c = np.zeros(Y.shape[1])
    for _ in range(iters):
        for i in range(Y.shape[0]):
            o = mask[i]
            if o.any():
                r[i] = np.mean(Y[i, o] - mu - c[o])
        for j in range(Y.shape[1]):
            o = mask[:, j]
            if o.any():
                c[j] = np.mean(Y[o, j] - mu - r[o])
    # B3: a row with no observations keeps r[i]=0, which is wrong for absolute error.
    # Impute it with the mean effect of the observed rows.
    seen_rows = np.array([mask[i].any() for i in range(Y.shape[0])])
    if seen_rows.any() and not seen_rows.all():
        r = np.where(seen_rows, r, r[seen_rows].mean())
    return mu, r, c


def fit_biased_mf(Y, mask, rank=2, steps=4000, lr=0.02, reg=0.06, seed=0):
    rng = np.random.default_rng(seed)
    mu, r, c = fit_additive(Y, mask)
    n, m = Y.shape
    P = 0.03 * rng.standard_normal((n, rank))
    Q = 0.03 * rng.standard_normal((m, rank))
    idx = np.argwhere(mask)
    for _ in range(steps):
        rng.shuffle(idx)
        for i, j in idx:
            e = Y[i, j] - (mu + r[i] + c[j] + P[i] @ Q[j])
            pi, qj = P[i].copy(), Q[j].copy()
            P[i] += lr * (e * qj - reg * pi)
            Q[j] += lr * (e * pi - reg * qj)
            r[i] += lr * (e - reg * r[i])
            c[j] += lr * (e - reg * c[j])
    return lambda i, j: mu + r[i] + c[j] + P[i] @ Q[j]


def fit_hybrid(Y, mask, Facc, rank=2, ridge=1.0):
    """
    Machine-side features drive the column effect and the column embedding, so an
    unobserved accelerator still gets a prediction. Row effects stay free.
    """
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mu, r, c = fit_additive(Y, mask)
    seen = np.array([mask[:, j].any() for j in range(Y.shape[1])])
    # column bias as a function of hardware features
    pipe_c = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ridge))
    pipe_c.fit(Facc[seen], c[seen])
    # B2: only substitute the spec-based prediction where there is NO measured column
    # effect. Overwriting well-estimated measured effects with a noisy regression was
    # the cause of the hybrid losing to a column-mean baseline on load cold start.
    c_hat = np.where(seen, c, pipe_c.predict(Facc))
    # residual after additive-with-predicted-column, factorised, column factors also from features
    R = Y - (mu + r[:, None] + c_hat[None, :])
    Rf = np.where(mask, R, 0.0)
    U, S, Vt = np.linalg.svd(Rf, full_matrices=False)
    k = min(rank, len(S))
    P = U[:, :k] * S[:k]
    Qs = Vt[:k].T
    pipes_q = []
    for d in range(k):
        p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ridge))
        p.fit(Facc[seen], Qs[seen, d])
        pipes_q.append(p)
    Qh = np.column_stack([p.predict(Facc) for p in pipes_q]) if k else np.zeros((Y.shape[1], 0))
    if k:                      # B2 (cont.): same rule for the interaction factors
        Qh = np.where(seen[:, None], Qs, Qh)
    return lambda i, j: mu + r[i] + c_hat[j] + (P[i] @ Qh[j] if k else 0.0)


# ---------------- evaluation ----------------
def coldstart_metrics(Y, mask_full, pred, j0):
    """
    Machine cold start: rank the held-out column against the OBSERVED columns of the
    same row. The plain within-row metric cannot do this, because holding out a whole
    column leaves one test cell per row and every row is skipped (returns NaN).
    Y here is log10 efficiency, so HIGHER is better.
    """
    ok = tot = 0
    regret = []
    for i in range(Y.shape[0]):
        if not mask_full[i, j0]:
            continue
        others = [j for j in range(Y.shape[1]) if j != j0 and mask_full[i, j]]
        if not others:
            continue
        for jb in others:
            t = Y[i, j0] - Y[i, jb]
            p = pred(i, j0) - pred(i, jb)
            if t != 0:
                tot += 1
                ok += (t > 0) == (p > 0)
        cand = others + [j0]
        best = max(cand, key=lambda j: Y[i, j])
        pick = max(cand, key=lambda j: pred(i, j))
        regret.append(10 ** Y[i, best] / 10 ** Y[i, pick])
    return (100 * ok / tot if tot else float("nan"),
            float(np.mean(regret)) if regret else float("nan"), tot)


def rank_metrics(Y, pred, pairs):
    """pairwise ranking accuracy + placement regret over held-out cells, per row."""
    correct = tot = 0
    regret = []
    byrow = defaultdict(list)
    for (i, j) in pairs:
        byrow[i].append(j)
    for i, js in byrow.items():
        if len(js) < 2:
            continue
        for a in range(len(js)):
            for b in range(a + 1, len(js)):
                ja, jb = js[a], js[b]
                t = Y[i, ja] - Y[i, jb]
                p = pred(i, ja) - pred(i, jb)
                if t != 0:
                    tot += 1
                    correct += (t > 0) == (p > 0)
        # regret: efficiency lost by picking argmax(pred) instead of argmax(true)
        # Y is log10 efficiency, higher is better
        best = max(js, key=lambda j: Y[i, j])
        pick = max(js, key=lambda j: pred(i, j))
        regret.append(10 ** Y[i, best] / 10 ** Y[i, pick])
    return (100 * correct / tot if tot else float("nan"),
            float(np.mean(regret)) if regret else float("nan"),
            tot, len(regret))


def mae(Y, pred, cells):
    return float(np.mean([abs(Y[i, j] - pred(i, j)) for i, j in cells]))


def run(scenario="Offline", confident_only=False, folds=5, seed=0):
    models, accs, Y, NPN = build(scenario, confident_only)
    mask_all = ~np.isnan(Y)
    n_obs = int(mask_all.sum())
    print(f"\n{'='*78}\nscenario={scenario}  confident_only={confident_only}")
    print(f"matrix {Y.shape[0]} models x {Y.shape[1]} accelerators, "
          f"{n_obs} observed ({100*n_obs/Y.size:.1f}% dense)")
    Facc = acc_features(accs)
    rng = np.random.default_rng(seed)

    # ---------- Task A: random cell hold-out ----------
    obs_cells = [tuple(x) for x in np.argwhere(mask_all)]
    rng.shuffle(obs_cells)
    res = defaultdict(list)
    for f in range(folds):
        test = [c for k, c in enumerate(obs_cells) if k % folds == f]
        tr_mask = mask_all.copy()
        for i, j in test:
            tr_mask[i, j] = False
        if not all(tr_mask.any(axis=1)) or not all(tr_mask.any(axis=0)):
            pass
        mu, r, c = fit_additive(Y, tr_mask)
        preds = {
            "global mean": (lambda i, j, mu=mu: mu),
            "accelerator-only (column)": (lambda i, j, mu=mu, c=c: mu + c[j]),
            "model-only (row)": (lambda i, j, mu=mu, r=r: mu + r[i]),
            "additive (row+column)": (lambda i, j, mu=mu, r=r, c=c: mu + r[i] + c[j]),
            "biased MF rank-1": fit_biased_mf(Y, tr_mask, rank=1, seed=f),
            "biased MF rank-2": fit_biased_mf(Y, tr_mask, rank=2, seed=f),
            "hybrid (features, rank-2)": fit_hybrid(Y, tr_mask, Facc, rank=2),
        }
        for name, p in preds.items():
            acc_, reg, npairs, nrows = rank_metrics(Y, p, test)
            res[name].append((mae(Y, p, test), acc_, reg))
    print(f"\nTASK A  pair completion ({folds}-fold random cell hold-out)")
    print(f"{'method':30} {'MAE(log10)':>11} {'pair acc %':>11} {'regret x':>10}")
    for name in res:
        a = np.array(res[name], dtype=float)
        print(f"{name:30} {np.nanmean(a[:,0]):11.3f} {np.nanmean(a[:,1]):11.1f} {np.nanmean(a[:,2]):10.3f}")

    # ---------- Task B: machine cold start ----------
    print(f"\nTASK B  machine cold start (hold out an entire accelerator column)")
    cand = [j for j in range(Y.shape[1]) if mask_all[:, j].sum() >= 4]
    print(f"  accelerators with >=4 observed models: {len(cand)}")
    resB = defaultdict(list)
    for j0 in cand:
        tr_mask = mask_all.copy()
        tr_mask[:, j0] = False
        test = [(i, j0) for i in range(Y.shape[0]) if mask_all[i, j0]]
        mu, r, c = fit_additive(Y, tr_mask)
        # a column never observed has no c[j]; additive/MF fall back to mu+r
        preds = {
            "additive, no column info": (lambda i, j, mu=mu, r=r: mu + r[i]),
            "class-mean column proxy": None,  # filled below
            "hybrid (features, rank-2)": fit_hybrid(Y, tr_mask, Facc, rank=2),
        }
        # proxy: mean column effect of same-class accelerators
        same = [j for j in range(Y.shape[1]) if j != j0 and tr_mask[:, j].any()
                and SPECS[accs[j]]["klass"] == SPECS[accs[j0]]["klass"]]
        cprox = float(np.mean([c[j] for j in same])) if same else 0.0
        preds["class-mean column proxy"] = (lambda i, j, mu=mu, r=r, cp=cprox: mu + r[i] + cp)
        for name, p in preds.items():
            # B1: use the cold-start-aware metric; rank_metrics returns NaN here
            a_, reg, npairs = coldstart_metrics(Y, mask_all, p, j0)
            resB[name].append((mae(Y, p, test), a_, reg))
    print(f"{'method':30} {'MAE(log10)':>11} {'pair acc %':>11} {'regret x':>10}")
    for name in resB:
        a = np.array(resB[name], dtype=float)
        print(f"{name:30} {np.nanmean(a[:,0]):11.3f} {np.nanmean(a[:,1]):11.1f} {np.nanmean(a[:,2]):10.3f}")

    # ---------- Task C: load cold start ----------
    print(f"\nTASK C  load cold start (hold out an entire model row)")
    candR = [i for i in range(Y.shape[0]) if mask_all[i].sum() >= 4]
    print(f"  models with >=4 observed accelerators: {len(candR)}")
    resC = defaultdict(list)
    for i0 in candR:
        tr_mask = mask_all.copy()
        tr_mask[i0, :] = False
        test = [(i0, j) for j in range(Y.shape[1]) if mask_all[i0, j]]
        mu, r, c = fit_additive(Y, tr_mask)
        preds = {
            "global mean": (lambda i, j, mu=mu: mu),
            "accelerator-only (column)": (lambda i, j, mu=mu, c=c: mu + c[j]),
            "hybrid (features, rank-2)": fit_hybrid(Y, tr_mask, Facc, rank=2),
        }
        for name, p in preds.items():
            a_, reg, npairs, nrows = rank_metrics(Y, p, test)
            resC[name].append((mae(Y, p, test), a_, reg))
    print(f"{'method':30} {'MAE(log10)':>11} {'pair acc %':>11} {'regret x':>10}")
    for name in resC:
        a = np.array(resC[name], dtype=float)
        print(f"{name:30} {np.nanmean(a[:,0]):11.3f} {np.nanmean(a[:,1]):11.1f} {np.nanmean(a[:,2]):10.3f}")


if __name__ == "__main__":
    run("Offline", confident_only=False)
    run("Offline", confident_only=True)
