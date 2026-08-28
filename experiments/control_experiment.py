# -*- coding: utf-8 -*-
"""
CONTROL. Run the same model comparison on the controlled pilot grid, where the
configuration axis is held fixed across machines, then on MLPerf, where each cell
is a vendor's best-tuned configuration.

If interaction modelling wins on the controlled grid and loses on MLPerf, the
negative result is a property of the DATA, not of the method.
"""
import io, json, math, sys
import numpy as np
from collections import defaultdict

RNG = np.random.default_rng(0)


def fit_additive(Y, mask, iters=300):
    mu = float(Y[mask].mean())
    r = np.zeros(Y.shape[0]); c = np.zeros(Y.shape[1])
    for _ in range(iters):
        for i in range(Y.shape[0]):
            o = mask[i]
            if o.any(): r[i] = np.mean(Y[i, o] - mu - c[o])
        for j in range(Y.shape[1]):
            o = mask[:, j]
            if o.any(): c[j] = np.mean(Y[o, j] - mu - r[o])
    return mu, r, c


def fit_mf(Y, mask, rank=1, steps=3000, lr=0.02, reg=0.08, seed=0):
    rng = np.random.default_rng(seed)
    mu, r, c = fit_additive(Y, mask)
    P = 0.02 * rng.standard_normal((Y.shape[0], rank))
    Q = 0.02 * rng.standard_normal((Y.shape[1], rank))
    idx = np.argwhere(mask)
    for _ in range(steps):
        rng.shuffle(idx)
        for i, j in idx:
            e = Y[i, j] - (mu + r[i] + c[j] + P[i] @ Q[j])
            pi, qj = P[i].copy(), Q[j].copy()
            P[i] += lr * (e * qj - reg * pi); Q[j] += lr * (e * pi - reg * qj)
    return lambda i, j: mu + r[i] + c[j] + P[i] @ Q[j]


def pairwise_and_regret(Y, pred, test_cells, mask_full):
    """
    For each row, rank the held-out cells of that row. Correct-pair rate plus
    regret = true_best / true_value_of_predicted_pick  (Y is log10 ENERGY: lower better).
    """
    byrow = defaultdict(list)
    for i, j in test_cells:
        byrow[i].append(j)
    ok = tot = 0; regrets = []
    for i, js in byrow.items():
        if len(js) < 2: continue
        for a in range(len(js)):
            for b in range(a + 1, len(js)):
                ja, jb = js[a], js[b]
                t = Y[i, ja] - Y[i, jb]; p = pred(i, ja) - pred(i, jb)
                if t != 0:
                    tot += 1; ok += (t > 0) == (p > 0)
        best = min(js, key=lambda j: Y[i, j])
        pick = min(js, key=lambda j: pred(i, j))
        regrets.append(10 ** Y[i, pick] / 10 ** Y[i, best])
    return (100 * ok / tot if tot else np.nan,
            float(np.mean(regrets)) if regrets else np.nan)


def pilot_matrix():
    d = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
    rows = [r for b in d for r in b["rows"] if r.get("status") == "ok"]
    MACH = ["T4", "L4", "A10G", "L40S", "A100-40GB"]
    keys = sorted({(r["load"], r["precision"], r["batch"]) for r in rows})
    Y = np.zeros((len(keys), len(MACH)))
    ki = {k: i for i, k in enumerate(keys)}; mi = {m: j for j, m in enumerate(MACH)}
    for r in rows:
        Y[ki[(r["load"], r["precision"], r["batch"])], mi[r["machine"]]] = math.log10(r["energy_per_sample_mj"])
    return keys, MACH, Y


def run_control(hold_frac=0.30, reps=20):
    keys, MACH, Y = pilot_matrix()
    n, m = Y.shape
    full = np.ones_like(Y, dtype=bool)
    print(f"CONTROL GRID: {n} (load,precision,batch) rows x {m} machines, {Y.size} cells, fully observed")
    mu, r, c = fit_additive(Y, full)
    resid = Y - (mu + r[:, None] + c[None, :])
    ss = ((Y - Y.mean()) ** 2).sum()
    print(f"  interaction residual: {100*(resid**2).sum()/ss:.1f}% of variance, sd {resid.std():.3f} log10\n")

    res = defaultdict(list)
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        mask = np.ones_like(Y, dtype=bool)
        cells = [tuple(x) for x in np.argwhere(full)]
        rng.shuffle(cells)
        # hold out cells but keep >=2 per row so ranking is evaluable
        test = []
        per_row = defaultdict(int)
        for (i, j) in cells:
            if len(test) >= int(hold_frac * Y.size): break
            if per_row[i] < m - 2:
                test.append((i, j)); per_row[i] += 1; mask[i, j] = False
        if not mask.all(axis=1).any(): continue
        mu, r, c = fit_additive(Y, mask)
        preds = {
            "machine-only (fixed ranking)": (lambda i, j, mu=mu, c=c: mu + c[j]),
            "additive (row+column)": (lambda i, j, mu=mu, r=r, c=c: mu + r[i] + c[j]),
            "biased MF rank-1": fit_mf(Y, mask, rank=1, seed=rep),
            "biased MF rank-2": fit_mf(Y, mask, rank=2, seed=rep),
        }
        for name, p in preds.items():
            mae = float(np.mean([abs(Y[i, j] - p(i, j)) for i, j in test]))
            pa, rg = pairwise_and_regret(Y, p, test, full)
            res[name].append((mae, pa, rg))

    print(f"{'method':32} {'MAE(log10)':>11} {'pair acc %':>11} {'regret x':>10}")
    print("-" * 68)
    for name in ["machine-only (fixed ranking)", "additive (row+column)",
                 "biased MF rank-1", "biased MF rank-2"]:
        a = np.array(res[name], dtype=float)
        print(f"{name:32} {np.nanmean(a[:,0]):11.4f} {np.nanmean(a[:,1]):11.1f} {np.nanmean(a[:,2]):10.4f}")
    print(f"\n{reps} repeats, {int(hold_frac*100)}% of cells held out, >=2 cells kept per row")
    return res


if __name__ == "__main__":
    run_control()
