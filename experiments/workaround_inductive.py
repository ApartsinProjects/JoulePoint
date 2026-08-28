# -*- coding: utf-8 -*-
"""
WORKAROUND for the negative result.

Diagnosis: free per-entity embeddings need ~29 parameters to express a 2.5%-of-
variance interaction from ~84 training cells, so regularisation either crushes the
interaction to zero (MF collapses to additive) or it fits noise.

Fix: the interaction is not arbitrary. Its rank-1 row factor correlates with
precision (-0.52) and batch size (+0.39), so parameterise the row factor as a
LINEAR FUNCTION OF CONFIGURATION FEATURES instead of a free vector. That reduces
the interaction to a handful of parameters and makes it defined for loads never
seen in training.

    E(load, machine) = mu + r(load) + c(machine) + (w . x_load) * v_machine

Evaluated under leave-one-load-family-out, the hard cold-start split.
"""
import io, json, math
import numpy as np
from collections import defaultdict
from sklearn.linear_model import RidgeCV

MACH = ["T4", "L4", "A10G", "L40S", "A100-40GB"]


def load_grid():
    d = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
    rows = [r for b in d for r in b["rows"] if r.get("status") == "ok"]
    keys = sorted({(r["load"], r["precision"], r["batch"]) for r in rows})
    Y = np.zeros((len(keys), len(MACH)))
    ki = {k: i for i, k in enumerate(keys)}
    mi = {m: j for j, m in enumerate(MACH)}
    for r in rows:
        Y[ki[(r["load"], r["precision"], r["batch"])], mi[r["machine"]]] = math.log10(r["energy_per_sample_mj"])
    return keys, Y


def config_features(keys):
    """Descriptors available BEFORE running anything: precision, batch, arch family."""
    fam = sorted({k[0] for k in keys})
    X = []
    for k in keys:
        is32 = 1.0 if k[1] == "fp32" else 0.0
        lb = math.log2(k[2])
        row = [is32, lb, is32 * lb, lb ** 2]
        row += [1.0 if k[0] == f else 0.0 for f in fam]   # architecture family one-hot
        X.append(row)
    return np.array(X), fam


def fit_additive(Y, mask):
    mu = float(Y[mask].mean())
    r = np.zeros(Y.shape[0]); c = np.zeros(Y.shape[1])
    for _ in range(300):
        for i in range(Y.shape[0]):
            o = mask[i]
            if o.any(): r[i] = np.mean(Y[i, o] - mu - c[o])
        for j in range(Y.shape[1]):
            o = mask[:, j]
            if o.any(): c[j] = np.mean(Y[o, j] - mu - r[o])
    return mu, r, c


def evaluate(Y, pred_rows, test_rows):
    """pred_rows: (n_test, n_mach) predictions. Ranking + regret on held-out rows."""
    ok = tot = 0; regrets = []
    for t, i in enumerate(test_rows):
        for a in range(len(MACH)):
            for b in range(a + 1, len(MACH)):
                tr = Y[i, a] - Y[i, b]; pr = pred_rows[t, a] - pred_rows[t, b]
                if tr != 0:
                    tot += 1; ok += (tr > 0) == (pr > 0)
        best = int(np.argmin(Y[i])); pick = int(np.argmin(pred_rows[t]))
        regrets.append(10 ** Y[i, pick] / 10 ** Y[i, best])
    return 100 * ok / tot, float(np.mean(regrets)), float(np.mean([r > 1.0001 for r in regrets]) * 100)


def main():
    keys, Y = load_grid()
    X, fam = config_features(keys)
    loads = sorted({k[0] for k in keys})
    print(f"grid {Y.shape[0]} rows x {Y.shape[1]} machines; leave-one-load-family-out\n")

    agg = defaultdict(list)
    for Lout in loads:
        tr = [i for i, k in enumerate(keys) if k[0] != Lout]
        te = [i for i, k in enumerate(keys) if k[0] == Lout]
        mask = np.zeros_like(Y, dtype=bool); mask[tr] = True

        mu, r, c = fit_additive(Y, mask)
        # row effect for an unseen load: regress r on config features
        rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], r[tr])
        r_te = rr.predict(X[te])

        base = mu + r_te[:, None] + c[None, :]

        # --- interaction, learned in feature space ---
        R = Y[tr] - (mu + r[tr][:, None] + c[None, :])
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        v1 = Vt[0]                                  # machine-side interaction direction
        s_tr = R @ v1                               # per-row interaction score
        sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], s_tr)
        s_te = sm.predict(X[te])
        inter = base + np.outer(s_te, v1)

        # --- rank-2 version ---
        v2 = Vt[1]
        s2_tr = R @ v2
        sm2 = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], s2_tr)
        inter2 = inter + np.outer(sm2.predict(X[te]), v2)

        preds = {
            "machine-only (fixed ranking)": np.tile(mu + c, (len(te), 1)),
            "additive (row+column)": base,
            "inductive interaction rank-1": inter,
            "inductive interaction rank-2": inter2,
            "oracle": Y[te],
        }
        for name, P in preds.items():
            pa, rg, badpct = evaluate(Y, P, te)
            agg[name].append((pa, rg, badpct))

    print(f"{'method':32} {'pair acc %':>11} {'regret x':>10} {'suboptimal picks %':>20}")
    print("-" * 78)
    for name in ["machine-only (fixed ranking)", "additive (row+column)",
                 "inductive interaction rank-1", "inductive interaction rank-2", "oracle"]:
        a = np.array(agg[name], dtype=float)
        print(f"{name:32} {a[:,0].mean():11.1f} {a[:,1].mean():10.4f} {a[:,2].mean():20.1f}")
    print("\nper-load-family regret (rank-1 inductive vs additive):")
    for i, L in enumerate(loads):
        print(f"  {L:12} additive {agg['additive (row+column)'][i][1]:.4f}   "
              f"inductive {agg['inductive interaction rank-1'][i][1]:.4f}")


if __name__ == "__main__":
    main()
