# -*- coding: utf-8 -*-
"""
Can collaborative filtering recover the decisive descriptor an operator is never told?

This is the setting Table 10 says is the common one. An orchestrated container declares a
memory request, a device count and an image name. It does NOT declare numerical precision,
and Section 7's descriptor ablation shows precision is the descriptor that decides the
placement: withhold it and the model collapses exactly onto the additive ceiling of 81.7 per
cent, the same score as knowing nothing at all. Full descriptors reach 87.9.

So the operator is in neither of the two regimes tested so far. Descriptors are not fully
available, so the feature-based model of Section 7.3 is crippled. They are not fully absent
either, so throwing them away for a purely free-embedding model discards real information and
forfeits cold start.

The natural estimator is a hybrid, and it is what this script tests:

    loading_i  =  <w, x_i^obs>  +  u_i

The first term uses whatever IS declared. The second is a free per-workload factor learned
from that workload's own observed placements. With no history u_i shrinks to zero and the
model degrades gracefully to features-only, so it can never be worse than what the operator
could already do. With history, u_i has the opportunity to encode what the missing descriptor
would have said, inferring precision from behaviour rather than from declaration.

PREDICTIONS STATED IN ADVANCE:
  H1  features-only WITHOUT precision reproduces the additive ceiling, since Table 5 already
      shows the surviving descriptors carry no decision information on their own. If it does
      not, the descriptor set here differs from Table 5's and the comparison is invalid.
  H2  the hybrid climbs above that ceiling as observations per workload increase, and
      approaches the full-descriptor model that was told the precision outright.
  H3  at zero observations per workload the hybrid must EQUAL features-only exactly. This is
      the graceful-degradation guarantee and it is a correctness check, not a result.
  H4  the hybrid should not exceed the full-descriptor model by much. Recovering a hidden
      descriptor from behaviour should approach knowing it, not beat it. Substantially beating
      it would suggest leakage.

Invariants:
  S1  no estimator may read a cell it did not observe
  S2  the fitted precision-free feature set genuinely excludes precision, asserted directly
  S3  accuracy is measured only on held-out cells
"""
import io, json, math, sys, warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

SANITY = []
def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

keys, Ylog, Tput = load_grid()
Y = np.array(Ylog)
R, C = Y.shape
FAM = sorted({k[0] for k in keys})

def feats(k, with_precision):
    load, prec, batch = k
    f = [1.0 if load == g else 0.0 for g in FAM]
    f.append(math.log2(batch) / 7.0)
    if with_precision:
        f.append(1.0 if prec == "fp32" else 0.0)
    return np.array(f)

Xfull = np.array([feats(k, True) for k in keys])
Xpart = np.array([feats(k, False) for k in keys])
sane("S2 the partial descriptor set genuinely excludes precision",
     Xpart.shape[1] == Xfull.shape[1] - 1,
     "{} features with precision, {} without".format(Xfull.shape[1], Xpart.shape[1]))

def fit_add(Yo, M):
    mu = Yo[M].mean(); r = np.zeros(R); c = np.zeros(C)
    for _ in range(80):
        for i in range(R):
            s = M[i]; r[i] = (Yo[i][s] - mu - c[s]).mean() if s.any() else 0.0
        for j in range(C):
            s = M[:, j]; c[j] = (Yo[s, j] - mu - r[s]).mean() if s.any() else 0.0
    return mu, r, c

def fit(Yo, M, X, free, ridge=3e-2, it=150):
    """loading = <w,x> + (u if free else 0); alternating least squares on observed cells."""
    mu, r, c = fit_add(Yo, M)
    Res = np.array([Yo[i] - mu - r[i] - c for i in range(R)])
    P = X.shape[1]
    rng = np.random.default_rng(0)
    # BUG FOUND AND FIXED. The first version initialised w and u to zero and updated v FIRST.
    # That makes z = X@w + u = 0, so v collapses to 0, so the w-solve sees an all-zero normal
    # matrix and returns 0, so z stays 0 forever: the alternating least squares starts exactly
    # on a degenerate fixed point and never leaves it. Every variant then reduced to the
    # additive model and all four columns of the results table were byte-identical, which is
    # what exposed it. The fix is to give the loading a non-zero start and update the machine
    # sensitivities from it, rather than the other way round.
    w = rng.normal(0, 0.01, P); u = np.zeros(R); v = rng.normal(0, 0.01, C)
    for _ in range(it):
        z = X @ w + u
        if not np.any(np.abs(z) > 1e-12):
            z = rng.normal(0, 0.01, R)
        for j in range(C):
            s = M[:, j]
            den = (z[s] ** 2).sum() + ridge
            v[j] = (z[s] * Res[s, j]).sum() / den if den > 0 else 0.0
        A = np.zeros((P, P)); b = np.zeros(P)
        for i in range(R):
            for j in range(C):
                if M[i, j]:
                    A += (v[j] ** 2) * np.outer(X[i], X[i])
                    b += v[j] * (Res[i, j] - u[i] * v[j]) * X[i]
        w = np.linalg.solve(A + ridge * np.eye(P), b)
        if free:
            zx = X @ w
            for i in range(R):
                s = M[i]
                den = (v[s] ** 2).sum() + ridge
                u[i] = ((Res[i][s] - zx[i] * v[s]) * v[s]).sum() / den if den > 0 else 0.0
        else:
            u[:] = 0.0
    z = X @ w + u
    return np.array([mu + r[i] + c + z[i] * v for i in range(R)])

def acc(Pred, M):
    # CORRECTED. The earlier version overwrote revealed cells with ground truth before
    # scoring. A parallel experiment showed that substitution lets an additive model beat a
    # hindsight-optimal fixed ranking and breaks Proposition 1, because it mixes a fitted
    # surface with exact values on an inconsistent footing. Every estimator is now scored on
    # its own fitted matrix, and only pairs with at least one held-out cell are counted.
    Pred = Pred.copy()
    ok = n = 0
    for i in range(R):
        hid = ~M[i]
        for a in range(C):
            for b in range(a + 1, C):
                if hid[a] or hid[b]:
                    n += 1; ok += (Pred[i, a] < Pred[i, b]) == (Y[i, a] < Y[i, b])
    return ok / n if n else float("nan")

print("\nour grid, {} workload-configurations x {} accelerators".format(R, C))
print("{:>9}{:>14}{:>16}{:>16}{:>12}".format(
    "obs/row", "fixed", "partial (no prec)", "HYBRID", "full (told)"))
rows = []
for k in range(0, C):
    got = defaultdict(list)
    for sd in range(40):
        rng = np.random.default_rng(17 * sd + k)
        M = np.zeros((R, C), bool)
        for i in range(R):
            if k:
                M[i, rng.choice(C, size=k, replace=False)] = True
        if not M.any():
            M[0, 0] = True
        Yo = np.where(M, Y, 0.0)
        mu, r, c = fit_add(Yo, M)
        got["fixed"].append(acc(np.array([mu + r[i] + c for i in range(R)]), M))
        got["partial"].append(acc(fit(Yo, M, Xpart, False), M))
        got["hybrid"].append(acc(fit(Yo, M, Xpart, True), M))
        got["full"].append(acc(fit(Yo, M, Xfull, False), M))
    row = {"obs_per_row": k}
    for kk, v in got.items():
        row[kk] = float(np.mean(v)); row[kk + "_sd"] = float(np.std(v))
    rows.append(row)
    print("{:>9}{:>14.1%}{:>16.1%}{:>16.1%}{:>12.1%}".format(
        k, row["fixed"], row["partial"], row["hybrid"], row["full"]))

print()
sane("H1 features without precision sit at the fixed-ranking ceiling",
     abs(rows[1]["partial"] - rows[1]["fixed"]) < 0.03,
     "partial {:.1%} vs fixed {:.1%} at one observation; Table 5 reports both at 81.7%".format(
         rows[1]["partial"], rows[1]["fixed"]))
gain = [r["hybrid"] - r["partial"] for r in rows]
sane("H2 the hybrid climbs above features-only as history accumulates",
     max(gain) > 0.02,
     "largest gain {:+.1f} points at {} observations per workload".format(
         100 * max(gain), rows[int(np.argmax(gain))]["obs_per_row"]))
sane("H3 with no history the hybrid equals features-only exactly",
     abs(rows[0]["hybrid"] - rows[0]["partial"]) < 1e-9,
     "hybrid {:.4f} vs partial {:.4f}".format(rows[0]["hybrid"], rows[0]["partial"]))
best = max(rows, key=lambda r: r["hybrid"])
sane("H4 the hybrid approaches but does not greatly exceed being told the descriptor",
     best["hybrid"] <= best["full"] + 0.05,
     "best hybrid {:.1%} against full-descriptor {:.1%} at the same density".format(
         best["hybrid"], best["full"]))

rec = [(r["obs_per_row"],
        (r["hybrid"] - r["partial"]) / (r["full"] - r["partial"]) if r["full"] > r["partial"] else float("nan"))
       for r in rows]
print("fraction of the missing-descriptor gap recovered from behaviour alone:")
for k, f in rec:
    if not math.isnan(f):
        print("   {} observations per workload: {:.0%}".format(k, f))

OUT = dict(shape=[R, C], sweep=rows, recovered_fraction=rec, sanity=SANITY,
           note="partial descriptors withhold numerical precision, the descriptor Table 5 "
                "identifies as decisive and Table 10 says orchestrated containers do not declare")
json.dump(OUT, io.open("experiments/results/cf_partial_observability.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/cf_partial_observability.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
