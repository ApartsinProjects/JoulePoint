# -*- coding: utf-8 -*-
"""A2 re-diagnosis: the row-effect hypothesis was wrong. Find the real cause."""
import io, json, sys, warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlperf_experiment import build, acc_features, fit_additive

models, accs, Y, NPN = build("Offline", confident_only=False)
M = ~np.isnan(Y); F = acc_features(accs)

i0 = max(range(Y.shape[0]), key=lambda i: M[i].sum())
tm = M.copy(); tm[i0, :] = False
mu, r, c = fit_additive(Y, tm)

print("STEP 1  does the row effect even matter for within-row ranking?")
print("  A row-constant shifts every machine in that row equally, so it cancels in")
print("  pairwise comparisons AND in argmax. Predicted effect of imputing r: none.")
print("  Observed effect in the previous run: none (76.5/4.269 identical). Hypothesis refuted.\n")

print("STEP 2  what actually differs between the hybrid and the column-only baseline?")
seen = np.array([tm[:, j].any() for j in range(Y.shape[1])])
pc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
pc.fit(F[seen], c[seen]); c_hat = pc.predict(F)
err = c_hat[seen] - c[seen]
print(f"  columns with data in this split: {seen.sum()}/{len(seen)}")
print(f"  spec-predicted column effect vs MEASURED column effect:")
print(f"    RMSE {np.sqrt((err**2).mean()):.3f} log10, max |err| {np.abs(err).max():.3f} log10")
print(f"    measured c spread: {c[seen].min():.2f} .. {c[seen].max():.2f}")
print(f"    correlation: {np.corrcoef(c_hat[seen], c[seen])[0,1]:.3f}")
print("\n  ROOT CAUSE: on LOAD cold start every column still has data, so c is well estimated.")
print("  The hybrid discards it and substitutes a noisy spec-based prediction. That is a design")
print("  error, not a property of the data: specs should only be used for columns with NO data.\n")

def hybrid(Y, mask, F, rank=2, use_specs_only_for_unseen=True):
    mu, r, c = fit_additive(Y, mask)
    seen = np.array([mask[:, j].any() for j in range(Y.shape[1])])
    p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    p.fit(F[seen], c[seen]); pred_c = p.predict(F)
    c_use = np.where(seen, c, pred_c) if use_specs_only_for_unseen else pred_c
    R = np.where(mask, Y - (mu + r[:, None] + c_use[None, :]), 0.0)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    k = min(rank, len(S)); P = U[:, :k] * S[:k]; Qs = Vt[:k].T
    pq = []
    for d in range(k):
        q = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        q.fit(F[seen], Qs[seen, d]); pq.append(q)
    Qh = np.column_stack([q.predict(F) for q in pq]) if k else np.zeros((Y.shape[1], 0))
    Qu = np.where(seen[:, None], Qs, Qh) if k else Qh
    return lambda i, j: mu + r[i] + c_use[j] + (P[i] @ Qu[j] if k else 0.0)

print("STEP 3  re-run Task C with specs used ONLY for unseen columns")
candR = [i for i in range(Y.shape[0]) if M[i].sum() >= 4]
res = defaultdict(list)
for i0 in candR:
    tmk = M.copy(); tmk[i0, :] = False
    mu, r, c = fit_additive(Y, tmk)
    test = [j for j in range(Y.shape[1]) if M[i0, j]]
    preds = {
        "column-only baseline": (lambda i, j, mu=mu, c=c: mu + c[j]),
        "hybrid, specs replace everything (old)": hybrid(Y, tmk, F, use_specs_only_for_unseen=False),
        "hybrid, specs only for unseen (fixed)": hybrid(Y, tmk, F, use_specs_only_for_unseen=True),
    }
    for name, p in preds.items():
        ok = tot = 0
        for a in range(len(test)):
            for b in range(a+1, len(test)):
                t = Y[i0, test[a]] - Y[i0, test[b]]; pp = p(i0, test[a]) - p(i0, test[b])
                if t != 0: tot += 1; ok += (t > 0) == (pp > 0)
        best = max(test, key=lambda j: Y[i0, j]); pick = max(test, key=lambda j: p(i0, j))
        if tot: res[name].append((100*ok/tot, 10**Y[i0,best]/10**Y[i0,pick]))
print(f"{'method':42} {'pair acc %':>11} {'regret x':>10}")
for name in ["column-only baseline", "hybrid, specs replace everything (old)",
             "hybrid, specs only for unseen (fixed)"]:
    a = np.array(res[name], dtype=float)
    print(f"{name:42} {a[:,0].mean():11.1f} {a[:,1].mean():10.3f}")
