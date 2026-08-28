# -*- coding: utf-8 -*-
"""
A1  MLPerf Task B returned NaN for pairwise accuracy and regret.
A2  MLPerf Task C hybrid scored regret 4.269, far worse than a column-mean baseline.

Both are suspected code defects rather than findings. Diagnose, fix, re-run.
"""
import io, json, math, sys, warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from accelerator_specs import SPECS
from mlperf_experiment import build, acc_features, fit_additive

FIX = []


def note(tag, verdict, detail):
    FIX.append(dict(tag=tag, verdict=verdict, detail=detail))
    print(f"\n[{tag}] {verdict}\n    {detail}")


models, accs, Y, NPN = build("Offline", confident_only=False)
M = ~np.isnan(Y)
F = acc_features(accs)
print(f"matrix {Y.shape}, {M.sum()} observed")

# ------------------------------------------------------------------ A1
print("\n" + "=" * 84)
print("A1  why were Task B metrics NaN?")
print("=" * 84)
j0 = max(range(Y.shape[1]), key=lambda j: M[:, j].sum())
test_cells = [(i, j0) for i in range(Y.shape[0]) if M[i, j0]]
byrow = defaultdict(list)
for i, j in test_cells:
    byrow[i].append(j)
print(f"  held-out column {accs[j0]}: {len(test_cells)} test cells")
print(f"  test cells per row: {sorted(set(len(v) for v in byrow.values()))}")
note("A1", "CONFIRMED BUG in the metric, not a finding",
     "rank_metrics compares cells WITHIN a row. Holding out a whole column leaves exactly one "
     "test cell per row, so the len(js)<2 guard skips every row and both metrics return NaN. "
     "The correct machine-cold-start question is whether the held-out accelerator is ranked "
     "correctly AGAINST THE OBSERVED ones, so the comparison set must include observed cells.")


def coldstart_metrics(Y, M, pred, j0):
    """Rank the held-out column against the observed columns of the same row."""
    ok = tot = 0
    regret = []
    for i in range(Y.shape[0]):
        if not M[i, j0]:
            continue
        others = [j for j in range(Y.shape[1]) if j != j0 and M[i, j]]
        if not others:
            continue
        for jb in others:
            t = Y[i, j0] - Y[i, jb]
            p = pred(i, j0) - pred(i, jb)
            if t != 0:
                tot += 1
                ok += (t > 0) == (p > 0)
        cand = others + [j0]
        # Y is log10 efficiency here: HIGHER is better
        best = max(cand, key=lambda j: Y[i, j])
        pick = max(cand, key=lambda j: pred(i, j))
        regret.append(10 ** Y[i, best] / 10 ** Y[i, pick])
    return (100 * ok / tot if tot else float("nan"),
            float(np.mean(regret)) if regret else float("nan"), tot)


# ------------------------------------------------------------------ A2
print("\n" + "=" * 84)
print("A2  why did the hybrid collapse on Task C (whole row held out)?")
print("=" * 84)
i0 = max(range(Y.shape[0]), key=lambda i: M[i].sum())
tr_mask = M.copy(); tr_mask[i0, :] = False
mu, r, c = fit_additive(Y, tr_mask)
print(f"  held-out row {models[i0]}: fitted row effect r[{i0}] = {r[i0]:.6f}")
print(f"  other rows' effects range: {r[[i for i in range(len(r)) if i != i0]].min():.3f} "
      f"to {r[[i for i in range(len(r)) if i != i0]].max():.3f}")
note("A2", "CONFIRMED BUG, unhandled cold row",
     f"fit_additive only updates r[i] for rows with observations, so a fully held-out row keeps "
     f"r[i]=0 (measured: {r[i0]:.4f}) while real row effects span "
     f"{r[[i for i in range(len(r)) if i!=i0]].min():.2f}..{r[[i for i in range(len(r)) if i!=i0]].max():.2f}. "
     f"The hybrid adds that biased row term, the column-only baseline does not use r at all, which "
     f"is exactly why the baseline looked better. The row effect must be imputed (row mean of the "
     f"training rows, or regressed from load features) before the comparison is meaningful.")

# ------------------------------------------------------------------ re-run B and C, fixed
print("\n" + "=" * 84)
print("RE-RUN with both defects fixed")
print("=" * 84)


def hybrid(Y, mask, F, rank=2, ridge=1.0, impute_rows=True):
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mu, r, c = fit_additive(Y, mask)
    if impute_rows:                       # <-- the A2 fix
        seen_rows = np.array([mask[i].any() for i in range(Y.shape[0])])
        if seen_rows.any():
            r = np.where(seen_rows, r, r[seen_rows].mean())
    seen = np.array([mask[:, j].any() for j in range(Y.shape[1])])
    pc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ridge))
    pc.fit(F[seen], c[seen]); c_hat = pc.predict(F)
    R = np.where(mask, Y - (mu + r[:, None] + c_hat[None, :]), 0.0)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    k = min(rank, len(S)); P = U[:, :k] * S[:k]; Qs = Vt[:k].T
    pq = []
    for d in range(k):
        p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ridge))
        p.fit(F[seen], Qs[seen, d]); pq.append(p)
    Qh = np.column_stack([p.predict(F) for p in pq]) if k else np.zeros((Y.shape[1], 0))
    return lambda i, j: mu + r[i] + c_hat[j] + (P[i] @ Qh[j] if k else 0.0)


print("\nTASK B (machine cold start), metric fixed:")
cand = [j for j in range(Y.shape[1]) if M[:, j].sum() >= 4]
resB = defaultdict(list)
for j0 in cand:
    tm = M.copy(); tm[:, j0] = False
    mu, r, c = fit_additive(Y, tm)
    seen_r = np.array([tm[i].any() for i in range(Y.shape[0])])
    r_imp = np.where(seen_r, r, r[seen_r].mean())
    same = [j for j in range(Y.shape[1]) if j != j0 and tm[:, j].any()
            and SPECS[accs[j]]["klass"] == SPECS[accs[j0]]["klass"]]
    cprox = float(np.mean([c[j] for j in same])) if same else 0.0
    preds = {
        "no column info": (lambda i, j, mu=mu, r=r_imp, c=c: mu + r[i] + (c[j] if j != j0 else 0.0)),
        "class-mean proxy": (lambda i, j, mu=mu, r=r_imp, c=c, cp=cprox: mu + r[i] + (c[j] if j != j0 else cp)),
        "hybrid (specs, rank-2)": hybrid(Y, tm, F, rank=2),
    }
    for name, p in preds.items():
        a, rg, tot = coldstart_metrics(Y, M, p, j0)
        if not math.isnan(a):
            resB[name].append((a, rg))
print(f"  {len(cand)} held-out accelerators")
print(f"{'method':28} {'pair acc %':>11} {'regret x':>10}")
for name in ["no column info", "class-mean proxy", "hybrid (specs, rank-2)"]:
    a = np.array(resB[name], dtype=float)
    print(f"{name:28} {a[:,0].mean():11.1f} {a[:,1].mean():10.3f}")

print("\nTASK C (load cold start), row-effect imputation fixed:")
candR = [i for i in range(Y.shape[0]) if M[i].sum() >= 4]
resC = defaultdict(list)
for i0 in candR:
    tm = M.copy(); tm[i0, :] = False
    mu, r, c = fit_additive(Y, tm)
    test = [(i0, j) for j in range(Y.shape[1]) if M[i0, j]]
    preds = {
        "column-only baseline": (lambda i, j, mu=mu, c=c: mu + c[j]),
        "hybrid, BROKEN (r=0)": hybrid(Y, tm, F, rank=2, impute_rows=False),
        "hybrid, FIXED (r imputed)": hybrid(Y, tm, F, rank=2, impute_rows=True),
    }
    for name, p in preds.items():
        byr = defaultdict(list)
        for i, j in test:
            byr[i].append(j)
        ok = tot = 0; rg = []
        for i, js in byr.items():
            for a_ in range(len(js)):
                for b_ in range(a_ + 1, len(js)):
                    t = Y[i, js[a_]] - Y[i, js[b_]]; pp = p(i, js[a_]) - p(i, js[b_])
                    if t != 0:
                        tot += 1; ok += (t > 0) == (pp > 0)
            best = max(js, key=lambda j: Y[i, j]); pick = max(js, key=lambda j: p(i, j))
            rg.append(10 ** Y[i, best] / 10 ** Y[i, pick])
        if tot:
            resC[name].append((100 * ok / tot, float(np.mean(rg))))
print(f"{'method':28} {'pair acc %':>11} {'regret x':>10}")
for name in ["column-only baseline", "hybrid, BROKEN (r=0)", "hybrid, FIXED (r imputed)"]:
    a = np.array(resC[name], dtype=float)
    print(f"{name:28} {a[:,0].mean():11.1f} {a[:,1].mean():10.3f}")

json.dump(FIX, io.open("experiments/results/audit_mlperf_bugs.json", "w", encoding="utf-8"), indent=1)
print(f"\nsaved -> experiments/results/audit_mlperf_bugs.json")
