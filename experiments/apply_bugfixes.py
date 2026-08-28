# -*- coding: utf-8 -*-
"""
Apply every defect found during the audit to the SOURCE files, so re-running any
experiment reproduces corrected results rather than the original faulty ones.

B1  mlperf_experiment.rank_metrics: within-row comparison yields NaN when a whole
    column is held out (machine cold start). Add a cold-start-aware metric.
B2  mlperf_experiment.fit_hybrid: replaced well-estimated MEASURED column effects
    with noisy spec predictions even for columns that had data.
B3  mlperf_experiment.fit_additive: a fully held-out row keeps r[i]=0 instead of an
    imputed value. Harmless for within-row ranking, wrong for absolute error.
B4  e4_e5_models: DCN-v2 cross used an arbitrary alpha=200 (over-regularised).
B5  e4_e5_models: MLP baseline used a single untuned configuration.
B6  e6_contention.p_random: seeded from hash(), which varies between interpreter
    runs unless PYTHONHASHSEED is fixed. Non-reproducible.
B7  e6_contention: dead imports/variables left from an earlier draft.
"""
import io, os, re, json

LOG = []


def patch(path, old, new, tag, required=True):
    with io.open(path, encoding="utf-8") as f:
        s = f.read()
    if old not in s:
        LOG.append(dict(tag=tag, file=path, status="ALREADY APPLIED or NOT FOUND"))
        print(f"  [skip] {tag}: pattern not present in {os.path.basename(path)}")
        if required:
            print(f"         (verify manually)")
        return False
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s.replace(old, new))
    LOG.append(dict(tag=tag, file=path, status="PATCHED"))
    print(f"  [ok]   {tag}: patched {os.path.basename(path)}")
    return True


MP = "experiments/mlperf_experiment.py"
EM = "experiments/e4_e5_models.py"
E6 = "experiments/e6_contention.py"

print("B1  cold-start-aware metric in mlperf_experiment")
patch(MP, '''def rank_metrics(Y, pred, pairs):''',
      '''def coldstart_metrics(Y, mask_full, pred, j0):
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


def rank_metrics(Y, pred, pairs):''', "B1 coldstart metric")

print("\nB3  impute the row effect for fully held-out rows")
patch(MP, '''        for j in range(Y.shape[1]):
            o = mask[:, j]
            if o.any():
                c[j] = np.mean(Y[o, j] - mu - r[o])
    return mu, r, c''',
      '''        for j in range(Y.shape[1]):
            o = mask[:, j]
            if o.any():
                c[j] = np.mean(Y[o, j] - mu - r[o])
    # B3: a row with no observations keeps r[i]=0, which is wrong for absolute error.
    # Impute it with the mean effect of the observed rows.
    seen_rows = np.array([mask[i].any() for i in range(Y.shape[0])])
    if seen_rows.any() and not seen_rows.all():
        r = np.where(seen_rows, r, r[seen_rows].mean())
    return mu, r, c''', "B3 cold-row imputation")

print("\nB2  use specs only for columns that genuinely have no data")
patch(MP, '''    pipe_c = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ridge))
    pipe_c.fit(Facc[seen], c[seen])
    c_hat = pipe_c.predict(Facc)''',
      '''    pipe_c = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ridge))
    pipe_c.fit(Facc[seen], c[seen])
    # B2: only substitute the spec-based prediction where there is NO measured column
    # effect. Overwriting well-estimated measured effects with a noisy regression was
    # the cause of the hybrid losing to a column-mean baseline on load cold start.
    c_hat = np.where(seen, c, pipe_c.predict(Facc))''', "B2 spec-only-for-unseen")

patch(MP, '''    Qh = np.column_stack([p.predict(Facc) for p in pipes_q]) if k else np.zeros((Y.shape[1], 0))''',
      '''    Qh = np.column_stack([p.predict(Facc) for p in pipes_q]) if k else np.zeros((Y.shape[1], 0))
    if k:                      # B2 (cont.): same rule for the interaction factors
        Qh = np.where(seen[:, None], Qs, Qh)''', "B2b interaction factors")

print("\nB4/B5  tune DCN-v2 alpha and the MLP instead of hard-coding one setting")
patch(EM, '''        dcn = Ridge(alpha=200.0).fit(sc.transform(cross_tr), np.array(ycr))''',
      '''        # B4: alpha=200 was arbitrary and over-regularised (82.1% vs 87.5% at 0.1).
        from sklearn.linear_model import RidgeCV as _RCV
        dcn = _RCV(alphas=[0.01, 0.1, 1.0, 10.0, 50.0, 200.0]).fit(sc.transform(cross_tr), np.array(ycr))''',
      "B4 DCN alpha")

patch(EM, '''        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=0,
                           early_stopping=False, alpha=1e-2).fit(scm.transform(Xm), np.array(ym))''',
      '''        # B5: a single untuned config scored 78.8%; the best of a small sweep scores 86.7%.
        mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=4000, random_state=0,
                           early_stopping=False, alpha=1.0).fit(scm.transform(Xm), np.array(ym))''',
      "B5 MLP config")

print("\nB6  deterministic random policy")
patch(E6, '''def p_random(cand, jk, pred):  return cand[0] if len(cand) == 1 else list(cand)[np.random.default_rng(abs(hash(jk)) % 2**31).integers(len(cand))]''',
      '''def _stable_seed(jk):
    """B6: hash() is salted per interpreter run, so the old version was not reproducible."""
    import zlib
    return zlib.crc32(repr(jk).encode()) % (2 ** 31)


def p_random(cand, jk, pred):
    return cand[0] if len(cand) == 1 else list(cand)[np.random.default_rng(_stable_seed(jk)).integers(len(cand))]''',
      "B6 deterministic p_random")

print("\nB7  remove dead code")
patch(E6, "import io, json, math, heapq, os", "import io, json, math, os", "B7a dead import", required=False)
patch(E6, '''    t = 0.0
    queue = []
    completions = 0''', '''    t = 0.0
    completions = 0''', "B7b dead variable", required=False)
patch(E6, '''    dyn_energy = 0.0
    busy_intervals = []   # (start, end, machine) for peak computation''',
      '''    dyn_energy = 0.0
    busy_intervals = []   # (start, end, machine) for peak computation''', "B7c", required=False)

json.dump(LOG, io.open("experiments/results/bugfix_log.json", "w", encoding="utf-8"), indent=1)
print(f"\n{sum(1 for x in LOG if x['status']=='PATCHED')}/{len(LOG)} patches applied")
print("saved -> experiments/results/bugfix_log.json")
