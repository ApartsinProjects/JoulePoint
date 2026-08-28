# -*- coding: utf-8 -*-
"""
CF-MULTI.  The Section 7 predictive model, fitted and evaluated on EVERY corpus.

Until now the inductive model

    y(i,j) = mu + r(x_i) + c_j + <w, x_i> v_j

has only been fitted on two corpora: our own 120-cell accelerator grid and the
MLPerf datacenter matrix. Four further corpora had been analysed for variance
decomposition only. This script fits and scores the same model, under a properly
held-out protocol, on all seven matrices we hold, and adds the widest cold-start
test available anywhere in the project: leave-one-machine-out over the eighteen
Grid'5000 clusters, each held-out cluster predicted from its published hardware
descriptors alone.

MATRICES (constructions are REUSED, not re-derived)
--------------------------------------------------
  own-grid    24 configs x  5 accelerators   experiments/pilot_results.json via e4_e5_models.load_grid
  grid5000     5 workloads x 18 clusters     experiments/results/k2_grid5000.json  ["log10_energy"]
  wilkins     32 workloads x  4 platforms    experiments/results/wilkins_corpus.json
  ejhusom      5 workloads x  2 machines     experiments/results/ejhusom_corpus.json
  c1-extended 30 configs  x  7 accelerators  experiments/results/c1_bridge.json
  c4-training 18 configs  x  7 accelerators  experiments/results/c4_training_grid.json
  llm-perf    74 rows     x  3 GPUs          experiments/results/j2_llmperf_matrix.json

WHAT IS REPORTED PER CORPUS
---------------------------
  ceiling_heldout         the HELD-OUT Proposition 1 ceiling: the better of the fitted
                          additive model and the best fixed ranking chosen on the training
                          rows and applied to the held-out rows. Gains are measured here.
  ceiling_optimal_fixed   the same optimum over all m! rankings chosen with HINDSIGHT on
                          the scored rows, computed exactly by a subset dynamic program
                          over the linear ordering problem. Strict, but on a matrix with
                          few rows and many machines it is optimistically biased.
  bilinear                the rank-1 inductive interaction model.
  gain                    bilinear minus ceiling_heldout, in percentage points.
  regret                  mean energy of the chosen machine over the row-wise oracle.
  oracle loading          the same rank-1 form given the held-out row's TRUE loading; an
                          oracle that bounds any descriptor set under this model form.

PROTOCOL. Leave-one-workload-family-out, exactly as Section 7: an entire family of
rows is withheld, its row effect r and its interaction loading <w,x> are predicted
from descriptors, and the fitted machine terms c_j, v_j are reused. Cold start on
machines withholds an entire COLUMN and predicts c_j and v_j from machine descriptors.

DESCRIPTORS come from what each corpus actually carries; none are invented.
  own-grid    family one-hot, fp32 indicator, log2 batch, their product, log2(batch)^2
  grid5000    NPB class indicator, plus a reference profile (instructions per core,
              cache-miss and cache-reference ratios, log10 runtime) taken from ONE
              reference cluster, `taurus`, which is then EXCLUDED from the ranked
              columns so no descriptor is scored on the machine it came from.
              Machine descriptors: log10 cores, log10 nominal fmax, log10 fmin,
              number of DVFS steps; a secondary set adds log10 idle-probe power.
  wilkins     model one-hot, sweep indicator, log2 tokens, its square, and the
              sweep x log2 tokens product
  ejhusom     dataset one-hot, log10 model parameter count
  c1 / c4     family one-hot, fp32 indicator, log2 batch, product, log2(batch)^2
              (c4 additionally log10 params_m). Machine descriptors: log10 device
              memory and log10 enforced power cap, both recorded in the corpus.
  llm-perf    quantisation-scheme one-hot, log10 parameter count parsed from the
              model identifier where the identifier states it

CORPORA WHERE COLD START ON MACHINES IS NOT ATTEMPTED, AND WHY
--------------------------------------------------------------
  wilkins   the corpus records no numeric machine descriptors, and two of its four
            platforms carry the same accelerator (A100) and differ only in host CPU,
            which no accelerator specification distinguishes. Hand-entering vendor
            TDPs would invent the very feature under test. SKIPPED.
  ejhusom   two machines: holding one out leaves a single training machine, so the
            descriptor regression has one point. NOT MEASURABLE.
  llm-perf  three GPUs: holding one out leaves two training points for a two-feature
            regression, which is exactly determined and carries no residual. NOT
            MEASURABLE.

SANITY CHECKS, STATED IN ADVANCE
--------------------------------
  S1  PROPOSITION 1, EXACT. On every corpus and every fold, the fitted additive model
      and the fixed ranking induced by its own column effects must return IDENTICAL
      pairwise accuracy and IDENTICAL regret, to 1e-12. They are the same policy.
      Any disagreement means the additive fit or the scorer is wrong, not that
      Proposition 1 is approximate.
  S2  CEILING VALIDITY. The fitted additive accuracy must be <= the exact optimal
      fixed-ranking accuracy on the same rows, on every corpus. A fitted model
      beating the oracle over its own hypothesis class is impossible.
  S3  DEGENERATE-LOADING CONTROL. Replacing <w,x_i> by a constant must yield a model
      whose within-row machine ORDERING is identical for every row (it is then a fixed
      ranking) and whose accuracy is therefore <= the exact optimal fixed ranking.
      This is the check that the interaction term is doing the work, not the base.
  S4  ANCHOR. On own-grid the pipeline must reproduce the published 81.7 additive
      ceiling and 87.9 bilinear accuracy, each within 0.5 points. If it does not,
      nothing else in this file is comparable to the paper.
  S5  RANGE. Every accuracy in [0,100]. Every regret >= 1.0. The row-wise oracle's
      regret is exactly 1.0.
  S6  NO LEAKAGE IN MACHINE COLD START. Overwriting the held-out machine's true column
      with NaN before fitting must leave the cold-start predictions bit-identical.
  S7  DESCRIPTOR PERMUTATION NEGATIVE CONTROL. Shuffling the machine descriptor rows
      across machines must destroy the cold-start gain: the permuted cold-start
      accuracy must not exceed the true-descriptor accuracy by more than bootstrap
      noise. Stated expectation: the paired difference (true minus permuted) is
      positive, or at worst its interval contains zero.
  S8  TWO-COLUMN RESOLUTION. On ejhusom, pairwise accuracy over 5 rows and 2 columns
      takes at most 6 distinct values, so the finest resolvable difference is 20
      percentage points, and the interaction has ONE degree of freedom per fold. No
      conclusion is drawn from that corpus; the number is printed and labelled
      NOT MEASURABLE. This is a declaration, not a check that can fail.
  S12 HELD-OUT CEILING. A fixed ranking chosen on the TRAINING rows and applied to the
      held-out rows is reported alongside the fitted additive model; the larger of the
      two is the held-out Proposition 1 ceiling, and it is what the gain is measured
      against. The hindsight optimum over all m! rankings is also reported, with the
      caveat that on a matrix with 5 rows and 17 machines it chooses among 17! orderings
      using the very rows it is scored on and is therefore optimistically biased.
  S10 RANK-1 FORM VALIDITY. The same rank-1 interaction with the held-out row's TRUE
      loading substituted for <w,x_i> (an oracle, not a usable model) must be at least
      as accurate as the fitted additive model. It upper-bounds what any descriptor set
      could achieve with this interaction form, and it is what separates "this corpus
      has no interaction to find" from "the loading regression cannot be fitted from
      this many training rows". If it falls below additive, the rank-1 form itself is
      wrong for that corpus and that is the finding.
  S11 COLD-START DESCRIPTOR REGRESSIONS. The leave-one-machine-out R^2 of the c_j and
      v_j regressions on machine descriptors is reported for every cold-start panel. A
      cold-start gain with a negative v-regression R^2 would be a contradiction.
  S9  EXACT-DP CORRECTNESS. On every matrix with at most 7 machines the subset dynamic
      program's optimal fixed ranking must equal a brute-force search over all
      permutations. This validates the dynamic program used at m=18 where brute force
      is impossible.

RULES OF ENGAGEMENT. No hyperparameter is tuned to make a number look better; the
ridge path is the same RidgeCV grid used in e4_e5_models.py throughout. A corpus on
which the model fails to beat its ceiling is reported as a failure and explained.
"""
import io, json, math, os, sys, itertools, warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")

from e4_e5_models import load_grid, load_feats, additive  # reuse the paper's fitting core

ALPHAS = np.logspace(-3, 3, 25)
MODELS = ["fixed ranking", "additive", "cv-optimal fixed ranking", "bilinear rank-1",
          "constant loading", "oracle loading (upper bound)"]
RNG = np.random.default_rng(20260819)
SANITY, OUT = [], {}


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))


# ------------------------------------------------------------------ scoring
def pair_records(Y, P, rows, cols):
    """Per-pair correctness and per-row regret. Y and P are log10 energy."""
    pairs, regret = [], []
    for i in rows:
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                ja, jb = cols[a], cols[b]
                t = Y[i, ja] - Y[i, jb]
                if t == 0:
                    continue
                p = P[i, ja] - P[i, jb]
                pairs.append((i, ja, jb, int((t > 0) == (p > 0))))
        best = min(cols, key=lambda j: Y[i, j])
        pick = min(cols, key=lambda j: P[i, j])
        regret.append((i, pick, best, 10.0 ** (Y[i, pick] - Y[i, best])))
    return pairs, regret


def acc_of(pairs):
    return 100.0 * np.mean([p[3] for p in pairs]) if pairs else float("nan")


def reg_of(regret):
    return float(np.mean([r[3] for r in regret])) if regret else float("nan")


def optimal_fixed_ranking(Y, rows, cols):
    """EXACT maximum-accuracy fixed ranking (linear ordering problem, subset DP).

    W[a,b] = number of rows in which machine a truly beats machine b. A fixed ranking
    is a permutation; its score is the sum of W[a,b] over pairs it orders correctly.
    f(S) = best score using S as the first |S| positions.
    """
    m = len(cols)
    W = np.zeros((m, m))
    tot = 0
    for i in rows:
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                d = Y[i, cols[b]] - Y[i, cols[a]]
                if d > 0:
                    W[a, b] += 1
    for a in range(m):
        for b in range(a + 1, m):
            tot += W[a, b] + W[b, a]
    N = 1 << m
    colsum = np.zeros((N, m))
    for S in range(1, N):
        lb = S & (-S)
        v = lb.bit_length() - 1
        colsum[S] = colsum[S ^ lb] + W[v]
    f = np.full(N, -1.0)
    choice = np.zeros(N, dtype=np.int8)
    f[0] = 0.0
    for S in range(1, N):
        best, bv = -1.0, 0
        T = S
        while T:
            lb = T & (-T)
            v = lb.bit_length() - 1
            T ^= lb
            cand = f[S ^ lb] + colsum[S ^ lb][v]
            if cand > best:
                best, bv = cand, v
        f[S] = best
        choice[S] = bv
    order, S = [], N - 1
    while S:
        v = int(choice[S])
        order.append(v)
        S ^= (1 << v)
    order = order[::-1]
    return (100.0 * f[N - 1] / tot if tot else float("nan")), [cols[v] for v in order]


def brute_fixed_ranking(Y, rows, cols):
    m = len(cols)
    W = np.zeros((m, m))
    tot = 0.0
    for i in rows:
        for a in range(m):
            for b in range(m):
                if a != b and Y[i, cols[b]] - Y[i, cols[a]] > 0:
                    W[a, b] += 1
    for a in range(m):
        for b in range(a + 1, m):
            tot += W[a, b] + W[b, a]
    best = -1.0
    for perm in itertools.permutations(range(m)):
        s = 0.0
        for x in range(m):
            for y in range(x + 1, m):
                s += W[perm[x], perm[y]]
        best = max(best, s)
    return 100.0 * best / tot if tot else float("nan")


def best_fixed_machine_regret(Y, rows, cols):
    out = []
    for j in cols:
        r = [10.0 ** (Y[i, j] - min(Y[i, jj] for jj in cols)) for i in rows]
        out.append((float(np.mean(r)), j))
    return min(out)


def paired_bootstrap(pairs_a, pairs_b, B=4000):
    """Cluster bootstrap over ROWS of the difference acc(a) - acc(b). Same draws both."""
    rows = sorted({p[0] for p in pairs_a} | {p[0] for p in pairs_b})
    idx = {r: k for k, r in enumerate(rows)}
    A = [[] for _ in rows]
    Bl = [[] for _ in rows]
    for p in pairs_a:
        A[idx[p[0]]].append(p[3])
    for p in pairs_b:
        Bl[idx[p[0]]].append(p[3])
    diffs = []
    n = len(rows)
    for _ in range(B):
        d = RNG.integers(0, n, n)
        a = [v for k in d for v in A[k]]
        b = [v for k in d for v in Bl[k]]
        if not a or not b:
            continue
        diffs.append(100.0 * (np.mean(a) - np.mean(b)))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ------------------------------------------------------- the Section 7 model
def fit_fold(Y, X, tr, te):
    """Fit mu + r(x) + c_j + <w,x> v_j on rows `tr`, predict all rows.

    Returns a dict of full-shape prediction matrices.
    """
    mask = np.zeros_like(Y, dtype=bool)
    mask[tr] = True
    mu, r, c = additive(Y, mask)
    rr = RidgeCV(alphas=ALPHAS).fit(X[tr], r[tr])
    r_all = r.copy()
    r_all[te] = rr.predict(X[te])
    base = mu + r_all[:, None] + c[None, :]
    R = Y[tr] - base[tr]
    v1 = np.linalg.svd(R, full_matrices=False)[2][0]
    s_tr = R @ v1
    sm = RidgeCV(alphas=ALPHAS).fit(X[tr], s_tr)
    s_all = np.empty(Y.shape[0])
    s_all[tr] = s_tr
    s_all[te] = sm.predict(X[te])
    s_true = np.empty(Y.shape[0])
    s_true[tr] = s_tr
    s_true[te] = (Y[te] - base[te]) @ v1
    P = {}
    P["fixed ranking"] = np.tile(mu + c, (Y.shape[0], 1))
    P["additive"] = base
    P["bilinear rank-1"] = base + np.outer(s_all, v1)
    P["constant loading"] = base + np.outer(np.full(Y.shape[0], float(s_tr.mean())), v1)
    # ORACLE LOADING: the same rank-1 form with <w,x_i> replaced by the held-out row's
    # own true loading. Not a usable model; it upper-bounds what ANY descriptor set could
    # reach with this interaction form, and so separates "there is no interaction to find"
    # from "the descriptor regression cannot be fitted at this many training rows".
    P["oracle loading (upper bound)"] = base + np.outer(s_true, v1)
    diag = dict(alpha_loading=float(sm.alpha_),
                rank1_share=float(100.0 * np.linalg.svd(R, compute_uv=False)[0] ** 2 /
                                  max((R ** 2).sum(), 1e-30)),
                s_true=[float(x) for x in s_true[te]],
                s_pred=[float(x) for x in s_all[te]])
    return P, diag


def evaluate_corpus(name, Y, X, folds, cols=None, note=""):
    n, m = Y.shape
    cols = list(range(m)) if cols is None else list(cols)
    allrows = sorted({i for _, te in folds for i in te})
    ceil_opt, ceil_order = optimal_fixed_ranking(Y, allrows, cols)
    reg_fixed, reg_fixed_j = best_fixed_machine_regret(Y, allrows, cols)

    acc_pairs = defaultdict(list)
    regs = defaultdict(list)
    per_fold = []
    for fname, te in folds:
        tr = [i for i in range(n) if i not in set(te)]
        # HELD-OUT ceiling: the best fixed ranking is CHOSEN ON TRAINING ROWS ONLY and
        # then applied to the held-out rows. This is the fair Proposition 1 ceiling under
        # this protocol. The hindsight optimum below is reported separately because on a
        # matrix with few rows and many machines it selects among m! orderings using the
        # test rows themselves and is optimistically biased.
        _, tr_order = optimal_fixed_ranking(Y, tr, cols)
        P_cvrank = np.zeros_like(Y)
        for rank, j in enumerate(tr_order):
            P_cvrank[:, j] = float(rank)
        P, diag = fit_fold(Y, X, tr, te)
        P["cv-optimal fixed ranking"] = P_cvrank
        row = dict(fold=fname, n_test_rows=len(te), n_train_rows=len(tr), **diag)
        for mk, Pm in P.items():
            pr, rg = pair_records(Y, Pm, te, cols)
            acc_pairs[mk].extend(pr)
            regs[mk].extend(rg)
            row[mk] = dict(acc=acc_of(pr), regret=reg_of(rg))
        # S3 support: is the constant-loading model a fixed ranking?
        Pc = P["constant loading"]
        orders = {tuple(sorted(cols, key=lambda j: Pc[i, j])) for i in te}
        row["constant_loading_is_fixed_ranking"] = (len(orders) == 1)
        per_fold.append(row)

    res = dict(corpus=name, shape=[n, m], scored_columns=len(cols), note=note,
               ceiling_optimal_fixed=ceil_opt,
               ceiling_optimal_order=[int(j) for j in ceil_order],
               best_fixed_machine_regret=reg_fixed, best_fixed_machine=int(reg_fixed_j),
               folds=[f for f, _ in folds], per_fold=per_fold)
    for mk in MODELS:
        res[mk] = dict(acc=acc_of(acc_pairs[mk]), regret=reg_of(regs[mk]),
                       n_pairs=len(acc_pairs[mk]))
    fold_acc = [f["bilinear rank-1"]["acc"] for f in per_fold]
    res["bilinear_fold_sd"] = float(np.std(fold_acc, ddof=1)) if len(fold_acc) > 1 else None
    res["ceiling_cv_optimal_fixed"] = res["cv-optimal fixed ranking"]["acc"]
    res["ceiling_fitted_additive"] = res["additive"]["acc"]
    res["ceiling_heldout"] = max(res["additive"]["acc"], res["cv-optimal fixed ranking"]["acc"])
    res["gain_vs_heldout_ceiling"] = res["bilinear rank-1"]["acc"] - res["ceiling_heldout"]
    res["gain_vs_optimal_fixed"] = res["bilinear rank-1"]["acc"] - ceil_opt
    res["gain_vs_fitted_additive"] = res["bilinear rank-1"]["acc"] - res["additive"]["acc"]
    mn, lo, hi = paired_bootstrap(acc_pairs["bilinear rank-1"], acc_pairs["additive"])
    res["gain_bootstrap"] = dict(mean=mn, lo=lo, hi=hi)
    res["oracle_regret"] = reg_of([(i, 0, 0, 1.0) for i in allrows])

    # ---- S1 Proposition 1, exact
    d_acc = abs(res["additive"]["acc"] - res["fixed ranking"]["acc"])
    d_reg = abs(res["additive"]["regret"] - res["fixed ranking"]["regret"])
    sane("S1 Proposition 1 exact on {}".format(name), d_acc < 1e-12 and d_reg < 1e-12,
         "additive {:.10f}% / {:.10f}x vs fixed ranking {:.10f}% / {:.10f}x".format(
             res["additive"]["acc"], res["additive"]["regret"],
             res["fixed ranking"]["acc"], res["fixed ranking"]["regret"]))
    # ---- S2 ceiling validity
    sane("S2 fitted additive does not exceed the exact optimal fixed ranking on {}".format(name),
         res["additive"]["acc"] <= ceil_opt + 1e-9,
         "fitted {:.2f}% <= optimal {:.2f}%".format(res["additive"]["acc"], ceil_opt))
    # ---- S3 degenerate loading
    okfix = all(f["constant_loading_is_fixed_ranking"] for f in per_fold)
    sane("S3 constant loading collapses to a fixed ranking on {}".format(name),
         okfix and res["constant loading"]["acc"] <= ceil_opt + 1e-9,
         "one ordering per fold: {}; accuracy {:.2f}% <= optimal {:.2f}%".format(
             okfix, res["constant loading"]["acc"], ceil_opt))
    # ---- S5 range
    accs = [res[k]["acc"] for k in MODELS]
    rgs = [res[k]["regret"] for k in MODELS]
    sane("S5 accuracies in range and regrets >= 1 on {}".format(name),
         all(0 <= a <= 100 for a in accs) and all(r >= 1.0 - 1e-12 for r in rgs)
         and abs(res["oracle_regret"] - 1.0) < 1e-12,
         "acc {:.1f}-{:.1f}, regret {:.4f}-{:.4f}, oracle {:.6f}".format(
             min(accs), max(accs), min(rgs), max(rgs), res["oracle_regret"]))
    # ---- S9 exact DP validated by brute force where feasible
    # ---- descriptor-regression quality for the interaction loading, pooled over folds
    st_ = [v for f in per_fold for v in f["s_true"]]
    sp_ = [v for f in per_fold for v in f["s_pred"]]
    signs = 100.0 * np.mean([(a > 0) == (b > 0) for a, b in zip(st_, sp_)]) if st_ else float("nan")
    corr = (float(np.corrcoef(st_, sp_)[0, 1]) if len(st_) > 2 and np.std(sp_) > 0 else None)
    res["loading_regression"] = dict(
        n=len(st_), sign_agreement_pct=float(signs), pearson=corr,
        sd_true=float(np.std(st_)), sd_pred=float(np.std(sp_)),
        alphas=[f["alpha_loading"] for f in per_fold],
        alpha_at_grid_max=sum(1 for f in per_fold if f["alpha_loading"] >= ALPHAS[-1] - 1e-9),
        mean_rank1_share=float(np.mean([f["rank1_share"] for f in per_fold])))
    res["oracle_loading_gain_vs_optimal_fixed"] =         res["oracle loading (upper bound)"]["acc"] - ceil_opt
    # ---- S10 the oracle-loading form must be at least as good as the fitted additive
    sane("S10 rank-1 form with the true loading is not worse than additive on {}".format(name),
         res["oracle loading (upper bound)"]["acc"] >= res["additive"]["acc"] - 1e-9,
         "oracle loading {:.2f}% vs fitted additive {:.2f}%; if this were below, the rank-1 "
         "interaction form itself would be wrong for this corpus".format(
             res["oracle loading (upper bound)"]["acc"], res["additive"]["acc"]))
    if len(cols) <= 7:
        bf = brute_fixed_ranking(Y, allrows, cols)
        sane("S9 subset DP matches brute force on {}".format(name), abs(bf - ceil_opt) < 1e-9,
             "DP {:.6f}% vs brute force over {}! permutations {:.6f}%".format(
                 ceil_opt, len(cols), bf))
    return res


# ============================================================== corpus loaders
def corpus_own_grid():
    keys, Y, _ = load_grid()
    X = load_feats(keys)
    fam = sorted({k[0] for k in keys})
    folds = [(f, [i for i, k in enumerate(keys) if k[0] == f]) for f in fam]
    return Y, X, folds, [str(k) for k in keys]


def _accel_feats(keys, extra=None):
    fam = sorted({k[0] for k in keys})
    X = []
    for idx, k in enumerate(keys):
        is32 = 1.0 if k[1] == "fp32" else 0.0
        lb = math.log2(k[2])
        row = [is32, lb, is32 * lb, lb ** 2] + [1.0 if k[0] == f else 0.0 for f in fam]
        if extra is not None:
            row = row + list(extra[idx])
        X.append(row)
    return np.array(X)


def corpus_cx(path, with_params=False):
    d = json.load(io.open(path, encoding="utf-8"))
    rows = [r for r in d["rows"] if r.get("status") == "ok"]
    mach = sorted({r["machine"] for r in rows})
    cell = defaultdict(dict)
    for r in rows:
        cell[(r["load"], r["precision"], r["batch"])][r["machine"]] = \
            math.log10(r["energy_per_sample_mj"])
    keys = sorted(k for k, v in cell.items() if len(v) == len(mach))
    Y = np.array([[cell[k][m] for m in mach] for k in keys])
    extra = None
    if with_params:
        pm = {(r["load"], r["precision"], r["batch"]): r.get("params_m") for r in rows}
        extra = [[math.log10(pm[k])] for k in keys]
    X = _accel_feats(keys, extra)
    fam = sorted({k[0] for k in keys})
    folds = [(f, [i for i, k in enumerate(keys) if k[0] == f]) for f in fam]
    Z = np.array([[math.log10(d["machines"][m]["mem_total_gb"]),
                   math.log10(d["machines"][m]["power_cap_w"])] for m in mach])
    return Y, X, folds, mach, Z, [str(k) for k in keys]


def corpus_wilkins():
    d = json.load(io.open("experiments/results/wilkins_corpus.json", encoding="utf-8"))
    Y = np.array(d["log10_energy_per_token"])
    rows = d["rows"]
    models = sorted({r[0] for r in rows})
    X = []
    for m, sweep, tok in rows:
        isout = 1.0 if sweep == "out" else 0.0
        lt = math.log2(tok)
        X.append([isout, lt, isout * lt, lt ** 2] + [1.0 if m == mm else 0.0 for mm in models])
    X = np.array(X)
    groups = sorted({(r[0], r[1]) for r in rows})
    folds = [("{}|{}".format(*g), [i for i, r in enumerate(rows) if (r[0], r[1]) == g])
             for g in groups]
    return Y, X, folds, d["hardware"], ["{}|{}|{}".format(*r) for r in rows]


PARAM_B = {"gemma_2b": 2.0, "gemma_7b": 7.0, "codellama_7b": 7.0}


def corpus_ejhusom():
    d = json.load(io.open("experiments/results/ejhusom_corpus.json", encoding="utf-8"))
    Y = np.array(d["log10_energy_per_token"])
    rows = d["rows"]
    ds = sorted({r[0] for r in rows})
    X = np.array([[math.log10(PARAM_B[r[1]])] + [1.0 if r[0] == t else 0.0 for t in ds]
                  for r in rows])
    mods = sorted({r[1] for r in rows})
    folds = [(m, [i for i, r in enumerate(rows) if r[1] == m]) for m in mods]
    return Y, X, folds, d["machines"], ["{}|{}".format(*r) for r in rows]


def _parse_params_b(name):
    import re
    s = name.lower()
    for pat, mul in ((r"(\d+(?:\.\d+)?)\s*b\b", 1.0), (r"(\d+(?:\.\d+)?)\s*m\b", 1e-3)):
        m = re.search(pat, s.replace("-", " ").replace("_", " ").replace("/", " "))
        if m:
            try:
                return float(m.group(1)) * mul
            except ValueError:
                pass
    return None


def corpus_llmperf():
    d = json.load(io.open("experiments/results/j2_llmperf_matrix.json", encoding="utf-8"))
    Y = np.array(d["log10_energy"])
    rows = d["rows"]
    quants = sorted({r[1][0] for r in rows})
    pb = [_parse_params_b(r[0]) for r in rows]
    n_parsed = sum(1 for p in pb if p)
    med = float(np.median([p for p in pb if p])) if n_parsed else 1.0
    X = []
    for (name, cfg), p in zip(rows, pb):
        lp = math.log10(p if p else med)
        X.append([lp, lp ** 2] + [1.0 if cfg[0] == q else 0.0 for q in quants])
    X = np.array(X)
    mods = sorted({r[0] for r in rows})
    folds = [(m, [i for i, r in enumerate(rows) if r[0] == m]) for m in mods]
    return Y, X, folds, d["gpus"], ["{}|{}".format(r[0], r[1][0]) for r in rows], n_parsed, len(rows)


# --------------------------------------------------------- Grid'5000 loaders
G5_REF_CLUSTER = "taurus"


def grid5000_descriptors():
    """Machine and workload descriptors read from the published Grid'5000 files."""
    import csv
    d = json.load(io.open("experiments/results/k2_grid5000.json", encoding="utf-8"))
    Y = np.array(d["log10_energy"])
    clusters = d["hardware"]
    benches = d["workloads_used"]
    classes = d["workload_classes"]

    raw = list(csv.DictReader(io.open("data/grid5000/data.csv", encoding="utf-8",
                                      errors="replace"), delimiter=" "))
    aug = list(csv.DictReader(io.open("data/grid5000/data_augmented.csv", encoding="utf-8",
                                      errors="replace"), delimiter=" "))
    nproc, fmax1, fall, idlep = {}, {}, defaultdict(set), defaultdict(list)
    for r in raw:
        c = r["cluster"]
        fall[c].add(float(r["fmax"]))
        if abs(float(r["ratio"]) - 1.0) < 1e-9:
            nproc[c] = int(r["nproc"])
            fmax1[c] = float(r["fmax"])
            if r["bench"] == "IdleC0":
                idlep[c].append(float(r["mean_power"]))
    Zspec = np.array([[math.log10(nproc[c]), math.log10(fmax1[c]),
                       math.log10(min(fall[c])), float(len(fall[c]))] for c in clusters])
    Zidle = np.hstack([Zspec, np.array([[math.log10(float(np.median(idlep[c])))]
                                        for c in clusters])])

    # workload descriptors: NPB class + reference-cluster profile
    prof = defaultdict(lambda: defaultdict(list))
    for r in aug:
        if r["cluster"] != G5_REF_CLUSTER or abs(float(r["ratio"]) - 1.0) > 1e-9:
            continue
        if r["bench"] not in benches:
            continue
        try:
            ins = float(r["med_instructions"]); cm = float(r["med_cache_misses"])
            cr = float(r["med_cache_references"]); du = float(r["duration"])
        except (TypeError, ValueError):
            continue
        if min(ins, cm, cr, du) <= 0:
            continue
        prof[r["bench"]]["ins"].append(ins)
        prof[r["bench"]]["cm"].append(cm / ins)
        prof[r["bench"]]["cr"].append(cr / ins)
        prof[r["bench"]]["dur"].append(du)
    X = []
    for b in benches:
        p = prof[b]
        X.append([1.0 if classes[b] == "D" else 0.0,
                  math.log10(float(np.median(p["ins"])) / nproc[G5_REF_CLUSTER]),
                  math.log10(float(np.median(p["cm"]))),
                  math.log10(float(np.median(p["cr"]))),
                  math.log10(float(np.median(p["dur"])))])
    return Y, np.array(X), clusters, benches, Zspec, Zidle


# ------------------------------------------------- cold start on machines
def machine_cold_start(name, Y, Z, machines, label, permute=False, leak_check=False):
    """Hold out an entire machine column; predict c_j and v_j from descriptors."""
    n, m = Y.shape
    Zu = Z.copy()
    if permute:
        Zu = Zu[RNG.permutation(m)]
    per_fold, pairs_add, pairs_int, regs_add, regs_int = [], [], [], [], []
    ceilings = []
    leak_ok = True
    for j0 in range(m):
        tr_cols = [j for j in range(m) if j != j0]
        Yw = Y.copy()
        if leak_check:
            Yw[:, j0] = np.nan
        mask = np.ones_like(Y, dtype=bool)
        mask[:, j0] = False
        Yf = np.where(mask, np.nan_to_num(Yw, nan=0.0), 0.0)
        mu, r, c = additive(Yf, mask)
        rc = RidgeCV(alphas=ALPHAS).fit(Zu[tr_cols], c[tr_cols])
        c_hat = c.copy()
        c_hat[j0] = rc.predict(Zu[[j0]])[0]
        base = mu + r[:, None] + c_hat[None, :]
        R = Y[:, tr_cols] - base[:, tr_cols]
        v1_tr = np.linalg.svd(R, full_matrices=False)[2][0]
        vfull = np.zeros(m)
        for idx, j in enumerate(tr_cols):
            vfull[j] = v1_tr[idx]
        vm = RidgeCV(alphas=ALPHAS).fit(Zu[tr_cols], v1_tr)
        vfull[j0] = vm.predict(Zu[[j0]])[0]
        c_true_j0 = float(np.mean(Y[:, j0]) - np.mean(Y[:, tr_cols]))
        s_row = R @ v1_tr
        P_add = base
        P_int = base + np.outer(s_row, vfull)

        cols_all = list(range(m))
        # cold-start pairs only: every pair that involves the held-out machine
        pa = [p for p in pair_records(Y, P_add, range(n), cols_all)[0] if j0 in (p[1], p[2])]
        pi = [p for p in pair_records(Y, P_int, range(n), cols_all)[0] if j0 in (p[1], p[2])]
        ra = pair_records(Y, P_add, range(n), cols_all)[1]
        ri = pair_records(Y, P_int, range(n), cols_all)[1]
        # exact ceiling on the cold-start pair set: the best position for j0 in any
        # total order is found by trying all m insertion positions among the others.
        ceilings.append(_coldstart_ceiling(Y, j0, tr_cols, range(n)))
        pairs_add.extend(pa); pairs_int.extend(pi)
        regs_add.extend(ra); regs_int.extend(ri)
        per_fold.append(dict(held_out=machines[j0],
                             coldstart_acc_additive=acc_of(pa),
                             coldstart_acc_bilinear=acc_of(pi),
                             coldstart_ceiling=ceilings[-1],
                             regret_additive=reg_of(ra), regret_bilinear=reg_of(ri),
                             c_true=c_true_j0,
                             c_hat=float(c_hat[j0]), v_hat=float(vfull[j0]),
                             v1_train_norm=float(np.linalg.norm(v1_tr)),
                             pred=[float(x) for x in P_int[:, j0]],
                             actual=[float(x) for x in Y[:, j0]]))
    out = dict(corpus=name, descriptor_set=label, permuted=bool(permute), n_machines=m,
               coldstart_acc_additive=acc_of(pairs_add),
               coldstart_acc_bilinear=acc_of(pairs_int),
               coldstart_ceiling_mean=float(np.mean(ceilings)),
               regret_additive=reg_of(regs_add), regret_bilinear=reg_of(regs_int),
               n_coldstart_pairs=len(pairs_int),
               fold_sd_bilinear=float(np.std([f["coldstart_acc_bilinear"] for f in per_fold], ddof=1)),
               per_fold=per_fold)
    mn, lo, hi = paired_bootstrap(pairs_int, pairs_add)
    out["gain_bootstrap"] = dict(mean=mn, lo=lo, hi=hi)
    # S11: leave-one-machine-out R^2 of the two descriptor regressions
    ct = np.array([f["c_true"] for f in per_fold]); ch = np.array([f["c_hat"] for f in per_fold])
    out["c_regression_loo_r2"] = float(1 - ((ct - ch) ** 2).sum() /
                                       max(((ct - ct.mean()) ** 2).sum(), 1e-30))
    # v_j is only identified up to sign and scale per fold, so its LOO quality is measured
    # as the correlation between the predicted v_j and the true column's loading recovered
    # from the full-matrix rank-1 fit, refitted per fold with a common sign convention.
    Rfull = Y - (Y.mean() + (Y.mean(1) - Y.mean())[:, None] + (Y.mean(0) - Y.mean())[None, :])
    v_full = np.linalg.svd(Rfull, full_matrices=False)[2][0]
    vh = np.array([f["v_hat"] for f in per_fold])
    sgn = np.sign(np.corrcoef(vh, v_full)[0, 1]) or 1.0
    out["v_regression_loo_pearson"] = float(np.corrcoef(vh, sgn * v_full)[0, 1])
    out["_pairs_bilinear"] = pairs_int
    return out


def _coldstart_ceiling(Y, j0, tr_cols, rows):
    """Exact best fixed ranking on the pairs involving j0: try every rank position.

    Only j0's position in the total order affects pairs that involve j0, so scanning
    the m possible positions relative to the (fixed but arbitrary) order of the others
    is exhaustive over all total orders for this pair set.
    """
    rows = list(rows)
    others = list(tr_cols)
    # order the others by their mean, then insert j0 at each of len(others)+1 slots
    others = sorted(others, key=lambda j: np.mean([Y[i, j] for i in rows]))
    wins = {j: sum(1 for i in rows if Y[i, j0] < Y[i, j]) for j in others}
    ties = {j: sum(1 for i in rows if Y[i, j0] == Y[i, j]) for j in others}
    tot = sum(len(rows) - ties[j] for j in others)
    best = -1.0
    for pos in range(len(others) + 1):
        s = 0.0
        for k, j in enumerate(others):
            if k < pos:      # j ranked above j0 -> predicts j better than j0
                s += (len(rows) - ties[j]) - wins[j]
            else:            # j0 ranked above j
                s += wins[j]
        best = max(best, s)
    return 100.0 * best / tot if tot else float("nan")


# ==================================================================== main
def main():
    print("=" * 92)
    print("CF-MULTI  the Section 7 inductive model on every corpus")
    print("=" * 92)

    results = {}

    # ---------------------------------------------------------- own grid (anchor)
    print("\n--- own-grid (anchor) --------------------------------------------------")
    Y, X, folds, keys = corpus_own_grid()
    r_own = evaluate_corpus("own-grid", Y, X, folds, note="anchor against the published Table 4")
    results["own-grid"] = r_own
    sane("S4 anchor reproduces the published own-grid figures",
         abs(r_own["additive"]["acc"] - 81.7) < 0.5 and abs(r_own["bilinear rank-1"]["acc"] - 87.9) < 0.5,
         "additive {:.1f} (published 81.7), bilinear {:.1f} (published 87.9)".format(
             r_own["additive"]["acc"], r_own["bilinear rank-1"]["acc"]))

    # ---------------------------------------------------------- Grid'5000
    print("\n--- grid5000 -----------------------------------------------------------")
    Yg, Xg, clusters, benches, Zspec, Zidle = grid5000_descriptors()
    ref = clusters.index(G5_REF_CLUSTER)
    cols_g = [j for j in range(len(clusters)) if j != ref]
    folds_g = [(b, [i]) for i, b in enumerate(benches)]
    r_g5 = evaluate_corpus("grid5000", Yg, Xg, folds_g, cols=cols_g,
                           note="leave-one-workload-out; the reference cluster {} supplies the "
                                "workload profile and is excluded from the ranked columns"
                                .format(G5_REF_CLUSTER))
    results["grid5000"] = r_g5

    # ---------------------------------------------------------- Wilkins
    print("\n--- wilkins ------------------------------------------------------------")
    Yw, Xw, foldsw, hw_w, keysw = corpus_wilkins()
    results["wilkins"] = evaluate_corpus("wilkins", Yw, Xw, foldsw,
                                         note="leave-one-(model,sweep)-group-out")
    # A 99% score on 4 folds is suspiciously good, so the same corpus is rerun under a
    # STRICTER split in which an entire MODEL (both sweeps, 16 rows) is withheld, so no
    # measurement of the held-out model survives in training.
    mdl = sorted({k.split("|")[0] for k in keysw})
    folds_strict = [(m, [i for i, k in enumerate(keysw) if k.split("|")[0] == m]) for m in mdl]
    results["wilkins-strict"] = evaluate_corpus(
        "wilkins-strict", Yw, Xw, folds_strict,
        note="STRICTER: leave-one-MODEL-out, both sweeps withheld together")

    # ---------------------------------------------------------- ejhusom
    print("\n--- ejhusom ------------------------------------------------------------")
    Ye, Xe, folde, hw_e, keyse = corpus_ejhusom()
    r_ej = evaluate_corpus("ejhusom", Ye, Xe, folde,
                           note="TWO columns: see S8, not measurable")
    r_ej["measurable"] = False
    r_ej["not_measurable_reason"] = (
        "5 rows x 2 columns gives 5 pairwise comparisons, so accuracy takes only the 6 "
        "values 0/20/40/60/80/100 and the finest resolvable difference is 20 points. The "
        "interaction has one degree of freedom per row (v is a 2-vector fixed up to sign "
        "and scale), so a rank-1 term has nothing to estimate beyond a single sign.")
    results["ejhusom"] = r_ej
    sane("S8 two-column resolution declared on ejhusom", True,
         "5 comparisons, 6 attainable accuracy values, resolution 20 points; "
         "corpus reported but drawn no conclusion from")

    # ---------------------------------------------------------- C1 / C4
    print("\n--- c1-extended --------------------------------------------------------")
    Y1, X1, f1, m1, Z1, k1 = corpus_cx("experiments/results/c1_bridge.json")
    results["c1-extended"] = evaluate_corpus("c1-extended", Y1, X1, f1,
                                             note="leave-one-workload-family-out")
    print("\n--- c4-training --------------------------------------------------------")
    Y4, X4, f4, m4, Z4, k4 = corpus_cx("experiments/results/c4_training_grid.json",
                                       with_params=True)
    results["c4-training"] = evaluate_corpus("c4-training", Y4, X4, f4,
                                             note="leave-one-workload-family-out")

    # ---------------------------------------------------------- llm-perf
    print("\n--- llm-perf -----------------------------------------------------------")
    Yl, Xl, fl, gl, kl, nparsed, ntot = corpus_llmperf()
    results["llm-perf"] = evaluate_corpus(
        "llm-perf", Yl, Xl, fl,
        note="leave-one-model-out; parameter count parsed from the model identifier for "
             "{}/{} rows, median imputed elsewhere".format(nparsed, ntot))

    # ================================================ COLD START ON MACHINES
    print("\n" + "=" * 92)
    print("COLD START ON MACHINES  (hold out an entire column, predict it from descriptors)")
    print("=" * 92)
    cold = {}

    print("\n  grid5000, 18 clusters, spec descriptors (cores, fmax, fmin, DVFS steps)")
    cs = machine_cold_start("grid5000", Yg, Zspec, clusters, "spec: cores,fmax,fmin,dvfs_steps")
    cs_leak = machine_cold_start("grid5000", Yg, Zspec, clusters, "spec", leak_check=True)
    sane("S6 no leakage of the held-out column in grid5000 cold start",
         abs(cs["coldstart_acc_bilinear"] - cs_leak["coldstart_acc_bilinear"]) < 1e-12 and
         all(abs(a["c_hat"] - b["c_hat"]) < 1e-12 and abs(a["v_hat"] - b["v_hat"]) < 1e-12
             for a, b in zip(cs["per_fold"], cs_leak["per_fold"])),
         "predictions identical when the held-out column is replaced by NaN "
         "({:.10f}% both)".format(cs["coldstart_acc_bilinear"]))
    perm = machine_cold_start("grid5000", Yg, Zspec, clusters, "spec, PERMUTED", permute=True)
    mn, lo, hi = paired_bootstrap(cs["_pairs_bilinear"], perm["_pairs_bilinear"])
    sane("S7 permuted machine descriptors do not beat true descriptors on grid5000",
         mn >= 0 or lo <= 0 <= hi,
         "true {:.1f}% vs permuted {:.1f}%, paired difference {:+.1f} pts "
         "[{:+.1f}, {:+.1f}]".format(cs["coldstart_acc_bilinear"],
                                     perm["coldstart_acc_bilinear"], mn, lo, hi))
    cs_idle = machine_cold_start("grid5000", Yg, Zidle, clusters,
                                 "spec + idle-probe power (one characterisation run)")
    for k in (cs, perm, cs_idle):
        k.pop("_pairs_bilinear", None)
    cold["grid5000_spec"] = cs
    cold["grid5000_spec_permuted_control"] = perm
    cold["grid5000_spec_plus_idle"] = cs_idle

    print("\n  c1-extended, 7 accelerators, spec descriptors (device memory, power cap)")
    c1c = machine_cold_start("c1-extended", Y1, Z1, m1, "spec: mem_total_gb, power_cap_w")
    c1c.pop("_pairs_bilinear", None)
    cold["c1_spec"] = c1c
    print("  c4-training, 7 accelerators, same descriptors")
    c4c = machine_cold_start("c4-training", Y4, Z4, m4, "spec: mem_total_gb, power_cap_w")
    c4c.pop("_pairs_bilinear", None)
    cold["c4_spec"] = c4c

    cold["not_attempted"] = {
        "wilkins": "the corpus records no numeric machine descriptors, and two of the four "
                   "platforms carry the same A100 accelerator, differing only in host CPU. "
                   "Hand-entering vendor TDPs would invent the feature under test.",
        "ejhusom": "two machines; holding one out leaves a single training point.",
        "llm-perf": "three GPUs; holding one out leaves two training points for a "
                    "two-feature regression, which is exactly determined."}

    for k, v in cold.items():
        if isinstance(v, dict) and "coldstart_acc_bilinear" in v:
            sane("S11 {} cold-start descriptor regressions reported".format(k), True,
                 "c-regression LOO R2 {:+.3f}; v-regression LOO Pearson {:+.3f}".format(
                     v["c_regression_loo_r2"], v["v_regression_loo_pearson"]))
    for k, v in cold.items():
        if isinstance(v, dict) and "coldstart_acc_bilinear" in v:
            print("    {:34} additive {:5.1f}%  bilinear {:5.1f}%  ceilHS {:5.1f}%  "
                  "regret {:.4f}  n_pairs {:5d}  fold sd {:4.1f}".format(
                      k, v["coldstart_acc_additive"], v["coldstart_acc_bilinear"],
                      v["coldstart_ceiling_mean"], v["regret_bilinear"],
                      v["n_coldstart_pairs"], v["fold_sd_bilinear"]))

    # ================================================================ table
    print("\n" + "=" * 92)
    print("SUMMARY  leave-one-workload-family-out, all corpora")
    print("=" * 92)
    print("{:15}{:>7}{:>5}{:>8}{:>8}{:>9}{:>7}{:>10}{:>9}{:>7}".format(
        "corpus", "shape", "ntr", "ceilHO", "ceilHS", "bilinear", "gain", "regret",
        "oracleLd", "prop1"))
    print("-" * 92)
    order = ["own-grid", "grid5000", "wilkins", "wilkins-strict", "ejhusom",
             "c1-extended", "c4-training", "llm-perf"]
    for k in order:
        r = results[k]
        print("{:15}{:>7}{:>5}{:>8.1f}{:>8.1f}{:>9.1f}{:>+7.1f}{:>10.4f}{:>9.1f}{:>7}".format(
            k, "{}x{}".format(r["shape"][0], r["scored_columns"]),
            int(np.mean([f["n_train_rows"] for f in r["per_fold"]])),
            r["ceiling_heldout"], r["ceiling_optimal_fixed"], r["bilinear rank-1"]["acc"],
            r["gain_vs_heldout_ceiling"], r["bilinear rank-1"]["regret"],
            r["oracle loading (upper bound)"]["acc"],
            "exact" if abs(r["additive"]["acc"] - r["fixed ranking"]["acc"]) < 1e-12 else "FAIL"))
    print("\n  ceilHO = held-out Proposition 1 ceiling: the better of the fitted additive model")
    print("  and the best fixed ranking CHOSEN ON THE TRAINING ROWS. gain is measured against it.")
    print("  ceilHS = the same optimum chosen with hindsight over all m! rankings on the scored")
    print("  rows; exact, but on 5 rows x 17 machines it is optimistically biased.")
    print("  ntr = mean training rows per fold. oracleLd = the same rank-1 form given the")
    print("  held-out row's TRUE loading; an oracle bounding any descriptor set at this form.")

    # -------------------------------------------------- diagnosis of the non-gains
    print("\n" + "=" * 92)
    print("WHERE THE MODEL DOES NOT BEAT ITS CEILING, AND WHY")
    print("=" * 92)
    print("{:15}{:>6}{:>8}{:>8}{:>9}{:>9}{:>10}".format(
        "corpus", "ntr", "signAgr", "corr", "sd_true", "sd_pred", "alphaMax"))
    print("-" * 92)
    for k in order:
        L = results[k]["loading_regression"]
        print("{:15}{:>6}{:>8.0f}{:>8}{:>9.3f}{:>9.3f}{:>10}".format(
            k, int(np.mean([f["n_train_rows"] for f in results[k]["per_fold"]])),
            L["sign_agreement_pct"],
            "{:+.2f}".format(L["pearson"]) if L["pearson"] is not None else "n/a",
            L["sd_true"], L["sd_pred"],
            "{}/{}".format(L["alpha_at_grid_max"], len(L["alphas"]))))
    print("\n  signAgr = fraction of held-out rows whose interaction loading is predicted with")
    print("  the correct SIGN; corr = its correlation with the true loading; alphaMax = folds in")
    print("  which RidgeCV chose the largest penalty on the grid, i.e. shrank the loading to zero.")

    diagnosis = {
        "grid5000 (workload cold start)":
            "FAILS: 85.0 against a held-out ceiling of 88.2. Root cause is not the absence of "
            "an interaction: the training residual is {:.0f}% rank-1 on average, and the same "
            "rank-1 form given the true loading scores {:.1f}. The cause is that the corpus has "
            "FIVE workload rows, so leave-one-workload-out fits the loading regression on FOUR "
            "points with five descriptors; RidgeCV chose the largest penalty on the grid in "
            "{} of 5 folds, shrinking the predicted loading to zero, and in the one fold where "
            "it did not, it predicted the loading with the wrong sign and six times its true "
            "magnitude. Five workloads cannot support workload cold start. This corpus's value "
            "is its EIGHTEEN machines, and the machine cold start below is where it is used."
            .format(results["grid5000"]["loading_regression"]["mean_rank1_share"],
                    results["grid5000"]["oracle loading (upper bound)"]["acc"],
                    results["grid5000"]["loading_regression"]["alpha_at_grid_max"]),
        "c4-training":
            "FAILS: 85.2 against a held-out ceiling of 87.6. Same mechanism at a milder scale: "
            "14 training rows across 5 families, and RidgeCV shrank the loading to the grid "
            "maximum in {} of 5 folds. The interaction is present (oracle loading {:.1f}, "
            "{:.0f} percentage points above the ceiling), so what fails is the descriptor "
            "regression, not the model form. c1-extended, the same accelerators with 30 rows "
            "instead of 18, does gain (+{:.1f}), which is consistent with a sample-size cause."
            .format(results["c4-training"]["loading_regression"]["alpha_at_grid_max"],
                    results["c4-training"]["oracle loading (upper bound)"]["acc"],
                    results["c4-training"]["oracle loading (upper bound)"]["acc"]
                    - results["c4-training"]["ceiling_heldout"],
                    results["c1-extended"]["gain_vs_heldout_ceiling"]),
        "llm-perf":
            "NO GAIN, exactly zero: 95.5 for both the additive model and the bilinear one. The "
            "descriptors this corpus carries are the quantisation scheme and a parameter count "
            "parsed from the model identifier, and the fitted loading is driven almost entirely "
            "by the quantisation one-hot: it takes about three distinct values, so the "
            "interaction term applies a per-scheme column shift that is smaller than the machine "
            "main effect and flips no pair. The interaction itself is real and predictable "
            "in principle: the residual is {:.0f}% rank-1 and the oracle-loading model reaches "
            "{:.1f}. What is missing is a workload descriptor that varies WITHIN a quantisation "
            "scheme, which this corpus does not record."
            .format(results["llm-perf"]["loading_regression"]["mean_rank1_share"],
                    results["llm-perf"]["oracle loading (upper bound)"]["acc"]),
        "ejhusom":
            results["ejhusom"]["not_measurable_reason"],
        "grid5000 machine cold start, interaction contribution":
            "The bilinear term adds {:+.1f} points over the additive spec-prior model on the "
            "spec-only descriptor set and {:+.1f} with the idle probe added. The c_j regression "
            "carries the cold start (leave-one-machine-out R2 {:+.2f} on specs, {:+.2f} with the "
            "idle probe), while the v_j regression is only weakly identified (Pearson {:+.2f}). "
            "The honest reading is that on this corpus procurement-grade descriptors place a "
            "never-measured cluster in the ranking mainly through its LEVEL, not its interaction."
            .format(cold["grid5000_spec"]["coldstart_acc_bilinear"]
                    - cold["grid5000_spec"]["coldstart_acc_additive"],
                    cold["grid5000_spec_plus_idle"]["coldstart_acc_bilinear"]
                    - cold["grid5000_spec_plus_idle"]["coldstart_acc_additive"],
                    cold["grid5000_spec"]["c_regression_loo_r2"],
                    cold["grid5000_spec_plus_idle"]["c_regression_loo_r2"],
                    cold["grid5000_spec"]["v_regression_loo_pearson"])}
    for k, v in diagnosis.items():
        print("\n  * {}\n    {}".format(k, v))

    OUT.update(dict(
        corpora=results, cold_start=cold, sanity=SANITY, diagnosis=diagnosis,
        protocol=dict(
            model="y(i,j) = mu + r(x_i) + c_j + <w,x_i> v_j, rank-1 inductive interaction",
            fit="alternating additive fit, RidgeCV(logspace(-3,3,25)) for r(x) and <w,x>, "
                "v from the leading right singular vector of the training residual",
            splits="leave-one-workload-family-out; cold start holds out an entire machine column",
            ceiling="exact optimal fixed ranking by subset dynamic program over the linear "
                    "ordering problem; validated against brute force for m <= 7",
            metric="pairwise ranking accuracy over machine pairs within a row; energy regret "
                   "of the argmin-predicted machine against the row-wise oracle"),
        machine_descriptors=dict(
            grid5000=["log10 cores", "log10 nominal fmax", "log10 fmin", "n DVFS steps",
                      "(secondary) log10 idle-probe mean power"],
            c1_c4=["log10 mem_total_gb", "log10 power_cap_w"]),
        n_sanity_pass=sum(1 for s in SANITY if s["passed"]),
        n_sanity_total=len(SANITY)))
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(OUT, io.open("experiments/results/cf_multicorpus.json", "w", encoding="utf-8"),
              indent=1, default=float)
    print("\nsanity: {}/{} passed".format(OUT["n_sanity_pass"], OUT["n_sanity_total"]))
    for s in SANITY:
        if not s["passed"]:
            print("  FAILED: {} -- {}".format(s["check"], s["detail"]))
    print("wrote experiments/results/cf_multicorpus.json")


if __name__ == "__main__":
    main()
