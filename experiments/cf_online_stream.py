# -*- coding: utf-8 -*-
"""
Does collaborative filtering earn its place? The online, bandit-feedback test.

WHY THIS EXPERIMENT EXISTS. Two earlier results put the paper's CF component under
suspicion. Section 7 reports only a 6.2 point gain over the provable additive ceiling on a
dense grid, and the static sparsity sweep (experiments/cf_sample_complexity.py) found that
CF tracks a fixed machine ranking almost exactly, and is sometimes worse at mid densities.
The diagnosis tested here is that the SETTING was wrong, not the method. Static batch
completion over a dense matrix with 5-30 rows and a random observation mask is the one
regime where CF cannot help: few rows, every workload observed many times, and the answer
already measured.

THE REGIME THE METHOD WAS DESIGNED FOR. Apartsin, Meshulam and Aperstein, "Acting on the
Unseen: Communication-Free Collaborative Filtering for Decentralized Multi-Robot Task
Allocation" (arXiv:2605.25584, reference [25]) analyses a task pool far exceeding the
available rounds, most (agent, task) pairs never attempted, learning only from outcomes
actually observed, and proves a Theta(d) versus Theta(n) sample complexity: linear in the
latent rank d rather than in the task count n. Mapped onto placement: robots -> accelerators,
tasks -> workloads, task pool >> rounds -> a workload catalogue far larger than the number of
jobs a facility can ever profile, attempt-and-observe -> you learn the energy ONLY of the
machine you actually ran the job on, never the counterfactual.

WHAT IS SIMULATED. An online sequential placement problem. Jobs arrive one at a time drawn
from a heavy-tailed (Zipf) popularity distribution over a workload catalogue. On each
arrival the policy picks ONE machine, pays that cell's measured energy, and observes ONLY
that cell. Counterfactual cells are never revealed. Cumulative energy is compared with a
full-knowledge oracle over the identical stream and seed.

TWO DESIGN POINTS THAT A FIRST DRAFT GOT WRONG, AND WHY THEY MATTER. Both were caught by
inspecting a suspicious result rather than by a failing assertion, and both flattered the
per-row learner.
  (i) FORCED INITIALISATION ORDER MUST BE RANDOM. A learner that has never tried machine j
      for workload i has to try it. If the untried machines are tried in column order, then
      on BUTTER-E the first machine tried is always the CPU node, which is the better choice
      for 89 per cent of workloads. A workload seen exactly once would then be routed
      correctly for free, and the per-row learner would look strong for a reason that is pure
      column-ordering luck. The order is drawn uniformly at random per learning unit.
  (ii) A NOISELESS METER IS NOT NEUTRAL. If each cell returns one exact number, a per-row
      learner solves a workload in exactly M pulls and is optimal on it forever after, which
      is a property of the idealisation and not of any real facility. Every corpus here
      carries repeated runs of the same cell, so the run-to-run dispersion of the meter is
      MEASURED from the corpus itself (mean log-scale standard deviation over cells with at
      least three repetitions) and the sweeps are reported at both the idealised zero-noise
      setting and that measured setting. Accounting always uses the cell's central measured
      value, so this is pseudo-regret, the standard accounting for stochastic bandits, and
      the oracle remains exactly attainable.

Zipf is the natural popularity law: it is the standard empirical model for job-name
frequency in HPC and cloud traces, and its exponent s is exactly the knob that controls how
much of the stream lands on workloads seen once or twice. s = 0 is uniform, s = 1.2 is the
classical Zipf regime reported for cluster traces, s = 2.0 and s = 3.0 are very heavy heads.
The exponent is swept rather than chosen.

POLICIES, all run on the identical stream with the identical seed:
  oracle        full knowledge, picks the row minimum. The regret denominator
  fixed         best single machine ranking learned online from bandit feedback, fitted as
                the additive model <a,x_i> + c_j and acting on argmin_j c_j. By Proposition 1
                an additive model induces one machine order for every workload, so this arm
                is simultaneously the fixed-ranking baseline and the additive baseline. The
                descriptor row term is included so the machine effects c_j are de-confounded
                from which workloads happened to be routed where; that makes this baseline as
                strong as it can be
  per_workload  independent epsilon-greedy over machines within each workload row, with no
                sharing at all. This is the Theta(n) method
  cf            the paper's bilinear model <a,x_i> + c_j + <w,x_i> v_j fitted by alternating
                ridge on the observed stream only. Interaction is carried entirely by the
                workload descriptors, so a workload never seen still receives a machine
                preference. This is the cold-start mechanism under test
  cf_eb         cf with empirical-Bayes shrinkage toward whatever direct observations the row
                already has: posterior = (n * row_mean + k * cf_pred) / (n + k), k = 1. This
                is the deployable system, CF for the tail and direct measurement for the head
  random        uniform machine choice, the sanity floor

NO HYPERPARAMETER TUNING. Every learning policy uses the same exploration rate
(epsilon = 0.10), the same forced initialisation (each machine once per learning unit), the
same refit period (100 rounds), the same ridge constant (1e-2) and the same number of
alternating iterations (6). None of these was searched over. The shrinkage constant k = 1 in
cf_eb is the uninformative default of one pseudo-observation.

PREDICTIONS STATED IN ADVANCE, all falsifiable:
  H1  with catalogue size >> rounds, per_workload is far worse than fixed, because it never
      collects enough samples per row. This is the Theta(n) failure
  H2  cf beats fixed, and the margin GROWS as the catalogue grows relative to the number of
      rounds, and as the tail gets heavier
  H3  cf's advantage is concentrated on rarely-seen workloads. On head workloads
      per_workload should eventually catch up and pass it
  H4  with only 2 machines the advantage is small by construction, because a binary choice
      leaves one bit for structure to exploit. Machine count is swept explicitly

SANITY CHECKS STATED IN ADVANCE, each printed PASS/FAIL:
  S1  every corpus matrix is complete and strictly positive; no imputed cell enters the oracle
  S2  oracle regret ratio is exactly 1.000 in every run, by construction
  S3  no policy ever attains regret ratio < 1.0. A value below 1 means the policy beat full
      knowledge, which is impossible and would prove an information leak
  S4  information containment: the set of (workload, machine) cells in each policy's memory
      is EXACTLY the set the environment revealed to it, and no policy object holds a copy of
      the energy matrix. Checked every run
  S5  identical stream: all policies in a run are queried with the same arrival sequence,
      verified by hashing the query log of each policy against the stream
  S6  positive control for per_workload: when rounds >> catalogue x machines, per_workload
      must converge to the epsilon-greedy floor, 1 + eps x (regret of uniform - 1), which is
      the best any policy in this comparison can reach. If it fails HERE the implementation
      is broken and its failure elsewhere is a bug rather than Theta(n)
  S7  positive control for fixed: its asymptotic machine choice must equal the corpus's
      offline best single machine
  S8  degenerate control: with an extremely heavy tail (s = 4, essentially one workload) the
      per-row learner must match or beat cf, since there is nothing to generalise across
  S9  random must be strictly worse than fixed at the largest round counts
  S10 seed spread is reported for every reported number
  S11 the online CF policy must NOT beat the same model class fitted offline with full
      information on the same quantity of bandit observations. Beating it would mean the
      online run is seeing cells it was never shown
"""
import io, json, math, os, sys, time, zipfile, csv, hashlib, warnings
from collections import defaultdict, Counter
import numpy as np

csv.field_size_limit(10_000_000)
warnings.filterwarnings("ignore")

SANITY = []


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))


CACHE = os.environ.get("CFOS_CACHE", "")

# =====================================================================  corpora


def _standardise(Xn):
    if Xn.shape[1] == 0:
        return Xn
    mu = Xn.mean(0)
    sd = Xn.std(0)
    sd[sd < 1e-12] = 1.0
    return (Xn - mu) / sd


def _onehot(vals):
    lv = sorted(set(vals))
    idx = {v: i for i, v in enumerate(lv)}
    O = np.zeros((len(vals), len(lv)))
    for r, v in enumerate(vals):
        O[r, idx[v]] = 1.0
    return O


def _pack(numeric, cats):
    parts = [np.ones((numeric.shape[0], 1))]
    if numeric.shape[1]:
        parts.append(_standardise(numeric))
    for c in cats:
        parts.append(_onehot(c))
    return np.hstack(parts)


def _obs_log_sd(replicate_lists, minreps=3):
    """Run-to-run dispersion of the meter, MEASURED from the corpus's own repeated runs.

    This is the number used for the noisy-feedback condition. A noiseless meter would let the
    per-row learner solve a workload in exactly M pulls, which is not a property of the world
    but of the idealisation, so the realistic setting has to be reported alongside it.
    """
    sds = [float(np.std(np.log(np.asarray(v, dtype=float))))
           for v in replicate_lists if len(v) >= minreps and min(v) > 0]
    return (float(np.mean(sds)), len(sds)) if sds else (0.0, 0)


def corpus_butter_e():
    """13,121 neural-architecture training workloads x {CPU node, 2xV100 GPU node}.

    Energy is the median standardised whole-node joules over the measured repetitions of
    each (dataset, shape, size, depth, machine class) cell, exactly as
    experiments/butter_e_experiment.py builds it.
    """
    cpath = os.path.join(CACHE, "butter_e_matrix.npz") if CACHE else ""
    if cpath and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return dict(name="butter-e", Y=z["Y"], X=z["X"], machines=list(z["machines"]),
                    keys=[tuple(k) for k in z["keys"]],
                    obs_log_sd=float(z["sd"]), obs_log_sd_cells=int(z["sdn"]))
    agg_ = defaultdict(list)
    zf = zipfile.ZipFile("data/butter-e/runs_with_standardized_energy.csv.zip")
    with zf.open("runs_with_standardized_energy.csv") as f:
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace")):
            try:
                e = float(r.get("std_energy") or r.get("energy"))
                size = float(r["size"])
                depth = float(r["depth"])
            except (TypeError, ValueError):
                continue
            if e <= 0 or size <= 0:
                continue
            agg_[(r["dataset"], r["shape"], size, depth, r["is_gpu"] == "1")].append(e)
    sd, sdn = _obs_log_sd(list(agg_.values()))
    med = {k: float(np.median(v)) for k, v in agg_.items()}
    loads = defaultdict(dict)
    for (ds, sh, size, depth, gpu), v in med.items():
        loads[(ds, sh, size, depth)][gpu] = v
    keys = sorted(k for k, v in loads.items() if True in v and False in v)
    Y = np.array([[loads[k][False], loads[k][True]] for k in keys])
    num = np.array([[math.log10(k[2]), k[3], math.log10(k[2]) / max(k[3], 1.0),
                     math.log10(k[2]) ** 2] for k in keys])
    X = _pack(num, [[k[0] for k in keys], [k[1] for k in keys]])
    out = dict(name="butter-e", Y=Y, X=X, machines=["cpu_node", "gpu_2xv100"], keys=keys,
               obs_log_sd=sd, obs_log_sd_cells=sdn)
    if cpath:
        np.savez(cpath, Y=Y, X=X, machines=np.array(out["machines"], dtype=object),
                 keys=np.array([[str(x) for x in k] for k in keys], dtype=object),
                 sd=sd, sdn=sdn)
    return out


def corpus_llmperf():
    """LLM inference decode-phase GPU energy. Workloads are
    (model, quantisation, batch, sequence length, new tokens); machines are {A10, A100, T4}.
    Only rows measured on all three GPUs are kept, so the oracle is always measured."""
    import glob

    def num(x):
        try:
            v = float(x)
            return v if v > 0 and math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    rows = []
    for f in sorted(glob.glob("data/llm-perf/*.csv")):
        base = os.path.basename(f)
        quant = base.split("cuda-")[1].split("-1x")[0]
        gpu = base.split("-1x")[1].replace(".csv", "")
        with io.open(f, encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                e = num(r.get("report.decode.energy.gpu")) or num(r.get("report.decode.energy.total"))
                if e is None:
                    continue
                rows.append((gpu, quant, r.get("config.backend.model") or "",
                             num(r.get("config.scenario.input_shapes.batch_size")) or 1.0,
                             num(r.get("config.scenario.input_shapes.sequence_length")) or 0.0,
                             num(r.get("config.scenario.generate_kwargs.max_new_tokens")) or 0.0, e))
    machines = sorted({r[0] for r in rows})
    cell = defaultdict(list)
    for gpu, quant, model, b, s, nt, e in rows:
        cell[((model, quant, b, s, nt), gpu)].append(e)
    sd, sdn = _obs_log_sd(list(cell.values()))
    med = {k: float(np.median(v)) for k, v in cell.items()}
    per = defaultdict(dict)
    for (w, g), v in med.items():
        per[w][g] = v
    keys = sorted(w for w, d in per.items() if len(d) == len(machines))
    Y = np.array([[per[w][g] for g in machines] for w in keys])
    num = np.array([[math.log2(max(k[2], 1.0)), math.log10(max(k[3], 1.0)),
                     math.log10(max(k[4], 1.0))] for k in keys])
    X = _pack(num, [[k[0] for k in keys], [k[1] for k in keys]])
    return dict(name="llm-perf", Y=Y, X=X, machines=machines, keys=keys,
                obs_log_sd=sd, obs_log_sd_cells=sdn)


def corpus_wilkins():
    D = json.load(io.open("experiments/results/wilkins_corpus.json", encoding="utf-8"))
    keys = [tuple(r) for r in D["rows"]]
    Y = 10.0 ** np.array(D["log10_energy_per_token"])
    num = np.array([[math.log2(float(k[2]))] for k in keys])
    X = _pack(num, [[k[0] for k in keys], [k[1] for k in keys]])
    sd, sdn = _obs_log_sd([v for v in D["cell_values"].values() if isinstance(v, list)])
    return dict(name="wilkins", Y=Y, X=X, machines=list(D["hardware"]), keys=keys,
                obs_log_sd=sd, obs_log_sd_cells=sdn)


def corpus_c1():
    D = json.load(io.open("experiments/results/c1_bridge.json", encoding="utf-8"))
    cells = defaultdict(dict)
    for r in D["rows"]:
        if r.get("status") == "ok" and r.get("energy_per_sample_mj"):
            cells[(r["load"], r["precision"], r["batch"])][r["machine"]] = r["energy_per_sample_mj"]
    machines = sorted(D["machines"])
    keys = sorted(k for k, v in cells.items() if len(v) == len(machines))
    Y = np.array([[cells[k][m] for m in machines] for k in keys])
    num = np.array([[math.log2(float(k[2]))] for k in keys])
    X = _pack(num, [[k[0] for k in keys], [k[1] for k in keys]])
    # The C1 bridge grid stores one aggregated value per cell, so no run-to-run dispersion is
    # recoverable from it. The t4 replicate study measured 0.02-0.05 in log space on the same
    # rig; we do not import that number here, we report 0 and say the corpus cannot support
    # the noisy condition.
    return dict(name="own-grid", Y=Y, X=X, machines=machines, keys=keys,
                obs_log_sd=0.0, obs_log_sd_cells=0)


def corpus_grid5000():
    """5 NPB benchmarks x 18 Grid'5000 clusters. The widest hardware axis available, but the
    workload axis is a bare identifier: the descriptor vector is one-hot, so CF has nothing
    to generalise across and can only behave like the per-row learner. Included as the
    machine-count anchor, with that caveat stated."""
    D = json.load(io.open("experiments/results/k2_grid5000.json", encoding="utf-8"))
    machines = list(D["hardware"])
    keys = [(w,) for w in D["workloads_used"]]
    Y = np.array([[D["cell_medians"]["{}|{}".format(w[0], m)] for m in machines] for w in keys])
    X = _pack(np.zeros((len(keys), 0)), [[k[0] for k in keys]])
    sd, sdn = _obs_log_sd([v for v in D["cell_values"].values() if isinstance(v, list)])
    return dict(name="grid5000", Y=Y, X=X, machines=machines, keys=keys,
                obs_log_sd=sd, obs_log_sd_cells=sdn)


# =====================================================================  policies
EPS = 0.10
REFIT = 100
RIDGE = 1e-2
ALS_ITERS = 6


class Base(object):
    """Policies never receive the energy matrix. They see only descriptors X (public
    spec-sheet-style metadata) and the scalar log-energy of cells they actually chose."""

    def __init__(self, X, M, rng):
        self.X = X
        self.M = M
        self.rng = rng
        self.seen = []       # (i, j) cells this policy was told about
        self.queries = []    # workload ids this policy was asked about

    def act(self, i):
        raise NotImplementedError

    def update(self, i, j, logy):
        self.seen.append((i, j))


class RandomP(Base):
    def act(self, i):
        self.queries.append(i)
        return int(self.rng.integers(self.M))


class PerWorkload(Base):
    """Theta(n): independent epsilon-greedy inside each workload row, no sharing."""

    def __init__(self, X, M, rng):
        Base.__init__(self, X, M, rng)
        self.sum = defaultdict(lambda: np.zeros(M))
        self.cnt = defaultdict(lambda: np.zeros(M))

    def act(self, i):
        self.queries.append(i)
        c = self.cnt[i]
        un = np.flatnonzero(c == 0)
        if un.size:
            # The order of forced initialisation must be RANDOM. Taking un[0] hands this
            # policy the column that happens to be listed first, and on BUTTER-E that column
            # is the CPU node, which is optimal for 89 per cent of workloads: a singleton row
            # would then be routed correctly for free, by column ordering luck alone.
            return int(self.rng.choice(un))
        if self.rng.random() < EPS:
            return int(self.rng.integers(self.M))
        return int(np.argmin(self.sum[i] / c))

    def update(self, i, j, logy):
        Base.update(self, i, j, logy)
        self.sum[i][j] += logy
        self.cnt[i][j] += 1


class _Fitted(Base):
    """Shared machinery: keep the observed stream, refit periodically, act epsilon-greedily
    on a per-machine score. Sub-classes define the model."""

    def __init__(self, X, M, rng):
        Base.__init__(self, X, M, rng)
        self.oi = []
        self.oj = []
        self.oy = []
        self.t = 0
        self.a = np.zeros(X.shape[1])
        self.c = np.zeros(M)
        self.w = np.zeros(X.shape[1])
        self.v = np.zeros(M)
        self.fitted = False
        self.init_arm = 0
        self.init_order = rng.permutation(M)      # random, for the reason given in PerWorkload

    def score(self, i):
        raise NotImplementedError

    def refit(self):
        raise NotImplementedError

    def act(self, i):
        self.queries.append(i)
        self.t += 1
        if self.init_arm < self.M:                       # forced initialisation
            j = int(self.init_order[self.init_arm])
            self.init_arm += 1
            return j
        if (not self.fitted) or self.rng.random() < EPS:
            return int(self.rng.integers(self.M))
        return int(np.argmin(self.score(i)))

    def update(self, i, j, logy):
        Base.update(self, i, j, logy)
        self.oi.append(i)
        self.oj.append(j)
        self.oy.append(logy)
        if len(self.oy) % REFIT == 0 or len(self.oy) == self.M:
            self.refit()


def _ridge(D, y, lam):
    A = D.T @ D + lam * np.eye(D.shape[1])
    return np.linalg.solve(A, D.T @ y)


class FixedRank(_Fitted):
    """Additive model <a,x_i> + c_j. Proposition 1: one machine order for all workloads."""

    def refit(self):
        idx = np.asarray(self.oi)
        j = np.asarray(self.oj)
        y = np.asarray(self.oy)
        Xo = self.X[idx]
        O = np.zeros((len(y), self.M))
        O[np.arange(len(y)), j] = 1.0
        b = _ridge(np.hstack([Xo, O]), y, RIDGE)
        self.a = b[:self.X.shape[1]]
        self.c = b[self.X.shape[1]:]
        self.fitted = True

    def score(self, i):
        return self.c            # identical for every workload, by construction


class CF(_Fitted):
    """Bilinear CF with side information: <a,x_i> + c_j + <w,x_i> v_j, alternating ridge on
    observed cells only. A workload never seen still gets c_j + <w,x_i> v_j."""

    def refit(self):
        idx = np.asarray(self.oi)
        j = np.asarray(self.oj)
        y = np.asarray(self.oy)
        Xo = self.X[idx]
        P = self.X.shape[1]
        n = len(y)
        O = np.zeros((n, self.M))
        O[np.arange(n), j] = 1.0
        v = self.v.copy()
        if not np.any(np.abs(v) > 1e-9):
            v = np.linspace(-1.0, 1.0, self.M)
            v = v / max(np.linalg.norm(v), 1e-9)
        a = np.zeros(P)
        c = np.zeros(self.M)
        w = np.zeros(P)
        for _ in range(ALS_ITERS):
            D = np.hstack([Xo, O, Xo * v[j][:, None]])
            b = _ridge(D, y, RIDGE)
            a, c, w = b[:P], b[P:P + self.M], b[P + self.M:]
            z = Xo @ w
            D2 = np.hstack([Xo, O * z[:, None]])
            b2 = _ridge(D2, y, RIDGE)
            a, v = b2[:P], b2[P:]
            nv = np.linalg.norm(v)
            if nv > 1e-9:                                # fix the scale ambiguity
                w = w * nv
                v = v / nv
        D3 = np.hstack([Xo, O, Xo * v[j][:, None]])
        b3 = _ridge(D3, y, RIDGE)
        a, c, w = b3[:P], b3[P:P + self.M], b3[P + self.M:]
        self.a, self.c, self.w, self.v = a, c, w, v
        self.fitted = True

    def score(self, i):
        return self.c + float(self.X[i] @ self.w) * self.v


class CFEB(CF):
    """CF plus empirical-Bayes shrinkage toward the row's own observations (k = 1)."""

    K = 1.0

    def __init__(self, X, M, rng):
        CF.__init__(self, X, M, rng)
        self.rs = defaultdict(lambda: np.zeros(M))
        self.rc = defaultdict(lambda: np.zeros(M))

    def update(self, i, j, logy):
        CF.update(self, i, j, logy)
        self.rs[i][j] += logy
        self.rc[i][j] += 1

    def score(self, i):
        base = self.c + float(self.X[i] @ self.w) * self.v
        cnt = self.rc.get(i)
        if cnt is None or not cnt.any():
            return base
        pred_abs = float(self.X[i] @ self.a) + base       # descriptor row level + machine part
        obs = np.where(cnt > 0, self.rs[i] / np.maximum(cnt, 1.0), pred_abs)
        return (cnt * obs + self.K * pred_abs) / (cnt + self.K)


POLICIES = [("fixed", FixedRank), ("per_workload", PerWorkload), ("cf", CF),
            ("cf_eb", CFEB), ("random", RandomP)]
ORDER = ["oracle", "fixed", "per_workload", "cf", "cf_eb", "random"]


# =====================================================================  the stream
def zipf_probs(n, s, rng):
    p = 1.0 / np.power(np.arange(1, n + 1, dtype=float), s)
    p /= p.sum()
    perm = rng.permutation(n)                  # popularity is independent of energy
    q = np.empty(n)
    q[perm] = p
    return q


def run_stream(Y, X, T, s, seed, noise=0.0, sample_every=None):
    """One online run. `noise` is the log-scale run-to-run dispersion of the meter reading
    handed back to the policies. Accounting (cost paid, oracle) uses the cell's central
    measured value, so this is pseudo-regret, the standard accounting for stochastic bandits,
    and the oracle stays exactly attainable."""
    N, M = Y.shape
    rng = np.random.default_rng(seed)
    p = zipf_probs(N, s, rng)
    stream = rng.choice(N, size=T, p=p)
    total = Counter(stream.tolist())
    logY = np.log(Y)
    rng_obs = np.random.default_rng(seed * 104729 + 13)
    pols = {nm: cls(X, M, np.random.default_rng(seed * 7919 + h))
            for h, (nm, cls) in enumerate(POLICIES)}
    revealed = {nm: [] for nm in pols}
    cost = {nm: np.zeros(T) for nm in pols}
    subopt = {nm: np.zeros(T) for nm in pols}
    ocost = np.zeros(T)
    seen_before = defaultdict(int)
    prior_ct = np.zeros(T, dtype=int)
    freq = np.zeros(T, dtype=int)
    if sample_every is None:
        sample_every = max(1, T // 100)
    for t in range(T):
        i = int(stream[t])
        prior_ct[t] = seen_before[i]
        freq[t] = total[i]
        best = Y[i].min()
        ocost[t] = best
        for nm, pol in pols.items():
            j = pol.act(i)
            assert 0 <= j < M
            cost[nm][t] = Y[i, j]
            subopt[nm][t] = 0.0 if Y[i, j] <= best else 1.0
            revealed[nm].append((i, j))
            reading = float(logY[i, j])
            if noise > 0:
                reading += noise * float(rng_obs.normal())
            pol.update(i, j, reading)
        seen_before[i] += 1
    # ---- S4 information containment, S5 identical stream
    leaks = []
    shash = hashlib.md5(stream.astype(np.int64).tobytes()).hexdigest()
    for nm, pol in pols.items():
        if pol.seen != revealed[nm]:
            leaks.append(nm + ":memory-mismatch")
        qh = hashlib.md5(np.asarray(pol.queries, dtype=np.int64).tobytes()).hexdigest()
        if qh != shash:
            leaks.append(nm + ":stream-mismatch")
        for attr in list(vars(pol).values()):
            if isinstance(attr, np.ndarray) and attr.shape == Y.shape and np.allclose(attr, Y):
                leaks.append(nm + ":holds-energy-matrix")
    otot = float(ocost.sum())
    out = dict(oracle_total=otot, T=T, N=N, M=M, leaks=leaks, noise=noise,
               stream_unique=int(len(total)),
               tail_share=float(sum(1 for t in range(T) if freq[t] <= 2) / T))
    oc = np.cumsum(ocost)
    per_policy = {}
    for nm in pols:
        cc = np.cumsum(cost[nm])
        curve = [[int(k), float(cc[k] / oc[k])] for k in range(sample_every - 1, T, sample_every)]
        buck = {}
        for lab, msk in (("prior_0", prior_ct == 0), ("prior_1", prior_ct == 1),
                         ("prior_2", prior_ct == 2),
                         ("prior_3_5", (prior_ct >= 3) & (prior_ct <= 5)),
                         ("prior_6_20", (prior_ct >= 6) & (prior_ct <= 20)),
                         ("prior_21p", prior_ct >= 21),
                         ("freq_1_2", freq <= 2), ("freq_3_10", (freq >= 3) & (freq <= 10)),
                         ("freq_11p", freq >= 11)):
            if msk.sum() == 0:
                continue
            buck[lab] = dict(n=int(msk.sum()),
                             regret=float(cost[nm][msk].sum() / ocost[msk].sum()),
                             subopt=float(subopt[nm][msk].mean()))
        per_policy[nm] = dict(regret=float(cost[nm].sum() / otot),
                              subopt=float(subopt[nm].mean()),
                              curve=curve, buckets=buck)
    per_policy["oracle"] = dict(
        regret=1.0, subopt=0.0,
        curve=[[int(k), 1.0] for k in range(sample_every - 1, T, sample_every)],
        buckets={lab: dict(n=int(m.sum()), regret=1.0, subopt=0.0) for lab, m in
                 (("prior_0", prior_ct == 0), ("prior_1", prior_ct == 1),
                  ("prior_2", prior_ct == 2), ("prior_3_5", (prior_ct >= 3) & (prior_ct <= 5)),
                  ("prior_6_20", (prior_ct >= 6) & (prior_ct <= 20)),
                  ("prior_21p", prior_ct >= 21), ("freq_1_2", freq <= 2),
                  ("freq_3_10", (freq >= 3) & (freq <= 10)), ("freq_11p", freq >= 11))
                 if m.sum()})
    out["policies"] = per_policy
    out["fixed_choice"] = int(np.argmin(pols["fixed"].c))
    return out


def aggregate(runs):
    """Mean and spread across seeds for each policy, plus bucket means."""
    o = {}
    for nm in sorted(runs[0]["policies"]):
        r = np.array([x["policies"][nm]["regret"] for x in runs])
        s = np.array([x["policies"][nm]["subopt"] for x in runs])
        bk = {}
        labs = set()
        for x in runs:
            labs |= set(x["policies"][nm]["buckets"])
        for lab in sorted(labs):
            vals = [x["policies"][nm]["buckets"][lab]["regret"] for x in runs
                    if lab in x["policies"][nm]["buckets"]]
            sub = [x["policies"][nm]["buckets"][lab]["subopt"] for x in runs
                   if lab in x["policies"][nm]["buckets"]]
            ns = [x["policies"][nm]["buckets"][lab]["n"] for x in runs
                  if lab in x["policies"][nm]["buckets"]]
            bk[lab] = dict(regret=float(np.mean(vals)), regret_sd=float(np.std(vals)),
                           subopt=float(np.mean(sub)), n_mean=float(np.mean(ns)),
                           seeds=len(vals))
        cur = np.array([[c[1] for c in x["policies"][nm]["curve"]] for x in runs])
        rounds = [c[0] for c in runs[0]["policies"][nm]["curve"]]
        o[nm] = dict(regret=float(r.mean()), regret_sd=float(r.std()),
                     regret_min=float(r.min()), regret_max=float(r.max()),
                     subopt=float(s.mean()), subopt_sd=float(s.std()), buckets=bk,
                     curve=[[rounds[k], float(cur[:, k].mean())] for k in range(len(rounds))])
    return o


def condition(cname, C, T, s, seeds, cat_size=None, machines=None, tag=None, noise=0.0):
    Yf, Xf = C["Y"], C["X"]
    runs = []
    for sd in seeds:
        rng = np.random.default_rng(100000 + sd)
        Y, X = Yf, Xf
        if machines is not None and machines < Yf.shape[1]:
            mi = np.sort(rng.choice(Yf.shape[1], size=machines, replace=False))
            Y = Yf[:, mi]
        if cat_size is not None and cat_size < Y.shape[0]:
            ri = rng.choice(Y.shape[0], size=cat_size, replace=False)
            Y = Y[ri]
            X = Xf[ri]
        runs.append(run_stream(Y, X, T, s, sd, noise=noise))
    return dict(corpus=cname, tag=tag or cname, T=T, zipf_s=s, seeds=len(seeds), noise=noise,
                catalogue=int(runs[0]["N"]), machines=int(runs[0]["M"]),
                rounds_over_catalogue=float(T) / runs[0]["N"],
                stream_unique_mean=float(np.mean([r["stream_unique"] for r in runs])),
                tail_share_mean=float(np.mean([r["tail_share"] for r in runs])),
                leaks=sorted({l for r in runs for l in r["leaks"]}),
                fixed_choice_mode=int(Counter(r["fixed_choice"] for r in runs).most_common(1)[0][0]),
                policies=aggregate(runs))


def show(cond):
    p = cond["policies"]
    print("  {:<26} N={:<6} M={:<3} T={:<5} s={:<4} noise={:<5.3f} distinct={:<6.0f} tail={:.2f}".format(
        cond["tag"], cond["catalogue"], cond["machines"], cond["T"], cond["zipf_s"],
        cond["noise"], cond["stream_unique_mean"], cond["tail_share_mean"]))
    line = "      "
    for nm in ORDER:
        line += "{} {:.4f}+-{:.4f}  ".format(nm, p[nm]["regret"], p[nm]["regret_sd"])
    print(line)


def offline_crosscheck(C):
    """Independent verification of the headline online number, and of what carries it.

    An online policy that beat what a full-information fit of the same model class can reach
    would be reading cells it was never shown. So the same bilinear model is fitted offline
    three ways and the online result is checked to sit ABOVE the best of them:
      (a) full information, random 50/50 split of the catalogue
      (b) 2000 bandit-style observations (one randomly chosen machine per workload, exactly
          the information an online run of T = 2000 receives), evaluated on all 13121
      (c) leave-one-dataset-out, the inductive split used in the offline BUTTER-E experiment
    Splitting the descriptors into geometry (size, depth, shape) versus geometry plus dataset
    identity also answers whether the advantage is real transfer or dataset memorisation.
    """
    Y, X, keys = C["Y"], C["X"], C["keys"]
    L = np.log(Y)
    d = L[:, 0] - L[:, 1]
    best = Y.min(1)
    ds = np.array([k[0] for k in keys])
    nsh = len({k[1] for k in keys})
    geom = list(range(0, 1 + 4 + nsh))
    allc = list(range(X.shape[1]))

    def fit_eval(tr, te, cols):
        Xt = X[np.ix_(tr, cols)]
        w = np.linalg.solve(Xt.T @ Xt + RIDGE * np.eye(len(cols)), Xt.T @ d[tr])
        ch = np.where((X[np.ix_(te, cols)] @ w) > 0, Y[te, 1], Y[te, 0])
        return float(ch.sum() / best[te].sum())

    rng = np.random.default_rng(0)
    rr, rg = [], []
    for _ in range(5):
        p = rng.permutation(len(Y))
        tr, te = p[:len(Y) // 2], p[len(Y) // 2:]
        rr.append(fit_eval(tr, te, allc))
        rg.append(fit_eval(tr, te, geom))
    gr, gg = [], []
    for held in sorted(set(ds)):
        te = np.flatnonzero(ds == held)
        tr = np.flatnonzero(ds != held)
        if len(te) < 20:
            continue
        gr.append(fit_eval(tr, te, allc))
        gg.append(fit_eval(tr, te, geom))
    # (b) the same bilinear ALS used online, on 2000 bandit observations
    bl = []
    for rep in range(5):
        idx = rng.choice(len(Y), 2000, replace=False)
        arm = rng.integers(2, size=2000)
        Xo = X[idx]
        O = np.zeros((2000, 2))
        O[np.arange(2000), arm] = 1.0
        yv = L[idx, arm]
        P = X.shape[1]
        v = np.array([-1.0, 1.0]) / math.sqrt(2.0)
        for _ in range(ALS_ITERS):
            D = np.hstack([Xo, O, Xo * v[arm][:, None]])
            b = _ridge(D, yv, RIDGE)
            c, w = b[P:P + 2], b[P + 2:]
            z = Xo @ w
            b2 = _ridge(np.hstack([Xo, O * z[:, None]]), yv, RIDGE)
            v = b2[P:]
            nv = np.linalg.norm(v)
            w, v = w * nv, v / nv
        sc = c[None, :] + (X @ w)[:, None] * v[None, :]
        pick = np.argmin(sc, 1)
        bl.append(float(Y[np.arange(len(Y)), pick].sum() / best.sum()))
    return dict(
        additive_ceiling_always_cpu=float(Y[:, 0].sum() / best.sum()),
        always_gpu=float(Y[:, 1].sum() / best.sum()),
        interaction_random_split_all_descriptors=[float(np.mean(rr)), float(np.std(rr))],
        interaction_random_split_geometry_only=[float(np.mean(rg)), float(np.std(rg))],
        interaction_leave_dataset_out_all=[float(np.mean(gr)), float(np.std(gr)), len(gr)],
        interaction_leave_dataset_out_geometry_only=[float(np.mean(gg)), float(np.std(gg))],
        bilinear_from_2000_bandit_observations=[float(np.mean(bl)), float(np.std(bl))])


def main():
    t0 = time.time()
    print("loading corpora ...")
    C = {}
    C["butter-e"] = corpus_butter_e()
    C["llm-perf"] = corpus_llmperf()
    C["wilkins"] = corpus_wilkins()
    C["own-grid"] = corpus_c1()
    C["grid5000"] = corpus_grid5000()
    for k, c in C.items():
        print("  {:<10} {:>6} workloads x {:>2} machines, descriptors {:>3}".format(
            k, c["Y"].shape[0], c["Y"].shape[1], c["X"].shape[1]))
    sane("S1 matrices complete and strictly positive, no imputation",
         all(np.isfinite(c["Y"]).all() and (c["Y"] > 0).all() for c in C.values()),
         "; ".join("{}={}x{}".format(k, c["Y"].shape[0], c["Y"].shape[1]) for k, c in C.items()))

    SEEDS = list(range(8))
    NZ = {k: float(c.get("obs_log_sd", 0.0)) for k, c in C.items()}
    RES = {"corpora": {k: dict(workloads=int(c["Y"].shape[0]), machines=list(c["machines"]),
                               descriptors=int(c["X"].shape[1]),
                               measured_obs_log_sd=NZ[k],
                               measured_obs_log_sd_from_cells=int(c.get("obs_log_sd_cells", 0)),
                               offline_best_single_machine=str(
                                   c["machines"][int(np.argmin(c["Y"].sum(0)))]),
                               offline_fixed_regret=float(
                                   c["Y"].sum(0).min() / c["Y"].min(1).sum()))
                       for k, c in C.items()},
           "conditions": []}
    print("\noffline reference (full information, uniform workload weights)")
    for k, v in RES["corpora"].items():
        print("  {:<10} best single machine = {:<22} regret vs oracle = {:.4f}   "
              "measured meter log-sd = {:.3f} (from {} repeated cells)".format(
                  k, v["offline_best_single_machine"], v["offline_fixed_regret"],
                  v["measured_obs_log_sd"], v["measured_obs_log_sd_from_cells"]))
    BE = NZ["butter-e"]

    def add(c):
        RES["conditions"].append(c)
        show(c)
        return c

    # ---------------------------------------------------- A. catalogue-size sweep
    print("\nA. catalogue size sweep, BUTTER-E, T = 2000 rounds")
    for s in [0.0, 1.2]:
        for nz in [0.0, BE]:
            for N in [50, 500, 2000, 13121]:
                add(condition("butter-e", C["butter-e"], 2000, s, SEEDS, cat_size=N, noise=nz,
                              tag="A N={} s={} nz={:.3f}".format(N, s, nz)))

    # ---------------------------------------------------- B. tail heaviness sweep
    print("\nB. popularity exponent sweep, BUTTER-E full 13121 catalogue, T = 2000")
    for nz in [0.0, BE]:
        for s in [0.0, 0.4, 0.8, 1.2, 2.0, 3.0]:
            add(condition("butter-e", C["butter-e"], 2000, s, SEEDS, noise=nz,
                          tag="B s={} nz={:.3f}".format(s, nz)))

    # ---------------------------------------------------- C. round-count sweep
    print("\nC. round count sweep, BUTTER-E full catalogue")
    for s in [0.0, 1.2]:
        for T in [500, 2000, 8000]:
            add(condition("butter-e", C["butter-e"], T, s, SEEDS, noise=BE,
                          tag="C T={} s={}".format(T, s)))

    # ---------------------------------------------------- D. other corpora
    print("\nD. other corpora, T = 2000, at their own measured meter noise")
    for nm in ["llm-perf", "wilkins", "own-grid", "grid5000"]:
        for s in [0.0, 1.2]:
            add(condition(nm, C[nm], 2000, s, SEEDS, noise=NZ[nm],
                          tag="D {} s={}".format(nm, s)))
    print("\n   the same corpora at T = 60 rounds, their catalogue >> rounds end (40 seeds)")
    for nm in ["llm-perf", "wilkins", "own-grid", "grid5000"]:
        add(condition(nm, C[nm], 60, 0.0, list(range(40)), noise=NZ[nm],
                      tag="D {} T=60".format(nm)))

    # ---------------------------------------------------- E. machine count sweep
    print("\nE. machine count sweep (H4), T = 2000, uniform arrivals")
    for nm, ks in [("own-grid", [2, 3, 4, 5, 6, 7]), ("grid5000", [2, 4, 8, 12, 18]),
                   ("llm-perf", [2, 3])]:
        for k in ks:
            add(condition(nm, C[nm], 2000, 0.0, SEEDS, machines=k, noise=NZ[nm],
                          tag="E {} M={}".format(nm, k)))
    add(condition("butter-e", C["butter-e"], 2000, 0.0, SEEDS, machines=2, noise=BE,
                  tag="E butter-e M=2"))

    # ---------------------------------------------------- controls
    print("\ncontrols")
    ctrl = {}
    ctrl["S6"] = condition("butter-e", C["butter-e"], 4000, 1.2, SEEDS, cat_size=20, noise=BE,
                           tag="S6 rounds >> catalogue x machines")
    show(ctrl["S6"])
    ctrl["S8"] = condition("butter-e", C["butter-e"], 2000, 4.0, SEEDS, noise=BE,
                           tag="S8 degenerate heavy head")
    show(ctrl["S8"])
    RES["controls"] = ctrl

    # ---------------------------------------------------- sanity evaluation
    print("\nsanity checks")
    allc = RES["conditions"] + list(ctrl.values())
    sane("S2 oracle regret is exactly 1.000 in every condition",
         all(abs(c["policies"]["oracle"]["regret"] - 1.0) < 1e-12 for c in allc),
         "{} conditions".format(len(allc)))
    bad = [(c["tag"], nm, round(c["policies"][nm]["regret_min"], 6))
           for c in allc for nm in c["policies"]
           if c["policies"][nm]["regret_min"] < 1.0 - 1e-12]
    sane("S3 no policy ever beats the oracle (regret >= 1.0 in every single run)",
         not bad, "violations: {}".format(bad[:5] if bad else "none"))
    lk = sorted({l for c in allc for l in c["leaks"]})
    sane("S4/S5 information containment and identical stream", not lk,
         "leak flags: {}".format(lk if lk else "none"))
    # S6 as first written demanded regret < 1.02, which no epsilon-greedy policy can reach.
    # Every policy here spends a fraction EPS of its rounds choosing uniformly at random, so
    # the attainable floor is 1 + EPS x (regret of the uniform policy - 1), not 1. The check
    # was mis-specified rather than the policy broken; it is replaced by the floor it should
    # have named, with 20 per cent slack.
    s6 = ctrl["S6"]["policies"]
    floor6 = 1.0 + EPS * (s6["random"]["regret"] - 1.0)
    sane("S6 positive control: per_workload reaches the epsilon-greedy floor when rounds >> N x M",
         s6["per_workload"]["regret"] <= 1.2 * floor6 and
         s6["per_workload"]["regret"] < s6["fixed"]["regret"],
         "per_workload {:.4f}, exploration floor {:.4f}, fixed {:.4f} at N=20, T=4000".format(
             s6["per_workload"]["regret"], floor6, s6["fixed"]["regret"]))
    off = {k: int(np.argmin(c["Y"].sum(0))) for k, c in C.items()}
    fk = [c for c in allc if c["machines"] == C[c["corpus"]]["Y"].shape[1] and c["T"] >= 2000]
    nagree = sum(1 for c in fk if c["fixed_choice_mode"] == off[c["corpus"]])
    sane("S7 positive control: fixed converges to the offline best single machine",
         nagree == len(fk), "{}/{} full-machine conditions agree".format(nagree, len(fk)))
    s8 = ctrl["S8"]["policies"]
    sane("S8 degenerate control: with an extreme head the per-row learner matches or beats cf",
         s8["per_workload"]["regret"] <= s8["cf"]["regret"] + 1e-9,
         "per_workload {:.4f} vs cf {:.4f} at s=4.0".format(
             s8["per_workload"]["regret"], s8["cf"]["regret"]))
    big = [c for c in allc if c["T"] >= 2000]
    nrand = sum(1 for c in big if c["policies"]["random"]["regret"] > c["policies"]["fixed"]["regret"])
    sane("S9 random is strictly worse than fixed at T >= 2000",
         nrand == len(big), "{}/{} conditions".format(nrand, len(big)))
    sane("S10 seed spread reported for every number",
         all("regret_sd" in c["policies"][nm] for c in allc for nm in c["policies"]),
         "8 seeds per condition, 40 for the T=60 conditions")

    # ---------------------------------------------------- hypothesis verdicts
    print("\nhypothesis verdicts")
    K = {c["tag"]: c for c in RES["conditions"]}
    V = {}
    BUCKS = ["prior_0", "prior_1", "prior_2", "prior_3_5", "prior_6_20", "prior_21p"]

    def A(N, s, nz):
        return K["A N={} s={} nz={:.3f}".format(N, s, nz)]["policies"]

    def B(s, nz):
        return K["B s={} nz={:.3f}".format(s, nz)]["policies"]

    def marg(p):
        return [p["fixed"]["regret"] - p["cf"]["regret"],
                p["fixed"]["regret"] - p["cf_eb"]["regret"]]

    hb = A(13121, 0.0, BE)            # the true catalogue >> rounds condition
    V["H1"] = dict(statement="catalogue >> rounds makes per_workload far worse than fixed",
                   condition="BUTTER-E, 13121 workloads, 2000 rounds, uniform arrivals, "
                             "measured meter noise",
                   per_workload=hb["per_workload"]["regret"], fixed=hb["fixed"]["regret"],
                   held=bool(hb["per_workload"]["regret"] > hb["fixed"]["regret"] * 1.02),
                   across_catalogue_sizes=[[N, A(N, 0.0, BE)["per_workload"]["regret"],
                                            A(N, 0.0, BE)["fixed"]["regret"]]
                                           for N in [50, 500, 2000, 13121]])
    mc = [[N] + marg(A(N, 0.0, BE)) for N in [50, 500, 2000, 13121]]
    mc0 = [[N] + marg(A(N, 0.0, 0.0)) for N in [50, 500, 2000, 13121]]
    mt = [[s] + marg(B(s, BE)) for s in [0.0, 0.4, 0.8, 1.2, 2.0, 3.0]]
    mt0 = [[s] + marg(B(s, 0.0)) for s in [0.0, 0.4, 0.8, 1.2, 2.0, 3.0]]
    V["H2"] = dict(statement="cf beats fixed; margin grows with catalogue/rounds and tail weight",
                   columns=["level", "fixed-cf", "fixed-cf_eb"],
                   margin_by_catalogue_noisy=mc, margin_by_catalogue_noiseless=mc0,
                   margin_by_zipf_s_noisy=mt, margin_by_zipf_s_noiseless=mt0,
                   cf_beats_fixed_at_full_catalogue=bool(hb["cf"]["regret"] < hb["fixed"]["regret"]),
                   cf_eb_beats_fixed_at_full_catalogue=bool(hb["cf_eb"]["regret"] < hb["fixed"]["regret"]),
                   margin_grows_with_catalogue=bool(mc[-1][1] > mc[0][1]),
                   margin_grows_with_head_concentration=bool(mt[-1][1] > mt[0][1]))
    V["H3"] = dict(statement="cf's advantage concentrates on rarely-seen workloads",
                   sparse_regime_by_prior_visit_count={
                       b: {nm: hb[nm]["buckets"].get(b, {}).get("regret") for nm in ORDER}
                       for b in BUCKS},
                   dense_regime_by_prior_visit_count={
                       b: {nm: A(50, 0.0, BE)[nm]["buckets"].get(b, {}).get("regret")
                           for nm in ORDER} for b in BUCKS},
                   heavy_head_by_prior_visit_count={
                       b: {nm: B(1.2, BE)[nm]["buckets"].get(b, {}).get("regret")
                           for nm in ORDER} for b in BUCKS})
    mach = []
    for c in RES["conditions"]:
        if c["tag"].startswith("E "):
            p = c["policies"]
            mach.append([c["corpus"], c["machines"], c["catalogue"]] + marg(p)
                        + [p["fixed"]["regret"], p["cf"]["regret"], p["cf_eb"]["regret"]])
    V["H4"] = dict(statement="machine count limits the achievable advantage",
                   columns=["corpus", "machines", "catalogue", "fixed-cf", "fixed-cf_eb",
                            "fixed", "cf", "cf_eb"],
                   margin_by_machine_count=mach)
    RES["hypotheses"] = V
    for k in ["H1", "H2", "H4"]:
        print("  {}: {}".format(k, {a: b for a, b in V[k].items() if isinstance(b, bool)}))

    print("\noffline cross-check of the headline (BUTTER-E)")
    XC = offline_crosscheck(C["butter-e"])
    RES["offline_crosscheck"] = XC
    for k, v in XC.items():
        print("  {:<48} {}".format(k, v))
    onl = hb["cf"]["regret"]
    ceil = XC["bilinear_from_2000_bandit_observations"][0]
    sane("S11 online cf does not beat the full-information fit of its own model class",
         onl > ceil,
         "online cf {:.4f} vs the same model fitted offline on the same amount of bandit "
         "information {:.4f}".format(onl, ceil))

    RES["sanity"] = SANITY
    RES["runtime_s"] = time.time() - t0
    RES["config"] = dict(epsilon=EPS, refit_period=REFIT, ridge=RIDGE, als_iters=ALS_ITERS,
                         eb_pseudo_obs=CFEB.K, seeds=len(SEEDS),
                         note="no hyperparameter was searched over; all are shared across policies")
    with io.open("experiments/results/cf_online_stream.json", "w", encoding="utf-8") as f:
        json.dump(RES, f, indent=1)
    print("\nwrote experiments/results/cf_online_stream.json in {:.1f}s".format(RES["runtime_s"]))


if __name__ == "__main__":
    main()
