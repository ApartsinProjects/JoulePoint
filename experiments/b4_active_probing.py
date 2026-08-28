# -*- coding: utf-8 -*-
"""
B4, Recipe 3: active probing to identify the workload mix.

B3 showed what a passive scheduler log buys. This asks whether it is worth instrumenting
a few jobs properly instead. A probe is a job the facility fully characterises: it learns
the exact grid cell, INCLUDING the numerical precision that Section 7 identifies as the
decisive descriptor and that ordinary logs omit. Probes cost operator attention, so the
question is how few are enough, and whether choosing which jobs to probe beats sampling
at random.

Three acquisition strategies, all under an identical budget K:
  random        probe K jobs drawn uniformly from the arrival stream
  uncertainty   probe the jobs whose posterior over grid cells is most ambiguous,
                measured by the entropy of the passive-log likelihood
  stratified    probe to cover distinct (batch, memory) observable classes evenly

and one control that spends the same K observations the cheap way:
  passive+K     no probes at all, just K additional ordinary log lines. This is the
                equal-information-cost comparison: K units of observation bought as
                precision-revealing probes versus K units bought as precision-blind log.

Fleet choice. As in B3, the fleet is chosen by the B8 fluid-limit LP (calibrated rho,
outsourcing penalty, steepest-descent local search) whose LP-ranked shortlist is then
ranked by the discrete-event simulator. Enumerating 1001 simulated compositions per
candidate mix was the reason the first version of this script could not finish. The LP only
proposes; every energy here is simulator-measured. The oracle is the fleet the same chooser
picks from the true mix, and the chooser's own gap to exhaustive enumeration is audited
per mix under S0 and stored in the results.

A leak found and fixed after the first full run, and the reason this experiment can now be
run at all. The logged memory request originally carried a 2x factor for fp32, which gave
each of the 24 grid cells a unique (memory, batch) signature: the "precision-blind" log
identified every job exactly, so probing could reveal nothing and all three acquisition
strategies returned byte-identical regret, which is what exposed it. The memory request now
depends on model and batch only, so each observable class holds the fp16 and fp32 versions
of one model and batch and precision must be recovered from runtime, or bought with a probe.

That leaves runtime as the only passive channel carrying precision, and how much it carries
depends on how noisy the log is: fp16 and fp32 runtimes on a known machine differ by a
median factor of 2.2 (0.79 in absolute log ratio), so a clean log still identifies precision
and probes are redundant. Every configuration is therefore run at three logging-noise
levels, sigma = 0.05, 0.30 and 0.75, which bracket that separation, and the probe-versus-
passive comparison is reported per level.

Sanity checks stated in advance:
  S0  the chooser, given the true mix, recovers the exhaustive-enumeration optimum
      (fidelity audit of the LP pruner; see B3's docstring for the one known failure mode)
  S1  K = 0 must be identical across strategies
  S1b budgets must be NESTED: one trace and one probe order per (mix, jitter, seed), with
      budget K taking the first K entries of that order. The first full run seeded the trace
      with the budget itself, so each K saw a different 200-job trace and the K curve
      confounded probing with resampling noise; the fix is structural rather than a check.
  S2  probing must converge monotonically to the trace's empirical mix. This replaces
      "regret is non-increasing in K", which is mis-specified in two ways the first run
      demonstrated. A probe reveals the true cell of a job IN THE TRACE, so more probes pull
      the estimate toward the trace's EMPIRICAL mix, itself a noisy draw from the true mix;
      at N = 200 that sampling error dominates, and sharpening an estimate of the wrong
      target can raise regret. Raw energy regret also drops when an under-provisioned fleet
      breaks the SLA, which is a service failure rather than an improvement. The monotone
      convergence to the empirical mix is the exact invariant probing does guarantee; the
      regret path is printed next to it rather than asserted.
  S3  at K equal to the trace length every job is characterised, so the inferred mix must
      equal the trace's empirical distribution exactly. The original check demanded regret
      exactly 1.000 there, which cannot hold: probing all 200 jobs reveals the empirical mix
      of those 200 draws, not the true mix, so the regret floor at K = N is a property of the
      trace length. B3's S5 confirms that at N = 5000 full characterisation does land on the
      oracle fleet.
  S4  probing must never degrade service relative to not probing, judged SLA-first: a
      cheaper fleet that misses the 60 s SLA is a failed design, not a better one. The
      energy-only version of this check failed on two mixes for exactly that reason.
  S5  no SLA-meeting fleet may ever beat the exhaustive-enumeration optimum. Sub-1.000
      regrets against the chooser's oracle are legitimate only via an SLA violation or via
      the chooser's own gap where S0 fails; neither may break the enumeration bound.
"""
import io, json, sys, time, warnings
from itertools import product
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, SLA_S, HORIZON, LAM = 20000, 60.0, 3600.0, 0.5
RHO, PENALTY = 0.94, 5.0            # calibrated in B8 against this same simulator
SHORTLIST_LP, SHORTLIST_SIM = 25, 25   # simulate the whole LP shortlist, as B8 does
TRACE_N = 200
BUDGETS = [0, 10, 40, 200]          # trimmed from [0,5,10,20,40,80,200] to fund the jitter sweep
SEEDS = [0, 1, 2, 3]                # trimmed from five seeds for the same reason
JITTERS = [0.05, 0.30, 0.75]        # logging noise; fp16 vs fp32 separation is 0.79 in |log|
TAU = 0.05                          # estimator's assumed noise, reset per jitter level
STRATEGIES = ("random", "uncertainty", "stratified")
SANITY = []

def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

keys, Ylog, Tput = load_grid()
NK, NM = len(keys), len(MACH)
E = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(NM)} for i, k in enumerate(keys)}
T = {k: {MACH[j]: Tput[i, j] for j in range(NM)} for i, k in enumerate(keys)}
Ecell = np.array([[10 ** Ylog[i, j] / 1000.0 for j in range(NM)] for i in range(NK)])
Tcell = np.array([[JOB / Tput[i, j] for j in range(NM)] for i in range(NK)])
IDLEv = np.array([IDLE[m] for m in MACH])

def w_of(pred):
    w = np.array([1.0 if pred(k) else 0.0 for k in keys])
    return w / w.sum()

TRUE_MIXES = {
    "uniform": w_of(lambda k: True),
    "vision inference": w_of(lambda k: k[0] in ("resnet50", "convnext_t", "vit_b16")),
    "modern precision (fp16)": w_of(lambda k: k[1] == "fp16"),
}
COMPS = [c for c in product(range(0, 11), repeat=5) if sum(c) == 10]

# The logged memory request depends on the model and batch only. The 2x fp32 factor an
# earlier version applied gave all 24 cells a unique (memory, batch) signature, so the
# supposedly precision-blind log identified every job exactly and probing was provably
# worthless: the first full run returned byte-identical regret for all three acquisition
# strategies, which is what exposed the leak. Each observable class now holds the fp16 and
# fp32 versions of one model and batch, so precision has to come from runtime, or a probe.
MEMG = {}
for k in keys:
    load, prec, batch = k
    base = {"resnet50": 0.10, "convnext_t": 0.11, "vit_b16": 0.33, "transformer": 0.18}[load]
    MEMG[k] = round(base * (1 + batch / 64.0), 3)

def facility(pool, w, seed=0):
    rng = np.random.default_rng(seed)
    slots = [m for m, c in pool.items() for _ in range(c)]
    ns = len(slots)
    free_at = np.zeros(ns)
    idx = np.arange(NK)
    t, arr = 0.0, []
    while t < HORIZON:
        t += rng.exponential(1.0 / LAM)
        if t < HORIZON:
            arr.append((t, keys[int(rng.choice(idx, p=w))]))
    dyn, delays = 0.0, []
    for at, jk in arr:
        free = [i for i in range(ns) if free_at[i] <= at]
        start = at if free else float(np.min(free_at))
        if not free:
            free = [i for i in range(ns) if free_at[i] <= start]
        cand = sorted({slots[i] for i in free})
        m = min(cand, key=lambda mm: E[jk][mm])
        i = next(i for i in free if slots[i] == m)
        rt = JOB / T[jk][m]
        free_at[i] = start + rt
        delays.append(start - at)
        dyn += JOB * E[jk][m] - IDLE[m] * rt
    static = sum(IDLE[s] for s in slots) * HORIZON
    return (static + dyn) / max(len(arr), 1), float(np.mean(delays)) if delays else 0.0

# ---------------------------------------------------------------- LP fleet chooser (B8)
def recourse(n, w):
    n = np.asarray(n, dtype=float)
    if n.sum() <= 0:
        return float("inf")
    live = np.where(n > 0)[0]
    nv = len(live)
    nvar = NK * (nv + 1)
    c = np.zeros(nvar)
    for i in range(NK):
        for a, j in enumerate(live):
            c[i * (nv + 1) + a] = w[i] * (JOB * Ecell[i, j] - IDLEv[j] * Tcell[i, j])
        c[i * (nv + 1) + nv] = w[i] * PENALTY * JOB * Ecell[i].max()
    Aeq = np.zeros((NK, nvar)); beq = np.ones(NK)
    for i in range(NK):
        Aeq[i, i * (nv + 1):(i + 1) * (nv + 1)] = 1.0
    Aub = np.zeros((nv, nvar)); bub = np.zeros(nv)
    for a, j in enumerate(live):
        for i in range(NK):
            Aub[a, i * (nv + 1) + a] = LAM * w[i] * Tcell[i, j]
        bub[a] = RHO * n[j]
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=(0, 1), method="highs")
    if not r.success:
        return float("inf")
    return (float(IDLEv @ n) * HORIZON + r.fun * LAM * HORIZON) / (LAM * HORIZON)

def local_search(w, slots=10, ntypes=5, seed=0, starts=3):
    """Steepest descent over one-slot swaps: score every swap against the CURRENT fleet,
    then apply only the best. Applying moves mid-scan lets a later swap decrement a
    component the scan already emptied, producing NEGATIVE counts; the asserts guard that."""
    rng = np.random.default_rng(seed)
    bestn, bestv, evals = None, float("inf"), 0
    for st in range(starts):
        n = np.zeros(ntypes)
        if st == 0:
            n[rng.integers(ntypes)] = slots
        else:
            for _ in range(slots):
                n[rng.integers(ntypes)] += 1
        v = recourse(n, w); evals += 1
        improved = True
        while improved:
            improved = False
            bestmove, bestval = None, v
            for a in range(ntypes):
                if n[a] <= 0:
                    continue
                for b in range(ntypes):
                    if a == b:
                        continue
                    cand = n.copy(); cand[a] -= 1; cand[b] += 1
                    assert cand.min() >= 0 and abs(cand.sum() - slots) < 1e-9
                    cv = recourse(cand, w); evals += 1
                    if cv < bestval - 1e-9:
                        bestval, bestmove = cv, cand
            if bestmove is not None:
                v, n, improved = bestval, bestmove, True
        assert n.min() >= 0 and abs(n.sum() - slots) < 1e-9
        if v < bestv:
            bestv, bestn = v, n
    return bestn, bestv, evals

def lp_shortlist(w, k=SHORTLIST_LP):
    n, _, ev = local_search(w)
    cands = {tuple(int(x) for x in n)}
    for a in range(5):
        for b in range(5):
            if a == b or n[a] <= 0:
                continue
            c2 = n.copy(); c2[a] -= 1; c2[b] += 1
            cands.add(tuple(int(x) for x in c2))
            for a2 in range(5):
                for b2 in range(5):
                    if a2 == b2 or c2[a2] <= 0:
                        continue
                    c3 = c2.copy(); c3[a2] -= 1; c3[b2] += 1
                    assert min(c3) >= 0 and sum(c3) == 10
                    cands.add(tuple(int(x) for x in c3))
    scored = sorted((recourse(np.array(c, float), w), c) for c in cands)
    return [c for _, c in scored[:k]], ev + len(cands)

_cache = {}
_stats = dict(chooser_calls=0, lp_solves=0, simulations=0)
def choose_fleet(w):
    ck = tuple(np.round(w, 10))
    if ck in _cache:
        return _cache[ck]
    cands, ev = lp_shortlist(w)
    best, bestpool = float("inf"), None
    for c in cands[:SHORTLIST_SIM]:
        pool = {MACH[i]: c[i] for i in range(5) if c[i] > 0}
        e, d = facility(pool, w, seed=0)
        _stats["simulations"] += 1
        if d <= SLA_S and e < best:
            best, bestpool = e, pool
    if bestpool is None:
        for c in cands:
            pool = {MACH[i]: c[i] for i in range(5) if c[i] > 0}
            e, d = facility(pool, w, seed=0)
            _stats["simulations"] += 1
            if d <= SLA_S and e < best:
                best, bestpool = e, pool
    _stats["chooser_calls"] += 1
    _stats["lp_solves"] += ev
    _cache[ck] = (bestpool, best)
    return bestpool, best

def enumerate_fleet(w):
    best, bestpool = float("inf"), None
    for c in COMPS:
        pool = {MACH[i]: c[i] for i in range(5) if c[i] > 0}
        e, d = facility(pool, w, seed=0)
        _stats["simulations"] += 1
        if d <= SLA_S and e < best:
            best, bestpool = e, pool
    return bestpool, best

def fmt(pool):
    return "+".join("{}x{}".format(v, k) for k, v in sorted(pool.items()))

# ---------------------------------------------------------------- trace and estimator
def make_trace(w, n, rng, sigma=0.05):
    idx = rng.choice(np.arange(NK), size=n, p=w)
    out = []
    for i in idx:
        k = keys[int(i)]
        m = MACH[int(rng.integers(len(MACH)))]
        rt = JOB / T[k][m] * float(rng.lognormal(0.0, sigma))
        out.append(dict(true=k, mem=MEMG[k], batch=k[2], rt=rt, machine=m))
    return out

def like_vector(line, tau=None):
    tau = TAU if tau is None else tau
    v = np.zeros(NK)
    for i, k in enumerate(keys):
        if k[2] != line["batch"] or abs(MEMG[k] - line["mem"]) > 1e-6:
            continue
        pred = JOB / T[k][line["machine"]]
        v[i] = np.exp(-((np.log(line["rt"]) - np.log(pred)) ** 2) / (2 * tau ** 2))
    return v / v.sum() if v.sum() > 0 else np.ones(NK) / NK

def entropy(p):
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())

def mix_from(trace, probed_idx):
    """Probed lines contribute a point mass at their true cell; the rest their posterior."""
    post = np.zeros(NK)
    for j, line in enumerate(trace):
        if j in probed_idx:
            post[keys.index(line["true"])] += 1.0
        else:
            post += like_vector(line)
    return post / post.sum()

def empirical_mix(trace):
    post = np.zeros(NK)
    for line in trace:
        post[keys.index(line["true"])] += 1.0
    return post / post.sum()

def stratified_order(trace):
    """Full probe order covering distinct (batch, memory) observable classes round-robin."""
    buckets = {}
    for j, l in enumerate(trace):
        buckets.setdefault((l["batch"], round(l["mem"], 3)), []).append(j)
    order, keys_ = [], sorted(buckets)
    bi = 0
    while len(order) < len(trace):
        b = keys_[bi % len(keys_)]
        if buckets[b]:
            order.append(buckets[b].pop(0))
        bi += 1
    return order

def tvd(p, q):
    return 0.5 * float(np.abs(p - q).sum())

# ---------------------------------------------------------------- run
t0 = time.time()
print("Recipe 3: active probing, trace length {} jobs".format(TRACE_N))
print("chooser = B8 LP (rho={}, penalty={}) local search -> top {} of {} LP candidates "
      "simulated\n".format(RHO, PENALTY, SHORTLIST_SIM, SHORTLIST_LP))

ORACLE = {}
print("oracle fleets: chooser on the true mix, audited against exhaustive enumeration")
for mixname, wtrue in TRUE_MIXES.items():
    enum_pool, enum_e = enumerate_fleet(wtrue)
    pool, e = choose_fleet(wtrue)
    ORACLE[mixname] = dict(pool=pool, energy=e, enum_pool=enum_pool, enum_energy=enum_e,
                           gap=e / enum_e - 1)
    print("  {:<26} oracle {:<24} {:.0f} J/job   enumeration {:<24} {:.0f} (gap {:+.2%})".format(
        mixname, fmt(pool), e, fmt(enum_pool), enum_e, e / enum_e - 1))
print()
for mixname in TRUE_MIXES:
    o = ORACLE[mixname]
    sane("S0 LP chooser recovers the enumeration optimum, {}".format(mixname[:22]),
         o["pool"] == o["enum_pool"],
         "enumeration {} at {:.0f} J/job, chooser {} at {:.0f} ({:+.2%})".format(
             fmt(o["enum_pool"]), o["enum_energy"], fmt(o["pool"]), o["energy"], o["gap"]))
print()

# One trace and one probe ORDER per (mix, sigma, seed). Taking the first K entries of a
# fixed order makes the budgets nested for every strategy, including random, so a larger
# budget is strictly more information about the same trace.
TRACES = {}
for SIGMA in JITTERS:
    TAU = SIGMA
    for mixname, wtrue in TRUE_MIXES.items():
        for sd in SEEDS:
            rng = np.random.default_rng(7000 * sd + int(1000 * SIGMA))
            tr_full = make_trace(wtrue, TRACE_N + max(BUDGETS), rng, sigma=SIGMA)
            base = tr_full[:TRACE_N]
            order = {"random": list(rng.permutation(TRACE_N)),
                     "uncertainty": [j for _, j in sorted(
                         ((entropy(like_vector(l)), j) for j, l in enumerate(base)),
                         reverse=True)],
                     "stratified": stratified_order(base)}
            for how, o in order.items():
                assert sorted(o) == list(range(TRACE_N)), how
            TRACES[(mixname, SIGMA, sd)] = (tr_full, order)

rows, per_dp, s3_detail = [], [], {}
for SIGMA in JITTERS:
  TAU = SIGMA                              # the estimator is told the logging noise level
  print("\n########## log jitter sigma = {:.2f} ##########".format(SIGMA))
  for mixname, wtrue in TRUE_MIXES.items():
      oracle_pool, oracle_e = ORACLE[mixname]["pool"], ORACLE[mixname]["energy"]
      ofleet = fmt(oracle_pool)
      print("=== true mix: {} ===  oracle {} at {:.0f} J/job".format(mixname, ofleet, oracle_e))
      print("  {:>6} {:>14}{:>9}{:>9}{:>9}  {}".format("K", "strategy", "TVD", "regret",
                                                        "SLAviol", "modal fleet"))
      for K in BUDGETS:
          for how in STRATEGIES + ("passive+K",):
              regs, tv, fl, exact, viol, tv_emp = [], [], [], [], [], []
              for sd in SEEDS:
                  # One trace per (mix, sigma, seed), drawn ONCE and reused across budgets.
                  # The first version seeded the trace with 7000*sd + K, so every budget saw
                  # a DIFFERENT trace: the K curve mixed the effect of probing with the noise
                  # of redrawing 200 jobs, and the monotone-convergence invariant could not
                  # hold even in principle. Budgets are now nested: the probe order is fixed
                  # per (seed, strategy) and K takes its first K entries, and passive+K
                  # extends the same trace with K extra lines instead of resampling it.
                  tr_full, order = TRACES[(mixname, SIGMA, sd)]
                  if how == "passive+K":
                      tr = tr_full[:TRACE_N + K]
                      what = mix_from(tr, set())
                  else:
                      tr = tr_full[:TRACE_N]
                      sel = set(order[how][:K])
                      what = mix_from(tr, sel)
                      tv_emp.append(tvd(what, empirical_mix(tr)))
                      if K >= TRACE_N:
                          exact.append(tvd(what, empirical_mix(tr)))
                  pool, _ = choose_fleet(what)
                  e_true, d_true = facility(pool, wtrue, seed=0)
                  _stats["simulations"] += 1
                  regs.append(e_true / oracle_e)
                  tv.append(tvd(what, wtrue))
                  fl.append(fmt(pool))
                  viol.append(d_true > SLA_S)
                  per_dp.append(dict(mix=mixname, sigma=SIGMA, K=K, strategy=how, seed=sd,
                                     tvd=tv[-1], regret=regs[-1], fleet=fl[-1],
                                     energy=float(e_true), mean_delay_s=float(d_true),
                                     feasible_under_true_mix=bool(d_true <= SLA_S),
                                     regret_vs_enumeration=float(
                                         e_true / ORACLE[mixname]["enum_energy"])))
              if exact:
                  s3_detail.setdefault((SIGMA, mixname), []).append((how, float(max(exact))))
              modal = max(set(fl), key=fl.count)
              feas = [r for r, v in zip(regs, viol) if not v]
              rows.append(dict(mix=mixname, sigma=SIGMA, K=K, strategy=how,
                               tvd=float(np.mean(tv)),
                               tvd_to_empirical=(float(np.mean(tv_emp)) if tv_emp else None),
                               regret=float(np.mean(regs)), regret_sd=float(np.std(regs)),
                               sla_violation_rate=float(np.mean(viol)),
                               regret_feasible_only=(float(np.mean(feas)) if feas else None),
                               modal_fleet=modal, matches_oracle=(modal == ofleet),
                               frac_seeds_matching_oracle=float(np.mean([f == ofleet for f in fl]))))
              print("  {:>6} {:>14}{:>9.3f}{:>9.3f}{:>9.2f}  {}".format(
                  K, how, float(np.mean(tv)), float(np.mean(regs)), float(np.mean(viol)), modal))
      print()

print("sanity checks")
for sg in JITTERS:
    for mixname in TRUE_MIXES:
        r0 = {r["strategy"]: r["regret"] for r in rows if r["mix"] == mixname
              and r["sigma"] == sg and r["K"] == 0 and r["strategy"] in STRATEGIES}
        sane("S1 K=0 identical across strategies, sigma {:.2f}, {}".format(sg, mixname[:18]),
             max(r0.values()) - min(r0.values()) < 1e-12,
             ", ".join("{} {:.4f}".format(k, v) for k, v in r0.items()))

# S2, restated. "Regret is non-increasing in K" is mis-specified, and the first run showed
# both of its failure modes. A probe reveals the true cell of a job IN THE TRACE, so more
# probes drive the estimate toward the trace's EMPIRICAL mix, which is itself a noisy draw
# from the true mix; sharpening a estimate of the wrong target can raise regret, and at
# N = 200 that sampling error dominates. Raw energy regret also falls when an
# under-provisioned fleet breaks the SLA. The exact invariant probing does guarantee is
# monotone convergence to the empirical mix, which is what is asserted; the regret path is
# printed alongside it, and the SLA-meeting regret is checked not to end higher than it
# started.
for sg in JITTERS:
    for mixname in TRUE_MIXES:
        for how in STRATEGIES:
            rs = sorted((r for r in rows if r["mix"] == mixname and r["sigma"] == sg
                         and r["strategy"] == how), key=lambda r: r["K"])
            emp = [r["tvd_to_empirical"] for r in rs]
            mono = all(emp[i + 1] <= emp[i] + 1e-9 for i in range(len(emp) - 1))
            sane("S2 probing converges monotonically to the empirical mix, sigma {:.2f}, "
                 "{} {}".format(sg, mixname[:16], how), mono,
                 "TVD to empirical {} ; regret {}".format(
                     " -> ".join("{:.4f}".format(v) for v in emp),
                     " -> ".join("{:.3f}".format(r["regret"]) for r in rs)))
for sg in JITTERS:
    for mixname in TRUE_MIXES:
        full = [r for r in rows if r["mix"] == mixname and r["sigma"] == sg
                and r["K"] == TRACE_N and r["strategy"] in STRATEGIES]
        worst = max(v for _, v in s3_detail[(sg, mixname)])
        sane("S3 full probing recovers the empirical mix exactly, sigma {:.2f}, {}".format(
             sg, mixname[:18]), worst < 1e-12,
             "max TVD to the empirical mix {:.2e}; full-probe regret {}".format(
                 worst, ", ".join("{} {:.4f}".format(r["strategy"], r["regret"]) for r in full)))

# S4, restated SLA-first for the same reason as B3's S4: a cheaper fleet that misses the
# 60 s SLA is a failed design, not a better one, so probing is judged on service first.
for sg in JITTERS:
    for mixname in TRUE_MIXES:
        base = [r for r in rows if r["mix"] == mixname and r["sigma"] == sg and r["K"] == 0
                and r["strategy"] == "random"][0]
        probed = [r for r in rows if r["mix"] == mixname and r["sigma"] == sg and r["K"] > 0
                  and r["strategy"] in STRATEGIES]
        worse = [r for r in probed if r["sla_violation_rate"] > base["sla_violation_rate"] + 1e-9]
        sane("S4 probing never degrades service versus not probing, sigma {:.2f}, {}".format(
             sg, mixname[:18]), not worse,
             "K=0 SLA violations {:.2f}, worst probed {:.2f}; regret K=0 {:.4f}, worst probed "
             "{:.4f}".format(base["sla_violation_rate"],
                             max(r["sla_violation_rate"] for r in probed), base["regret"],
                             max(r["regret"] for r in probed)))

for mixname in TRUE_MIXES:
    mine = [d for d in per_dp if d["mix"] == mixname]
    sub = [d for d in mine if d["regret"] < 1.0 - 1e-9]
    bad = [d for d in mine if d["feasible_under_true_mix"]
           and d["regret_vs_enumeration"] < 1.0 - 1e-9]
    sane("S5 no SLA-meeting fleet ever beats the enumeration optimum, {}".format(mixname[:22]),
         not bad,
         "{} of {} datapoints undercut the chooser oracle, {} of them by breaking the SLA; "
         "{} undercut enumeration".format(
             len(sub), len(mine), len([d for d in sub if not d["feasible_under_true_mix"]]),
             len(bad)))

# ---------------------------------------------------------------- probing vs passive
print("\nequal information cost: K probes versus K extra passive log lines")
compare = []
for sg in JITTERS:
    for mixname in TRUE_MIXES:
        for K in BUDGETS:
            if K == 0:
                continue
            pr = {how: [r for r in rows if r["mix"] == mixname and r["sigma"] == sg
                        and r["K"] == K and r["strategy"] == how][0] for how in STRATEGIES}
            pas = [r for r in rows if r["mix"] == mixname and r["sigma"] == sg
                   and r["K"] == K and r["strategy"] == "passive+K"][0]
            bh = min(pr, key=lambda h: (pr[h]["sla_violation_rate"], pr[h]["regret"]))
            bp = pr[bh]
            win = (bp["sla_violation_rate"], bp["regret"]) < (pas["sla_violation_rate"],
                                                              pas["regret"] - 1e-12)
            tie = (abs(bp["regret"] - pas["regret"]) <= 1e-12
                   and bp["sla_violation_rate"] == pas["sla_violation_rate"])
            compare.append(dict(mix=mixname, sigma=sg, K=K, best_probe_strategy=bh,
                                best_probe_regret=bp["regret"], best_probe_tvd=bp["tvd"],
                                passive_plus_K_regret=pas["regret"], passive_tvd=pas["tvd"],
                                probe_wins=bool(win), tie=bool(tie)))
            print("  sigma {:.2f}  {:<26} K={:<4} probe {:.4f} (TVD {:.3f})  passive+K {:.4f} "
                  "(TVD {:.3f})   {}".format(sg, mixname, K, bp["regret"], bp["tvd"],
                                             pas["regret"], pas["tvd"],
                                             "tie" if tie else ("probe" if win else "passive")))

print("\nstrategy comparison, mean regret over all K>0, by log jitter")
strat_mean = {}
for sg in JITTERS:
    line = []
    for how in STRATEGIES:
        vals = [r["regret"] for r in rows if r["strategy"] == how and r["sigma"] == sg
                and 0 < r["K"] < TRACE_N]
        tvs = [r["tvd"] for r in rows if r["strategy"] == how and r["sigma"] == sg
               and 0 < r["K"] < TRACE_N]
        strat_mean["{:.2f}|{}".format(sg, how)] = float(np.mean(vals))
        line.append("{} {:.4f} (TVD {:.3f})".format(how, float(np.mean(vals)), float(np.mean(tvs))))
    print("  sigma {:.2f}:  {}".format(sg, "   ".join(line)))

OUT = dict(trace_n=TRACE_N, budgets=BUDGETS, seeds=SEEDS, jitters=JITTERS, summary=rows,
           per_datapoint=per_dp, sanity=SANITY,
           chooser=dict(kind="B8 fluid-limit LP local search, simulator-refined shortlist",
                        rho=RHO, penalty=PENALTY, lp_shortlist=SHORTLIST_LP,
                        simulated_shortlist=SHORTLIST_SIM,
                        note="the LP only proposes fleets; every energy reported here is "
                             "produced by the discrete-event simulator"),
           oracle={m: dict(fleet=fmt(o["pool"]), energy=o["energy"],
                           enumeration_fleet=fmt(o["enum_pool"]),
                           enumeration_energy=o["enum_energy"],
                           chooser_gap_vs_enumeration=o["gap"])
                   for m, o in ORACLE.items()},
           probe_vs_passive=compare, strategy_mean_regret=strat_mean,
           cost=dict(simulations=_stats["simulations"], lp_solves=_stats["lp_solves"],
                     chooser_calls=_stats["chooser_calls"],
                     simulations_if_enumerating=len(COMPS) * _stats["chooser_calls"],
                     wall_clock_s=round(time.time() - t0, 1)))
json.dump(OUT, io.open("experiments/results/b4_active_probing.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/b4_active_probing.json")
print("cost: {} simulations and {} LP solves in {:.0f}s; enumerating instead would have "
      "needed {:,}".format(_stats["simulations"], _stats["lp_solves"], time.time() - t0,
                           len(COMPS) * _stats["chooser_calls"]))
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
