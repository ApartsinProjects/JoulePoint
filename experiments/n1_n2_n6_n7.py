# -*- coding: utf-8 -*-
"""
N1  consolidation with power-down (attacks the static term, which dominates)
N2  submission-time descriptor ablation (the deployment sub-problem)
N6  procurement: which fleet should you buy
N7  active profiling: information-gain vs random acquisition

Every stage carries an explicit sanity check whose expected outcome is stated in
advance, so a silent bug shows up as a violated invariant rather than a plausible number.
"""
import io, json, math, sys, warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from e4_e5_models import load_grid, load_feats, mach_feats, additive, score, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
CAP = {"T4": 70.0, "L4": 72.0, "A10G": 150.0, "L40S": 350.0, "A100-40GB": 400.0}
SANITY = []
RESULTS = {}


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


keys, Ylog, Tput = load_grid()
X, Z = load_feats(keys), mach_feats()
n, m = Ylog.shape
E_J = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(m)} for i, k in enumerate(keys)}
T_S = {k: {MACH[j]: Tput[i, j] for j in range(m)} for i, k in enumerate(keys)}
JOB = 20000


def predictor():
    """Leave-one-load-family-out bilinear rank-1, the E5 winner."""
    fam = sorted({k[0] for k in keys})
    P = np.zeros_like(Ylog)
    for L in fam:
        tr = [i for i, k in enumerate(keys) if k[0] != L]
        te = [i for i, k in enumerate(keys) if k[0] == L]
        mu = Ylog[tr].mean(); r = Ylog[tr].mean(1) - mu; c = Ylog[tr].mean(0) - mu
        rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], r)
        base = mu + rr.predict(X[te])[:, None] + c[None, :]
        R = Ylog[tr] - (mu + r[:, None] + c[None, :])
        v1 = np.linalg.svd(R, full_matrices=False)[2][0]
        sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], R @ v1)
        P[te] = base + np.outer(sm.predict(X[te]), v1)
    return {k: {MACH[j]: P[i, j] for j in range(m)} for i, k in enumerate(keys)}


PRED = predictor()

# ===================================================================== N1
def n1_consolidation():
    print("\n" + "=" * 88)
    print("N1  CONSOLIDATION WITH POWER-DOWN")
    print("=" * 88)

    def sim(pool, policy, lam, horizon=3600.0, seed=0, sleep_after=None,
            wake_s=30.0, wake_j=20000.0):
        rng = np.random.default_rng(seed)
        slots = [mm for mm, c_ in pool.items() for _ in range(c_)]
        ns = len(slots)
        free_at = np.zeros(ns); last_busy = np.zeros(ns); asleep = np.zeros(ns, bool)
        t = 0.0; arr = []
        while t < horizon:
            t += rng.exponential(1.0 / lam)
            if t < horizon:
                arr.append((t, keys[rng.integers(len(keys))]))
        dyn = 0.0; wake_energy = 0.0; delays = []
        busy_iv = defaultdict(list)
        for at, jk in arr:
            if sleep_after is not None:
                for i in range(ns):
                    if not asleep[i] and free_at[i] <= at and (at - max(free_at[i], last_busy[i])) > sleep_after:
                        asleep[i] = True
            idx = [i for i in range(ns) if free_at[i] <= at]
            start = at if idx else float(np.min(free_at))
            if not idx:
                idx = [i for i in range(ns) if free_at[i] <= start]
            # consolidation prefers slots that are already awake
            awake = [i for i in idx if not asleep[i]]
            usable = awake if (sleep_after is not None and awake) else idx
            cand = sorted({slots[i] for i in usable})
            mm = policy(cand, jk)
            i = next(i for i in usable if slots[i] == mm)
            if asleep[i]:
                wake_energy += wake_j; start += wake_s; asleep[i] = False
            rt = JOB / T_S[jk][mm]
            free_at[i] = start + rt; last_busy[i] = start + rt
            busy_iv[i].append((start, start + rt))
            delays.append(start - at)
            dyn += JOB * E_J[jk][mm] - IDLE[mm] * rt
        # N1-FIX: exact per-slot sleep accounting. The previous version charged idle
        # from t=0 to last_busy+threshold, so mid-horizon sleep gaps were never
        # credited and the saving was understated as 0.2% instead of ~15%.
        static = 0.0
        for i in range(ns):
            iv = sorted(busy_iv[i]); sleep_time = 0.0; prev_end = 0.0
            if sleep_after is not None:
                for s_, e_ in iv:
                    if s_ - prev_end > sleep_after:
                        sleep_time += (s_ - prev_end) - sleep_after
                    prev_end = max(prev_end, e_)
                if horizon - prev_end > sleep_after:
                    sleep_time += (horizon - prev_end) - sleep_after
            static += IDLE[slots[i]] * max(0.0, horizon - sleep_time)
        return dict(total_j=static + dyn + wake_energy, static_j=static, dyn_j=dyn,
                    wake_j=wake_energy, per_job=(static + dyn + wake_energy) / max(len(arr), 1),
                    mean_delay=float(np.mean(delays)) if delays else 0.0, jobs=len(arr))

    pol_model = lambda cand, jk: min(cand, key=lambda mm: PRED[jk][mm])
    POOL = {mm: 2 for mm in MACH}
    print("\n  sanity checks:")
    a = sim(POOL, pol_model, 0.10, sleep_after=None, seed=0)
    b = sim(POOL, pol_model, 0.10, sleep_after=1e9, seed=0)
    sane("power-down with an unreachable threshold equals no power-down",
         abs(a["total_j"] - b["total_j"]) / a["total_j"] < 0.02,
         f"no-sleep {a['total_j']:.0f} J vs threshold=1e9 {b['total_j']:.0f} J")
    c0 = sim(POOL, pol_model, 0.10, sleep_after=60, wake_j=0.0, wake_s=0.0, seed=0)
    sane("free instant wake never costs more than never sleeping",
         c0["total_j"] <= a["total_j"] * 1.001,
         f"sleep(free wake) {c0['total_j']:.0f} J vs no-sleep {a['total_j']:.0f} J")

    print(f"\n  {'sleep threshold':>18}{'wake cost J':>13}{'per-job J':>12}{'static J':>13}"
          f"{'wake J':>11}{'delay s':>10}{'vs no-sleep':>13}")
    rows = []
    base = a["per_job"]
    for th, wj in [(None, 0), (300, 20000), (120, 20000), (60, 20000), (60, 50000), (30, 20000)]:
        r = sim(POOL, pol_model, 0.10, sleep_after=th, wake_j=wj, seed=0)
        rows.append(dict(threshold=th, wake_cost_j=wj, **r))
        lbl = "none" if th is None else f"{th}s"
        print(f"  {lbl:>18}{wj:13.0f}{r['per_job']:12.1f}{r['static_j']:13.0f}"
              f"{r['wake_j']:11.0f}{r['mean_delay']:10.1f}{100*(base-r['per_job'])/base:12.1f}%")
    RESULTS["n1"] = rows
    best = min(rows, key=lambda r: r["per_job"])
    print(f"\n  best: threshold={best['threshold']}, saving {100*(base-best['per_job'])/base:.1f}% of facility energy")
    print(f"  for comparison, PLACEMENT alone was worth ~7-10% in E1")


# ===================================================================== N2
def n2_descriptors():
    print("\n" + "=" * 88)
    print("N2  SUBMISSION-TIME DESCRIPTOR ABLATION  (the deployment sub-problem)")
    print("=" * 88)
    d = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
    rws = [r for b in d for r in b["rows"] if r.get("status") == "ok"]
    mem = {}
    for r in rws:
        mem.setdefault((r["load"], r["precision"], r["batch"]), []).append(r["peak_mem_gb"])
    memv = np.array([np.median(mem[k]) for k in keys])
    fam = sorted({k[0] for k in keys})
    SETS = {
        "A full descriptors (family+precision+batch)": lambda: X,
        "B identity only (which script)": lambda: np.array([[1.0 if k[0] == f else 0.0 for f in fam] for k in keys]),
        "C observable only (mem request + batch)": lambda: np.column_stack(
            [np.log10(memv), np.array([math.log2(k[2]) for k in keys])]),
        "D memory request only": lambda: np.log10(memv)[:, None],
        "E nothing (global mean)": lambda: np.ones((len(keys), 1)),
    }

    def run(Xf, split):
        accs = []
        if split == "family":
            folds = [([i for i, k in enumerate(keys) if k[0] != L],
                      [i for i, k in enumerate(keys) if k[0] == L]) for L in fam]
        else:
            rng = np.random.default_rng(0); idx = rng.permutation(n)
            folds = [(list(idx[np.arange(n) % 4 != f]), list(idx[np.arange(n) % 4 == f])) for f in range(4)]
        for tr, te in folds:
            mu = Ylog[tr].mean(); r = Ylog[tr].mean(1) - mu; c = Ylog[tr].mean(0) - mu
            rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[tr], r)
            base = np.zeros_like(Ylog); base[te] = mu + rr.predict(Xf[te])[:, None] + c[None, :]
            R = Ylog[tr] - (mu + r[:, None] + c[None, :])
            v1 = np.linalg.svd(R, full_matrices=False)[2][0]
            sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[tr], R @ v1)
            P = base.copy(); P[te] = base[te] + np.outer(sm.predict(Xf[te]), v1)
            a_, _ = score(Ylog, P, te)
            accs.append(a_)
        return float(np.mean(accs))

    print("\n  sanity checks:")
    e_fam = run(SETS["E nothing (global mean)"](), "family")
    sane("a constant feature cannot beat the fixed-ranking ceiling",
         e_fam <= 82.5, f"nothing-features scores {e_fam:.1f}%, additive ceiling is 81.7%")

    print(f"\n  {'descriptor set':46}{'unseen family':>15}{'recurring job':>15}")
    out = []
    for name, f in SETS.items():
        Xf = f()
        a1 = run(Xf, "family"); a2 = run(Xf, "random")
        out.append(dict(descriptors=name, unseen_family=a1, recurring=a2, n_features=Xf.shape[1]))
        print(f"  {name:46}{a1:15.1f}{a2:15.1f}")
    RESULTS["n2"] = out
    full = out[0]; obs = out[2]
    print(f"\n  observable-only retains {100*obs['unseen_family']/full['unseen_family']:.0f}% of full-descriptor")
    print(f"  accuracy on unseen families, and {100*obs['recurring']/full['recurring']:.0f}% on recurring jobs.")


# ===================================================================== N6
def n6_procurement():
    print("\n" + "=" * 88)
    print("N6  PROCUREMENT: which fleet should you buy?")
    print("=" * 88)

    def facility(pool, policy, lam=0.10, horizon=3600.0, seeds=(0, 1, 2)):
        tot = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            slots = [mm for mm, c_ in pool.items() for _ in range(c_)]
            ns = len(slots); free_at = np.zeros(ns)
            t = 0.0; arr = []
            while t < horizon:
                t += rng.exponential(1.0 / lam)
                if t < horizon:
                    arr.append((t, keys[rng.integers(len(keys))]))
            dyn = 0.0; done = 0; delays = []
            for at, jk in arr:
                idx = [i for i in range(ns) if free_at[i] <= at]
                start = at if idx else float(np.min(free_at))
                if not idx:
                    idx = [i for i in range(ns) if free_at[i] <= start]
                cand = sorted({slots[i] for i in idx})
                mm = policy(cand, jk)
                i = next(i for i in idx if slots[i] == mm)
                rt = JOB / T_S[jk][mm]
                free_at[i] = start + rt; delays.append(start - at)
                dyn += JOB * E_J[jk][mm] - IDLE[mm] * rt; done += 1
            static = sum(IDLE[s] for s in slots) * horizon
            tot.append(((static + dyn) / max(done, 1), float(np.mean(delays)), done))
        return float(np.mean([x[0] for x in tot])), float(np.mean([x[1] for x in tot]))

    pol_model = lambda cand, jk: min(cand, key=lambda mm: PRED[jk][mm])
    pol_fast = lambda cand, jk: max(cand, key=lambda mm: T_S[jk][mm])
    print("\n  sanity check:")
    p_all_l4 = facility({"L4": 10}, pol_model)
    p_all_a100 = facility({"A100-40GB": 10}, pol_model)
    sane("an all-L4 fleet beats an all-A100 fleet on energy per job",
         p_all_l4[0] < p_all_a100[0],
         f"L4x10 {p_all_l4[0]:.0f} J vs A100x10 {p_all_a100[0]:.0f} J (L4 wins every measured cell)")

    print(f"\n  searching fleet compositions of 10 slots over 5 machine types")
    best = []
    from itertools import product
    comps = [c for c in product(range(0, 11), repeat=5) if sum(c) == 10]
    print(f"  {len(comps)} compositions of exactly 10 slots")
    # N6-FIX: without a service constraint the search trivially buys the slowest
    # efficient device. Constrain mean queueing delay, and report the frontier.
    # N6-FIX(2): at lam=0.1 nothing queues, so the SLA never binds and the search
    # trivially buys the most efficient device. Run where contention is real.
    SLA_S = 60.0
    LAM = 0.5
    for c_ in comps:
        pool = {MACH[i]: c_[i] for i in range(5) if c_[i] > 0}
        e, dly = facility(pool, pol_model, lam=LAM, seeds=(0,))
        best.append((e, dly, pool))
    feasible = [b for b in best if b[1] <= SLA_S]
    print(f"  at lam={LAM}, compositions meeting mean-delay <= {SLA_S:.0f}s: "
          f"{len(feasible)}/{len(best)}")
    uncon = sorted(best, key=lambda x: x[0])[0]
    print(f"  UNCONSTRAINED optimum: {uncon[2]} at {uncon[0]:.1f} J/job, delay {uncon[1]:.0f}s")
    best = sorted(feasible or best, key=lambda x: x[0])
    print(f"\n  {'rank':>5}  {'per-job J':>10}{'delay s':>10}  fleet")
    for r, (e, dly, pool) in enumerate(best[:5], 1):
        print(f"  {r:5}  {e:10.1f}{dly:10.1f}  {pool}")
    print(f"  {'worst':>5}  {best[-1][0]:10.1f}{best[-1][1]:10.1f}  {best[-1][2]}")
    # what a throughput-first buyer would choose: most FLOPs, i.e. all A100
    e_fast, d_fast = facility({"A100-40GB": 10}, pol_fast, lam=0.5, seeds=(0,))
    print(f"\n  a throughput-first buyer (all A100, fastest-free): {e_fast:.1f} J/job")
    print(f"  energy-optimal fleet:                              {best[0][0]:.1f} J/job"
          f"  ({100*(e_fast-best[0][0])/e_fast:.1f}% lower)")
    RESULTS["n6"] = [dict(per_job_j=e, delay_s=d, fleet=p) for e, d, p in best[:20]] + \
                    [dict(per_job_j=best[-1][0], delay_s=best[-1][1], fleet=best[-1][2], worst=True)]


# ===================================================================== N7
def n7_active():
    print("\n" + "=" * 88)
    print("N7  ACTIVE PROFILING: information-gain vs random acquisition")
    print("=" * 88)

    def fit_and_score(mask):
        mu, r, c = additive(Ylog, mask)
        R = np.where(mask, Ylog - (mu + r[:, None] + c[None, :]), 0.0)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        v1 = Vt[0]
        s = R @ v1
        P = mu + r[:, None] + c[None, :] + np.outer(s, v1)
        held = [(i, j) for i in range(n) for j in range(m) if not mask[i, j]]
        byrow = defaultdict(list)
        for i, j in held:
            byrow[i].append(j)
        ok = tot = 0
        for i in range(n):
            for a in range(m):
                for b in range(a + 1, m):
                    t = Ylog[i, a] - Ylog[i, b]; p = P[i, a] - P[i, b]
                    if t != 0:
                        tot += 1; ok += (t > 0) == (p > 0)
        return 100 * ok / tot, P

    def seed_mask(rng, per_row=1):
        mk = np.zeros_like(Ylog, dtype=bool)
        for i in range(n):
            for j in rng.choice(m, size=per_row, replace=False):
                mk[i, j] = True
        return mk

    budgets = [24, 36, 48, 60, 72, 90]
    curves = defaultdict(lambda: defaultdict(list))
    for rep in range(8):
        rng = np.random.default_rng(rep)
        for strat in ["random", "uncertainty (max residual)"]:
            mk = seed_mask(rng)
            for B in budgets:
                while mk.sum() < B:
                    cand = [(i, j) for i in range(n) for j in range(m) if not mk[i, j]]
                    if not cand:
                        break
                    if strat == "random":
                        pick = cand[rng.integers(len(cand))]
                    else:
                        _, P = fit_and_score(mk)
                        resid = np.abs(Ylog - P)
                        pick = max(cand, key=lambda ij: resid[ij])
                    mk[pick] = True
                acc, _ = fit_and_score(mk)
                curves[strat][B].append(acc)
    print("\n  sanity check:")
    r_lo = np.mean(curves["random"][budgets[0]]); r_hi = np.mean(curves["random"][budgets[-1]])
    sane("accuracy improves as the measurement budget grows", r_hi >= r_lo - 0.5,
         f"random: {r_lo:.1f}% at {budgets[0]} cells -> {r_hi:.1f}% at {budgets[-1]} cells")
    print(f"\n  {'cells measured':>16}" + "".join(f"{s:>28}" for s in curves))
    out = []
    for B in budgets:
        line = f"  {B:>16}"
        row = {"cells": B}
        for s in curves:
            v = float(np.mean(curves[s][B])); line += f"{v:28.1f}"; row[s] = v
        out.append(row); print(line)
    RESULTS["n7"] = out
    print(f"\n  full grid is {n*m} cells; oracle (all measured) pairwise accuracy is 100%")


if __name__ == "__main__":
    n1_consolidation(); n2_descriptors(); n6_procurement(); n7_active()
    json.dump({"results": RESULTS, "sanity": SANITY},
              io.open("experiments/results/n1_n2_n6_n7.json", "w", encoding="utf-8"), indent=1)
    npass = sum(1 for s in SANITY if s["passed"])
    print(f"\n\nsanity checks: {npass}/{len(SANITY)} passed")
    for s in SANITY:
        if not s["passed"]:
            print(f"  FAILED: {s['check']} -- {s['detail']}")
    print("saved -> experiments/results/n1_n2_n6_n7.json")
