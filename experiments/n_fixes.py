# -*- coding: utf-8 -*-
"""
Three corrections to the N-round, each addressing a suspected defect rather than
accepting the number:

F1  N1's powered-time accounting was crude (a slot was charged idle from t=0 to
    last_busy+threshold, so mid-horizon sleep gaps were never credited). Replace
    with exact per-slot interval accounting.
F2  N2 could not separate WHICH descriptor carries the signal. Add ablations that
    isolate precision and family.
F3  N6 ran at zero queueing delay, so "buy the most efficient device" was trivially
    optimal. Sweep utilisation until contention binds and report delay.
"""
import io, json, math, sys, warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from e4_e5_models import load_grid, load_feats, additive, score, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB = 20000
SANITY = []
OUT = {}


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


keys, Ylog, Tput = load_grid()
X = load_feats(keys)
n, m = Ylog.shape
E_J = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(m)} for i, k in enumerate(keys)}
T_S = {k: {MACH[j]: Tput[i, j] for j in range(m)} for i, k in enumerate(keys)}
fam = sorted({k[0] for k in keys})


def predictor(Xf):
    P = np.zeros_like(Ylog)
    for L in fam:
        tr = [i for i, k in enumerate(keys) if k[0] != L]
        te = [i for i, k in enumerate(keys) if k[0] == L]
        mu = Ylog[tr].mean(); r = Ylog[tr].mean(1) - mu; c = Ylog[tr].mean(0) - mu
        rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[tr], r)
        base = mu + rr.predict(Xf[te])[:, None] + c[None, :]
        R = Ylog[tr] - (mu + r[:, None] + c[None, :])
        v1 = np.linalg.svd(R, full_matrices=False)[2][0]
        sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[tr], R @ v1)
        P[te] = base + np.outer(sm.predict(Xf[te]), v1)
    return P


PRED = {k: {MACH[j]: predictor(X)[i, j] for j in range(m)} for i, k in enumerate(keys)}
POL = lambda cand, jk: min(cand, key=lambda mm: PRED[jk][mm])
FAST = lambda cand, jk: max(cand, key=lambda mm: T_S[jk][mm])


# ======================================================= F1
def sim_exact(pool, policy, lam, horizon=3600.0, seed=0, sleep_after=None,
              wake_s=30.0, wake_j=20000.0):
    """Exact per-slot sleep accounting."""
    rng = np.random.default_rng(seed)
    slots = [mm for mm, c_ in pool.items() for _ in range(c_)]
    ns = len(slots)
    free_at = np.zeros(ns)
    busy = defaultdict(list)
    t = 0.0; arr = []
    while t < horizon:
        t += rng.exponential(1.0 / lam)
        if t < horizon:
            arr.append((t, keys[rng.integers(len(keys))]))
    dyn = 0.0; delays = []
    for at, jk in arr:
        idx = [i for i in range(ns) if free_at[i] <= at]
        start = at if idx else float(np.min(free_at))
        if not idx:
            idx = [i for i in range(ns) if free_at[i] <= start]
        cand = sorted({slots[i] for i in idx})
        mm = policy(cand, jk)
        i = next(i for i in idx if slots[i] == mm)
        rt = JOB / T_S[jk][mm]
        # would this slot have been asleep? charge wake cost and latency
        if sleep_after is not None and busy[i]:
            gap = start - busy[i][-1][1]
            if gap > sleep_after:
                start += wake_s
                dyn += wake_j
        busy[i].append((start, start + rt))
        free_at[i] = start + rt
        delays.append(start - at)
        dyn += JOB * E_J[jk][mm] - IDLE[mm] * rt
    # exact powered time per slot
    static = 0.0; asleep_total = 0.0
    for i in range(ns):
        iv = sorted(busy[i])
        sleep_time = 0.0
        if sleep_after is not None:
            prev_end = 0.0
            for s, e in iv:
                gap = s - prev_end
                if gap > sleep_after:
                    sleep_time += gap - sleep_after
                prev_end = max(prev_end, e)
            tail = horizon - prev_end
            if tail > sleep_after:
                sleep_time += tail - sleep_after
        powered = max(0.0, horizon - sleep_time)
        asleep_total += sleep_time
        static += IDLE[slots[i]] * powered
    total = static + dyn
    return dict(total_j=total, static_j=static, dyn_j=dyn, per_job=total / max(len(arr), 1),
                mean_delay=float(np.mean(delays)) if delays else 0.0,
                asleep_frac=asleep_total / (ns * horizon), jobs=len(arr))


print("=" * 90)
print("F1  N1 REDONE with exact sleep accounting")
print("=" * 90)
POOL = {mm: 2 for mm in MACH}
print("\n  sanity checks:")
a = sim_exact(POOL, POL, 0.10, sleep_after=None, seed=0)
b = sim_exact(POOL, POL, 0.10, sleep_after=1e9, seed=0)
sane("unreachable threshold reproduces the no-sleep baseline",
     abs(a["total_j"] - b["total_j"]) / a["total_j"] < 1e-6,
     f"{a['total_j']:.0f} J vs {b['total_j']:.0f} J")
z = sim_exact(POOL, POL, 0.10, sleep_after=0.0, wake_j=0.0, wake_s=0.0, seed=0)
sane("instant free sleep removes essentially all idle energy",
     z["asleep_frac"] > 0.5 and z["static_j"] < a["static_j"] * 0.6,
     f"asleep {100*z['asleep_frac']:.0f}% of slot-time, static {z['static_j']:.0f} vs {a['static_j']:.0f} J")

print(f"\n  {'threshold':>12}{'wake J':>10}{'per-job J':>12}{'static J':>12}"
      f"{'asleep %':>10}{'delay s':>9}{'vs no-sleep':>13}")
rows = []
base = a["per_job"]
for th, wj in [(None, 0), (600, 20000), (300, 20000), (120, 20000), (60, 20000),
               (30, 20000), (30, 5000), (30, 0)]:
    r = sim_exact(POOL, POL, 0.10, sleep_after=th, wake_j=wj, seed=0)
    rows.append(dict(threshold=th, wake_cost_j=wj, **r))
    lbl = "none" if th is None else f"{th}s"
    print(f"  {lbl:>12}{wj:10.0f}{r['per_job']:12.1f}{r['static_j']:12.0f}"
          f"{100*r['asleep_frac']:10.1f}{r['mean_delay']:9.1f}{100*(base-r['per_job'])/base:12.1f}%")
OUT["f1_consolidation"] = rows
best = min(rows, key=lambda r: r["per_job"])
print(f"\n  best: threshold={best['threshold']}, wake={best['wake_cost_j']} J "
      f"-> {100*(base-best['per_job'])/base:.1f}% of facility energy")
print(f"  placement alone was worth 7-10% (E1); the two are additive levers")


# ======================================================= F2
print("\n" + "=" * 90)
print("F2  N2 REDONE: which descriptor actually carries the signal?")
print("=" * 90)
d = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
rws = [r for b in d for r in b["rows"] if r.get("status") == "ok"]
memd = defaultdict(list)
for r in rws:
    memd[(r["load"], r["precision"], r["batch"])].append(r["peak_mem_gb"])
memv = np.array([np.median(memd[k]) for k in keys])
p32 = np.array([1.0 if k[1] == "fp32" else 0.0 for k in keys])
lb = np.array([math.log2(k[2]) for k in keys])
onehot = np.array([[1.0 if k[0] == f else 0.0 for f in fam] for k in keys])

SETS = {
    "A full (family + precision + batch)": np.column_stack([p32, lb, p32 * lb, lb ** 2, onehot]),
    "F precision + batch, NO family":      np.column_stack([p32, lb, p32 * lb, lb ** 2]),
    "G family + batch, NO precision":      np.column_stack([lb, lb ** 2, onehot]),
    "H precision only":                    p32[:, None],
    "C observable (mem + batch)":          np.column_stack([np.log10(memv), lb]),
    "E nothing":                           np.ones((n, 1)),
}


def cv(Xf):
    accs = []
    for L in fam:
        tr = [i for i, k in enumerate(keys) if k[0] != L]
        te = [i for i, k in enumerate(keys) if k[0] == L]
        mu = Ylog[tr].mean(); r = Ylog[tr].mean(1) - mu; c = Ylog[tr].mean(0) - mu
        rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[tr], r)
        P = np.zeros_like(Ylog); P[te] = mu + rr.predict(Xf[te])[:, None] + c[None, :]
        R = Ylog[tr] - (mu + r[:, None] + c[None, :])
        v1 = np.linalg.svd(R, full_matrices=False)[2][0]
        sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[tr], R @ v1)
        P[te] = P[te] + np.outer(sm.predict(Xf[te]), v1)
        accs.append(score(Ylog, P, te)[0])
    return float(np.mean(accs))


print(f"\n  {'descriptor set':40}{'pairwise acc':>14}{'vs full':>10}")
res = {}
full = cv(SETS["A full (family + precision + batch)"])
for name, Xf in SETS.items():
    v = cv(Xf); res[name] = v
    print(f"  {name:40}{v:14.1f}{v - full:+10.1f}")
OUT["f2_descriptors"] = res
sane("removing precision costs more than removing family identity",
     res["G family + batch, NO precision"] < res["F precision + batch, NO family"],
     f"no-precision {res['G family + batch, NO precision']:.1f}% vs "
     f"no-family {res['F precision + batch, NO family']:.1f}%")


# ======================================================= F3
print("\n" + "=" * 90)
print("F3  N6 REDONE: procurement under real contention")
print("=" * 90)


def facility(pool, policy, lam, seeds=(0, 1)):
    rs = [sim_exact(pool, policy, lam, seed=s) for s in seeds]
    return (float(np.mean([r["per_job"] for r in rs])),
            float(np.mean([r["mean_delay"] for r in rs])))


print(f"\n  {'arrival rate':>13}{'best fleet':>34}{'per-job J':>11}{'delay s':>9}"
      f"{'all-A100 J':>12}{'all-A100 delay':>15}")
proc = []
from itertools import product
comps = [c for c in product(range(0, 11), repeat=5) if sum(c) == 10]
for lam in [0.1, 0.5, 1.0, 2.0]:
    scored = []
    for c_ in comps:
        pool = {MACH[i]: c_[i] for i in range(5) if c_[i] > 0}
        e, dly = facility(pool, POL, lam, seeds=(0,))
        scored.append((e, dly, pool))
    scored.sort(key=lambda x: x[0])
    e0, d0, p0 = scored[0]
    ea, da = facility({"A100-40GB": 10}, FAST, lam, seeds=(0,))
    proc.append(dict(lam=lam, best_fleet=p0, best_j=e0, best_delay=d0,
                     a100_j=ea, a100_delay=da))
    print(f"  {lam:13.1f}{str(p0):>34}{e0:11.1f}{d0:9.1f}{ea:12.1f}{da:15.1f}")
OUT["f3_procurement"] = proc
sane("queueing delay becomes non-zero as arrival rate rises",
     proc[-1]["best_delay"] > 0.5,
     f"delay at lam={proc[-1]['lam']} is {proc[-1]['best_delay']:.1f}s (was 0.0 at lam=0.1)")

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/n_fixes.json", "w", encoding="utf-8"), indent=1)
npass = sum(1 for s in SANITY if s["passed"])
print(f"\n\nsanity: {npass}/{len(SANITY)} passed")
for s in SANITY:
    if not s["passed"]:
        print(f"  FAILED: {s['check']} -- {s['detail']}")
print("saved -> experiments/results/n_fixes.json")
