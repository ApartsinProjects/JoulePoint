# -*- coding: utf-8 -*-
"""
K6. Where the 1.9 per cent gap between two simulator variants comes from, and whether the
headline numbers are seed-stable.

The gap. Recipe 1 (fig2_pareto.py) draws each arrival's workload with

    keys[rng.integers(len(keys))]

while B8 draws it with

    keys[int(rng.choice(idx, p=w))]

For a uniform mix these two are distributionally IDENTICAL, but they are not the same
call: they consume different amounts of generator state and therefore produce different
arrival sequences from the same seed. Everything else in the two simulators, the idle
table, JOB, HORIZON, LAM, the greedy energy-first placement and the static-plus-dynamic
accounting, is identical line for line. So the hypothesis is that the 1.9 per cent is
single-seed Monte Carlo variation, not a modelling difference.

That hypothesis is worth testing rather than asserting, because if it is right it implies
something the paper currently does not report: the headline figures in Section 8 are
single-seed point estimates, and they need an interval.

Hypothesis and predictions, stated in advance:
  H   the two samplers differ only in generator consumption, not in distribution.
  P1  over many seeds the two samplers must agree in MEAN energy for the same fleet, to
      within Monte Carlo error. If they disagree, H is wrong and there is a real bug.
  P2  the seed-to-seed spread of a single fleet's energy must be of the same order as the
      observed 1.9 per cent gap. If the spread is far smaller, H is wrong.
  P3  under a NON-uniform mix the two samplers must genuinely diverge, because
      rng.integers ignores the mix weights entirely. This is a positive control: it
      confirms the test can detect a real difference when one exists.

Then, the question that actually matters for the paper:
  Q1  is the energy-optimal fleet stable across seeds, or does it move?
  Q2  what is the seed interval on the 34.3 per cent composition saving?
"""
import io, json, sys, warnings
from itertools import product
import numpy as np
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, HORIZON, LAM, SLA = 20000, 3600.0, 0.5, 60.0
SANITY = []

def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

keys, Ylog, Tput = load_grid()
NK = len(keys)
E = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(len(MACH))} for i, k in enumerate(keys)}
T = {k: {MACH[j]: Tput[i, j] for j in range(len(MACH))} for i, k in enumerate(keys)}

def facility(pool, w=None, sampler="choice", policy="energy", seed=0):
    """One simulator, two arrival samplers, so the comparison isolates exactly one line."""
    rng = np.random.default_rng(seed)
    slots = [m for m, c in pool.items() for _ in range(c)]
    ns = len(slots)
    free_at = np.zeros(ns)
    idx = np.arange(NK)
    t, arr = 0.0, []
    while t < HORIZON:
        t += rng.exponential(1.0 / LAM)
        if t < HORIZON:
            if sampler == "integers":
                arr.append((t, keys[rng.integers(NK)]))          # Recipe 1, ignores w
            else:
                arr.append((t, keys[int(rng.choice(idx, p=w))]))  # B8, honours w
    dyn, delays = 0.0, []
    for at, jk in arr:
        free = [i for i in range(ns) if free_at[i] <= at]
        start = at if free else float(np.min(free_at))
        if not free:
            free = [i for i in range(ns) if free_at[i] <= start]
        cand = sorted({slots[i] for i in free})
        m = (min(cand, key=lambda mm: E[jk][mm]) if policy == "energy"
             else max(cand, key=lambda mm: T[jk][mm]))
        i = next(i for i in free if slots[i] == m)
        rt = JOB / T[jk][m]
        free_at[i] = start + rt
        delays.append(start - at)
        dyn += JOB * E[jk][m] - IDLE[m] * rt
    static = sum(IDLE[s] for s in slots) * HORIZON
    return (static + dyn) / max(len(arr), 1), float(np.mean(delays)) if delays else 0.0

UNIFORM = np.ones(NK) / NK
OPT = {"L4": 4, "L40S": 6}
REF = {"A100-40GB": 10}
SEEDS = list(range(60))

# ------------------------------------------------------------------ P1, P2
print("single-fleet, single-seed reproduction of the two published numbers")
e_i, _ = facility(OPT, UNIFORM, sampler="integers", seed=0)
e_c, _ = facility(OPT, UNIFORM, sampler="choice", seed=0)
print("  4xL4+6xL40S, seed 0: Recipe 1 sampler {:.1f} J/job, B8 sampler {:.1f} J/job, "
      "gap {:+.2f}%".format(e_i, e_c, 100 * (e_c - e_i) / e_i))

vi = np.array([facility(OPT, UNIFORM, sampler="integers", seed=s)[0] for s in SEEDS])
vc = np.array([facility(OPT, UNIFORM, sampler="choice", seed=s)[0] for s in SEEDS])
se = np.sqrt(vi.var(ddof=1) / len(vi) + vc.var(ddof=1) / len(vc))
diff = vc.mean() - vi.mean()
z = diff / se if se > 0 else 0.0
print("\nover {} seeds, uniform mix, same fleet:".format(len(SEEDS)))
print("  Recipe 1 sampler  mean {:.1f}  sd {:.1f}  ({:.2f}% of mean)".format(
    vi.mean(), vi.std(ddof=1), 100 * vi.std(ddof=1) / vi.mean()))
print("  B8 sampler        mean {:.1f}  sd {:.1f}  ({:.2f}% of mean)".format(
    vc.mean(), vc.std(ddof=1), 100 * vc.std(ddof=1) / vc.mean()))
print("  difference of means {:+.1f} J/job, standard error {:.1f}, z = {:+.2f}".format(diff, se, z))
sane("P1 the two samplers agree in mean under a uniform mix", abs(z) < 2.0,
     "z = {:+.2f}, difference {:+.2f}% of mean".format(z, 100 * diff / vi.mean()))
spread = 100 * vi.std(ddof=1) / vi.mean()
sane("P2 seed spread is of the same order as the observed 1.9% gap",
     0.5 <= spread <= 6.0,
     "seed-to-seed sd is {:.2f}% of the mean; the single-seed gap was 1.9%".format(spread))

# ------------------------------------------------------------------ P3 positive control
SKEW = np.array([3.0 if k[0] == "transformer" else 1.0 for k in keys])
SKEW = SKEW / SKEW.sum()
si = np.array([facility(OPT, SKEW, sampler="integers", seed=s)[0] for s in SEEDS])
sc = np.array([facility(OPT, SKEW, sampler="choice", seed=s)[0] for s in SEEDS])
se2 = np.sqrt(si.var(ddof=1) / len(si) + sc.var(ddof=1) / len(sc))
z2 = (sc.mean() - si.mean()) / se2
print("\npositive control, transformer-heavy mix (rng.integers cannot see the weights):")
print("  Recipe 1 sampler mean {:.1f}, B8 sampler mean {:.1f}, z = {:+.1f}".format(
    si.mean(), sc.mean(), z2))
sane("P3 the samplers DO diverge when the mix is not uniform", abs(z2) > 3.0,
     "z = {:+.1f}, so the test can detect a genuine difference".format(z2))

# ------------------------------------------------------------------ Q1, Q2
print("\nis the optimal fleet stable across seeds?")
COMPS = [c for c in product(range(0, 11), repeat=5) if sum(c) == 10]
winners, savings, opt_e = [], [], []
for s in range(12):
    best, bestc = float("inf"), None
    for c in COMPS:
        pool = {MACH[i]: c[i] for i in range(5) if c[i] > 0}
        e, d = facility(pool, UNIFORM, sampler="integers", seed=s)
        if d <= SLA and e < best:
            best, bestc = e, c
    ref, _ = facility(REF, UNIFORM, sampler="integers", policy="fastest", seed=s)
    winners.append(bestc)
    opt_e.append(best)
    savings.append(100 * (ref - best) / ref)
    print("  seed {:>2}: {:<26} {:>7.0f} J/job, saving {:.1f}%".format(
        s, "+".join("{}x{}".format(v, MACH[i]) for i, v in enumerate(bestc) if v > 0), best, savings[-1]))

from collections import Counter
cnt = Counter(winners)
modal, modal_n = cnt.most_common(1)[0]
sav = np.array(savings)
print("\n  modal fleet {} in {} of {} seeds".format(
    "+".join("{}x{}".format(v, MACH[i]) for i, v in enumerate(modal) if v > 0), modal_n, len(winners)))
print("  composition saving: mean {:.1f}%, sd {:.1f}, range {:.1f} to {:.1f}".format(
    sav.mean(), sav.std(ddof=1), sav.min(), sav.max()))
print("  95% interval on the mean: {:.1f} to {:.1f}%".format(
    sav.mean() - 1.96 * sav.std(ddof=1) / np.sqrt(len(sav)),
    sav.mean() + 1.96 * sav.std(ddof=1) / np.sqrt(len(sav))))
sane("Q1 the modal optimal fleet is the published one",
     modal == (0, 4, 0, 6, 0),
     "modal {} chosen in {}/{} seeds".format(modal, modal_n, len(winners)))
sane("Q2 the published 34.3% saving lies inside the seed range",
     sav.min() <= 34.3 <= sav.max(),
     "seed range {:.1f} to {:.1f}%, mean {:.1f}%".format(sav.min(), sav.max(), sav.mean()))

OUT = dict(single_seed=dict(recipe1=e_i, b8=e_c, gap_pct=100 * (e_c - e_i) / e_i),
           uniform=dict(integers=vi.tolist(), choice=vc.tolist(), z=float(z),
                        seed_sd_pct=float(spread)),
           skewed_control=dict(integers=si.tolist(), choice=sc.tolist(), z=float(z2)),
           per_seed_optimum=[dict(seed=s, fleet=list(w), energy=e, saving_pct=sv)
                             for s, w, e, sv in zip(range(12), winners, opt_e, savings)],
           modal_fleet=list(modal), modal_count=modal_n,
           saving_mean=float(sav.mean()), saving_sd=float(sav.std(ddof=1)),
           saving_min=float(sav.min()), saving_max=float(sav.max()),
           sanity=SANITY)
json.dump(OUT, io.open("experiments/results/k6_seed_variance.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/k6_seed_variance.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
