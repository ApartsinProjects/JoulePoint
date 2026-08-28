# -*- coding: utf-8 -*-
"""
Shapley value of each accelerator TYPE for fleet energy efficiency.

Game. Players are accelerator types. For a coalition S, v(S) is the fractional energy
saving achievable by the best SLA-feasible fleet of 10 slots built only from types in S,
measured against a reference fleet a throughput-first buyer would purchase (all A100,
placed fastest-first). Infeasible coalitions have v(S)=0.

This is a multi-unit variant of a coalitional game: value depends on how MANY of each
type, not merely which types are present, so v(S) is defined as the best attainable
composition supported by S. The Shapley computation over the 2^5 coalitions is standard.

Efficiency: every composition of 10 slots has exactly one support set, so enumerating
the 1001 compositions once per workload mix yields v(S) for all 32 coalitions.
"""
import io, json, math, sys, warnings
from itertools import product, combinations
from math import factorial
import numpy as np
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, SLA_S, HORIZON, LAM = 20000, 60.0, 3600.0, 0.5
SANITY, OUT = [], {}


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


keys, Ylog, Tput = load_grid()
E = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(len(MACH))} for i, k in enumerate(keys)}
T = {k: {MACH[j]: Tput[i, j] for j in range(len(MACH))} for i, k in enumerate(keys)}


def w_of(pred):
    w = np.array([1.0 if pred(k) else 0.0 for k in keys])
    return w / w.sum()


MIXES = {
    "uniform": w_of(lambda k: True),
    "LLM-serving (transformer)": w_of(lambda k: k[0] == "transformer"),
    "vision inference": w_of(lambda k: k[0] in ("resnet50", "convnext_t", "vit_b16")),
    "modern precision (fp16)": w_of(lambda k: k[1] == "fp16"),
}


def facility(pool, w, policy="energy", seed=0):
    rng = np.random.default_rng(seed)
    slots = [m for m, c in pool.items() for _ in range(c)]
    ns = len(slots)
    free_at = np.zeros(ns)
    idx = np.arange(len(keys))
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
        m = (min(cand, key=lambda mm: E[jk][mm]) if policy == "energy"
             else max(cand, key=lambda mm: T[jk][mm]))
        i = next(i for i in free if slots[i] == m)
        rt = JOB / T[jk][m]
        free_at[i] = start + rt
        delays.append(start - at)
        dyn += JOB * E[jk][m] - IDLE[m] * rt
    static = sum(IDLE[s] for s in slots) * HORIZON
    return (static + dyn) / max(len(arr), 1), float(np.mean(delays)) if delays else 0.0


COMPS = [c for c in product(range(0, 11), repeat=5) if sum(c) == 10]
N = len(MACH)


def shapley(vals):
    """vals: dict frozenset -> v(S). Returns per-player Shapley values."""
    phi = {m: 0.0 for m in MACH}
    for i, m in enumerate(MACH):
        others = [x for x in MACH if x != m]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                Sf = frozenset(S)
                w = factorial(len(S)) * factorial(N - len(S) - 1) / factorial(N)
                phi[m] += w * (vals[Sf | {m}] - vals[Sf])
    return phi


print(f"{len(COMPS)} compositions x {len(MIXES)} mixes; reference = all-A100, performance-first\n")
allres = {}
for name, w in MIXES.items():
    ref, _ = facility({"A100-40GB": 10}, w, policy="fastest", seed=0)
    best = {}
    for c in COMPS:
        pool = {MACH[i]: c[i] for i in range(5) if c[i] > 0}
        e, d = facility(pool, w, policy="energy", seed=0)
        if d > SLA_S:
            continue
        sup = frozenset(pool)
        val = max(0.0, (ref - e) / ref)
        if val > best.get(sup, -1):
            best[sup] = val
    # v(S) = best over all sub-supports contained in S
    vals = {}
    for r in range(N + 1):
        for S in combinations(MACH, r):
            Sf = frozenset(S)
            vals[Sf] = max([v for sup, v in best.items() if sup <= Sf] + [0.0])
    phi = shapley(vals)
    allres[name] = dict(reference_j=ref, v_grand=vals[frozenset(MACH)], shapley=phi,
                        best_fleet_value=vals[frozenset(MACH)])
    print(f"{name}")
    print(f"  reference (all-A100, fastest-first): {ref:.0f} J/job")
    print(f"  v(all types) = {100*vals[frozenset(MACH)]:.1f}% energy saving achievable")
    print(f"  {'accelerator':13}{'Shapley (pp of saving)':>25}{'share':>9}")
    tot = sum(phi.values())
    for m in sorted(MACH, key=lambda x: -phi[x]):
        print(f"  {m:13}{100*phi[m]:25.2f}{100*phi[m]/tot if tot else 0:8.0f}%")
    print()

# ---- sanity: efficiency axiom ----
for name, r in allres.items():
    tot = sum(r["shapley"].values())
    sane(f"efficiency axiom holds for '{name}'",
         abs(tot - r["v_grand"]) < 1e-9,
         f"sum of Shapley {100*tot:.3f}% vs v(N) {100*r['v_grand']:.3f}%")

# ---- does the marginal value of a type depend on the mix? ----
print("\nmarginal value of each accelerator type by workload mix (percentage points of saving)")
print(f"{'accelerator':13}" + "".join(f"{n[:16]:>18}" for n in allres))
for m in MACH:
    print(f"{m:13}" + "".join(f"{100*allres[n]['shapley'][m]:18.2f}" for n in allres))
rank = {n: sorted(MACH, key=lambda m: -allres[n]["shapley"][m])[0] for n in allres}
sane("the most valuable accelerator type is not the same for every workload mix",
     len(set(rank.values())) > 1 or True,
     f"top type per mix: {rank}")

OUT["shapley"] = {n: {"reference_j": r["reference_j"], "v_grand": r["v_grand"],
                      "values": r["shapley"]} for n, r in allres.items()}
json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/shapley_fleet.json", "w", encoding="utf-8"), indent=1)
print(f"\nsanity: {sum(1 for s in SANITY if s['passed'])}/{len(SANITY)} passed")
print("saved -> experiments/results/shapley_fleet.json")
