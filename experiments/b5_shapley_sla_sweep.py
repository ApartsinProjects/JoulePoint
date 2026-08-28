# -*- coding: utf-8 -*-
"""
B5. Shapley value of each accelerator type across a sweep of service constraints.

The paper reports marginal values at a single 60-second SLA. That leaves an obvious
question open: is the ranking of accelerator types a property of the workload mix, or is
it an artefact of where the delay constraint happens to sit? A buyer facing a stricter
contract needs to know whether the answer changes.

Sanity checks stated in advance:
  S1  efficiency axiom: the Shapley values must sum exactly to v(grand coalition)
  S2  monotone feasibility: relaxing the SLA cannot shrink the feasible set
  S3  monotone value: relaxing the SLA cannot reduce v(N), since the feasible set grows
  S4  a type absent from every optimal fleet at every SLA should carry a small value
"""
import io, json, sys, warnings
from itertools import product, combinations
from math import factorial
import numpy as np
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, HORIZON, LAM = 20000, 3600.0, 0.5
SLA_GRID = [15.0, 30.0, 60.0, 120.0, 300.0, 900.0]
SANITY = []

def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

keys, Ylog, Tput = load_grid()
E = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(len(MACH))} for i, k in enumerate(keys)}
T = {k: {MACH[j]: Tput[i, j] for j in range(len(MACH))} for i, k in enumerate(keys)}

def w_of(pred):
    w = np.array([1.0 if pred(k) else 0.0 for k in keys])
    return w / w.sum()

MIXES = {"uniform": w_of(lambda k: True),
         "modern precision (fp16)": w_of(lambda k: k[1] == "fp16"),
         "vision inference": w_of(lambda k: k[0] in ("resnet50", "convnext_t", "vit_b16"))}

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
    phi = {m: 0.0 for m in MACH}
    for m in MACH:
        others = [x for x in MACH if x != m]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                Sf = frozenset(S)
                w = factorial(len(S)) * factorial(N - len(S) - 1) / factorial(N)
                phi[m] += w * (vals[Sf | {m}] - vals[Sf])
    return phi

# energy and delay of every composition, computed ONCE per mix and reused at every SLA,
# so the SLA sweep varies only the feasibility filter and nothing else.
print("{} compositions x {} mixes, evaluated once; {} SLA levels applied as a filter\n".format(
    len(COMPS), len(MIXES), len(SLA_GRID)))

OUT = {"sla_grid": SLA_GRID, "mixes": {}, "per_composition": {}}
for name, w in MIXES.items():
    ref, refd = facility({"A100-40GB": 10}, w, policy="fastest", seed=0)
    table = []
    for c in COMPS:
        pool = {MACH[i]: c[i] for i in range(5) if c[i] > 0}
        e, d = facility(pool, w, policy="energy", seed=0)
        table.append((pool, e, d))
    OUT["per_composition"][name] = [
        {"pool": p, "energy_j": round(e, 1), "delay_s": round(d, 3)} for p, e, d in table]

    per_sla = {}
    print("=== mix: {} ===".format(name))
    print("  reference all-A100 fastest-first: {:.0f} J/job at {:.1f} s delay".format(ref, refd))
    print("  {:>8}{:>10}{:>10}  {:<26}".format("SLA s", "feasible", "v(N) %", "optimal fleet")
          + "".join("{:>12}".format(m) for m in MACH))
    for sla in SLA_GRID:
        best, bestfleet = {}, None
        bestval = -1.0
        for pool, e, d in table:
            if d > sla:
                continue
            sup = frozenset(pool)
            val = max(0.0, (ref - e) / ref)
            if val > best.get(sup, -1):
                best[sup] = val
            if val > bestval:
                bestval, bestfleet = val, pool
        nfeas = sum(1 for _, _, d in table if d <= sla)
        vals = {}
        for r in range(N + 1):
            for S in combinations(MACH, r):
                Sf = frozenset(S)
                vals[Sf] = max([v for sup, v in best.items() if sup <= Sf] + [0.0])
        phi = shapley(vals)
        per_sla[sla] = dict(feasible=nfeas, v_grand=vals[frozenset(MACH)],
                            optimal_fleet=bestfleet, shapley=phi)
        fleetstr = "+".join("{}x{}".format(v, k) for k, v in sorted(bestfleet.items())) if bestfleet else "none"
        print("  {:>8.0f}{:>10}{:>10.1f}  {:<26}".format(sla, nfeas, 100 * vals[frozenset(MACH)], fleetstr)
              + "".join("{:>12.2f}".format(100 * phi[m]) for m in MACH))
    OUT["mixes"][name] = dict(reference_j=ref, per_sla={str(k): v for k, v in per_sla.items()})
    print()

# ------------------------------------------------------------------ sanity
print("sanity checks")
for name, r in OUT["mixes"].items():
    for sla, d in r["per_sla"].items():
        tot = sum(d["shapley"].values())
        sane("S1 efficiency, {} @ {}s".format(name[:18], sla), abs(tot - d["v_grand"]) < 1e-9,
             "sum {:.6f} vs v(N) {:.6f}".format(tot, d["v_grand"]))
for name, r in OUT["mixes"].items():
    feas = [r["per_sla"][str(s)]["feasible"] for s in SLA_GRID]
    sane("S2 feasibility monotone in SLA, {}".format(name[:18]),
         all(a <= b for a, b in zip(feas, feas[1:])), " -> ".join(map(str, feas)))
    vs = [r["per_sla"][str(s)]["v_grand"] for s in SLA_GRID]
    sane("S3 v(N) monotone in SLA, {}".format(name[:18]),
         all(a <= b + 1e-12 for a, b in zip(vs, vs[1:])),
         " -> ".join("{:.3f}".format(v) for v in vs))

# S4: does the ranking of types change with the SLA?
print("\nis the ranking of accelerator types stable across the SLA sweep?")
for name, r in OUT["mixes"].items():
    ranks = []
    for s in SLA_GRID:
        phi = r["per_sla"][str(s)]["shapley"]
        ranks.append(tuple(sorted(MACH, key=lambda m: -phi[m])))
    distinct = len(set(ranks))
    print("  {:26} {} distinct orderings across {} SLA levels".format(name, distinct, len(SLA_GRID)))
    for s, rk in zip(SLA_GRID, ranks):
        print("     {:>6.0f}s  {}".format(s, " > ".join(rk)))
    OUT["mixes"][name]["distinct_orderings"] = distinct

OUT["sanity"] = SANITY
json.dump(OUT, io.open("experiments/results/b5_shapley_sla_sweep.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/b5_shapley_sla_sweep.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
