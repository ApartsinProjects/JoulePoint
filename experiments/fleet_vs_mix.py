# -*- coding: utf-8 -*-
"""
Does the energy-optimal fleet depend on the workload mix?

This is the decisive test of the paper's practical claim. If the optimal fleet is
invariant to workload composition, procurement is a lookup and no model is needed.
If it shifts with the mix, then fleet design requires predicting energy for
workload-hardware pairs, and procurement becomes the primary application.

Free: a simulator over measured tables. No compute cost.
"""
import io, json, math, sys, warnings
import numpy as np
from itertools import product
from collections import defaultdict
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

warnings.filterwarnings("ignore")
IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB = 20000
SLA_S = 60.0
HORIZON = 3600.0
SANITY, OUT = [], {}


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


keys, Ylog, Tput = load_grid()
E = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(len(MACH))} for i, k in enumerate(keys)}
T = {k: {MACH[j]: Tput[i, j] for j in range(len(MACH))} for i, k in enumerate(keys)}

# ---- workload mixes: each is a weight over the 24 (load, precision, batch) cells ----
def weights(pred):
    w = np.array([1.0 if pred(k) else 0.0 for k in keys])
    return w / w.sum()

MIXES = {
    "uniform (all workloads)":        weights(lambda k: True),
    "LLM-serving-like (transformer)": weights(lambda k: k[0] == "transformer"),
    "vision inference (CNN + ViT)":   weights(lambda k: k[0] in ("resnet50", "convnext_t", "vit_b16")),
    "modern precision only (fp16)":   weights(lambda k: k[1] == "fp16"),
    "legacy precision (fp32)":        weights(lambda k: k[1] == "fp32"),
    "latency-sensitive (batch 8)":    weights(lambda k: k[2] == 8),
    "throughput-oriented (batch128)": weights(lambda k: k[2] == 128),
}


def facility(pool, w, lam, seed=0):
    """Arrival replay drawing workloads from mix w; returns energy per job and mean delay."""
    rng = np.random.default_rng(seed)
    slots = [m for m, c in pool.items() for _ in range(c)]
    ns = len(slots)
    free_at = np.zeros(ns)
    idx = np.arange(len(keys))
    t = 0.0
    arr = []
    while t < HORIZON:
        t += rng.exponential(1.0 / lam)
        if t < HORIZON:
            arr.append((t, keys[int(rng.choice(idx, p=w))]))
    dyn = 0.0
    delays = []
    for at, jk in arr:
        free = [i for i in range(ns) if free_at[i] <= at]
        start = at if free else float(np.min(free_at))
        if not free:
            free = [i for i in range(ns) if free_at[i] <= start]
        cand = sorted({slots[i] for i in free})
        m = min(cand, key=lambda mm: E[jk][mm])      # energy-optimal among available
        i = next(i for i in free if slots[i] == m)
        rt = JOB / T[jk][m]
        free_at[i] = start + rt
        delays.append(start - at)
        dyn += JOB * E[jk][m] - IDLE[m] * rt
    static = sum(IDLE[s] for s in slots) * HORIZON
    return (static + dyn) / max(len(arr), 1), float(np.mean(delays)) if delays else 0.0


COMPS = [c for c in product(range(0, 11), repeat=5) if sum(c) == 10]
print(f"searching {len(COMPS)} fleet compositions of 10 slots, per workload mix\n")

LAM = 0.5
results = {}
for name, w in MIXES.items():
    scored = []
    for c in COMPS:
        pool = {MACH[i]: c[i] for i in range(5) if c[i] > 0}
        e, d = facility(pool, w, LAM, seed=0)
        scored.append((e, d, pool))
    feas = [x for x in scored if x[1] <= SLA_S]
    best = sorted(feas or scored, key=lambda x: x[0])[0]
    # what the uniform-optimal fleet would have cost under THIS mix
    results[name] = dict(fleet=best[2], per_job_j=best[0], delay_s=best[1],
                         n_feasible=len(feas))
    print(f"{name:32} -> {str(best[2]):40} {best[0]:8.0f} J/job  delay {best[1]:5.1f}s"
          f"  ({len(feas)}/{len(scored)} feasible)")

# ---- how much does using the WRONG fleet cost? ----
print("\ncost of provisioning for the wrong mix (rows = fleet chosen, cols = mix actually run)")
names = list(MIXES)
ref = results["uniform (all workloads)"]["fleet"]
print(f"\n{'fleet chosen for':32}" + "".join(f"{n.split(' ')[0][:11]:>13}" for n in names))
penalty = {}
for chosen in names:
    pool = results[chosen]["fleet"]
    line = f"{chosen[:30]:32}"
    row = {}
    for actual in names:
        e, d = facility(pool, MIXES[actual], LAM, seed=0)
        opt = results[actual]["per_job_j"]
        pen = 100 * (e - opt) / opt
        row[actual] = pen
        line += f"{pen:12.1f}%"
    penalty[chosen] = row
    print(line)

OUT["optimal_fleets"] = {k: dict(fleet=v["fleet"], per_job_j=v["per_job_j"],
                                 delay_s=v["delay_s"]) for k, v in results.items()}
OUT["misprovision_penalty_pct"] = penalty

distinct = {tuple(sorted(v["fleet"].items())) for v in results.values()}
sane("the energy-optimal fleet is not invariant to workload mix",
     len(distinct) > 1,
     f"{len(distinct)} distinct optimal fleets across {len(MIXES)} mixes")
worst = max((penalty[c][a], c, a) for c in names for a in names if c != a)
sane("provisioning for the wrong mix carries a measurable penalty",
     worst[0] > 2.0,
     f"worst case {worst[0]:.1f}%, fleet chosen for '{worst[1][:26]}' running '{worst[2][:26]}'")

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/fleet_vs_mix.json", "w", encoding="utf-8"), indent=1)
print(f"\nsanity: {sum(1 for s in SANITY if s['passed'])}/{len(SANITY)} passed")
print("saved -> experiments/results/fleet_vs_mix.json")
