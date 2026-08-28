# -*- coding: utf-8 -*-
"""
B2. Workload-level execution-idle: which workload-hardware pairs strand the most energy.
B1. Multi-output prediction: predict peak memory alongside energy so the model stops
    recommending machines the job cannot fit on (4 of 31 infeasible picks previously).
"""
import io, json, math, sys, warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
SANITY, OUT = [], {}


def sane(n, ok, d):
    SANITY.append(dict(check=n, passed=bool(ok), detail=d))
    print("    [" + ("PASS" if ok else "FAIL") + "] " + n + ": " + d)


# ---------------------------------------------------------------- B2
print("=" * 84)
print("B2  EXECUTION-IDLE BY WORKLOAD-HARDWARE PAIR")
print("=" * 84)
import re
raw = io.open("experiments/eiw.log", encoding="utf-8", errors="replace").read()
m = re.search(r"===EIW_JSON_START===\s*(.*?)\s*===EIW_JSON_END===", raw, re.S)
if not m:
    print("  no data block; skipping B2")
else:
    d = json.loads(m.group(1))
    json.dump(d, io.open("experiments/results/exec_idle_workload.json", "w", encoding="utf-8"))
    rows = [r for blk in d for r in blk["rows"]]
    print("  cells: " + str(len(rows)))
    by = defaultdict(dict)
    res_idle = {}
    for r in rows:
        if r.get("energy_per_sample_mj"):
            by[(r["machine"], r["load"])][r["duty"]] = r["energy_per_sample_mj"]
        res_idle[(r["machine"], r["load"])] = r.get("resident_idle_w")
    print()
    print("  Energy per sample at low duty relative to full duty.")
    print("  A ratio of 1.0 means the pair is energy-proportional; higher means energy is")
    print("  stranded while the device is allocated but under-driven.")
    print()
    print("  {:12}{:14}{:>12}{:>12}{:>12}".format("machine", "load", "d=10%", "d=25%", "d=50%"))
    tab = []
    for (mm, ld), v in sorted(by.items()):
        if 1.0 in v and all(x in v for x in (0.10, 0.25, 0.50)):
            r10, r25, r50 = v[0.10] / v[1.0], v[0.25] / v[1.0], v[0.50] / v[1.0]
            tab.append((mm, ld, r10, r25, r50))
            print("  {:12}{:14}{:12.2f}{:12.2f}{:12.2f}".format(mm, ld, r10, r25, r50))
    if tab:
        worst = max(tab, key=lambda x: x[2])
        best = min(tab, key=lambda x: x[2])
        print()
        print("  worst pair at 10% duty: {} on {} pays {:.2f}x".format(worst[1], worst[0], worst[2]))
        print("  best  pair at 10% duty: {} on {} pays {:.2f}x".format(best[1], best[0], best[2]))
        spread = worst[2] / best[2]
        print("  spread across pairs: {:.2f}x".format(spread))
        OUT["exec_idle_pairs"] = [dict(machine=a, load=b, r10=c, r25=d_, r50=e)
                                  for a, b, c, d_, e in tab]
        sane("stranded energy is a property of the PAIR, not of the machine alone",
             spread > 1.3,
             "worst/best ratio across pairs {:.2f}x".format(spread))

# ---------------------------------------------------------------- B1
print()
print("=" * 84)
print("B1  MULTI-OUTPUT: PREDICT FEASIBILITY, NOT ONLY ENERGY")
print("=" * 84)
MACH = ["T4", "L4", "A10G", "L40S", "A100-40GB"]
MEMGB = {"T4": 16.1, "L4": 24.2, "A10G": 24.1, "L40S": 48.3, "A100-40GB": 85.9}
d4 = json.load(io.open("experiments/results/n4_redesign.json", encoding="utf-8"))
rows = [x for r in d4 for x in r["rows"]]
keys = sorted({(x["load"], x["mode"], x["precision"], x["batch"]) for x in rows})
n, m = len(keys), len(MACH)
Y = np.full((n, m), np.nan); MEM = np.full((n, m), np.nan); FEAS = np.zeros((n, m), bool)
ki = {k: i for i, k in enumerate(keys)}
for x in rows:
    i, j = ki[(x["load"], x["mode"], x["precision"], x["batch"])], MACH.index(x["machine"])
    if x.get("status") == "ok":
        Y[i, j] = math.log10(x["energy_per_sample_mj"]); MEM[i, j] = x["peak_mem_gb"]; FEAS[i, j] = True
M = ~np.isnan(Y)
loads = sorted({k[0] for k in keys})
X = np.array([[1.0 if k[2] == "fp32" else 0.0, math.log2(k[3]), 1.0 if k[1] == "train" else 0.0,
               (1.0 if k[2] == "fp32" else 0.0) * math.log2(k[3])]
              + [1.0 if k[0] == L else 0.0 for L in loads] for k in keys])

# memory model: peak memory is nearly machine-independent for a given (load, mode, prec, batch)
obs_mem = np.nanmean(np.where(M, MEM, np.nan), axis=1)
cv = np.nanstd(np.where(M, MEM, np.nan), axis=1) / np.maximum(obs_mem, 1e-9)
print("  peak memory across machines for the same configuration:")
print("    median coefficient of variation {:.3f}".format(float(np.nanmedian(cv))))
sane("peak memory is essentially a property of the workload, not the machine",
     float(np.nanmedian(cv)) < 0.10,
     "median CV across machines {:.3f}".format(float(np.nanmedian(cv))))

pred_mem = np.zeros(n)
for L in loads:
    tr = [i for i, k in enumerate(keys) if k[0] != L and not math.isnan(obs_mem[i])]
    te = [i for i, k in enumerate(keys) if k[0] == L]
    if not tr or not te:
        continue
    rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], np.log10(obs_mem[tr]))
    pred_mem[te] = 10 ** rr.predict(X[te])
ok_mem = [i for i in range(n) if not math.isnan(obs_mem[i]) and pred_mem[i] > 0]
err = np.abs(pred_mem[ok_mem] - obs_mem[ok_mem]) / obs_mem[ok_mem]
print("  held-out memory prediction: median relative error {:.1f}%".format(100 * float(np.median(err))))

SAFETY = 1.25
inf_before = inf_after = 0
for i in range(n):
    for j in range(m):
        if not M[i, j] and FEAS[i].any():
            inf_before += 1
            if pred_mem[i] * SAFETY > MEMGB[MACH[j]]:
                inf_after += 1
print("  infeasible cells correctly excluded by the memory model with a {:.2f}x margin: {}/{}".format(
    SAFETY, inf_after, inf_before))
OUT["memory_model"] = dict(median_rel_err=float(np.median(err)),
                           infeasible_caught=inf_after, infeasible_total=inf_before)
sane("the memory model excludes most genuinely infeasible machines",
     inf_before > 0 and inf_after / inf_before > 0.6,
     "{}/{} caught".format(inf_after, inf_before))

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/b1_b2.json", "w", encoding="utf-8"), indent=1)
print()
print("sanity: {}/{} passed".format(sum(1 for s in SANITY if s["passed"]), len(SANITY)))
print("saved -> experiments/results/b1_b2.json")
