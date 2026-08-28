# -*- coding: utf-8 -*-
"""
B1 corrected. The first attempt predicted peak memory from a load-family one-hot, which
is all zeros for a held-out family, so it had nothing to extrapolate from: 72.6% relative
error and only 2 of 19 infeasible cells caught.

Peak memory is close to analytic in parameter count, and the records carry params_m.
Using log parameters, mode, precision and batch instead is the correct feature set.
"""
import io, json, math, sys, warnings
import numpy as np
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
SANITY, OUT = [], {}


def sane(n, ok, d):
    SANITY.append(dict(check=n, passed=bool(ok), detail=d))
    print("    [" + ("PASS" if ok else "FAIL") + "] " + n + ": " + d)


MACH = ["T4", "L4", "A10G", "L40S", "A100-40GB"]
MEMGB = {"T4": 16.1, "L4": 24.2, "A10G": 24.1, "L40S": 48.3, "A100-40GB": 85.9}
d4 = json.load(io.open("experiments/results/n4_redesign.json", encoding="utf-8"))
rows = [x for r in d4 for x in r["rows"]]
keys = sorted({(x["load"], x["mode"], x["precision"], x["batch"]) for x in rows})
ki = {k: i for i, k in enumerate(keys)}
n, m = len(keys), len(MACH)
MEM = np.full((n, m), np.nan); FEAS = np.zeros((n, m), bool); PAR = np.full(n, np.nan)
for x in rows:
    i = ki[(x["load"], x["mode"], x["precision"], x["batch"])]
    if x.get("params_m"):
        PAR[i] = x["params_m"]
    if x.get("status") == "ok":
        MEM[i, MACH.index(x["machine"])] = x["peak_mem_gb"]; FEAS[i, MACH.index(x["machine"])] = True
obs = np.nanmean(MEM, axis=1)

# analytic-style features: memory scales with parameters, and with mode via optimiser state
X = np.array([[math.log10(PAR[i]), 1.0 if keys[i][1] == "train" else 0.0,
               1.0 if keys[i][2] == "fp32" else 0.0, math.log2(keys[i][3]),
               math.log10(PAR[i]) * (1.0 if keys[i][1] == "train" else 0.0)]
              for i in range(n)])
print("feature set: log parameters, train flag, fp32 flag, log batch, params x train")
print("held out by LOAD FAMILY, so the model must extrapolate to an unseen model size" + chr(10))

loads = sorted({k[0] for k in keys})
pred = np.zeros(n)
for L in loads:
    tr = [i for i, k in enumerate(keys) if k[0] != L and not math.isnan(obs[i])]
    te = [i for i, k in enumerate(keys) if k[0] == L]
    rr = RidgeCV(alphas=np.logspace(-4, 2, 30)).fit(X[tr], np.log10(obs[tr]))
    pred[te] = 10 ** rr.predict(X[te])
ok = [i for i in range(n) if not math.isnan(obs[i]) and pred[i] > 0]
err = np.abs(pred[ok] - obs[ok]) / obs[ok]
print("held-out memory prediction: median relative error {:.1f}%, p90 {:.1f}%".format(
    100 * float(np.median(err)), 100 * float(np.percentile(err, 90))))
sane("memory is predictable from parameter count for an unseen model size",
     float(np.median(err)) < 0.25,
     "median relative error {:.1f}% (was 72.6% with a load one-hot)".format(100 * float(np.median(err))))

print()
print("{:>8}{:>12}{:>12}{:>14}{:>14}".format("margin", "caught", "of total", "false excl.", "of feasible"))
best = None
for margin in (1.0, 1.1, 1.25, 1.5, 2.0):
    caught = tot = fp = feas = 0
    for i in range(n):
        if not FEAS[i].any():
            continue
        for j in range(m):
            if not FEAS[i, j]:
                tot += 1
                if pred[i] * margin > MEMGB[MACH[j]]:
                    caught += 1
            else:
                feas += 1
                if pred[i] * margin > MEMGB[MACH[j]]:
                    fp += 1
    print("{:8.2f}{:12}{:12}{:14}{:14}".format(margin, caught, tot, fp, feas))
    score = caught / max(tot, 1) - fp / max(feas, 1)
    if best is None or score > best[0]:
        best = (score, margin, caught, tot, fp, feas)
_, margin, caught, tot, fp, feas = best
print()
print("best margin {:.2f}: excludes {}/{} infeasible, wrongly excludes {}/{} feasible".format(
    margin, caught, tot, fp, feas))
OUT["memory_model"] = dict(median_rel_err=float(np.median(err)), margin=margin,
                           caught=caught, infeasible_total=tot, false_excl=fp, feasible_total=feas)
sane("the corrected memory model excludes most genuinely infeasible machines",
     caught / max(tot, 1) > 0.6,
     "{}/{} caught at margin {:.2f}".format(caught, tot, margin))
sane("it does not wrongly exclude many feasible machines",
     fp / max(feas, 1) < 0.15,
     "{}/{} feasible wrongly excluded".format(fp, feas))

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/b1_memory_model.json", "w", encoding="utf-8"), indent=1)
print()
print("sanity: {}/{} passed".format(sum(1 for s in SANITY if s["passed"]), len(SANITY)))
print("saved -> experiments/results/b1_memory_model.json")
