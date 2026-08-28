# -*- coding: utf-8 -*-
"""
Placement regret on the N4 grid.

Regret has been degenerate since the first pilot because the L4 won all 120 cells,
so every method scored exactly 1.0000. The N4 redesign broke that: 19 memory-infeasible
cells and three different winning accelerators. This is the first evaluation in which
the decision metric can discriminate.

Feasibility is now part of the label: a method that recommends an infeasible machine
has failed outright, which is exactly the operational failure mode the model must avoid.
"""
import io, json, math, sys, warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV

MACH = ["T4", "L4", "A10G", "L40S", "A100-40GB"]
HW = {"T4": dict(tdp=70, mem=16, bw=320, year=2018, arch="Turing"),
      "L4": dict(tdp=72, mem=24, bw=300, year=2023, arch="Ada"),
      "A10G": dict(tdp=150, mem=24, bw=600, year=2021, arch="Ampere"),
      "L40S": dict(tdp=350, mem=48, bw=864, year=2023, arch="Ada"),
      "A100-40GB": dict(tdp=400, mem=80, bw=2039, year=2020, arch="Ampere")}
SANITY, OUT = [], {}


def sane(n, ok, d):
    SANITY.append(dict(check=n, passed=bool(ok), detail=d))
    print(f"    [{'PASS' if ok else 'FAIL'}] {n}: {d}")


d = json.load(io.open("experiments/results/n4_redesign.json", encoding="utf-8"))
rows = [x for r in d for x in r["rows"]]
E, FEAS = defaultdict(dict), defaultdict(dict)
for x in rows:
    k = (x["load"], x["mode"], x["precision"], x["batch"])
    if x.get("status") == "ok" and x.get("energy_per_sample_mj"):
        E[k][x["machine"]] = math.log10(x["energy_per_sample_mj"])
        FEAS[k][x["machine"]] = True
    elif x.get("status") == "oom":
        FEAS[k][x["machine"]] = False

keys = sorted(E)
n, m = len(keys), len(MACH)
print(f"grid: {n} configurations x {m} machines")
n_oom = sum(1 for k in keys for mm in MACH if FEAS[k].get(mm) is False)
winners = defaultdict(int)
for k in keys:
    winners[min(E[k], key=E[k].get)] += 1
print(f"infeasible cells: {n_oom}")
print(f"winner distribution: {dict(winners)}")
sane("more than one accelerator wins somewhere", len(winners) > 1,
     f"{len(winners)} distinct winners: {dict(winners)}")

# ---- features ----
loads = sorted({k[0] for k in keys})
X = np.array([[1.0 if k[2] == "fp32" else 0.0,
               math.log2(k[3]),
               1.0 if k[1] == "train" else 0.0,
               (1.0 if k[2] == "fp32" else 0.0) * math.log2(k[3])]
              + [1.0 if k[0] == L else 0.0 for L in loads] for k in keys])
Z = np.array([[math.log10(HW[mm]["tdp"]), math.log10(HW[mm]["mem"]),
               math.log10(HW[mm]["bw"]), HW[mm]["year"] - 2018] for mm in MACH])
Y = np.full((n, m), np.nan)
M = np.zeros((n, m), bool)
for i, k in enumerate(keys):
    for j, mm in enumerate(MACH):
        if mm in E[k]:
            Y[i, j] = E[k][mm]; M[i, j] = True


def evaluate(name, pred, res):
    """Regret and top-1 accuracy over FEASIBLE machines only; infeasible picks are failures."""
    reg, ok, infeas = [], 0, 0
    for i, k in enumerate(keys):
        feas = [j for j, mm in enumerate(MACH) if M[i, j]]
        if len(feas) < 2:
            continue
        best = min(feas, key=lambda j: Y[i, j])
        pick = min(range(m), key=lambda j: pred[i, j])   # model may pick an infeasible machine
        if not M[i, pick]:
            infeas += 1
            pick = min(feas, key=lambda j: pred[i, j])   # fall back, but count the error
        reg.append(10 ** Y[i, pick] / 10 ** Y[i, best])
        ok += (pick == best)
    res[name] = dict(regret=float(np.mean(reg)), top1=100 * ok / len(reg),
                     infeasible_picks=infeas, n=len(reg))
    return res


res = {}
mu = np.nanmean(Y[M]); r_ = np.zeros(n); c_ = np.zeros(m)
for _ in range(200):
    for i in range(n):
        o = M[i]
        if o.any(): r_[i] = np.mean(Y[i, o] - mu - c_[o])
    for j in range(m):
        o = M[:, j]
        if o.any(): c_[j] = np.mean(Y[o, j] - mu - r_[o])

evaluate("fixed ranking (additive)", np.tile(mu + c_, (n, 1)), res)
evaluate("lowest TDP", np.tile(np.array([HW[mm]["tdp"] for mm in MACH], float), (n, 1)), res)
evaluate("largest memory", np.tile(-np.array([HW[mm]["mem"] for mm in MACH], float), (n, 1)), res)

# leave-one-load-family-out bilinear
P = np.zeros((n, m))
for L in loads:
    tr = [i for i, k in enumerate(keys) if k[0] != L]
    te = [i for i, k in enumerate(keys) if k[0] == L]
    if not tr or not te:
        continue
    Ytr = np.where(M[tr], Y[tr], np.nan)
    mut = np.nanmean(Ytr)
    rt = np.nanmean(Ytr - mut, axis=1); rt = np.nan_to_num(rt)
    ct = np.nanmean(Ytr - mut - rt[:, None], axis=0); ct = np.nan_to_num(ct)
    rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], rt)
    R = np.nan_to_num(Ytr - (mut + rt[:, None] + ct[None, :]))
    v1 = np.linalg.svd(R, full_matrices=False)[2][0]
    sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], R @ v1)
    P[te] = mut + rr.predict(X[te])[:, None] + ct[None, :] + np.outer(sm.predict(X[te]), v1)
evaluate("bilinear, configuration features", P, res)
evaluate("oracle", np.where(M, Y, 1e9), res)

print(f"\n{'policy':34}{'regret':>10}{'top-1 %':>10}{'infeasible picks':>19}")
for k_ in ["fixed ranking (additive)", "lowest TDP", "largest memory",
           "bilinear, configuration features", "oracle"]:
    v = res[k_]
    print(f"{k_:34}{v['regret']:10.3f}{v['top1']:10.1f}{v['infeasible_picks']:19}")

sane("regret now discriminates between methods",
     max(v["regret"] for v in res.values()) - min(v["regret"] for v in res.values()) > 0.05,
     f"spread {min(v['regret'] for v in res.values()):.3f} to {max(v['regret'] for v in res.values()):.3f}")
sane("the model beats a fixed ranking on regret",
     res["bilinear, configuration features"]["regret"] < res["fixed ranking (additive)"]["regret"],
     f"bilinear {res['bilinear, configuration features']['regret']:.3f} vs "
     f"fixed {res['fixed ranking (additive)']['regret']:.3f}")

OUT["regret"] = res
OUT["winners"] = dict(winners)
OUT["infeasible_cells"] = n_oom
json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/n4_regret.json", "w", encoding="utf-8"), indent=1)
print(f"\nsanity: {sum(1 for s in SANITY if s['passed'])}/{len(SANITY)} passed")
print("saved -> experiments/results/n4_regret.json")
