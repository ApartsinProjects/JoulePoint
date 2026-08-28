# -*- coding: utf-8 -*-
"""
B7. Exact assignment baseline and the prediction-versus-policy decomposition.

A first attempt batched arrivals and the sanity checks caught it: batch-optimal came out
WORSE than greedy at high load, which is impossible over the same choice set. Root cause,
found by inspection: the batch branch fixed the free-machine set at batch close while
greedy could use machines that freed mid-batch, so the two solved different problems.

This version is event-driven. At every scheduling epoch both policies see the identical
set of waiting jobs and the identical set of free machines, so the only difference is how
the assignment within that set is chosen. Invariant: joint assignment can never lose to
greedy under perfect information.
"""
import io, json, math, sys, warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from e4_e5_models import load_grid, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, HORIZON = 20000, 3600.0
SANITY, OUT = [], {}
NL = chr(10)


def sane(n, ok, d):
    SANITY.append(dict(check=n, passed=bool(ok), detail=d))
    print("    [" + ("PASS" if ok else "FAIL") + "] " + n + ": " + d)


keys, Ylog, Tput = load_grid()
E = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(len(MACH))} for i, k in enumerate(keys)}
T = {k: {MACH[j]: Tput[i, j] for j in range(len(MACH))} for i, k in enumerate(keys)}


def predictor():
    fam = sorted({k[0] for k in keys})
    X = np.array([[1.0 if k[1] == "fp32" else 0.0, math.log2(k[2]),
                   (1.0 if k[1] == "fp32" else 0.0) * math.log2(k[2]), math.log2(k[2]) ** 2]
                  + [1.0 if k[0] == f else 0.0 for f in fam] for k in keys])
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
    return {k: {MACH[j]: 10 ** P[i, j] / 1000.0 for j in range(len(MACH))} for i, k in enumerate(keys)}


PRED = predictor()


def simulate(pool, cost_table, assign, lam, seed=0):
    rng = np.random.default_rng(seed)
    slots = [m for m, c in pool.items() for _ in range(c)]
    ns = len(slots)
    free_at = np.zeros(ns)
    t, arr = 0.0, []
    while t < HORIZON:
        t += rng.exponential(1.0 / lam)
        if t < HORIZON:
            arr.append((t, keys[rng.integers(len(keys))]))
    queue, dyn, delays, done, ai, now = [], 0.0, [], 0, 0, 0.0
    while ai < len(arr) or queue:
        if not queue:
            now = max(now, arr[ai][0])
        while ai < len(arr) and arr[ai][0] <= now:
            queue.append(arr[ai]); ai += 1
        free = [i for i in range(ns) if free_at[i] <= now + 1e-12]
        if not free or not queue:
            nxt = []
            if ai < len(arr):
                nxt.append(arr[ai][0])
            if queue:
                busy = [free_at[i] for i in range(ns) if free_at[i] > now + 1e-12]
                if busy:
                    nxt.append(min(busy))
            if not nxt:
                break
            now = min(nxt)
            continue
        k = min(len(queue), len(free))
        grp = queue[:k]
        if assign == "greedy":
            avail, chosen = list(free), []
            for at, jk in grp:
                i = min(avail, key=lambda i: cost_table[jk][slots[i]])
                avail.remove(i); chosen.append(i)
        else:
            C = np.array([[cost_table[jk][slots[i]] for i in free] for _, jk in grp])
            rr_, cc_ = linear_sum_assignment(C)
            chosen = [None] * len(grp)
            for a, b in zip(rr_, cc_):
                chosen[a] = free[b]
        for (at, jk), i in zip(grp, chosen):
            rt = JOB / T[jk][slots[i]]
            free_at[i] = now + rt
            delays.append(now - at)
            dyn += JOB * E[jk][slots[i]] - IDLE[slots[i]] * rt
            done += 1
        del queue[:k]
    static = sum(IDLE[s] for s in slots) * HORIZON
    return (static + dyn) / max(done, 1), float(np.mean(delays)) if delays else 0.0


POOL = {m: 2 for m in MACH}
print("pool " + str(POOL) + ", event-driven, identical state per epoch" + NL)
hdr = "{:>6}{:>14}{:>14}{:>13}{:>14}{:>12}{:>9}"
print(hdr.format("lam", "greedy+true", "joint+true", "greedy+pred", "joint+pred", "joint gain", "delay s"))
rows = []
for lam in (0.1, 0.5, 1.0, 2.0, 4.0):
    cell = {}
    for an, asg in (("greedy", "greedy"), ("joint", "optimal")):
        for cn, tbl in (("true", E), ("pred", PRED)):
            v = [simulate(POOL, tbl, asg, lam, s) for s in range(3)]
            cell[(an, cn)] = (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])))
    gt, ot = cell[("greedy", "true")][0], cell[("joint", "true")][0]
    gp, op = cell[("greedy", "pred")][0], cell[("joint", "pred")][0]
    gain = 100 * (gt - ot) / gt
    rows.append(dict(lam=lam, greedy_true=gt, joint_true=ot, greedy_pred=gp, joint_pred=op,
                     joint_gain_pct=gain, delay_s=cell[("greedy", "true")][1]))
    print("{:6.1f}{:14.0f}{:14.0f}{:13.0f}{:14.0f}{:11.2f}%{:9.0f}".format(
        lam, gt, ot, gp, op, gain, cell[("greedy", "true")][1]))
OUT["load_sweep"] = rows

print()
worst = max(r["joint_true"] - r["greedy_true"] for r in rows)
sane("joint assignment never loses to greedy under perfect information",
     worst <= 1e-6, "max shortfall {:.3f} J/job".format(worst))

ref = rows[1]
gap = ref["greedy_pred"] - ref["joint_true"]
pol = ref["greedy_true"] - ref["joint_true"]
prd = ref["joint_pred"] - ref["joint_true"]
print("at lam=0.5, today's system is {:.1f} J/job above the true optimum ({:.2f}%)".format(
    gap, 100 * gap / ref["joint_true"]))
print("  policy alone (greedy, perfect prediction):  {:6.1f} J/job".format(pol))
print("  prediction alone (joint, imperfect model):  {:6.1f} J/job".format(prd))
OUT["decomposition_lam0.5"] = dict(total_gap_j=gap, policy_j=pol, prediction_j=prd,
                                   total_gap_pct=100 * gap / ref["joint_true"])
sane("both error sources are small relative to the facility bill",
     abs(gap) / ref["joint_true"] < 0.02,
     "total gap {:.2f}% of energy per job".format(100 * gap / ref["joint_true"]))

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/b7_assignment.json", "w", encoding="utf-8"), indent=1)
print()
print("sanity: {}/{} passed".format(sum(1 for s in SANITY if s["passed"]), len(SANITY)))
print("saved -> experiments/results/b7_assignment.json")
