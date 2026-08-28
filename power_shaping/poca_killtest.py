# -*- coding: utf-8 -*-
"""
PoC-A -- the public-data kill test  (REAL emerald DVFS data, continuous power caps).

Question: does selecting workloads by measured power elasticity beat simple policies
(uniform cap, priority-first, largest-power-first, static class order)?

Data: data/raw/dvfs_sweep.csv from ai-emerald/emerald-ai-demo-may-2025 (Nature Energy
2025, 256-GPU cluster). 8 measured LLM workloads x 6 GPU power caps (100..400 W), with
measured normalized throughput and power-per-GPU. We build a cluster pool by replicating
each measured workload type with small measurement-noise jitter (repo reports ~1.6%
within-cell sd), preserving the REAL curve shapes.

Action space: a continuous GPU power cap per workload (as nvidia-smi -pl sets, in watts),
throughput interpolated from the measured curve. Every policy allocates power under a cap
C = frac * total_baseline_power; service cost = sum_i w_i*(1 - q_i(x_i)).

Oracle: exact via fine-grid multiple-choice knapsack DP; cross-checked against continuous
water-filling. Invariants:
  INV-1 oracle cost <= every feasible heuristic cost (oracle is a global min)
  INV-2 homogeneous pool (all curves identical) -> oracle headroom over uniform ~ 0
Both must hold or results are void.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
RESULTS = os.path.join(HERE, "results")

# priority weight and class from the workload-name prefix
def classify(name: str):
    if name.startswith("infer"):
        return "online", 4.0          # inference serving: latency-sensitive, high priority
    if name.startswith("ft"):
        return "finetune", 2.0        # fine-tuning: medium
    if name.startswith("pt"):
        return "pretrain", 1.0        # pretraining: most deferrable, low priority
    return "other", 1.0

NGRID = 200  # fine power-grid resolution per workload


def _load_measured():
    df = pd.read_csv(os.path.join(RAW, "dvfs_sweep.csv"))
    wls = []
    for name, g in df.groupby("Workload"):
        g = g.sort_values("power per GPU")
        p = g["power per GPU"].to_numpy(float)
        q = g["normalized throughput"].to_numpy(float)
        q = np.clip(q, 1e-6, 1.0)
        cls, w = classify(name)
        wls.append(dict(name=name, cls=cls, w=w, p_meas=p, q_meas=q))
    return wls


def build_pool(replicate=5, seed=0, homogeneous=False):
    """Cluster pool = each measured workload type replicated `replicate` times (exact
    copies representing multiple concurrent instances of that measured workload).

    We deliberately do NOT jitter the throughput curve: a naive jitter with
    maximum.accumulate + renormalise can forge a flat q=1 plateau near the top, which
    an oracle then exploits as 'free' shed power and inflates its apparent gain. Keeping
    the measured curve exact preserves its (concave-throughput / convex-cost) shape, so
    heterogeneity comes only from the 8 real workload shapes and the priority weights.
    Per-instance power variation is modelled as a small multiplicative POWER scale, which
    shifts the curve without breaking convexity."""
    rng = np.random.default_rng(seed)
    meas = _load_measured()
    if homogeneous:                    # INV-2: everyone is a clone of one measured curve
        meas = [meas[0]] * len(meas)
    pool = []
    for base in meas:
        for k in range(replicate):
            p = base["p_meas"].copy()
            if not homogeneous and k > 0:
                p = p * (1.0 + rng.normal(0, 0.02))   # small per-instance power shift only
            xg = np.linspace(p.min(), p.max(), NGRID)
            qg = np.interp(xg, p, base["q_meas"])       # exact measured shape, monotone/concave
            pool.append(dict(
                name=base["name"], cls=base["cls"], w=base["w"], k=k,
                xg=xg, qg=qg, cost_g=base["w"] * (1.0 - qg),
                pmin=float(xg[0]), pmax=float(xg[-1]),
            ))
    return pool


# ---- power/cost evaluation on the fine grid --------------------------------
def q_at(wl, x):
    return float(np.interp(x, wl["xg"], wl["qg"]))

def cost_at(wl, x):
    return float(np.interp(x, wl["xg"], wl["cost_g"]))

def pool_cost_power(pool, xs):
    cost = sum(cost_at(pool[i], xs[i]) for i in range(len(pool)))
    power = sum(xs)
    return cost, power


# ---- baselines (all continuous) --------------------------------------------
def b0_none(pool, C):
    return [wl["pmax"] for wl in pool]

def b1_uniform(pool, C):
    """Uniform curtailment: every workload sheds the SAME fraction f of its own sheddable
    range (pmax - pmin), with f chosen so total power = C. Always feasible for
    C in [sum pmin, sum pmax] (a proportional cap on pmax alone can floor at pmin and
    overshoot the budget when workloads have a high pmin / narrow range)."""
    pmax = np.array([wl["pmax"] for wl in pool])
    pmin = np.array([wl["pmin"] for wl in pool])
    tot = pmax.sum()
    if C >= tot:
        return list(pmax)
    shed_range = (pmax - pmin).sum()
    f = min(1.0, (tot - C) / shed_range) if shed_range > 1e-9 else 1.0
    return list(pmax - f * (pmax - pmin))

def _greedy_shed(pool, C, order):
    """Set caps to pmax, then in `order` push each workload down to pmin until sum<=C,
    partially reducing the last one."""
    xs = [wl["pmax"] for wl in pool]
    _, pw = pool_cost_power(pool, xs)
    for i in order:
        if pw <= C:
            break
        room = xs[i] - pool[i]["pmin"]
        take = min(room, pw - C)
        xs[i] -= take
        pw -= take
    return xs

def b2_priority(pool, C):            # cap lowest-priority (low w) first
    order = sorted(range(len(pool)), key=lambda i: (pool[i]["w"], i))
    return _greedy_shed(pool, C, order)

def b3_largest_power(pool, C):       # cap highest-power first
    order = sorted(range(len(pool)), key=lambda i: -pool[i]["pmax"])
    return _greedy_shed(pool, C, order)

def b4_static_class(pool, C):        # fixed order: pretrain > finetune > online
    rank = {"pretrain": 0, "finetune": 1, "other": 2, "online": 3}
    order = sorted(range(len(pool)), key=lambda i: (rank.get(pool[i]["cls"], 9), i))
    return _greedy_shed(pool, C, order)

def _shared_curve_pool(pool):
    """Return a copy of the pool where every workload uses the POOL-AVERAGE throughput
    curve (on a common relative-power axis) but keeps its own priority weight and power
    range. A policy run on this pool knows priority + average elasticity but NOT the
    per-workload curve shape -- it isolates the value of per-workload elasticity."""
    rel = np.linspace(0.0, 1.0, NGRID)
    qs = []
    for wl in pool:
        relx = (wl["xg"] - wl["pmin"]) / (wl["pmax"] - wl["pmin"] + 1e-9)
        qs.append(np.interp(rel, relx, wl["qg"]))
    qbar = np.mean(qs, axis=0)
    shared = []
    for wl in pool:
        xg = wl["xg"]
        qg = np.interp((xg - wl["pmin"]) / (wl["pmax"] - wl["pmin"] + 1e-9), rel, qbar)
        shared.append(dict(name=wl["name"], cls=wl["cls"], w=wl["w"], k=wl["k"],
                           xg=xg, qg=qg, cost_g=wl["w"] * (1.0 - qg),
                           pmin=wl["pmin"], pmax=wl["pmax"]))
    return shared


def b5g_shared_elasticity(pool, C):
    """Profiled greedy that only knows priority + AVERAGE elasticity (no per-workload
    shape). Decisions made on the shared-curve pool, cost scored on the TRUE pool."""
    shared = _shared_curve_pool(pool)
    xs = b5_profiled(shared, C)
    return xs  # apply the same power targets to the real pool for scoring


def b5_profiled(pool, C, step_frac=0.004):
    """Continuous water-filling: repeatedly remove a small watt increment from whichever
    workload has the lowest marginal cost-per-watt. Exact optimum for convex cost."""
    xs = [wl["pmax"] for wl in pool]
    _, pw = pool_cost_power(pool, xs)
    steps = [(wl["pmax"] - wl["pmin"]) * step_frac for wl in pool]
    # precompute marginal cost-per-watt lookups lazily
    while pw > C:
        best = None
        for i in range(len(pool)):
            if xs[i] - steps[i] < pool[i]["pmin"] - 1e-9:
                continue
            dC = cost_at(pool[i], xs[i] - steps[i]) - cost_at(pool[i], xs[i])  # >=0
            dP = steps[i]
            score = dC / dP
            if best is None or score < best[0]:
                best = (score, i)
        if best is None:
            break
        i = best[1]
        take = min(steps[i], pw - C)
        xs[i] -= take
        pw -= take
    return xs


def b6_oracle(pool, C):
    """True optimum: for this convex continuous problem, fine water-filling is exact; the
    DP handles any non-convex extension. Return the lower-cost feasible of the two so the
    oracle is a genuine lower bound on achievable cost (INV-1)."""
    xs_dp = _oracle_dp(pool, C)
    xs_wf = b5_profiled(pool, C, step_frac=0.001)   # fine water-filling
    cands = [x for x in (xs_dp, xs_wf) if x is not None]
    best = min(cands, key=lambda x: pool_cost_power(pool, x)[0])
    return best


def _oracle_dp(pool, C, buckets=6000):
    """Exact multiple-choice knapsack over a fine per-workload power grid.
    Each level's power is rounded UP to buckets so the DP budget strictly upper-bounds
    the true allocated power -> the returned allocation always satisfies sum <= C."""
    pmax_tot = sum(wl["pmax"] for wl in pool)
    scale = buckets / pmax_tot
    cap_b = int(np.floor(C * scale))
    if cap_b < 0:
        return None
    INF = float("inf")
    dp = np.full(cap_b + 1, INF); dp[0] = 0.0
    parents = []
    LEVELS = 80
    for wl in pool:
        xs_levels = np.linspace(wl["pmin"], wl["pmax"], LEVELS)
        pw_b = np.ceil(xs_levels * scale).astype(int)   # conservative: bucket power >= true power
        cost_l = np.interp(xs_levels, wl["xg"], wl["cost_g"])
        ndp = np.full(cap_b + 1, INF)
        choice = np.full(cap_b + 1, -1, dtype=int)
        for li in range(LEVELS):
            w = int(pw_b[li])
            if w > cap_b:
                continue
            src = dp[: cap_b + 1 - w]
            cand = src + cost_l[li]
            tgt = np.arange(w, cap_b + 1)
            better = cand < ndp[tgt]
            ndp[tgt[better]] = cand[better]
            choice[tgt[better]] = li
        dp = ndp
        parents.append((choice, xs_levels, pw_b))
    if not np.isfinite(dp).any():
        return None
    best_b = int(np.argmin(dp))
    xs = [0.0] * len(pool)
    b = best_b
    for i in range(len(pool) - 1, -1, -1):
        choice, xs_levels, pw_b = parents[i]
        li = int(choice[b])
        if li < 0:
            return None
        xs[i] = float(xs_levels[li])
        b -= int(pw_b[li])
    # refine: the ceil-rounding leaves a little budget unspent; water-fill it back so the
    # oracle is a tight optimum (guarantees oracle <= every heuristic).
    return _waterfill_slack(pool, xs, C)


def _waterfill_slack(pool, xs, C, step=2.0):
    """Spend any remaining budget (C - sum xs) by adding power to whichever workload gives
    the largest cost reduction per watt, until budget exhausted or all at pmax."""
    xs = list(xs)
    pw = sum(xs)
    while C - pw > step:
        best = None
        for i in range(len(pool)):
            room = pool[i]["pmax"] - xs[i]
            if room < 1e-6:
                continue
            d = min(step, room)
            gain = cost_at(pool[i], xs[i]) - cost_at(pool[i], xs[i] + d)  # >=0
            if gain <= 0:
                continue
            score = gain / d
            if best is None or score > best[0]:
                best = (score, i, d)
        if best is None:
            break
        _, i, d = best
        d = min(d, C - pw)
        xs[i] += d
        pw += d
    return xs


POLICIES = {
    "B0_none": b0_none, "B1_uniform": b1_uniform, "B2_priority": b2_priority,
    "B3_largest_power": b3_largest_power, "B4_static_class": b4_static_class,
    "B5g_shared_elasticity": b5g_shared_elasticity,
    "B5_profiled": b5_profiled,
}


def heterogeneity_stats(pool):
    """Spread of normalized throughput across workload TYPES at a common relative cap."""
    meas = _load_measured()
    # at 50% of max power (~184 W), throughput per type
    q50 = {}
    for m in meas:
        x = 0.5 * m["p_meas"].max()
        q50[m["name"]] = float(np.interp(x, m["p_meas"], m["q_meas"]))
    vals = np.array(list(q50.values()))
    return {"q_at_50pct_power_by_type": q50,
            "spread": float(vals.max() - vals.min()),
            "min": float(vals.min()), "max": float(vals.max())}


def run_static(replicate=5, fracs=(0.95, 0.90, 0.85, 0.80, 0.75, 0.70)):
    pool = build_pool(replicate=replicate)
    total_base = sum(wl["pmax"] for wl in pool)
    fmin = sum(wl["pmin"] for wl in pool)
    rows = []
    for frac in fracs:
        C = frac * total_base
        curt = round((1 - frac) * 100)
        rec = {"curtailment_pct": curt, "cap_frac": frac,
               "feasible": bool(C >= fmin), "cap_w": C, "total_base_w": total_base}
        for name, fn in POLICIES.items():
            xs = fn(pool, C)
            cost, pw = pool_cost_power(pool, xs)
            rec[name] = {"cost": cost, "power_w": pw, "violation_w": max(0.0, pw - C),
                         "shed_w": total_base - pw}
        xs_o = b6_oracle(pool, C)
        if xs_o is not None:
            co, pwo = pool_cost_power(pool, xs_o)
        else:
            co, pwo = rec["B5_profiled"]["cost"], rec["B5_profiled"]["power_w"]
        rec["B6_oracle"] = {"cost": co, "power_w": pwo, "violation_w": max(0.0, pwo - C),
                            "shed_w": total_base - pwo}
        uni = rec["B1_uniform"]["cost"]; ora = co
        pri = rec["B2_priority"]["cost"]; prof = rec["B5_profiled"]["cost"]
        shared = rec["B5g_shared_elasticity"]["cost"]
        rec["profiled_gain_vs_uniform_pct"] = 100 * (uni - prof) / uni if uni > 1e-9 else 0.0
        rec["oracle_gain_vs_uniform_pct"] = 100 * (uni - ora) / uni if uni > 1e-9 else 0.0
        denom = uni - ora
        # A2: how much of the oracle's advantage over uniform does simple priority-first capture?
        rec["priority_capture_of_oracle_pct"] = (100 * (uni - pri) / denom) if denom > 1e-6 else float("nan")
        # Decomposition: shared-elasticity (priority + AVERAGE curve) capture of oracle...
        rec["shared_elasticity_capture_pct"] = (100 * (uni - shared) / denom) if denom > 1e-6 else float("nan")
        # ...and the residual attributable to PER-WORKLOAD elasticity knowledge:
        rec["per_workload_elasticity_gain_pct"] = 100 * (shared - ora) / shared if shared > 1e-9 else 0.0
        rows.append(rec)
    return pool, rows, total_base, fmin


def check_invariants(replicate=5):
    ok = True; msgs = []
    # INV-1 on the real pool
    pool = build_pool(replicate=replicate)
    tb = sum(wl["pmax"] for wl in pool)
    for frac in (0.9, 0.8, 0.7):
        C = frac * tb
        xs_o = b6_oracle(pool, C); co, _ = pool_cost_power(pool, xs_o)
        for name, fn in POLICIES.items():
            if name == "B0_none":
                continue
            c, pw = pool_cost_power(pool, fn(pool, C))
            if pw <= C + 1e-3 and c + 1e-4 < co:
                ok = False; msgs.append(f"INV-1 VIOLATED frac={frac}: {name} {c:.4f} < oracle {co:.4f}")
    # INV-2 homogeneous pool
    hp = build_pool(replicate=replicate, homogeneous=True)
    tbh = sum(wl["pmax"] for wl in hp)
    for frac in (0.9, 0.8, 0.75):
        C = frac * tbh
        xs_o = b6_oracle(hp, C); co, _ = pool_cost_power(hp, xs_o)
        uc, _ = pool_cost_power(hp, b1_uniform(hp, C))
        head = (uc - co) / uc if uc > 1e-9 else 0.0
        tag = "ok" if head <= 0.01 else "VIOLATED"
        if head > 0.01:
            ok = False
        msgs.append(f"INV-2 {tag} frac={frac}: homogeneous headroom {head*100:.3f}% (expect ~0)")
    return ok, msgs


def main():
    os.makedirs(RESULTS, exist_ok=True)
    ok, msgs = check_invariants()
    print("== invariants ==")
    for m in msgs: print("  " + m)
    print(f"  invariants pass: {ok}")
    if not ok:
        raise SystemExit("INVARIANT FAILURE -- results void")

    het = heterogeneity_stats(build_pool())
    pool, rows, tb, fmin = run_static()
    out = {"provenance": "real:emerald/dvfs_sweep.csv",
           "n_workload_types": 8, "n_pool": len(pool),
           "total_base_w": tb, "feasible_min_w": fmin,
           "heterogeneity": het,
           "invariants": {"pass": ok, "messages": msgs},
           "static": rows}
    with open(os.path.join(RESULTS, "poca_killtest.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n== heterogeneity (real, 8 LLM workload types) ==")
    print(f"  throughput at 50% power: min={het['min']:.3f} max={het['max']:.3f} spread={het['spread']:.3f}")
    print(f"\n== PoC-A static kill test (real emerald DVFS, {len(pool)} jobs) ==")
    print(f"{'curt%':>6} {'uniform':>8} {'priority':>8} {'largest':>8} {'profiled':>8} {'oracle':>8} "
          f"{'prof↑%':>7} {'orac↑%':>7} {'pri-cap%':>8}")
    for r in rows:
        print(f"{r['curtailment_pct']:>6} {r['B1_uniform']['cost']:>8.3f} {r['B2_priority']['cost']:>8.3f} "
              f"{r['B3_largest_power']['cost']:>8.3f} {r['B5_profiled']['cost']:>8.3f} {r['B6_oracle']['cost']:>8.3f} "
              f"{r['profiled_gain_vs_uniform_pct']:>7.1f} {r['oracle_gain_vs_uniform_pct']:>7.1f} "
              f"{r['priority_capture_of_oracle_pct']:>8.1f}")
    print(f"\nwritten -> results/poca_killtest.json")


if __name__ == "__main__":
    main()
