# -*- coding: utf-8 -*-
"""
Temporal per-pool heterogeneous controller: the end-to-end version of the section-5.6 result.
Unlike the homogeneous PoC-B (one elasticity curve, one cluster cap), here each request carries a
workload TYPE with its OWN measured power-cap elasticity curve, and the controller sets a PER-TYPE
power cap (plus per-class deferral). Priority classes CROSS-CUT types, so a critical memory-bound
job and an offline compute-bound job coexist and priority alone cannot separate them.

The real Azure arrival trace drives the tick simulator; a dynamic allowance C(t) dips mid-window;
all controllers are forced STRICTLY under C(t). We compare, at equal grid compliance:
  uniform     one cap for every type, class-blind (FIFO)
  priority    defer low-priority classes, one shared cap on the rest
  elasticity  per-type caps allocated by measured elasticity (marginal-cost water-filling) + defer

INVARIANT (pre-registered): give every type the SAME elasticity curve -> elasticity == priority
(no per-type variation to exploit), reproducing the homogeneous PoC-B and proving the gain is
elasticity, not a simulator asymmetry.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import pocb_sim as S

RAW = S.RAW
# 4 measured workload types spanning the elasticity range (curves from own_aws_sweep.csv):
#   gemm_fp16 compute-bound (steep), resnet50 moderate, embed_gather + membw memory-bound (flat).
TYPES = ["gemm_fp16", "resnet50", "embed_gather", "membw"]
TYPE_PROB = [0.30, 0.30, 0.20, 0.20]
CLASSES = S.CLASSES
# Inference-dominant mix: a latency-critical serving fleet where the NON-DEFERRABLE pool
# (critical+interactive = 80%) is itself large enough to bind under a severe dip, so deferral of the
# small batch tail cannot absorb the shortfall and in-place power shaping becomes the operative lever.
# (The paper's priority result uses the spec mix S.CLASS_PROB; this experiment isolates the regime
# where per-type elasticity adds value ON TOP of priority.)
CLASS_PROB = [0.25, 0.55, 0.10, 0.10]
CLASS_WEIGHT = S.CLASS_WEIGHT; CLASS_DEADLINE_S = S.CLASS_DEADLINE_S
N_GPU = S.N_GPU; P_GPU_MAX = S.P_GPU_MAX; P_IDLE = S.P_GPU_IDLE_FRAC
TOK = S.TOK_PER_GPU_S; DT = S.DT
STRICT = True; CAP_FLOOR = 0.2


def type_curves(homogeneous=False):
    """q_t(cap_frac) = normalized throughput vs cap fraction, per type, from measured A10G data."""
    d = pd.read_csv(os.path.join(RAW, "own_aws_sweep.csv"))
    curves = {}
    for t in TYPES:
        g = d[d["workload"] == t].groupby("cap_frac", as_index=False).agg(thr=("throughput", "mean"))
        g = g.sort_values("cap_frac"); q = g["thr"].to_numpy(float); q = q / q.max()
        curves[t] = (g["cap_frac"].to_numpy(float), np.maximum.accumulate(q))
    if homogeneous:                                   # invariant: all types share the mean curve
        caps = curves[TYPES[0]][0]
        mean = np.mean([np.interp(caps, curves[t][0], curves[t][1]) for t in TYPES], axis=0)
        curves = {t: (caps, mean) for t in TYPES}
    return curves


def qfrac(curves, t, cap):
    cf, q = curves[t]; return float(np.interp(cap, cf, q))


# The raw Azure trace slice under-fills a 20-GPU cluster, so a dip placed off the demand peak never
# binds. Scale request work so the UNCONTROLLED cluster runs near saturation (peak util ~0.9); only
# then does a grid dip force the must-run pool to shed and expose the elasticity lever.
LOAD_MULT = 3.0


def load_reqs(seed=0):
    df = pd.read_parquet(os.path.join(RAW, "azure_window.parquet"))
    rng = np.random.default_rng(seed)
    typ = rng.choice(len(TYPES), size=len(df), p=TYPE_PROB)
    cls = rng.choice(len(CLASSES), size=len(df), p=CLASS_PROB)      # class cross-cuts type
    work = LOAD_MULT * (S.C_PREFILL * df["ContextTokens"].to_numpy(float) + S.C_DECODE * df["GeneratedTokens"].to_numpy(float))
    return dict(t=df["t_s"].to_numpy(float), work=work, typ=typ, cls=cls)


# per-type GPU share (fixed, proportional to type load share)
GSHARE = {i: TYPE_PROB[i] * N_GPU for i in range(len(TYPES))}


def type_power(curves, ti, cap, util):
    return GSHARE[ti] * P_GPU_MAX * (P_IDLE + (1 - P_IDLE) * util) * cap


def type_capacity(ti, cap, curves):
    return GSHARE[ti] * TOK * qfrac(curves, TYPES[ti], cap) * DT


def controller(kind, curves, demand_t, gate_classes, C_t):
    """Return per-type caps (list) meeting C_t. demand_t[ti]=admitted work for type ti this tick.
    gate_classes handled by caller (which classes are admitted). Strict: power(caps) <= C_t."""
    nt = len(TYPES)
    def power_at(caps):
        # STRICT worst-case bound: charge every type at util=1 so the cap GUARANTEES power<=C_t
        # regardless of realized load. Applied identically to all three controllers -> equal
        # (near-100%) grid compliance, so the elasticity/priority contrast is not confounded by
        # one controller quietly drawing over the ceiling.
        return sum(type_power(curves, ti, caps[ti], 1.0) for ti in range(nt))
    if kind in ("uniform", "priority"):
        # single shared cap, reduce until <= C_t (strict)
        cap = 1.0
        for cg in np.linspace(1.0, CAP_FLOOR, 40):
            if power_at([cg]*nt) <= C_t:
                cap = cg; break
        else:
            cap = CAP_FLOOR
        return [cap]*nt
    # elasticity: greedy marginal-cost water-filling -- lower the cap of the type with the least
    # weighted-throughput-loss per watt saved, one small step at a time, until power <= C_t.
    caps = [1.0]*nt; step = 0.02
    for _ in range(400):
        if power_at(caps) <= C_t:
            break
        best, bg = None, 1e18
        for ti in range(nt):
            if caps[ti] <= CAP_FLOOR + 1e-9 or demand_t[ti] <= 0:
                continue
            c0 = list(caps); p0 = power_at(c0); q0 = qfrac(curves, TYPES[ti], caps[ti])
            c1 = list(caps); c1[ti] = max(CAP_FLOOR, caps[ti]-step); p1 = power_at(c1)
            q1 = qfrac(curves, TYPES[ti], c1[ti])
            dp = p0 - p1
            if dp <= 1e-6:
                continue
            dloss = (q0 - q1) * demand_t[ti]                 # throughput lost (weighted below by serve order)
            g = dloss / dp
            if g < bg:
                bg, best = g, ti
        if best is None:
            for ti in range(nt): caps[ti] = CAP_FLOOR
            break
        caps[best] = max(CAP_FLOOR, caps[best]-step)
    return caps


LADDER_DEFER = {"uniform": [], "priority": ["offline", "elastic"], "elasticity": ["offline", "elastic"]}


def simulate(reqs, C, kind, curves, seed=0):
    from collections import deque
    T = len(C); order = np.argsort(reqs["t"]); ai = reqs["t"][order]; n = len(reqs["t"])
    dq = deque()                                    # (work, arrival, typ, cls)
    ptr = 0; P = np.zeros(T); completed = []
    ncls = len(CLASSES); backlog_by_type = np.zeros(len(TYPES))
    demand_ema = np.zeros(len(TYPES))
    for k in range(T):
        # arrivals
        newwork = np.zeros(len(TYPES))
        while ptr < n and ai[ptr] < (k+1)*DT:
            j = order[ptr]; dq.append([reqs["work"][j], reqs["t"][j], int(reqs["typ"][j]), int(reqs["cls"][j])])
            newwork[int(reqs["typ"][j])] += reqs["work"][j]; ptr += 1
        demand_ema = 0.6*demand_ema + 0.4*newwork
        # feedforward per-type demand = arrival-rate estimate; controller sets caps strictly under C_t.
        # Deferral is IMPLICIT: priority-ordered serving under capacity limits leaves low-priority work
        # queued (it defers itself); uniform serves FIFO so it starves critical instead.
        demand_t = np.maximum(demand_ema, 1e-9)
        caps = controller(kind, curves, demand_t, set(range(ncls)), C[k])
        # serve: priority order (critical first) for priority/elasticity, FIFO for uniform
        cap_left = {ti: type_capacity(ti, caps[ti], curves) for ti in range(len(TYPES))}
        served_work = 0.0; now = (k+1)*DT
        pool = sorted(list(dq), key=lambda it: it[1]) if kind == "uniform" else sorted(list(dq), key=lambda it: it[3])
        remove = []
        for it in pool:
            ti = it[2]
            if cap_left[ti] <= 1e-9:
                continue
            take = min(it[0], cap_left[ti])
            it[0] -= take; cap_left[ti] -= take; served_work += take
            if it[0] <= 1e-9:
                lat = now - it[1]; completed.append((it[3], lat, lat <= CLASS_DEADLINE_S[CLASSES[it[3]]]))
                remove.append(it)
        for it in remove:
            dq.remove(it)
        # power = sum per-type
        pw = 0.0
        for ti in range(len(TYPES)):
            served_ti = type_capacity(ti, caps[ti], curves) - cap_left[ti]
            util = min(1.0, served_ti / (GSHARE[ti]*TOK*DT + 1e-9))
            pw += type_power(curves, ti, caps[ti], util)
        P[k] = pw
    # metrics
    peak_power = float(P.max()) if len(P) else 0.0
    comp = np.array([(c, l, m) for c, l, m in completed], float) if completed else np.zeros((0, 3))
    wcost = sum(CLASS_WEIGHT[CLASSES[int(c)]]*(0 if m else 1) for c, _, m in completed)
    wcost += sum(CLASS_WEIGHT[CLASSES[it[3]]] for it in dq)          # unfinished
    crit = comp[comp[:, 0] == 0] if len(comp) else comp
    inter = comp[comp[:, 0] == 1] if len(comp) else comp
    csl = float(100*(1-crit[:, 2].mean())) if len(crit) else 0.0
    isl = float(100*(1-inter[:, 2].mean())) if len(inter) else 0.0
    compl = float(100*(P <= C+1e-6).mean())
    return {"wcost": wcost, "crit_slo": csl, "inter_slo": isl, "compliance": compl, "peak_power": peak_power}


def run(dips=(0.55, 0.45, 0.4, 0.35, 0.3), seed=0):
    curves = type_curves(); reqs = load_reqs(seed)
    T = int(3600/DT)
    # Reference = the UNCONTROLLED realized DEMAND peak on this trace slice (not the theoretical
    # cluster max). Dipping relative to actual demand is what makes C(t) bind: an allowance set to
    # 40% of the theoretical max leaves the real (sub-peak) demand untouched and nothing shapes.
    ref_power = simulate(reqs, np.full(T, 1e15), "uniform", curves, seed)["peak_power"]
    full_cluster = sum(type_power(curves, ti, 1.0, 1.0) for ti in range(len(TYPES)))
    print(f"[load] uncontrolled peak power {ref_power:.0f} W = {100*ref_power/full_cluster:.0f}% of full-cluster ceiling (want ~90%)")
    rows = []
    for d in dips:
        C = np.full(T, ref_power*1.05)
        C[int(1200/DT):int(2400/DT)] = ref_power*d
        rec = {"dip_frac": d}
        for kind in ("uniform", "priority", "elasticity"):
            rec[kind] = simulate(reqs, C, kind, curves, seed)
        u, p, e = rec["uniform"]["wcost"], rec["priority"]["wcost"], rec["elasticity"]["wcost"]
        rec["priority_gain_vs_uniform_pct"] = round(100*(u-p)/u, 1) if u > 1e-9 else 0.0
        rec["elasticity_gain_vs_priority_pct"] = round(100*(p-e)/p, 1) if p > 1e-9 else 0.0
        rows.append(rec)
    return rows


def invariant():
    """Homogeneous curves -> elasticity == priority (elasticity gain ~0)."""
    curves = type_curves(homogeneous=True); reqs = load_reqs(0); T = int(3600/DT)
    ref_power = simulate(reqs, np.full(T, 1e15), "uniform", curves, 0)["peak_power"]
    C = np.full(T, ref_power*1.05); C[int(1200/DT):int(2400/DT)] = ref_power*0.4
    p = simulate(reqs, C, "priority", curves)["wcost"]; e = simulate(reqs, C, "elasticity", curves)["wcost"]
    gain = 100*(p-e)/p if p > 1e-9 else 0.0
    return abs(gain) < 3.0, gain


def main():
    ok, g = invariant()
    print(f"INVARIANT homogeneous curves -> elasticity gain {g:.1f}% (must be ~0): {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("invariant failed")
    rows = run()
    json.dump(rows, open(os.path.join(S.RESULTS, "het_pocb.json"), "w"), indent=2)
    print("\n== Temporal heterogeneous controller (equal strict grid compliance) ==")
    print(f"{'dip':>5} {'compl u/p/e':>14} {'crit u/p/e':>12} {'wcost u/p/e':>22} {'prio%':>6} {'elas%':>6}")
    for r in rows:
        u, p, e = r["uniform"], r["priority"], r["elasticity"]
        print(f"{r['dip_frac']:>5} {u['compliance']:4.0f}/{p['compliance']:3.0f}/{e['compliance']:3.0f}    "
              f"{u['crit_slo']:3.0f}/{p['crit_slo']:3.0f}/{e['crit_slo']:3.0f}  "
              f"{u['wcost']:6.0f}/{p['wcost']:6.0f}/{e['wcost']:6.0f}  "
              f"{r['priority_gain_vs_uniform_pct']:5.1f} {r['elasticity_gain_vs_priority_pct']:5.1f}")
    print("written -> results/het_pocb.json")


if __name__ == "__main__":
    main()
