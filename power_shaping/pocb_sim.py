# -*- coding: utf-8 -*-
"""
PoC-B track B1 -- realistic online-inference power shaping on the REAL Azure LLM trace.

A 1-second-tick simulator replays the extracted busy window (real arrivals, real context/
generated token counts). Serving cluster power is driven by token work and a GPU power-cap
elasticity taken from the emerald inference DVFS curve. A dynamic power envelope C(t) dips
mid-window; three controllers must keep facility power under C(t):

  uniform   : lower the GPU power cap for the WHOLE cluster (class-blind)
  priority  : keep critical/interactive at full power, shed offline/elastic first (delay)
  elasticity: shed offline first (cheapest service cost per watt), then batch, then power-cap
              -- workload-aware, uses class + the measured elasticity curve

Synthetic SLO classes are layered on the trace (varied in sensitivity). Metrics: envelope
compliance, per-class p95 latency and SLO-violation rate, weighted service cost.

INVARIANTS
  INV-B1 with C=inf all three controllers are identical and meet baseline SLOs
  INV-B2 when the envelope binds, uniform's CRITICAL-class SLO violations >= priority's
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
RESULTS = os.path.join(HERE, "results")

# ---- serving-cluster parameters -------------------------------------------
# N_GPU is sized to ~65% baseline utilisation for the replayed load (see load_requests);
# an oversized cluster would never queue and the envelope dip would not bite.
N_GPU = 20                  # serving GPUs (a real serving deployment for this trace slice)
P_GPU_MAX = 368.0           # W at full power cap (emerald infer_llama_8b max)
P_GPU_IDLE_FRAC = 0.30      # idle draw as fraction of max
TOK_PER_GPU_S = 2500.0      # token-equivalents/s per GPU at full power
C_PREFILL = 0.2             # work per context token (parallel prefill, cheap)
C_DECODE = 1.0              # work per generated token (sequential decode, expensive)

# emerald inference elasticity: normalized throughput vs GPU power cap (from dvfs_sweep)
_INFER = None
def infer_elasticity():
    global _INFER
    if _INFER is None:
        df = pd.read_csv(os.path.join(RAW, "dvfs_sweep.csv"))
        g = df[df["Workload"] == "infer_llama_8b"].sort_values("power per GPU")
        _INFER = (g["power per GPU"].to_numpy(float), g["normalized throughput"].to_numpy(float))
    return _INFER

def throughput_frac(cap_frac):
    """Relative serving throughput at a GPU power cap = cap_frac * P_GPU_MAX."""
    p, q = infer_elasticity()
    return float(np.interp(cap_frac * P_GPU_MAX, p, q))

def cluster_power(cap_frac, util):
    """Facility-agnostic IT power at a given power-cap fraction and utilization in [0,1]."""
    per_gpu = P_GPU_MAX * (P_GPU_IDLE_FRAC + (1 - P_GPU_IDLE_FRAC) * util) * cap_frac
    return N_GPU * per_gpu

# ---- SLO classes (synthetic labels on the real trace) ----------------------
CLASSES = ["critical", "interactive", "elastic", "offline"]
CLASS_PROB = [0.10, 0.50, 0.25, 0.15]        # spec's initial mix (varied in sensitivity)
# deadlines are queueing-delay budgets (>= tick so they are meetable at baseline)
CLASS_DEADLINE_S = {"critical": 1.0, "interactive": 4.0, "elastic": 20.0, "offline": 300.0}
CLASS_WEIGHT = {"critical": 8.0, "interactive": 4.0, "elastic": 1.0, "offline": 0.3}
DT = 0.5                                       # tick length (s), finer than critical deadline


def load_requests(seed=0, subsample=1.0):
    df = pd.read_parquet(os.path.join(RAW, "azure_window.parquet"))
    if subsample < 1.0:
        df = df.sample(frac=subsample, random_state=seed).sort_values("t_s")
    rng = np.random.default_rng(seed)
    cls = rng.choice(len(CLASSES), size=len(df), p=CLASS_PROB)
    work = C_PREFILL * df["ContextTokens"].to_numpy(float) + C_DECODE * df["GeneratedTokens"].to_numpy(float)
    return dict(t=df["t_s"].to_numpy(float), work=work, cls=cls)


def uncontrolled_peak(reqs, dt=DT, T=None):
    """Peak facility power with no shaping (cap=1, gates open), used as the envelope
    reference so dip_frac is a real curtailment of actual load."""
    if T is None:
        T = int(3600 / dt)
    m, ts = simulate(reqs, np.full(T, 1e12), ctrl_none, dt=dt)
    return float(np.max(ts["P"]))


def envelope(T, ref_power, dt=DT, dip_frac=0.75, dip_start_s=1200, dip_end_s=2400):
    """Dynamic envelope relative to the uncontrolled peak: flat ceiling above baseline,
    dipping to dip_frac * ref_power in the middle window."""
    C = np.full(T, ref_power * 1.02)           # flat ceiling just above baseline peak
    C[int(dip_start_s / dt):int(dip_end_s / dt)] = ref_power * dip_frac
    return C


def simulate(reqs, C, policy, dt=DT, seed=0, serve="priority", detail=None):
    """Tick simulator with a feedforward controller. `serve` sets the scheduling discipline:
    'fifo' (class-blind, offline can block critical) or 'priority' (critical served first).
    A class-blind UNIFORM baseline uses 'fifo'; workload-aware controllers use 'priority'."""
    from collections import deque
    T = len(C)
    order = np.argsort(reqs["t"])
    dq = {c: deque() for c in range(len(CLASSES))}   # per-class deques, arrival-sorted
    ptr = 0; ai = reqs["t"][order]
    P_series = np.zeros(T); util_series = np.zeros(T); cap_series = np.ones(T)
    completed = []
    n = len(reqs["t"])
    state = {"level": 0, "cap": 1.0}
    P_prev = 0.0
    backlog = 0.0                       # running total of remaining work (O(1) maintained)
    for t in range(T):
        work_this_tick = 0.0
        while ptr < n and ai[ptr] < (t + 1) * dt:
            j = order[ptr]; c = int(reqs["cls"][j])
            dq[c].append([reqs["work"][j], reqs["t"][j]])   # [remaining_work, arrival]
            work_this_tick += reqs["work"][j]
            ptr += 1
        backlog += work_this_tick
        state["demand"] = 0.7 * state.get("demand", work_this_tick) + 0.3 * work_this_tick
        # backlog-aware: fold pending work (cleared over a short horizon) into the demand
        # estimate so the controller predicts the recovery spike and caps for it.
        state["backlog"] = backlog
        cap_frac, gate = policy(state, C[t], P_prev, dt)
        cap_series[t] = cap_frac
        cap_tokens = N_GPU * TOK_PER_GPU_S * throughput_frac(cap_frac) * dt
        served_work = 0.0
        # gate[c] in [0,1]: 1 fully admitted, 0 deferred, fractional = admit only that fraction
        # of capacity to the class (fills leftover headroom with the marginal deferred class
        # instead of deferring a whole class -- avoids over-curtailment).
        classes_avail = [c for c in range(len(CLASSES)) if gate[c] > 0]
        cls_budget = {c: (float("inf") if gate[c] >= 1.0 else gate[c] * cap_tokens)
                      for c in range(len(CLASSES))}
        served_by = {c: 0.0 for c in range(len(CLASSES))}
        now = (t + 1) * dt
        while served_work < cap_tokens:
            # choose the next class to serve: priority = lowest class index; fifo = earliest front arrival
            cand = [c for c in classes_avail if dq[c] and served_by[c] < cls_budget[c] - 1e-9]
            if not cand:
                break
            if serve == "priority":
                c = min(cand)                                   # strict priority: critical class first
            elif serve == "edf":                                # earliest-deadline-first (front request)
                c = min(cand, key=lambda cc: dq[cc][0][1] + CLASS_DEADLINE_S[CLASSES[cc]])
            elif serve == "wfq":                                # weighted fair queueing: serve the class
                c = max(cand, key=lambda cc: CLASS_WEIGHT[CLASSES[cc]] / (served_by[cc] + 1e-9))  # most starved vs weight
            else:
                c = min(cand, key=lambda cc: dq[cc][0][1])       # fifo: earliest arrival across classes
            r = dq[c][0]
            take = min(r[0], cap_tokens - served_work, cls_budget[c] - served_by[c])
            r[0] -= take; served_work += take; served_by[c] += take
            if r[0] <= 1e-9:
                lat = now - r[1]
                completed.append((c, lat, lat <= CLASS_DEADLINE_S[CLASSES[c]]))
                if detail is not None:
                    detail.setdefault("comp", []).append((now, c, lat))
                dq[c].popleft()
            elif served_work >= cap_tokens - 1e-9:
                break                                            # global capacity exhausted mid-request
            # else: this class's fractional budget is spent -> it drops out of `cand`, serve others
        backlog -= served_work
        util = min(1.0, served_work / (N_GPU * TOK_PER_GPU_S * dt + 1e-9))
        util_series[t] = util
        P_series[t] = cluster_power(cap_frac, util)
        P_prev = P_series[t]
        if detail is not None:
            detail["gate"].append(list(gate))
            detail["qlen"].append([len(dq[c]) for c in range(len(CLASSES))])
            detail["util"].append(util)
            # p95 in-queue wait per class RIGHT NOW (now - arrival of still-queued requests).
            # Defined every tick (0 when a class's queue is empty), so the QoS curve has no gaps
            # and the delay shows up DURING the window as work waits -- not only when it drains.
            qage = []
            for c in range(len(CLASSES)):
                qage.append(float(np.percentile([now - it[1] for it in dq[c]], 95)) if dq[c] else 0.0)
            detail.setdefault("qage", []).append(qage)
            detail.setdefault("pred_P", []).append(float(state.get("pred_P", float("nan"))))
    queues = {c: list(dq[c]) for c in range(len(CLASSES))}     # leftover by class
    # metrics
    comp = np.array([(c, l, m) for c, l, m in completed], dtype=float) if completed else np.zeros((0, 3))
    metrics = {"n_completed": len(completed), "n_arrived": n}
    for c, name in enumerate(CLASSES):
        m = comp[comp[:, 0] == c] if len(comp) else comp
        if len(m):
            metrics[name] = {"count": int(len(m)), "p95_latency_s": float(np.percentile(m[:, 1], 95)),
                             "mean_latency_s": float(m[:, 1].mean()),
                             "slo_violation_pct": float(100 * (1 - m[:, 2].mean()))}
        else:
            metrics[name] = {"count": 0, "p95_latency_s": 0.0, "mean_latency_s": 0.0, "slo_violation_pct": 0.0}
    # weighted service cost = sum over completed of weight * (deadline miss) + tail unfinished
    unfinished = sum(len(q) for q in queues.values())
    wcost = sum(CLASS_WEIGHT[CLASSES[int(c)]] * (0 if met else 1) for c, _, met in completed)
    wcost += sum(CLASS_WEIGHT[CLASSES[c]] * len(queues[c]) for c in range(len(CLASSES)))  # unfinished penalised
    metrics["weighted_service_cost"] = wcost
    metrics["unfinished"] = unfinished
    # envelope compliance
    viol = np.maximum(0.0, P_series - C)
    metrics["envelope"] = {"compliance_pct": float(100 * (P_series <= C + 1e-6).mean()),
                           "violation_energy_kwh": float(viol.sum() * dt / 3.6e6),
                           "peak_violation_w": float(viol.max())}
    ts = {"P": P_series.tolist(), "C": C.tolist(), "cap": cap_series.tolist(),
          "util": util_series.tolist()}
    return metrics, ts


# ---- feedforward controllers (spec section 17) -----------------------------
# The envelope C(t) is known ahead, so each policy PREDICTS the power it would draw at each
# rung of its shed LADDER (from the current demand estimate) and picks the least-shedding
# rung whose predicted power <= C(t). One extra rung is added if last tick still violated.
# Policies differ only in the ladder: uniform caps power class-blind; priority/elasticity
# shed low-value classes first (protecting critical/interactive throughput).

CLASS_SHARE = np.array(CLASS_PROB)             # work share by class (≈ arrival share)

# ladder rung = (cap_fraction, gate[critical, interactive, elastic, offline])
LADDER_UNIFORM = [(f, [1, 1, 1, 1]) for f in
                  (1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.42, 0.36)]              # class-blind cap only
LADDER_PRIORITY = [                                                              # defer low classes, then cap
    (1.0, [1, 1, 1, 1]), (1.0, [1, 1, 1, 0]), (1.0, [1, 1, 0, 0]),
    (0.85, [1, 1, 0, 0]), (0.70, [1, 1, 0, 0]), (0.55, [1, 1, 0, 0]), (0.42, [1, 1, 0, 0])]
LADDER_ELASTICITY = [                    # cap through the full range first (like uniform);
    (1.00, [1, 1, 1, 1]), (0.90, [1, 1, 1, 1]), (0.80, [1, 1, 1, 1]),   # priority-serve already
    (0.70, [1, 1, 1, 1]), (0.60, [1, 1, 1, 1]),                          # protects critical here
    (0.55, [1, 1, 1, 0]), (0.48, [1, 1, 0, 0]), (0.42, [1, 1, 0, 0])]    # gate only when capping alone starves


def _predict_power(cap, gate, demand, dt):
    admitted = demand * float(np.dot(CLASS_SHARE, gate))     # work admitted this tick
    capacity = N_GPU * TOK_PER_GPU_S * throughput_frac(cap) * dt
    util = min(1.0, admitted / (capacity + 1e-9))
    return cluster_power(cap, util)


BACKLOG_HORIZON = 40         # ticks over which to plan to clear backlog
STRICT = False               # when True, every controller is forced strictly under C(t) (see _select)
CAP_FLOOR = 0.20             # hardest power cap the strict mode will apply

def _select(state, C_t, P_prev, dt, ladder):
    # effective demand includes backlog to be cleared over a short horizon (predicts the
    # recovery spike so the controller caps for it rather than violating then reacting)
    demand = state.get("demand", 0.0) + state.get("backlog", 0.0) / BACKLOG_HORIZON
    chosen = len(ladder) - 1
    for lvl, (cap, gate) in enumerate(ladder):
        if _predict_power(cap, gate, demand, dt) <= C_t:
            chosen = lvl
            break
    # one-rung feedback correction if we still overshot last tick
    bumped = False
    if P_prev > C_t * 1.01:
        chosen = min(chosen + 1, len(ladder) - 1); bumped = True
    state["level"] = chosen
    cap = ladder[chosen][0]; gate = list(ladder[chosen][1])      # copy (never mutate the ladder)
    # Anti-over-shed trim (skipped on a feedback-bump tick, where we deliberately shed harder).
    # The trim/fill size the delivered reduction to the CURRENT arrival rate (EMA, no backlog
    # inflation): the backlog term is for conservative rung SELECTION only; using it here would
    # keep the cap needlessly low and over-curtail. Real over/under-shoot is corrected by the
    # measured-power feedback below and the one-rung bump above.
    demand = state.get("demand", 0.0)
    if not bumped:
        # (1) cap axis: the ladder's cap rungs are discrete, so the chosen rung can drive power
        # well BELOW the allowance. With gating fixed, raise the cap to the LEAST-shedding value
        # whose predicted power still meets C_t (use full-power headroom first -- no throughput loss).
        if cap < 1.0:
            for cg in np.linspace(cap, 1.0, 24):
                if _predict_power(cg, gate, demand, dt) <= C_t:
                    cap = cg
        # (2) gate axis: deferring a whole class is coarse, and the backlog-aware demand
        # estimate biases the feedforward prediction high, so binary class-gating over-curtails.
        # Regulate the least-valuable DEFERRED class with FEEDBACK on the MEASURED last-tick
        # power: admit more of it while real power sits under the allowance, back off if it
        # rises above. Delivered reduction then tracks what the allowance requires, not the
        # (biased) prediction. Critical/interactive are never the marginal class here.
        # Admit deferred classes back MOST-VALUABLE FIRST (lowest class index): when headroom is
        # scarce we protect the higher-priority deferred class (elastic) and keep sacrificing the
        # least-valuable one (offline). Ascending order does this; reversing it inverts priority.
        deferred = sorted(c for c in range(len(gate)) if gate[c] == 0)   # elastic(2) before offline(3)
        if deferred:
            fill = state.get("fill", 0.0)                       # total admission across deferred classes
            if P_prev > 0:
                gap = (C_t - P_prev) / max(C_t, 1.0)            # >0 => unused headroom
                fill = min(float(len(deferred)), max(0.0, fill + 0.5 * gap * len(deferred)))
            rem = fill                                          # distribute most-valuable class first
            for c in deferred:
                g = min(1.0, rem); gate[c] = g; rem -= g
            state["fill"] = fill
        else:
            state["fill"] = 0.0
    else:
        state["fill"] = 0.0
    # STRICT compliance (a grid ceiling is a HARD constraint): guarantee REALIZED power <= C_t
    # regardless of utilization. Since cluster_power(cap,util) <= cluster_power(cap,1) = N*Pmax*cap,
    # capping at cap <= C_t/(N*Pmax) is a hard per-tick guarantee (worst-case util=1). This makes
    # EVERY controller strictly compliant, so uniform / priority / elasticity are compared at equal
    # grid compliance. It only lowers the cap (keeps each controller's class policy).
    if STRICT:
        cap_hard = C_t / (N_GPU * P_GPU_MAX)
        cap = max(0.05, min(cap, cap_hard))
    # expose the controller's OWN feedforward estimate of the power this action will draw
    # (the response-model prediction it acts on), so the figure can show model vs realized.
    state["pred_P"] = _predict_power(cap, gate, demand, dt)
    return cap, gate


def ctrl_none(state, C_t, P_prev, dt):       return 1.0, [1, 1, 1, 1]
def ctrl_uniform(state, C_t, P_prev, dt):    return _select(state, C_t, P_prev, dt, LADDER_UNIFORM)
def ctrl_priority(state, C_t, P_prev, dt):   return _select(state, C_t, P_prev, dt, LADDER_PRIORITY)
def ctrl_elasticity(state, C_t, P_prev, dt): return _select(state, C_t, P_prev, dt, LADDER_ELASTICITY)


CONTROLLERS = {"uniform": ctrl_uniform, "priority": ctrl_priority, "elasticity": ctrl_elasticity}
# uniform is class-blind -> FIFO service; workload-aware controllers serve by priority class
SERVE = {"uniform": "fifo", "priority": "priority", "elasticity": "priority"}


def run(subsample=1.0, dip_frac=0.55, seed=0, keep_ts=True):
    reqs = load_requests(seed=seed, subsample=subsample)
    T = int(3600 / DT)
    ref = uncontrolled_peak(reqs, T=T)
    C = envelope(T, ref, dip_frac=dip_frac)
    out = {"params": {"subsample": subsample, "dip_frac": dip_frac, "N_GPU": N_GPU,
                      "class_prob": CLASS_PROB}, "controllers": {}}
    for name, ctrl in CONTROLLERS.items():
        m, ts = simulate(reqs, C, ctrl, seed=seed, serve=SERVE[name])
        rec = {"metrics": m}
        if keep_ts:      # downsample ts to 1 s for figures
            step = int(1.0 / DT)
            rec["ts"] = {k: v[::step] for k, v in ts.items()}
        out["controllers"][name] = rec
    return out


def sweep_dips(subsample=1.0, dips=(0.75, 0.65, 0.55, 0.45, 0.40), seed=0):
    reqs = load_requests(seed=seed, subsample=subsample)
    T = int(3600 / DT)
    ref = uncontrolled_peak(reqs, T=T)
    rows = []
    for d in dips:
        C = envelope(T, ref, dip_frac=d)
        rec = {"dip_frac": d}
        for name, ctrl in CONTROLLERS.items():
            m, _ = simulate(reqs, C, ctrl, seed=seed, serve=SERVE[name])
            rec[name] = {"crit_slo": m["critical"]["slo_violation_pct"],
                         "inter_slo": m["interactive"]["slo_violation_pct"],
                         "off_slo": m["offline"]["slo_violation_pct"],
                         "wcost": m["weighted_service_cost"],
                         "compliance": m["envelope"]["compliance_pct"]}
        rows.append(rec)
    return rows


def fair_comparison(subsample=1.0, dips=(0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4), seed=0):
    """Equal-grid-compliance comparison. Forces STRICT compliance on every controller so the
    grid ceiling is a hard constraint for all, then reports per-class SLO and weighted service
    cost at matched (near-100%) compliance. Decomposes the gain over uniform into a PRIORITY
    component (priority vs uniform) and an ELASTICITY component (elasticity vs priority), which
    is the fair test of whether per-workload elasticity adds value beyond priority-aware
    deferral on this trace."""
    global STRICT
    reqs = load_requests(seed=seed, subsample=subsample)
    T = int(3600 / DT); ref = uncontrolled_peak(reqs, T=T)
    rows = []; STRICT = True
    try:
        for d in dips:
            C = envelope(T, ref, dip_frac=d)
            rec = {"dip_frac": d}
            for name, ctrl in CONTROLLERS.items():
                m, _ = simulate(reqs, C, ctrl, seed=seed, serve=SERVE[name])
                rec[name] = {"crit_slo": m["critical"]["slo_violation_pct"],
                             "inter_slo": m["interactive"]["slo_violation_pct"],
                             "off_slo": m["offline"]["slo_violation_pct"],
                             "wcost": m["weighted_service_cost"],
                             "compliance": m["envelope"]["compliance_pct"]}
            u, p, e = rec["uniform"]["wcost"], rec["priority"]["wcost"], rec["elasticity"]["wcost"]
            rec["priority_gain_vs_uniform_pct"] = 100 * (u - p) / u if u > 1e-9 else 0.0
            rec["elasticity_gain_vs_priority_pct"] = 100 * (p - e) / p if p > 1e-9 else 0.0
            rows.append(rec)
    finally:
        STRICT = False
    return rows


def check_invariants(subsample=0.3):
    msgs = []; ok = True
    reqs = load_requests(seed=0, subsample=subsample)
    T = int(3600 / DT)
    Cinf = np.full(T, 1e12)            # no envelope -> no controller sheds
    peak = {}
    for name, ctrl in CONTROLLERS.items():
        m, ts = simulate(reqs, Cinf, ctrl, seed=0, serve=SERVE[name])
        peak[name] = max(ts["P"])
    if max(peak.values()) - min(peak.values()) > 1.0:
        ok = False; msgs.append(f"INV-B1 VIOLATED: unconstrained peak power differs {peak}")
    else:
        msgs.append(f"INV-B1 ok: unconstrained peak power identical = {peak['uniform']:.0f} W (no shedding)")
    # binding envelope (deep dip so power-capping alone starves capacity)
    ref = uncontrolled_peak(reqs, T=T)
    C = envelope(T, ref, dip_frac=0.45)
    mu, _ = simulate(reqs, C, ctrl_uniform, seed=0, serve="fifo")
    mp, _ = simulate(reqs, C, ctrl_priority, seed=0, serve="priority")
    cu = mu["critical"]["slo_violation_pct"]; cp = mp["critical"]["slo_violation_pct"]
    if cu + 1e-9 < cp:
        ok = False; msgs.append(f"INV-B2 VIOLATED: uniform critical SLO viol {cu:.2f}% < priority {cp:.2f}%")
    else:
        msgs.append(f"INV-B2 ok: uniform critical SLO viol {cu:.2f}% >= priority {cp:.2f}%")
    return ok, msgs


def main():
    os.makedirs(RESULTS, exist_ok=True)
    ok, msgs = check_invariants()
    print("== PoC-B invariants =="); [print("  " + m) for m in msgs]
    if not ok:
        raise SystemExit("INVARIANT FAILURE -- results void")
    sweep = sweep_dips()
    out = run(dip_frac=0.42)
    out["invariants"] = {"pass": ok, "messages": msgs}
    out["dip_sweep"] = sweep
    with open(os.path.join(RESULTS, "pocb_azure.json"), "w") as f:
        json.dump(out, f)
    print("\n== PoC-B Azure replay: dip-depth sweep (60-min real trace, 20-min envelope dip) ==")
    print(f"{'dip':>5} | {'uniform: crit/inter/off SLO% wcost':>40} | {'elasticity: crit/inter/off SLO% wcost':>42}")
    for r in sweep:
        u = r["uniform"]; e = r["elasticity"]
        print(f"{r['dip_frac']:>5} | {u['crit_slo']:>6.1f} {u['inter_slo']:>6.1f} {u['off_slo']:>6.1f} {u['wcost']:>10.0f}      "
              f"| {e['crit_slo']:>6.1f} {e['inter_slo']:>6.1f} {e['off_slo']:>6.1f} {e['wcost']:>10.0f}")
    print("\nwritten -> results/pocb_azure.json")


if __name__ == "__main__":
    main()
