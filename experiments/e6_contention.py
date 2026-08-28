# -*- coding: utf-8 -*-
"""
E6  Contention replay at facility level.

The experiment that can refute the premise. Per-task energy savings are measured in
isolation; this asks whether they survive queueing, finite capacity, and the idle
floor of machines that stay powered whether or not they are used.

Facility energy over a horizon H:
    E = Σ_m P_idle,m · H              (every powered machine burns idle for the whole horizon)
      + Σ_jobs (E_active(job,m) − P_idle,m · runtime(job,m))    (dynamic excess while busy)

A job is a fixed amount of work (N samples). Runtime and energy on each machine come
from the measured pilot grid, so nothing here is simulated physics.
"""
import io, json, math, os
import numpy as np
from collections import defaultdict

MACH = ["T4", "L4", "A10G", "L40S", "A100-40GB"]
CAP = {"T4": 70.0, "L4": 72.0, "A10G": 150.0, "L40S": 350.0, "A100-40GB": 400.0}
# E0-measured warm idle power (W), NVML device-level. Note the large parts idle at a
# far LOWER fraction of cap than the small ones, which the flat-38% proxy got wrong.
IDLE_MEASURED = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB_SAMPLES = 20000


def load_tables():
    d = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
    rows = [r for b in d for r in b["rows"] if r.get("status") == "ok"]
    E, T = {}, {}
    for r in rows:
        k = (r["load"], r["precision"], r["batch"])
        E.setdefault(k, {})[r["machine"]] = r["energy_per_sample_mj"] / 1000.0   # J/sample
        T.setdefault(k, {})[r["machine"]] = r["throughput_sps"]
    keys = sorted(E)
    return keys, E, T


JOB_LOG = []
RUN_LOG = []


def simulate(keys, E, T, pool, policy, lam, idle_frac, horizon=3600.0, seed=0,
             predictor=None, sla_mult=None, tag=None, keep_jobs=False):
    """
    pool: dict machine -> count.  lam: arrivals/second.  idle_frac: idle power as
    fraction of cap.  policy: callable(free_machines, job_key) -> machine label.
    Returns facility energy (J), mean queue delay, peak power (W), completions.
    """
    rng = np.random.default_rng(seed)
    slots = []
    for m, cnt in pool.items():
        for _ in range(cnt):
            slots.append(m)
    n_slots = len(slots)
    free_at = np.zeros(n_slots)
    idle_w = (dict(IDLE_MEASURED) if idle_frac is None
              else {m: idle_frac * CAP[m] for m in MACH})

    global JOB_LOG
    jl_start = len(JOB_LOG)
    t = 0.0
    completions = 0
    delays = []
    dyn_energy = 0.0
    busy_intervals = []   # (start, end, machine) for peak computation
    sla_viol = 0

    arrivals = []
    while t < horizon:
        t += rng.exponential(1.0 / lam)
        if t < horizon:
            arrivals.append((t, keys[rng.integers(len(keys))]))

    for at, jk in arrivals:
        # slots free at arrival time
        free_idx = [i for i in range(n_slots) if free_at[i] <= at]
        if not free_idx:
            i = int(np.argmin(free_at))
            start = float(free_at[i])
            free_idx = [j for j in range(n_slots) if free_at[j] <= start]
        else:
            start = at
        cand = sorted({slots[i] for i in free_idx})
        m = policy(cand, jk, predictor)
        i = next(i for i in free_idx if slots[i] == m)
        rt = JOB_SAMPLES / T[jk][m]
        free_at[i] = start + rt
        delays.append(start - at)
        e_active = JOB_SAMPLES * E[jk][m]
        dyn_energy += e_active - idle_w[m] * rt
        busy_intervals.append((start, start + rt, m))
        JOB_LOG.append(dict(arrival=float(at), start=float(start), runtime_s=float(rt),
                            load=jk[0], precision=jk[1], batch=int(jk[2]), machine=m,
                            active_energy_j=float(e_active),
                            dynamic_excess_j=float(e_active - idle_w[m] * rt),
                            queue_delay_s=float(start - at)))
        completions += 1
        if sla_mult is not None:
            best_rt = min(JOB_SAMPLES / T[jk][mm] for mm in MACH)
            if (start - at) + rt > sla_mult * best_rt:
                sla_viol += 1

    static = sum(idle_w[m] for m in slots) * horizon
    total = static + dyn_energy

    # coincident peak power
    events = []
    for s, e, m in busy_intervals:
        events.append((s, CAP[m] - idle_w[m]))
        events.append((e, -(CAP[m] - idle_w[m])))
    events.sort()
    base = sum(idle_w[m] for m in slots)
    cur = base; peak = base
    for _, d in events:
        cur += d; peak = max(peak, cur)
    out = dict(total_j=total, static_j=static, dynamic_j=dyn_energy,
               per_job_j=total / max(completions, 1),
               mean_delay=float(np.mean(delays)) if delays else 0.0,
               peak_w=peak, completions=completions, sla_viol=sla_viol)
    jobs = JOB_LOG[jl_start:]
    if not keep_jobs:
        del JOB_LOG[jl_start:]
    else:
        for j in jobs:
            j["tag"] = tag
    RUN_LOG.append(dict(tag=tag, policy=getattr(policy, "__name__", str(policy)),
                        lam=float(lam), idle_frac=(None if idle_frac is None else float(idle_frac)),
                        idle_w={k: float(v) for k, v in idle_w.items()}, seed=int(seed),
                        horizon=float(horizon), pool={k: int(v) for k, v in pool.items()},
                        **out))
    return out


# ---------------- policies ----------------
def _stable_seed(jk):
    """B6: hash() is salted per interpreter run, so the old version was not reproducible."""
    import zlib
    return zlib.crc32(repr(jk).encode()) % (2 ** 31)


def p_random(cand, jk, pred):
    return cand[0] if len(cand) == 1 else list(cand)[np.random.default_rng(_stable_seed(jk)).integers(len(cand))]
def p_fastest(cand, jk, pred): return max(cand, key=lambda m: TT[jk][m])
def p_lowtdp(cand, jk, pred):  return min(cand, key=lambda m: CAP[m])
def p_static(cand, jk, pred):  return min(cand, key=lambda m: STATIC_RANK.index(m))
def p_oracle(cand, jk, pred):  return min(cand, key=lambda m: EE[jk][m])
def p_model(cand, jk, pred):   return min(cand, key=lambda m: pred[jk][m])


def build_predictor(keys, E):
    """Leave-one-load-family-out feature-space interaction model, as validated in E5."""
    from sklearn.linear_model import RidgeCV
    MACHI = {m: j for j, m in enumerate(MACH)}
    Y = np.array([[math.log10(E[k][m] * 1000) for m in MACH] for k in keys])
    fam = sorted({k[0] for k in keys})
    X = np.array([[1.0 if k[1] == "fp32" else 0.0, math.log2(k[2]),
                   (1.0 if k[1] == "fp32" else 0.0) * math.log2(k[2]), math.log2(k[2]) ** 2]
                  + [1.0 if k[0] == f else 0.0 for f in fam] for k in keys])
    P = np.zeros_like(Y)
    for Lout in fam:
        tr = [i for i, k in enumerate(keys) if k[0] != Lout]
        te = [i for i, k in enumerate(keys) if k[0] == Lout]
        mu = Y[tr].mean(); r = Y[tr].mean(1) - mu; c = Y[tr].mean(0) - mu
        rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], r)
        base_te = mu + rr.predict(X[te])[:, None] + c[None, :]
        R = Y[tr] - (mu + r[:, None] + c[None, :])
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        v1 = Vt[0]
        sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], R @ v1)
        P[te] = base_te + np.outer(sm.predict(X[te]), v1)
    return {k: {m: P[i, MACHI[m]] for m in MACH} for i, k in enumerate(keys)}


if __name__ == "__main__":
    keys, E, T = load_tables()
    EE, TT = E, T
    STATIC_RANK = sorted(MACH, key=lambda m: np.median([E[k][m] for k in keys]))
    pred = build_predictor(keys, E)
    print(f"static ranking learned from measurements: {STATIC_RANK}\n")

    POOL = {"T4": 2, "L4": 2, "A10G": 2, "L40S": 2, "A100-40GB": 2}
    policies = [("random-free", p_random), ("fastest-free", p_fastest),
                ("lowest-TDP-free", p_lowtdp), ("static ranking lookup", p_static),
                ("predicted energy (E5 model)", p_model), ("oracle (measured)", p_oracle)]

    print("=" * 96)
    print("FACILITY ENERGY PER JOB (J) vs UTILISATION      pool = 2 of each of 5 machines, idle = 38% of cap")
    print("=" * 96)
    lams = [0.02, 0.05, 0.10, 0.20, 0.40]
    print(f"{'policy':30}" + "".join(f"{'lam='+str(l):>13}" for l in lams))
    print("-" * 96)
    base_rows = {}
    for name, pol in policies:
        line = f"{name:30}"
        vals = []
        for lam in lams:
            rs = [simulate(keys, E, T, POOL, pol, lam, 0.38, seed=s, predictor=pred,
                           tag=f"util|{name}|lam{lam}", keep_jobs=(s == 0)) for s in range(3)]
            v = float(np.mean([r["per_job_j"] for r in rs]))
            vals.append(v); line += f"{v:13.1f}"
        base_rows[name] = vals
        print(line)
    print("\nsaving of predicted-energy policy vs each baseline, by utilisation:")
    for name in ["fastest-free", "static ranking lookup", "random-free"]:
        s = [100 * (base_rows[name][i] - base_rows["predicted energy (E5 model)"][i]) / base_rows[name][i]
             for i in range(len(lams))]
        print(f"  vs {name:26}" + "".join(f"{x:12.1f}%" for x in s))
    g = [100 * (base_rows["predicted energy (E5 model)"][i] - base_rows["oracle (measured)"][i]) / base_rows["oracle (measured)"][i] for i in range(len(lams))]
    print(f"  {'gap to oracle':29}" + "".join(f"{x:12.1f}%" for x in g))

    print("\n" + "=" * 96)
    print("SENSITIVITY TO THE IDLE FLOOR  (lam=0.10)")
    print("=" * 96)
    print(f"{'policy':30}" + "".join(f"{'idle='+str(int(f*100))+'%':>13}" for f in [0.0, 0.2, 0.38, 0.6]))
    for name, pol in policies:
        line = f"{name:30}"
        for f in [0.0, 0.2, 0.38, 0.6]:
            rs = [simulate(keys, E, T, POOL, pol, 0.10, f, seed=s, predictor=pred,
                           tag=f"idle|{name}|f{f}") for s in range(3)]
            line += f"{float(np.mean([r['per_job_j'] for r in rs])):13.1f}"
        print(line)

    print("\n" + "=" * 96)
    print("POOL HETEROGENEITY  (lam=0.10, idle=38%)")
    print("=" * 96)
    pools = {
        "homogeneous A100 (10)": {"A100-40GB": 10},
        "homogeneous L4 (10)":   {"L4": 10},
        "two types (5+5)":       {"L4": 5, "A100-40GB": 5},
        "full heterogeneity (2 each)": POOL,
    }
    print(f"{'pool':30}" + "".join(f"{n:>26}" for n in ["fastest-free", "predicted energy"]))
    for pname, pl in pools.items():
        r1 = np.mean([simulate(keys, E, T, pl, p_fastest, 0.10, 0.38, seed=s, predictor=pred,
                               tag=f"pool|fastest|{pname}")["per_job_j"] for s in range(3)])
        r2 = np.mean([simulate(keys, E, T, pl, p_model, 0.10, 0.38, seed=s, predictor=pred,
                               tag=f"pool|model|{pname}")["per_job_j"] for s in range(3)])
        print(f"{pname:30}{r1:26.1f}{r2:26.1f}   saving {100*(r1-r2)/r1:5.1f}%")

    os.makedirs("experiments/results", exist_ok=True)
    json.dump(RUN_LOG, io.open("experiments/results/e6_runs.json", "w", encoding="utf-8"))
    json.dump(JOB_LOG, io.open("experiments/results/e6_jobs.json", "w", encoding="utf-8"))
    json.dump({"energy_j_per_sample": {str(k): E[k] for k in keys},
               "throughput_sps": {str(k): T[k] for k in keys},
               "power_cap_w": CAP, "job_samples": JOB_SAMPLES,
               "predictor_log10_mj": {str(k): pred[k] for k in keys}},
              io.open("experiments/results/e6_inputs.json", "w", encoding="utf-8"))
    print(f"\nsaved {len(RUN_LOG)} simulation runs and {len(JOB_LOG)} per-job records to experiments/results/")
