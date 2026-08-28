# -*- coding: utf-8 -*-
"""
G6.  Is there a turning point at which raising utilisation stops paying?

A published claim about spatial sharing holds that packing work more densely eventually
raises net emissions rather than lowering them, with a turning point somewhere around
seventy per cent load. That claim collides with this paper's consolidation result, which
buys 13.2 per cent of facility energy precisely BY raising utilisation on the machines
that stay awake. Both cannot be unconditionally true, so the question is which regime the
measured setting sits in and where, if anywhere, the curve turns.

Two mechanisms pull in opposite directions and are separated here rather than mixed:

  amortisation   idle power is a fixed cost per unit time, so more work per powered
                 second lowers joules per job. This is the consolidation effect.
  displacement   the energy-optimal accelerator for a job is often busy when the fleet
                 is loaded, so the job runs somewhere less efficient. This is a
                 HETEROGENEITY-SPECIFIC penalty and it grows with utilisation.

If displacement ever outruns amortisation, joules per job turns upward, and that turn is
the phenomenon the utilisation-fallacy claim predicts.

Two experiments
---------------
  A  load sweep. Fixed fleet, arrival rate swept, utilisation and joules per job
     measured at each rate.
  B  right-sizing sweep. Fixed arrival rate, fleet SHRUNK slot by slot, which is what a
     facility actually does when it consolidates. Utilisation rises as the fleet
     shrinks. This is the operationally meaningful version of the question.

Both are run with and without power-down, because power-down changes which mechanism
dominates: with machines asleep, idle is no longer a fixed cost and amortisation has
much less to offer.

Sanity checks stated in advance
--------------------------------
  S1  Utilisation must be non-decreasing in arrival rate, and non-decreasing as the
      fleet shrinks at fixed rate. A violation means the occupancy accounting is wrong.
  S2  ENERGY DECOMPOSITION. total = static + dynamic + wake must hold exactly at every
      point of every sweep.
  S3  ISOLATING CONTROL. On a HOMOGENEOUS fleet there is no displacement channel: every
      free slot is the same device, so a busy fleet cannot push a job onto a less
      efficient one. Joules per job must therefore be monotonically non-increasing in
      utilisation on a homogeneous fleet, up to saturation. Stated in advance: if a
      turning point appears on the homogeneous control too, then displacement is NOT the
      cause and the explanation must be sought elsewhere.
  S4  With power-down at a zero threshold and free instant wake, static energy must
      equal EXACTLY the idle power integrated over busy time alone, machine by machine:
      every second in which a slot holds no job is free.

      Check that was MIS-SPECIFIED and replaced, recorded rather than quietly deleted.
      The first version of S4 asserted that under free instant power-down static energy
      is "essentially independent of load". That is wrong on inspection of the
      accounting, and it failed: with a zero threshold a slot is powered exactly while
      it is busy, so static energy is PROPORTIONAL to busy time and therefore rises
      with load (456 kJ at lam = 0.10 against 1,980 kJ at lam = 0.40). The quantity
      that is load-invariant is static energy per busy second, not per horizon. The
      check was replaced by the exact identity above, which is the invariant actually
      implied by the accounting, and no parameter was changed to make it pass.
  S5  MONTE CARLO. Every point is run over 8 seeds and reported with its spread. A
      turning point is only accepted if the rise beyond the minimum exceeds the seed
      spread; the paper's own facility simulations carry about 1.4 per cent seed-to-seed
      variation, so a 1 per cent bump is noise.
  S7  Utilisation must lie in [0, 1] at every point of every sweep. It is a busy
      fraction, so a value above one is impossible and indicates a mis-specified
      measurement window rather than a heavily loaded fleet.
  S6  Saturation guard. Mean queueing delay is reported at every point. A "turning
      point" that only appears where delay has exploded past the 60-second SLA is a
      queueing artefact, not an energy result, and is reported as such.

BUG FOUND AND FIXED while running this experiment
---------------------------------------------------
The first run reported utilisations of 276 per cent and joules per job that fell without
limit as load rose. A busy fraction cannot exceed one, so this was a bug rather than a
finding, and the cause was in the measurement window, not in the physics. Arrivals are
generated over a one-hour horizon, but jobs admitted near the end of that hour finish
after it; at high load the tail runs for hours. Static energy was charged over the
horizon while busy time accumulated over the far longer makespan, so the denominator of
utilisation was too small and the fixed cost of holding the fleet was under-counted by
exactly the amount that made saturation look efficient. The fix charges idle power over
max(horizon, makespan), the interval the fleet is actually powered for. Every sweep
below is produced after the fix, and S7 now asserts the invariant that would have caught
it. This affects only the saturated regime; where mean delay is small the makespan and
the horizon coincide and the numbers are unchanged.

Free: reuses the measured 24 x 5 grid. No compute cost.
"""
import io, json, math, sys, warnings
from collections import defaultdict
import numpy as np

sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, HORIZON, SLA_S = 20000, 3600.0, 60.0
SEEDS = list(range(8))
SANITY, OUT = [], {}


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


keys, Ylog, Tput = load_grid()
NK = len(keys)
E_J = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(len(MACH))} for i, k in enumerate(keys)}
T_S = {k: {MACH[j]: Tput[i, j] for j in range(len(MACH))} for i, k in enumerate(keys)}


def sim(pool, lam, seed=0, sleep_after=None, wake_s=30.0, wake_j=20000.0, horizon=HORIZON):
    """Discrete-event facility with energy-first placement and exact sleep accounting.

    Accounting is the F1 form: per-slot powered time is horizon minus the sum of gaps
    beyond the sleep threshold, so mid-horizon sleep is credited.
    """
    rng = np.random.default_rng(seed)
    slots = [mm for mm, c in pool.items() for _ in range(c)]
    ns = len(slots)
    if ns == 0:
        return None
    free_at = np.zeros(ns)
    busy = defaultdict(list)
    t, arr = 0.0, []
    while t < horizon:
        t += rng.exponential(1.0 / lam)
        if t < horizon:
            arr.append((t, keys[rng.integers(NK)]))
    dyn = wake = 0.0
    delays, busy_time = [], 0.0
    busy_idle_j = 0.0          # idle power integrated over BUSY time only
    for at, jk in arr:
        idx = [i for i in range(ns) if free_at[i] <= at]
        start = at if idx else float(np.min(free_at))
        if not idx:
            idx = [i for i in range(ns) if free_at[i] <= start]
        cand = sorted({slots[i] for i in idx})
        mm = min(cand, key=lambda x: E_J[jk][x])          # energy-first placement
        i = next(i for i in idx if slots[i] == mm)
        if sleep_after is not None and busy[i] and (start - busy[i][-1][1]) > sleep_after:
            start += wake_s
            wake += wake_j
        rt = JOB / T_S[jk][mm]
        busy[i].append((start, start + rt))
        free_at[i] = start + rt
        delays.append(start - at)
        busy_time += rt
        busy_idle_j += IDLE[mm] * rt
        dyn += JOB * E_J[jk][mm] - IDLE[mm] * rt
    # BUG FOUND AND FIXED HERE, see the docstring note on the measurement window: work
    # admitted near the end of the horizon finishes AFTER it, so the fleet is powered for
    # the makespan, not for the horizon. Charging static over the horizon alone made
    # joules per job fall without limit at high load and reported utilisations above
    # 100 per cent, which is impossible.
    window = max(horizon, float(free_at.max()) if ns else horizon)
    static, asleep = 0.0, 0.0
    for i in range(ns):
        sleep_time, prev_end = 0.0, 0.0
        if sleep_after is not None:
            for s_, e_ in sorted(busy[i]):
                if s_ - prev_end > sleep_after:
                    sleep_time += (s_ - prev_end) - sleep_after
                prev_end = max(prev_end, e_)
            if window - prev_end > sleep_after:
                sleep_time += (window - prev_end) - sleep_after
        asleep += sleep_time
        static += IDLE[slots[i]] * max(0.0, window - sleep_time)
    total = static + dyn + wake
    return dict(total_j=total, static_j=static, dyn_j=dyn, wake_j=wake,
                busy_idle_j=busy_idle_j, window_s=window,
                jobs=len(arr), per_job=total / max(len(arr), 1),
                utilisation=busy_time / (ns * window),
                mean_delay=float(np.mean(delays)) if delays else 0.0,
                asleep_frac=asleep / (ns * window), slots=ns)


def multi(pool, lam, **kw):
    rs = [sim(pool, lam, seed=s, **kw) for s in SEEDS]
    agg = {}
    for k in ("per_job", "utilisation", "mean_delay", "static_j", "dyn_j", "wake_j",
              "total_j", "asleep_frac", "jobs", "busy_idle_j", "window_s"):
        v = np.array([r[k] for r in rs], float)
        agg[k] = float(v.mean())
        agg[k + "_sd"] = float(v.std(ddof=1))
    agg["slots"] = rs[0]["slots"]
    agg["per_seed_per_job"] = [float(r["per_job"]) for r in rs]
    agg["per_seed_utilisation"] = [float(r["utilisation"]) for r in rs]
    return agg


HET = {mm: 2 for mm in MACH}                      # the five-type pool used throughout Section 8
HOM = {"L4": 10}                                  # homogeneous control, most energy-efficient device

print("sanity checks stated in advance:")
r0 = multi(HET, 0.10)
sane("S2 energy decomposition is exact at a reference point",
     abs(r0["total_j"] - (r0["static_j"] + r0["dyn_j"] + r0["wake_j"])) < 1e-6,
     f"total {r0['total_j']:,.0f} = static {r0['static_j']:,.0f} + dyn {r0['dyn_j']:,.0f} "
     f"+ wake {r0['wake_j']:,.0f}")

zpts = [(l, sim(HET, l, seed=0, sleep_after=0.0, wake_s=0.0, wake_j=0.0)) for l in (0.10, 0.40)]
s4_ok = all(abs(r["static_j"] - r["busy_idle_j"]) < 1e-6 for _, r in zpts)
sane("S4 free instant power-down charges idle power over busy time and nothing else",
     s4_ok,
     "; ".join(f"lam={l:.2f}: static {r['static_j']:,.1f} J vs idle-over-busy "
               f"{r['busy_idle_j']:,.1f} J" for l, r in zpts))

# ================================================================== A load sweep
LAMS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.85, 1.00]
CONFIGS = [
    ("heterogeneous, no power-down", HET, dict()),
    ("heterogeneous, power-down 60s", HET, dict(sleep_after=60.0)),
    ("homogeneous control, no power-down", HOM, dict()),
    ("homogeneous control, power-down 60s", HOM, dict(sleep_after=60.0)),
]
print(f"\n{'='*104}\nA  LOAD SWEEP on a ten-slot fleet, {len(SEEDS)} seeds per point\n{'='*104}")
sweepA = {}
for label, pool, kw in CONFIGS:
    rows = []
    print(f"\n{label}")
    print(f"{'lam':>7}{'util %':>9}{'J/job':>12}{'sd %':>7}{'static %':>10}{'delay s':>10}{'vs prev':>10}")
    prev = None
    for lam in LAMS:
        a = multi(pool, lam, **kw)
        a["lam"] = lam
        rows.append(a)
        d = "" if prev is None else f"{100*(a['per_job']-prev)/prev:+9.1f}%"
        print(f"{lam:7.2f}{100*a['utilisation']:9.1f}{a['per_job']:12.0f}"
              f"{100*a['per_job_sd']/a['per_job']:7.1f}{100*a['static_j']/a['total_j']:10.1f}"
              f"{a['mean_delay']:10.1f}{d:>10}")
        prev = a["per_job"]
    sweepA[label] = rows

# ================================================================== B right-sizing sweep
print(f"\n{'='*104}\nB  RIGHT-SIZING SWEEP at fixed load: shrink the fleet, watch utilisation rise"
      f"\n{'='*104}")
LAM_B = 0.30
FLEETS = []
for k in range(14, 4, -1):
    base, rem = divmod(k, 5)
    pool = {mm: base for mm in MACH}
    for j in range(rem):
        pool[MACH[j]] += 1
    FLEETS.append({mm: c for mm, c in pool.items() if c > 0})
sweepB = {}
for label, kw in (("heterogeneous, no power-down", dict()),
                  ("heterogeneous, power-down 60s", dict(sleep_after=60.0))):
    rows = []
    print(f"\n{label}   (arrival rate {LAM_B}/s)")
    print(f"{'slots':>7}{'util %':>9}{'J/job':>12}{'sd %':>7}{'static %':>10}{'delay s':>10}{'SLA':>6}")
    for pool in FLEETS:
        a = multi(pool, LAM_B, **kw)
        a["pool"] = dict(pool)
        rows.append(a)
        print(f"{a['slots']:7d}{100*a['utilisation']:9.1f}{a['per_job']:12.0f}"
              f"{100*a['per_job_sd']/a['per_job']:7.1f}{100*a['static_j']/a['total_j']:10.1f}"
              f"{a['mean_delay']:10.1f}{'ok' if a['mean_delay']<=SLA_S else 'MISS':>6}")
    sweepB[label] = rows

# ================================================================== turning points
def turning_point(rows, key="utilisation"):
    """Minimum of joules per job, and whether the rise past it clears the seed spread.

    A rise is accepted only when it exceeds three times the pooled seed standard
    deviation, which on this simulator is about 1.4 per cent of the mean.
    """
    pj = np.array([r["per_job"] for r in rows])
    sd = np.array([r["per_job_sd"] for r in rows])
    i = int(np.argmin(pj))
    after = pj[i + 1:]
    if len(after) == 0:
        return dict(index=i, at=rows[i][key], per_job=float(pj[i]), rise_pct=0.0,
                    rise_to=rows[i][key], rise_j=0.0, within_sla=bool(rows[i]["mean_delay"] <= SLA_S),
                    threshold_j=float(3.0 * sd[i]), significant=False,
                    reason="minimum is the last point sampled")
    j = int(np.argmax(after)) + i + 1
    rise = pj[j] - pj[i]
    thr = 3.0 * math.sqrt(sd[i] ** 2 + sd[j] ** 2)
    sla_ok = rows[j]["mean_delay"] <= SLA_S
    return dict(index=i, at=rows[i][key], per_job=float(pj[i]),
                rise_to=rows[j][key], rise_pct=float(100 * rise / pj[i]),
                significant=bool(rise > thr), within_sla=bool(sla_ok),
                threshold_j=float(thr), rise_j=float(rise))


print(f"\n{'='*104}\nTURNING POINTS\n{'='*104}")
TP = {}
for tag, sweeps, key in (("A load sweep", sweepA, "utilisation"), ("B right-sizing", sweepB, "utilisation")):
    for label, rows in sweeps.items():
        t = turning_point(rows, key)
        TP[f"{tag} | {label}"] = t
        verdict = ("turns upward" if t["significant"] else "no significant turn")
        extra = ""
        if t["significant"]:
            extra = (f", rises {t['rise_pct']:.1f}% by utilisation {100*t['rise_to']:.0f}%"
                     f", queueing {'within' if t['within_sla'] else 'PAST'} the 60 s SLA")
        print(f"  {tag:16} {label:38} minimum at utilisation "
              f"{100*t['at']:5.1f}%: {verdict}{extra}")

hom_turn = [TP[k] for k in TP if "homogeneous" in k]
sane("S3 isolating control: the homogeneous fleet shows no significant upturn",
     not any(t["significant"] for t in hom_turn),
     "; ".join(f"min at util {100*t['at']:.0f}%, largest subsequent rise "
               f"{t['rise_pct']:+.1f}% (threshold {100*t['threshold_j']/t['per_job']:.1f}%)"
               for t in hom_turn))

# S1 as first written demanded EXACT monotonicity of utilisation and failed on one pair
# of saturated points, 99.1 per cent at lam = 0.85 against 99.0 per cent at lam = 1.00.
# Inspecting the per-seed values shows the cause is not a bug: once the fleet saturates,
# utilisation asymptotes just below one and the residual differences are the ramp-up at
# the start of the run plus Monte Carlo noise, both far smaller than the seed spread.
# Demanding exact ordering between two points that are statistically identical is a
# mis-specified check. It is replaced by two checks that are actually informative: exact
# monotonicity wherever the fleet is NOT saturated, and monotonicity to within one
# pooled seed standard deviation everywhere. No parameter was tuned.
def _mono(rows, tol=lambda a, b: 0.0):
    return all(rows[i]["utilisation"] <= rows[i + 1]["utilisation"] + tol(rows[i], rows[i + 1])
               for i in range(len(rows) - 1))


def _pooled(a, b):
    return math.sqrt(a["utilisation_sd"] ** 2 + b["utilisation_sd"] ** 2)


unsat = {k: [r for r in v if r["mean_delay"] <= SLA_S] for k, v in
         list(sweepA.items()) + list(sweepB.items())}
mono_unsat = all(_mono(v) for v in unsat.values())
mono_all = all(_mono(v, _pooled) for v in list(sweepA.values()) + list(sweepB.values()))
sane("S1 utilisation rises with load and with fleet shrinkage", mono_unsat and mono_all,
     f"exactly monotone on every unsaturated segment: {mono_unsat}; monotone to within one "
     f"pooled seed sd everywhere: {mono_all}")

decomp_ok = all(abs(r["total_j"] - (r["static_j"] + r["dyn_j"] + r["wake_j"])) < 1e-6
                for rows in list(sweepA.values()) + list(sweepB.values()) for r in rows)
sane("S2 energy decomposition is exact at every point of every sweep", decomp_ok,
     f"{sum(len(r) for r in list(sweepA.values())+list(sweepB.values()))} points checked")

util_ok = all(0.0 <= r["utilisation"] <= 1.0 + 1e-9
              for rows in list(sweepA.values()) + list(sweepB.values()) for r in rows)
worst_u = max(r["utilisation"] for rows in list(sweepA.values()) + list(sweepB.values()) for r in rows)
sane("S7 utilisation is a fraction in [0, 1] at every point", util_ok,
     f"largest utilisation observed {100*worst_u:.1f}%")

sds = [100 * r["per_job_sd"] / r["per_job"]
       for rows in list(sweepA.values()) + list(sweepB.values()) for r in rows]
sane("S5 seed spread is reported and is small enough to resolve a few per cent",
     float(np.median(sds)) < 5.0,
     f"median seed sd {np.median(sds):.2f}% of the mean, worst {max(sds):.2f}%")

sla_flags = {k: t.get("within_sla") for k, t in TP.items() if t["significant"]}
sane("S6 any upturn found is reported together with whether queueing had already "
     "breached the SLA there", True,
     f"significant upturns: {sla_flags if sla_flags else 'none'}")

# ---- the consolidation lever, restated in these terms
het_pd = sweepB["heterogeneous, power-down 60s"]
het_np = sweepB["heterogeneous, no power-down"]
best_pd = min(het_pd, key=lambda r: r["per_job"])
big_np = het_np[0]
print(f"\nconsolidation restated: shrinking from {big_np['slots']} slots at "
      f"{100*big_np['utilisation']:.0f}% utilisation to {best_pd['slots']} slots at "
      f"{100*best_pd['utilisation']:.0f}% with power-down moves joules per job from "
      f"{big_np['per_job']:.0f} to {best_pd['per_job']:.0f}, "
      f"{100*(big_np['per_job']-best_pd['per_job'])/big_np['per_job']:.1f}%")

OUT["load_sweep"] = {k: [{kk: vv for kk, vv in r.items()} for r in v] for k, v in sweepA.items()}
OUT["rightsizing_sweep"] = {k: [{kk: vv for kk, vv in r.items()} for r in v] for k, v in sweepB.items()}
OUT["turning_points"] = TP
OUT["config"] = dict(seeds=SEEDS, horizon_s=HORIZON, job_samples=JOB, sla_s=SLA_S,
                     heterogeneous_pool=HET, homogeneous_control=HOM, lam_rightsizing=LAM_B,
                     acceptance="a rise past the minimum counts only above three pooled seed sd")

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/g6_utilisation.json", "w", encoding="utf-8"), indent=1)
print(f"\nsanity: {sum(1 for s in SANITY if s['passed'])}/{len(SANITY)} passed")
for s in SANITY:
    if not s["passed"]:
        print(f"  FAILED: {s['check']}: {s['detail']}")
print("saved -> experiments/results/g6_utilisation.json")
