# -*- coding: utf-8 -*-
"""
G4.  Who pays for a heterogeneous fleet?  Shapley cost sharing over TENANTS.

Section 8 values accelerator TYPES with a coalitional game. The same machinery answers a
question a shared facility has to answer every month: several teams submit different
workload mixes to one heterogeneous fleet, one electricity bill arrives, and the bill has
to be split. Naive rules split it by jobs submitted or by machine-seconds occupied. Both
ignore that a tenant whose work is easy to place cheaply is subsidising a tenant whose
work is not, because the second tenant is the reason the expensive, fast accelerators are
in the fleet at all.

The game
--------
Players are tenants. Each tenant t has a workload mix w_t over the 24 measured cells and
an arrival rate lam_t. For a coalition S, the STAND-ALONE COST c(S) is the least facility
energy, in joules over a one-hour horizon, of a fleet that serves exactly the combined
stream of S. The fleet is right-sized for the coalition, so c(empty) = 0 exactly and the
game is a proper cost game with no arbitrary residual to allocate. This is the classical
cost-allocation setup: what would this group of tenants pay if it ran its own facility?

Fleets are evaluated with the fluid linear-programming recourse model of B8, at the
utilisation ceiling RHO = 0.94 that B8 calibrated against the discrete-event simulator,
with the same off-fleet penalty price PENALTY = 5.0 making the recourse relatively
complete. B8 is imported by value, not by reference: the constants are restated here and
the LP is written out again so that this script does not depend on, or modify, B8.

Naive billing rules compared against the Shapley split
------------------------------------------------------
  per-job         bill split in proportion to jobs submitted
  per-second      bill split in proportion to accelerator-seconds occupied
  metered         each tenant charged the actual energy of its own jobs, with the
                  facility's unoccupied idle energy split by occupancy share

Sanity checks stated in advance
--------------------------------
  S0  On the grand coalition, the local-search fleet optimiser must find the SAME fleet
      and cost as exhaustive enumeration of every composition at every size in range.
      A search bug otherwise contaminates every c(S).
  S1  EFFICIENCY. The Shapley shares must sum to c(N) to floating-point precision.
  S2  SYMMETRY. In a control game where one tenant is split into two identical halves,
      the two halves must receive identical shares.
  S3  MONOTONICITY. c(S) <= c(T) whenever S is a subset of T. Adding a tenant cannot
      make the cheapest facility that serves everybody cheaper.
  S4  A singleton coalition's Shapley share in the one-player game equals its
      stand-alone cost.
  S5  SUBADDITIVITY. c(S union T) <= c(S) + c(T) for disjoint S and T, because the two
      groups can always simply run two separate facilities. A violation is a failure of
      the fleet search, not a property of the world, and is treated as a bug.
  S6  The right-sized fleet for every coalition must be INTERIOR: not pinned to the hard
      size cap, and not improvable by adding one more low-idle slot. Otherwise the search
      was truncated and every c(S) it produced is an overestimate.

Two checks FAILED on the first run and both were real; what was wrong and why
-----------------------------------------------------------------------------
The first run failed S5 (168 subadditivity violations) and S6 (26 coalitions pinned to a
size boundary), and produced the impossible-looking headline that every tenant pays MORE
sharing the facility than running alone. A cost game in which merging is always worse is
a bug, because two groups can always simply operate two separate facilities, so
c(S union T) <= c(S) + c(T) is available by construction.

Root cause, found by inspecting the fleets rather than by guessing: the fleet-size scan
was a fixed window of six sizes above the capacity lower bound, written on the assumption
that the cheapest fleet is close to the smallest one that can carry the load. That is
false in this cost structure. An L4 slot idles at 29.7 W against 88.7 W for an L40S, so
buying additional L4 capacity to keep work off the large parts keeps paying long past the
point where the fleet has enough throughput, and the optimum sits far above the capacity
bound. Small coalitions were therefore truncated hardest, their c(S) overstated least,
and the merged coalition's cost overstated most, which inverted the whole comparison.

Two changes fix it, neither of them a tuned parameter. The size scan now grows until four
consecutive sizes fail to improve. And the candidate pool is closed under merger: for
every pair of disjoint coalitions, the sum of their two best fleets is added as a
candidate, which is literally "run two separate facilities", so subadditivity becomes
attainable by the search rather than something the search has to rediscover.
  S7  DUMMY. A control tenant with zero arrival rate must receive a share of exactly
      zero.
  C1  Efficiency again, in the capacity-constrained variant of the game.
  C2  Monotonicity again, in the capacity-constrained variant.
  NOT a pass/fail check: whether the Shapley split lies in the CORE, that is whether
  every coalition pays at most its stand-alone cost. This is reported as a finding,
  since the Shapley value of a general cost game need not be in the core.

Free: reuses the measured 24 x 5 grid. No compute cost.
"""
import io, json, sys, warnings
from itertools import combinations, product
from math import factorial
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, HORIZON = 20000, 3600.0
RHO, PENALTY = 0.94, 5.0            # calibrated in B8; restated, not imported
SANITY, OUT = [], {}


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


keys, Ylog, Tput = load_grid()
NK, NM = len(keys), len(MACH)
Ecell = np.array([[10 ** Ylog[i, j] / 1000.0 for j in range(NM)] for i in range(NK)])  # J/sample
Tcell = np.array([[JOB / Tput[i, j] for j in range(NM)] for i in range(NK)])           # s/job
IDLEv = np.array([IDLE[m] for m in MACH])


def mix(pred):
    w = np.array([1.0 if pred(k) else 0.0 for k in keys])
    return w / w.sum()


# ------------------------------------------------------------------ tenants
TENANTS = [
    ("LLM serving",          mix(lambda k: k[0] == "transformer" and k[1] == "fp16"), 0.15),
    ("vision inference",     mix(lambda k: k[0] != "transformer" and k[1] == "fp16"), 0.15),
    ("legacy fp32 research", mix(lambda k: k[1] == "fp32"),                           0.05),
    ("large-batch offline",  mix(lambda k: k[2] == 128),                              0.10),
    ("interactive small-batch", mix(lambda k: k[2] == 8),                             0.05),
]
NT = len(TENANTS)
TNAME = [t[0] for t in TENANTS]


def combined(members, tenants):
    """Aggregate mix and arrival rate of a coalition."""
    lam = sum(tenants[i][2] for i in members)
    if lam <= 0:
        return None, 0.0
    w = sum(tenants[i][2] * tenants[i][1] for i in members) / lam
    return w, lam


# ------------------------------------------------------------------ LP recourse (B8 form)
def recourse(n, w, lam):
    """Total facility joules over the horizon for fleet counts n serving mix w at rate lam.

    Returns (total joules, off-fleet share, per-machine occupied seconds, per-cell routing).
    """
    n = np.asarray(n, dtype=float)
    if n.sum() <= 0 or lam <= 0:
        return (0.0, 0.0, np.zeros(NM), None) if lam <= 0 else (float("inf"), 1.0, None, None)
    live = np.where(n > 0)[0]
    nv = len(live)
    nvar = NK * (nv + 1)
    c = np.zeros(nvar)
    for i in range(NK):
        for a, j in enumerate(live):
            c[i * (nv + 1) + a] = w[i] * (JOB * Ecell[i, j] - IDLEv[j] * Tcell[i, j])
        c[i * (nv + 1) + nv] = w[i] * PENALTY * JOB * Ecell[i].max()
    Aeq = np.zeros((NK, nvar)); beq = np.ones(NK)
    for i in range(NK):
        Aeq[i, i * (nv + 1):(i + 1) * (nv + 1)] = 1.0
    Aub = np.zeros((nv, nvar)); bub = np.zeros(nv)
    for a, j in enumerate(live):
        for i in range(NK):
            Aub[a, i * (nv + 1) + a] = lam * w[i] * Tcell[i, j]
        bub[a] = RHO * n[j]
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=(0, 1), method="highs")
    if not r.success:
        return float("inf"), 1.0, None, None
    x = r.x.reshape(NK, nv + 1)
    total = float(IDLEv @ n) * HORIZON + float(r.fun) * lam * HORIZON
    occ = np.zeros(NM)
    for a, j in enumerate(live):
        occ[j] = float(np.sum(lam * w * x[:, a] * Tcell[:, j])) * HORIZON
    xfull = np.zeros((NK, NM))
    for a, j in enumerate(live):
        xfull[:, j] = x[:, a]
    return total, float((x[:, nv] * w).sum()), occ, xfull


def cost_of(n, w, lam):
    tot, outs, occ, _ = recourse(n, w, lam)
    return tot if (np.isfinite(tot) and outs < 1e-9) else float("inf")


# ------------------------------------------------------------------ fleet search
def min_slots(w, lam):
    """Capacity lower bound: even the fastest device per cell needs this much occupancy."""
    return int(np.ceil(lam * float(np.sum(w * Tcell.min(axis=1))) / RHO))


def local_search(w, lam, slots, seed=0, starts=3):
    rng = np.random.default_rng(seed)
    bestn, bestv = None, float("inf")
    for st in range(starts):
        n = np.zeros(NM)
        if st == 0:
            n[int(np.argmin(Ecell.mean(0)))] = slots
        else:
            for _ in range(slots):
                n[rng.integers(NM)] += 1
        v = recourse(n, w, lam)[0]
        improved = True
        while improved:
            improved = False
            bm, bv = None, v
            for a in range(NM):
                if n[a] <= 0:
                    continue
                for b in range(NM):
                    if a == b:
                        continue
                    cand = n.copy(); cand[a] -= 1; cand[b] += 1
                    cv = recourse(cand, w, lam)[0]
                    if cv < bv - 1e-9:
                        bv, bm = cv, cand
            if bm is not None:
                v, n, improved = bv, bm, True
        if v < bestv:
            bestv, bestn = v, n
    return bestn, bestv


SIZE_CAP = 60          # hard stop; S6 verifies it is never reached
STALL = 4              # stop growing after this many consecutive sizes without improvement


def search_fleets(w, lam, seed=0):
    """Candidate fleets for this coalition, scanning fleet SIZE adaptively.

    The scan was first written as a fixed window of six sizes above the capacity lower
    bound, on the assumption that the cheapest fleet sits close to the smallest feasible
    one. The checks caught that this assumption is wrong here: an extra L4 slot costs
    about 107 kJ of idle over the hour but can move a whole cell's work off an L40S or
    an A100, so the optimum sits well above the capacity bound. The scan now grows until
    four consecutive sizes fail to improve, and S6 verifies that the fleet finally chosen
    is interior rather than pinned to a boundary.
    """
    if lam <= 0:
        return []
    lo = max(1, min_slots(w, lam))
    out, best, stall, slots = [], float("inf"), 0, lo
    while slots <= SIZE_CAP and stall < STALL:
        n, v = local_search(w, lam, slots, seed=seed)
        out.append(tuple(int(x) for x in n))
        if v < best - 1e-9:
            best, stall = v, 0
        else:
            stall += 1
        slots += 1
    return out


# ------------------------------------------------------------------ game construction
def _score_pool(coalitions, tenants, pool):
    cost, fleet = {}, {}
    for S in coalitions:
        w, lam = combined(S, tenants)
        if lam <= 0:
            cost[S], fleet[S] = 0.0, None
            continue
        best, bn = float("inf"), None
        for nn in pool:
            v = cost_of(np.array(nn, float), w, lam)
            if v < best:
                best, bn = v, nn
        cost[S], fleet[S] = best, bn
    return cost, fleet


def build_game(tenants, tag, pool_extra=(), search=True, rounds=3):
    """c(S) for every coalition over a shared candidate pool, so costs are mutually consistent.

    After a first scoring pass the pool is CLOSED UNDER MERGER: for every pair of
    disjoint coalitions, the sum of their two best fleets is added as a candidate. That
    composite is exactly "run two separate facilities side by side", so having it in the
    pool is what makes subadditivity attainable rather than accidental. Without it the
    search reported a merged facility costing more than two separate ones, which is not a
    property of the world but a hole in the candidate set; S5 caught precisely that.
    """
    nt = len(tenants)
    coalitions = [frozenset(S) for r in range(nt + 1) for S in combinations(range(nt), r)]
    pool = set(pool_extra)
    if search:
        for S in coalitions:
            w, lam = combined(S, tenants)
            if lam > 0:
                pool.update(search_fleets(w, lam))
    cost, fleet = _score_pool(coalitions, tenants, sorted(pool))
    for _ in range(rounds):
        add = set()
        for S in coalitions:
            for T in coalitions:
                if S and T and not (S & T) and fleet[S] and fleet[T]:
                    merged = tuple(a + b for a, b in zip(fleet[S], fleet[T]))
                    if merged not in pool:
                        add.add(merged)
        if not add:
            break
        pool |= add
        cost, fleet = _score_pool(coalitions, tenants, sorted(pool))
    print(f"  [{tag}] {len(coalitions)} coalitions over a shared pool of {len(pool)} fleets")
    return coalitions, cost, fleet, pool


def shapley(cost, nt):
    phi = np.zeros(nt)
    for i in range(nt):
        others = [x for x in range(nt) if x != i]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                Sf = frozenset(S)
                wgt = factorial(len(S)) * factorial(nt - len(S) - 1) / factorial(nt)
                phi[i] += wgt * (cost[Sf | {i}] - cost[Sf])
    return phi


# ================================================================== main game
print("building the tenant cost game")
coals, COST, FLEET, POOL = build_game(TENANTS, "main")
PHI = shapley(COST, NT)
cN = COST[frozenset(range(NT))]

print("\nsanity checks stated in advance:")

# ---- S0 exhaustive validation of the fleet search on the grand coalition
wN, lamN = combined(range(NT), TENANTS)
nN0 = FLEET[frozenset(range(NT))]
ENUM_LO, ENUM_HI = max(1, sum(nN0) - 3), sum(nN0) + 3
enum_best, enum_n = float("inf"), None
for slots in range(ENUM_LO, ENUM_HI + 1):
    for c in product(range(slots + 1), repeat=NM):
        if sum(c) != slots:
            continue
        v = cost_of(np.array(c, float), wN, lamN)
        if v < enum_best:
            enum_best, enum_n = v, c
sane("S0 local search reproduces exhaustive enumeration on the grand coalition",
     enum_n == nN0 and abs(enum_best - cN) < 1e-6,
     f"exhaustive enumeration over sizes {ENUM_LO}-{ENUM_HI} gives {enum_n} at "
     f"{enum_best:,.0f} J; the search gives {nN0} at {cN:,.0f} J")

sane("S1 efficiency: Shapley shares sum to the total bill",
     abs(PHI.sum() - cN) < 1e-6, f"sum {PHI.sum():,.3f} J vs c(N) {cN:,.3f} J")

viol = [(S, T) for S in coals for T in coals if S < T and COST[S] > COST[T] + 1e-6]
sane("S3 monotonicity: no coalition costs more than a superset of it",
     not viol, f"{len(viol)} violations out of {sum(1 for S in coals for T in coals if S < T)} subset pairs")

sub_viol = []
for S in coals:
    for T in coals:
        if S and T and not (S & T) and COST[S | T] > COST[S] + COST[T] + 1e-6:
            sub_viol.append((sorted(S), sorted(T), COST[S | T] - COST[S] - COST[T]))
worst_sub = max((abs(v[2]) for v in sub_viol), default=0.0)
sane("S5 subadditivity: a merged facility never costs more than two separate ones",
     not sub_viol, f"{len(sub_viol)} violations, worst excess {worst_sub:,.0f} J")

CHEAP = int(np.argmin([IDLE[m] for m in MACH]))     # lowest-idle type, the one worth adding
bnd = []
for S in coals:
    if not S:
        continue
    sz = sum(FLEET[S])
    plus = np.array(FLEET[S], float); plus[CHEAP] += 1
    w_, l_ = combined(S, TENANTS)
    if sz >= SIZE_CAP:
        bnd.append((sorted(TNAME[i] for i in S), sz, "pinned at the hard size cap"))
    elif cost_of(plus, w_, l_) < COST[S] - 1e-6:
        bnd.append((sorted(TNAME[i] for i in S), sz, "one more low-idle slot is still cheaper"))
sane("S6 every right-sized fleet is interior: not at the size cap and not improvable by "
     "one more slot", not bnd,
     f"{len(bnd)} of {len(coals)-1} coalitions at a boundary" +
     ("" if not bnd else f"; e.g. {bnd[0]}") +
     f"; the grand-coalition fleet holds {sum(FLEET[frozenset(range(NT))])} slots against a cap of {SIZE_CAP}")

for i in range(NT):
    one = {frozenset(): 0.0, frozenset({0}): COST[frozenset({i})]}
    p1 = shapley(one, 1)
    if abs(p1[0] - COST[frozenset({i})]) > 1e-9:
        sane("S4 singleton game returns the stand-alone cost", False, f"{TNAME[i]}")
        break
else:
    sane("S4 singleton game returns the stand-alone cost", True,
         "all five singletons match their stand-alone cost exactly")

# ---- S2 symmetry control: split tenant 0 into two identical halves
name0, w0, l0 = TENANTS[0]
CTRL = [(name0 + " (half A)", w0, l0 / 2), (name0 + " (half B)", w0, l0 / 2)] + TENANTS[1:]
_, Cc, _, _ = build_game(CTRL, "symmetry control", pool_extra=POOL, search=False)
PHIc = shapley(Cc, len(CTRL))
sane("S2 symmetry: two identical half-tenants receive identical shares",
     abs(PHIc[0] - PHIc[1]) < 1e-6,
     f"{PHIc[0]:,.1f} J vs {PHIc[1]:,.1f} J (difference {PHIc[0]-PHIc[1]:.2e})")

# ---- S7 dummy control: a tenant with zero load
DUM = TENANTS + [("zero-load tenant", TENANTS[0][1], 0.0)]
_, Cd, _, _ = build_game(DUM, "dummy control", pool_extra=POOL, search=False)
PHId = shapley(Cd, len(DUM))
sane("S7 dummy: a zero-load tenant is charged exactly nothing",
     abs(PHId[-1]) < 1e-6, f"share {PHId[-1]:.3e} J")

# ================================================================== naive billing
wN, lamN = combined(range(NT), TENANTS)
nN = np.array(FLEET[frozenset(range(NT))], float)
totN, outsN, occN, xN = recourse(nN, wN, lamN)

jobs = np.array([TENANTS[i][2] * HORIZON for i in range(NT)])
occ_t = np.array([float(np.sum(TENANTS[i][2] * TENANTS[i][1][:, None] * xN * Tcell)) * HORIZON
                  for i in range(NT)])
job_energy = np.array([float(np.sum(TENANTS[i][2] * TENANTS[i][1][:, None] * xN * JOB * Ecell)) * HORIZON
                       for i in range(NT)])
idle_unocc = float(IDLEv @ nN) * HORIZON - float(IDLEv @ occN)

BILL = {
    "per-job": cN * jobs / jobs.sum(),
    "per-second (occupancy)": cN * occ_t / occ_t.sum(),
    "metered energy + idle by occupancy": job_energy + idle_unocc * occ_t / occ_t.sum(),
    "Shapley": PHI,
}
for k, v in BILL.items():
    assert abs(v.sum() - cN) < 1e-3 * cN, (k, v.sum(), cN)

print(f"\nfleet serving all tenants: "
      f"{'+'.join(f'{int(v)}x{MACH[i]}' for i, v in enumerate(nN) if v > 0)}"
      f"   total bill {cN/3.6e6:.2f} kWh ({cN:,.0f} J/hour)")
print(f"{'tenant':26}{'rate':>7}{'jobs %':>9}{'sec %':>8}" +
      "".join(f"{k[:14]:>16}" for k in BILL))
for i in range(NT):
    print(f"{TNAME[i]:26}{TENANTS[i][2]:7.2f}{100*jobs[i]/jobs.sum():9.1f}"
          f"{100*occ_t[i]/occ_t.sum():8.1f}" +
          "".join(f"{100*BILL[k][i]/cN:16.1f}" for k in BILL))
print(f"{'(percent of the bill)':26}")

print(f"\ndeparture of each naive rule from the Shapley-fair split, in percent of that "
      f"tenant's fair share")
print(f"{'tenant':26}{'stand-alone':>13}{'Shapley':>10}" +
      "".join(f"{k[:15]:>17}" for k in list(BILL)[:-1]))
DEV = {}
for i in range(NT):
    sa = COST[frozenset({i})]
    line = f"{TNAME[i]:26}{sa/3.6e6:13.2f}{PHI[i]/3.6e6:10.2f}"
    DEV[TNAME[i]] = {}
    for k in list(BILL)[:-1]:
        d = 100 * (BILL[k][i] - PHI[i]) / PHI[i]
        DEV[TNAME[i]][k] = d
        line += f"{d:+16.1f}%"
    print(line)
print("  (stand-alone and Shapley in kWh/hour; the rest are relative departures)")

worst = max((abs(DEV[t][k]), t, k) for t in DEV for k in DEV[t])
print(f"\nlargest single mis-charge: {worst[1]} under {worst[2]} billing, {worst[0]:.1f}% "
      f"away from its Shapley share")

# ---- core test (reported, not asserted)
core_viol = []
for S in coals:
    if not S or len(S) == NT:
        continue
    excess = float(PHI[list(S)].sum()) - COST[S]
    if excess > 1e-6:
        core_viol.append((sorted(TNAME[i] for i in S), excess))
in_core = not core_viol
print(f"\ncore test: the Shapley split is {'IN' if in_core else 'NOT IN'} the core "
      f"({len(core_viol)} of {len(coals)-2} proper coalitions pay more than standing alone)")
for S, e in sorted(core_viol, key=lambda x: -x[1])[:4]:
    print(f"    {', '.join(S)} pays {e/3.6e6:.3f} kWh/hour more than its stand-alone cost")

# ---- individual rationality
ir = [(TNAME[i], PHI[i], COST[frozenset({i})]) for i in range(NT)]
print(f"\nindividual rationality: every tenant's Shapley share versus running alone")
for nm, p, sa in ir:
    print(f"    {nm:26}{p/3.6e6:8.3f} kWh vs {sa/3.6e6:8.3f} standing alone "
          f"({100*(sa-p)/sa:+5.1f}% cheaper through sharing)")

# ================================================================== capacity-constrained variant
# The unconstrained game above lets each coalition buy as many slots as it likes, and the
# answer is that it buys one type: the cheapest facility for any mix is a large number of
# L4s. That is correct for the question as posed and it dissolves the phenomenon the paper
# is about, because Section 8's heterogeneity result lives in a REGIME, not in the parts:
# it appears when the slot count is capped and capacity binds, so a buyer must trade
# throughput against efficiency. The variant below restores that regime. Each coalition is
# entitled to rack space in proportion to its load, ceil(10 * lam_S / lam_N) slots, so the
# grand coalition holds exactly the ten slots of Section 8, and c(empty) is still zero.
# Where that entitlement admits no feasible fleet the budget is raised one slot at a time
# and the coalition is recorded, so the relaxation is visible rather than silent.
SLOTS_N = 10
print(f"\n{'='*100}\nCAPACITY-CONSTRAINED VARIANT: {SLOTS_N} slots for the grand coalition, "
      f"pro rata for every other\n{'='*100}")


def comps(k):
    return [c for c in product(range(k + 1), repeat=NM) if sum(c) == k]


COST_C, FLEET_C, BUDGET_C, RELAXED = {}, {}, {}, []
for S in coals:
    w_, l_ = combined(S, TENANTS)
    if l_ <= 0:
        COST_C[S], FLEET_C[S], BUDGET_C[S] = 0.0, None, 0
        continue
    k = max(1, int(np.ceil(SLOTS_N * l_ / lamN)))
    k0 = k
    while k <= SLOTS_N * 3:
        best, bn = float("inf"), None
        for c in comps(k):
            v = cost_of(np.array(c, float), w_, l_)
            if v < best:
                best, bn = v, c
        if bn is not None:
            break
        k += 1
    if k > k0:
        RELAXED.append((sorted(TNAME[i] for i in S), k0, k))
    COST_C[S], FLEET_C[S], BUDGET_C[S] = best, bn, k

PHI_C = shapley(COST_C, NT)
cN_C = COST_C[frozenset(range(NT))]
nC = np.array(FLEET_C[frozenset(range(NT))], float)
sane("C1 efficiency holds in the capacity-constrained game",
     abs(PHI_C.sum() - cN_C) < 1e-6, f"sum {PHI_C.sum():,.3f} J vs c(N) {cN_C:,.3f} J")
viol_c = [(S, T) for S in coals for T in coals if S < T and COST_C[S] > COST_C[T] + 1e-6]
sane("C2 monotonicity holds in the capacity-constrained game", not viol_c,
     f"{len(viol_c)} violations")
print(f"  budget relaxed above the pro-rata entitlement for {len(RELAXED)} coalitions"
      + (f"; e.g. {RELAXED[0]}" if RELAXED else ""))

totC, outsC, occC, xC = recourse(nC, wN, lamN)
occ_tC = np.array([float(np.sum(TENANTS[i][2] * TENANTS[i][1][:, None] * xC * Tcell)) * HORIZON
                   for i in range(NT)])
job_eC = np.array([float(np.sum(TENANTS[i][2] * TENANTS[i][1][:, None] * xC * JOB * Ecell)) * HORIZON
                   for i in range(NT)])
idle_uC = float(IDLEv @ nC) * HORIZON - float(IDLEv @ occC)
BILL_C = {
    "per-job": cN_C * jobs / jobs.sum(),
    "per-second (occupancy)": cN_C * occ_tC / occ_tC.sum(),
    "metered energy + idle by occupancy": job_eC + idle_uC * occ_tC / occ_tC.sum(),
    "Shapley": PHI_C,
}
print(f"\nconstrained fleet: "
      f"{'+'.join(f'{int(v)}x{MACH[i]}' for i, v in enumerate(nC) if v > 0)}"
      f"   ({len([1 for v in nC if v > 0])} distinct types)   "
      f"total bill {cN_C/3.6e6:.2f} kWh")
print(f"{'tenant':26}{'stand-alone':>13}{'Shapley':>10}" +
      "".join(f"{k[:15]:>17}" for k in list(BILL_C)[:-1]))
DEV_C = {}
for i in range(NT):
    DEV_C[TNAME[i]] = {k: float(100 * (BILL_C[k][i] - PHI_C[i]) / PHI_C[i]) for k in list(BILL_C)[:-1]}
    print(f"{TNAME[i]:26}{COST_C[frozenset({i})]/3.6e6:13.2f}{PHI_C[i]/3.6e6:10.2f}" +
          "".join(f"{DEV_C[TNAME[i]][k]:+16.1f}%" for k in list(BILL_C)[:-1]))
worst_c = max((abs(DEV_C[t][k]), t, k) for t in DEV_C for k in DEV_C[t])
print(f"\nlargest single mis-charge under capacity constraint: {worst_c[1]} under "
      f"{worst_c[2]} billing, {worst_c[0]:.1f}% away from its Shapley share")
core_c = [(sorted(TNAME[i] for i in S), float(PHI_C[list(S)].sum()) - COST_C[S])
          for S in coals if S and len(S) < NT and float(PHI_C[list(S)].sum()) - COST_C[S] > 1e-6]
print(f"core test, constrained: the Shapley split is {'IN' if not core_c else 'NOT IN'} the core "
      f"({len(core_c)} of {len(coals)-2} proper coalitions pay above stand-alone)")
for nm_ in range(NT):
    sa_ = COST_C[frozenset({nm_})]
    print(f"    {TNAME[nm_]:26}{PHI_C[nm_]/3.6e6:8.3f} kWh vs {sa_/3.6e6:8.3f} standing alone "
          f"({100*(sa_-PHI_C[nm_])/sa_:+5.1f}% cheaper through sharing)")

OUT["capacity_constrained"] = dict(
    slots_grand=SLOTS_N,
    budgets={"|".join(sorted(TNAME[i] for i in S)) or "(empty)": BUDGET_C[S] for S in coals},
    relaxed=[{"coalition": a, "pro_rata": b, "used": c} for a, b, c in RELAXED],
    coalition_costs_j={"|".join(sorted(TNAME[i] for i in S)) or "(empty)": COST_C[S] for S in coals},
    coalition_fleets={"|".join(sorted(TNAME[i] for i in S)) or "(empty)":
                      (None if FLEET_C[S] is None else
                       {MACH[j]: int(FLEET_C[S][j]) for j in range(NM) if FLEET_C[S][j] > 0})
                      for S in coals},
    grand_fleet={MACH[j]: int(nC[j]) for j in range(NM) if nC[j] > 0},
    total_bill_j=cN_C, total_bill_kwh=cN_C / 3.6e6,
    bills={k: {TNAME[i]: float(v[i]) for i in range(NT)} for k, v in BILL_C.items()},
    departure_from_shapley_pct=DEV_C,
    stand_alone_cost_j={TNAME[i]: COST_C[frozenset({i})] for i in range(NT)},
    core=dict(in_core=not core_c, violations=[{"coalition": a, "excess_j": b} for a, b in core_c]))

OUT["tenants"] = [dict(name=TNAME[i], rate=TENANTS[i][2],
                       mix_nonzero_cells=[list(keys[j]) for j in np.nonzero(TENANTS[i][1])[0].tolist()])
                  for i in range(NT)]
OUT["coalition_costs_j"] = {"|".join(sorted(TNAME[i] for i in S)) or "(empty)": COST[S] for S in coals}
OUT["coalition_fleets"] = {"|".join(sorted(TNAME[i] for i in S)) or "(empty)":
                           (None if FLEET[S] is None else
                            {MACH[j]: int(FLEET[S][j]) for j in range(NM) if FLEET[S][j] > 0})
                           for S in coals}
OUT["grand_fleet"] = {MACH[j]: int(nN[j]) for j in range(NM) if nN[j] > 0}
OUT["total_bill_j"] = cN
OUT["total_bill_kwh"] = cN / 3.6e6
OUT["bills"] = {k: {TNAME[i]: float(v[i]) for i in range(NT)} for k, v in BILL.items()}
OUT["bill_shares_pct"] = {k: {TNAME[i]: float(100 * v[i] / cN) for i in range(NT)} for k, v in BILL.items()}
OUT["departure_from_shapley_pct"] = DEV
OUT["stand_alone_cost_j"] = {TNAME[i]: COST[frozenset({i})] for i in range(NT)}
OUT["occupancy_seconds"] = {TNAME[i]: float(occ_t[i]) for i in range(NT)}
OUT["jobs"] = {TNAME[i]: float(jobs[i]) for i in range(NT)}
OUT["core"] = dict(in_core=in_core,
                   violations=[{"coalition": S, "excess_j": e} for S, e in core_viol])
OUT["config"] = dict(rho=RHO, penalty=PENALTY, horizon_s=HORIZON, job_samples=JOB,
                     size_cap=SIZE_CAP, size_scan_stall=STALL,
                     evaluator="fluid LP recourse, B8 form")

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/g4_cost_sharing.json", "w", encoding="utf-8"), indent=1)
print(f"\nsanity: {sum(1 for s in SANITY if s['passed'])}/{len(SANITY)} passed")
for s in SANITY:
    if not s["passed"]:
        print(f"  FAILED: {s['check']}: {s['detail']}")
print("saved -> experiments/results/g4_cost_sharing.json")
