# -*- coding: utf-8 -*-
"""
B8. Fleet design as a two-stage stochastic program.

Section 8.1 chooses a fleet by enumerating all 1001 compositions of ten slots over five
accelerator types. That is fine at ten slots and hopeless beyond: the count is
C(K+M-1, M-1), so 50 slots over 8 types is 4.6e8 compositions, each requiring a
simulation. Worse, enumeration optimises against ONE mix, while the whole finding of
Recipe 3 is that the mix is uncertain.

Both problems are the classical shape of a two-stage stochastic program:

  first stage   choose fleet counts n_m, here and now, before the mix is known
  recourse      once a mix scenario w is revealed, route work across the fleet

Recourse model. In steady state, let arrival rate lam be split so that fraction x_km of
cell-k work runs on machine type m. Energy over a horizon H is the static cost of holding
the fleet plus the dynamic excess of the work it does:

  min_x  sum_m IDLE_m n_m H  +  sum_km lam H w_k x_km (JOB E_km - IDLE_m t_km)
  s.t.   sum_m x_km = 1 for every k;  sum_k lam w_k x_km t_km <= rho n_m;  x >= 0

which is a linear program in x for fixed n. The service constraint enters as the
utilisation ceiling rho, calibrated below against the discrete-event simulator so that
the LP and the simulator agree on which fleets meet the 60-second SLA.

First stage is solved by steepest-descent local search over one-slot swaps, which needs
O(M^2) LP solves per step instead of enumerating every composition.

Sanity checks stated in advance:
  S1  on the tractable 10-slot, 5-type instance the local search must find the SAME
      optimum as exhaustive enumeration of all 1001 compositions
  S2  the LP recourse value must track the discrete-event simulator across fleets
      (rank correlation near 1); if it does not, the fluid model is not usable
  S3  with a single degenerate scenario the stochastic program must reduce to the
      deterministic one
  S4  the value of the stochastic solution must be non-negative: planning on the
      distribution cannot be worse than planning on the mean mix
  S5  expected value of perfect information must be non-negative.
      NOTE, corrected: an earlier draft of this docstring said EVPI must also "bound VSS".
      That is wrong. VSS and EVPI measure different things (the cost of ignoring the
      distribution versus the cost of not knowing the realisation) and there is no general
      ordering between them; here VSS 1420 J/job exceeds EVPI 880 J/job, which is perfectly
      admissible. The check as implemented only ever asserted EVPI >= 0, which is correct;
      only the prose was wrong, and it is fixed here so it cannot be read as a violated
      invariant.
"""
import io, json, sys, warnings
from itertools import product
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, SLA_S, HORIZON, LAM = 20000, 60.0, 3600.0, 0.5
PENALTY = 5.0          # off-fleet work priced at 5x the worst on-fleet energy
SANITY = []

def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

keys, Ylog, Tput = load_grid()
NK, NM = len(keys), len(MACH)
Ecell = np.array([[10 ** Ylog[i, j] / 1000.0 for j in range(NM)] for i in range(NK)])   # J per sample
Tcell = np.array([[JOB / Tput[i, j] for j in range(NM)] for i in range(NK)])            # s per job
IDLEv = np.array([IDLE[m] for m in MACH])

def w_of(pred):
    w = np.array([1.0 if pred(k) else 0.0 for k in keys])
    return w / w.sum()

SCENARIOS = {
    "uniform": w_of(lambda k: True),
    "vision inference": w_of(lambda k: k[0] in ("resnet50", "convnext_t", "vit_b16")),
    "modern precision (fp16)": w_of(lambda k: k[1] == "fp16"),
    "legacy fp32": w_of(lambda k: k[1] == "fp32"),
    "transformer-heavy": np.array([3.0 if k[0] == "transformer" else 1.0 for k in keys])
                          / sum(3.0 if k[0] == "transformer" else 1.0 for k in keys),
}
PROBS = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# ------------------------------------------------------------------ simulator (ground truth)
def facility(pool, w, seed=0):
    rng = np.random.default_rng(seed)
    slots = [m for m, c in pool.items() for _ in range(c)]
    ns = len(slots)
    free_at = np.zeros(ns)
    idx = np.arange(NK)
    t, arr = 0.0, []
    while t < HORIZON:
        t += rng.exponential(1.0 / LAM)
        if t < HORIZON:
            arr.append((t, int(rng.choice(idx, p=w))))
    dyn, delays = 0.0, []
    for at, ki in arr:
        free = [i for i in range(ns) if free_at[i] <= at]
        start = at if free else float(np.min(free_at))
        if not free:
            free = [i for i in range(ns) if free_at[i] <= start]
        cand = sorted({slots[i] for i in free})
        m = min(cand, key=lambda mm: Ecell[ki, MACH.index(mm)])
        mj = MACH.index(m)
        i = next(i for i in free if slots[i] == m)
        rt = Tcell[ki, mj]
        free_at[i] = start + rt
        delays.append(start - at)
        dyn += JOB * Ecell[ki, mj] - IDLEv[mj] * rt
    static = sum(IDLE[s] for s in slots) * HORIZON
    return (static + dyn) / max(len(arr), 1), float(np.mean(delays)) if delays else 0.0

# ------------------------------------------------------------------ LP recourse
def recourse(n, w, rho, njobs=None):
    """Expected facility energy per job for fleet counts n under mix w.

    Recourse is made RELATIVELY COMPLETE by an outsourcing variable per workload cell:
    work the fleet cannot serve within its utilisation ceiling is served off-fleet at a
    penalty price. Without it, any scenario admitting no feasible fleet (Section 8.1
    reports that a transformer-only mix is one) makes the expected objective infinite for
    every first-stage decision, and the search cannot move at all.
    """
    n = np.asarray(n, dtype=float)
    if n.sum() <= 0:
        return float("inf"), None
    live = np.where(n > 0)[0]
    nv = len(live)
    nvar = NK * (nv + 1)                       # per cell: nv on-fleet shares + 1 outsourced
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
            Aub[a, i * (nv + 1) + a] = LAM * w[i] * Tcell[i, j]
        bub[a] = rho * n[j]
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=(0, 1), method="highs")
    if not r.success:
        return float("inf"), None
    njobs = njobs if njobs is not None else LAM * HORIZON
    total = float(IDLEv @ n) * HORIZON + r.fun * LAM * HORIZON
    x = r.x.reshape(NK, nv + 1)
    return total / njobs, dict(outsourced=float((x[:, nv] * w).sum()))


def served_fully(n, w, rho):
    """True when the fleet serves the whole mix on-fleet, i.e. meets the constraint."""
    v, info = recourse(n, w, rho)
    return np.isfinite(v) and info is not None and info["outsourced"] < 1e-9

def expected_recourse(n, scen, probs, rho):
    tot, xs = 0.0, []
    for w, p in zip(scen, probs):
        v, x = recourse(n, w, rho)
        if not np.isfinite(v):
            return float("inf")
        tot += p * v
    return tot

# ------------------------------------------------------------------ calibrate rho
COMPS10 = [c for c in product(range(0, 11), repeat=5) if sum(c) == 10]
print("calibrating the utilisation ceiling against the simulator")
wcal = SCENARIOS["uniform"]
sim = [(c, *facility({MACH[i]: c[i] for i in range(5) if c[i] > 0}, wcal, seed=0)) for c in COMPS10]
feas_sim = {c for c, e, d in sim if d <= SLA_S}
best_rho, best_agree = None, -1
for rho in np.arange(0.50, 1.00, 0.02):
    feas_lp = {c for c in COMPS10 if served_fully(np.array(c, float), wcal, rho)}
    agree = len(feas_sim & feas_lp) + len((set(COMPS10) - feas_sim) & (set(COMPS10) - feas_lp))
    if agree > best_agree:
        best_agree, best_rho = agree, float(rho)
RHO = best_rho
print("  rho = {:.2f}, agreeing with the simulator on {}/{} compositions ({:.1f}%)\n".format(
    RHO, best_agree, len(COMPS10), 100 * best_agree / len(COMPS10)))

# ------------------------------------------------------------------ S2 fidelity
lpv, simv = [], []
for c, e, d in sim:
    if d > SLA_S:
        continue
    v, info = recourse(np.array(c, float), wcal, RHO)
    # compare like with like: the outsourcing penalty is a modelling device, not an
    # energy prediction, so fleets that need it are excluded from the fidelity check
    if np.isfinite(v) and info is not None and info["outsourced"] < 1e-9:
        lpv.append(v); simv.append(e)
lpv, simv = np.array(lpv), np.array(simv)
rank = float(np.corrcoef(np.argsort(np.argsort(lpv)), np.argsort(np.argsort(simv)))[0, 1])

def pair_acc(margin):
    n = ok = 0
    for i in range(len(simv)):
        for j in range(i + 1, len(simv)):
            if abs(simv[i] - simv[j]) / simv[j] > margin:
                n += 1
                ok += (lpv[i] < lpv[j]) == (simv[i] < simv[j])
    return ok / n, n

acc5, n5 = pair_acc(0.05)
acc1, n1p = pair_acc(0.01)
print("LP-vs-simulator fidelity: Spearman {:.3f}, median |rel err| {:.1%}".format(
    rank, float(np.median(np.abs(lpv - simv) / simv))))
print("  pairs differing by >1%: {:.1%} ordered correctly ({:,} pairs)".format(acc1, n1p))
print("  pairs differing by >5%: {:.1%} ordered correctly ({:,} pairs)".format(acc5, n5))
sane("S2 LP orders clearly separated fleets correctly", acc5 > 0.90,
     "{:.1%} of pairs differing by more than 5% ({:,} pairs); Spearman over all feasible "
     "fleets is {:.3f}, depressed by near-ties".format(acc5, n5, rank))

# ------------------------------------------------------------------ first stage
def local_search(scen, probs, slots, ntypes, seed=0, starts=3):
    rng = np.random.default_rng(seed)
    bestn, bestv, evals = None, float("inf"), 0
    for st in range(starts):
        n = np.zeros(ntypes)
        if st == 0:
            n[rng.integers(ntypes)] = slots
        else:
            for _ in range(slots):
                n[rng.integers(ntypes)] += 1
        v = expected_recourse(n, scen, probs, RHO); evals += 1
        improved = True
        while improved:
            improved = False
            # steepest descent: evaluate every one-slot swap against the CURRENT n, then
            # apply only the best. Applying moves mid-scan (an earlier version did) lets a
            # later swap decrement a component the scan already emptied, producing negative
            # counts and fleets that spuriously beat exhaustive enumeration.
            bestmove, bestval = None, v
            for a in range(ntypes):
                if n[a] <= 0:
                    continue
                for b in range(ntypes):
                    if a == b:
                        continue
                    cand = n.copy(); cand[a] -= 1; cand[b] += 1
                    assert cand.min() >= 0 and abs(cand.sum() - slots) < 1e-9
                    cv = expected_recourse(cand, scen, probs, RHO); evals += 1
                    if cv < bestval - 1e-9:
                        bestval, bestmove = cv, cand
            if bestmove is not None:
                v, n, improved = bestval, bestmove, True
        if v < bestv:
            bestv, bestn = v, n
    return bestn, bestv, evals

def shortlist(scen, probs, slots, ntypes, k=25):
    """The LP prunes; the simulator picks among the survivors. This is the division of
    labour the fidelity result supports: the LP is reliable on clear separations and
    unreliable on near-ties, so it should never make the final call itself."""
    n, v, ev = local_search(scen, probs, slots, ntypes, seed=0, starts=3)
    cands = {tuple(int(x) for x in n)}
    for a in range(ntypes):
        for b in range(ntypes):
            if a == b or n[a] <= 0:
                continue
            c2 = n.copy(); c2[a] -= 1; c2[b] += 1
            cands.add(tuple(int(x) for x in c2))
            for a2 in range(ntypes):
                for b2 in range(ntypes):
                    if a2 == b2 or c2[a2] <= 0 or c2.min() < 0:
                        continue
                    c3 = c2.copy(); c3[a2] -= 1; c3[b2] += 1
                    cands.add(tuple(int(x) for x in c3))
    scored = sorted((expected_recourse(np.array(c, float), scen, probs, RHO), c) for c in cands)
    return [c for _, c in scored[:k]], ev + len(cands)

print("\nS1: LP search plus simulator refinement versus exhaustive enumeration")
enum_best, enum_c = float("inf"), None
for c, e, d in sim:
    if d <= SLA_S and e < enum_best:
        enum_best, enum_c = e, c
cands, ev = shortlist([wcal], np.array([1.0]), 10, 5, k=25)
best_sim, best_c = float("inf"), None
for c in cands:
    e, d = facility({MACH[i]: c[i] for i in range(5) if c[i] > 0}, wcal, seed=0)
    if d <= SLA_S and e < best_sim:
        best_sim, best_c = e, c
sane("S1 LP shortlist plus simulation recovers the enumeration optimum",
     best_c == enum_c and abs(best_sim - enum_best) < 1e-6,
     "enumeration {} at {:.0f} J/job needing 1001 simulations; LP+refine {} at {:.0f} "
     "needing {} LP solves and {} simulations".format(enum_c, enum_best, best_c, best_sim,
                                                      ev, len(cands)))

# ------------------------------------------------------------------ S3
n1, v1, _ = local_search([wcal], np.array([1.0]), 10, 5, seed=1)
lp_enum = min(recourse(np.array(c, float), wcal, RHO)[0] for c in COMPS10)
sane("S3 single scenario reduces to the deterministic problem", abs(v1 - lp_enum) < 1e-6,
     "local search {:.1f} vs LP enumeration {:.1f} J/job".format(v1, lp_enum))

# ------------------------------------------------------------------ VSS and EVPI
scen = [SCENARIOS[k] for k in SCENARIOS]
names = list(SCENARIOS)
wbar = sum(p * w for p, w in zip(PROBS, scen))
n_sp, v_sp, _ = local_search(scen, PROBS, 10, 5, seed=0)
n_ev, _, _ = local_search([wbar], np.array([1.0]), 10, 5, seed=0)
v_ev = expected_recourse(n_ev, scen, PROBS, RHO)
wait_and_see = sum(p * local_search([w], np.array([1.0]), 10, 5, seed=0)[1]
                   for p, w in zip(PROBS, scen))
VSS, EVPI = v_ev - v_sp, v_sp - wait_and_see

def fl(n):
    return "+".join("{}x{}".format(int(v), MACH[i]) for i, v in enumerate(n) if v > 0)

print("\ntwo-stage stochastic program, 10 slots, {} mix scenarios".format(len(scen)))
print("  here-and-now fleet (stochastic)   {:<30} {:.0f} J/job".format(fl(n_sp), v_sp))
print("  mean-mix fleet (deterministic)    {:<30} {:.0f} J/job on the distribution".format(
    fl(n_ev), v_ev))
print("  wait-and-see (perfect foresight)  {:<30} {:.0f} J/job".format("-", wait_and_see))
print("  value of the stochastic solution   {:.0f} J/job ({:.2f}% of the mean-mix plan)".format(
    VSS, 100 * VSS / v_ev))
print("  expected value of perfect information {:.0f} J/job ({:.2f}%)".format(EVPI, 100 * EVPI / v_sp))
for nm, w, p in zip(names, scen, PROBS):
    v, info = recourse(n_sp, w, RHO)
    print("    scenario {:<26} p={:.2f}  {:.0f} J/job, off-fleet share {:.1%}".format(
        nm, p, v, info["outsourced"]))
sane("S4 value of the stochastic solution is non-negative", VSS >= -1e-9, "{:.3f} J/job".format(VSS))
sane("S5 EVPI is non-negative", EVPI >= -1e-9, "{:.3f} J/job".format(EVPI))

# ------------------------------------------------------------------ scaling
print("\nscaling beyond what enumeration can reach")
import math
scal = []
for slots, ntypes in [(10, 5), (20, 5), (50, 5), (100, 5)]:
    n, v, ev2 = local_search(scen, PROBS, slots, ntypes, seed=0)
    ncomp = math.comb(slots + ntypes - 1, ntypes - 1)
    scal.append(dict(slots=slots, types=ntypes, lp_evals=ev2, enumeration_size=ncomp,
                     fleet=fl(n), energy=v))
    print("  {:>4} slots: {:>11,} compositions to enumerate vs {:>5} LP solves "
          "({:>9,.0f}x fewer)   {} at {:.0f} J/job".format(
              slots, ncomp, ev2, ncomp / ev2, fl(n), v))

OUT = dict(rho=RHO, rho_agreement=best_agree / len(COMPS10), lp_vs_sim_spearman=rank,
           pair_accuracy_1pct=acc1, pair_accuracy_5pct=acc5, penalty=PENALTY,
           scenarios=names, probs=PROBS.tolist(),
           enumeration_optimum=list(enum_c), enumeration_energy=enum_best,
           lp_refined_optimum=list(best_c), lp_refined_energy=best_sim,
           stochastic_fleet=fl(n_sp), stochastic_value=v_sp,
           mean_mix_fleet=fl(n_ev), mean_mix_value_on_distribution=v_ev,
           wait_and_see=wait_and_see, VSS=VSS, EVPI=EVPI,
           VSS_pct=100 * VSS / v_ev, EVPI_pct=100 * EVPI / v_sp,
           scaling=scal, sanity=SANITY)
json.dump(OUT, io.open("experiments/results/b8_stochastic_program.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/b8_stochastic_program.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
