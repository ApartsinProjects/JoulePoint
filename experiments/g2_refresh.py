# -*- coding: utf-8 -*-
"""
G2.  Does staggered refresh buy heterogeneity for free?

Section 8 treats fleet composition as a single procurement decision, and finds it worth
about 34.5 per cent of facility energy. Real facilities do not buy once. They refresh in
waves, and the wave schedule is chosen for capital and depreciation reasons that have
nothing to do with energy. That schedule has an unremarked side effect: a facility that
replaces a third of its slots every year is ALWAYS holding three vintages at once, so it
is heterogeneous whether or not anybody decided it should be. A facility that replaces
everything every three years is homogeneous on the day of each purchase.

The question is therefore not whether heterogeneity pays, which Section 8 settles, but
whether the ordinary refresh calendar already delivers it.

Setup
-----
The five measured accelerators form a vintage ladder by introduction year: T4 in 2018,
A100 in 2020, A10G in 2021, and the Ada parts in 2023. A facility of ten slots starts in
2018 holding the then-current part and follows a refresh policy through 2026. At each
year, whatever the policy has left it holding is evaluated with the same discrete-event
facility simulator used everywhere else in the paper, at the same arrival rate, with
energy-first placement, over paired seeds.

Policies compared
-----------------
  never                   hold the 2018 fleet forever
  big bang every R years  replace all ten slots with the current part, R in {2,3,4}
  rolling, R-year cycle   replace the oldest 10/R slots each year, R in {2,3,4}
  oracle                  the best ten-slot composition buildable from every part
                          released so far; this is Section 8's search, rerun per year,
                          and it is the benchmark the policies are measured against

Reported alongside energy is mean queueing delay, because the published CarbonSim result
that mixed-generation clusters cut carbon by sixteen to twenty-six per cent also reports
a substantial runtime cost, and a refresh policy that buys energy with latency has not
bought anything for free.

Scope, stated plainly: this models OPERATIONAL energy only. Embodied carbon in the
replacement hardware, and the disposal of what it replaces, are not in this model, and a
refresh policy cannot be judged on carbon without them.

Sanity checks stated in advance
--------------------------------
  S1  A rolling policy with a one-year cycle must hold EXACTLY the same fleet, every
      year, as a big-bang policy with a one-year period. Both replace everything
      annually, so any difference is a bug in the wave bookkeeping.
  S2  The never-refresh policy must hold the 2018 fleet unchanged at every year.
  S3  SLOT CONSERVATION. Every policy holds exactly ten slots in every year. Refresh
      moves slots between types and never creates or destroys them.
  S4  The oracle must be at least as good as every policy in every year. Because oracle
      and policies are evaluated on the SAME seeds, and the oracle is a minimum over a
      set that contains every policy fleet, this must hold exactly, not merely on
      average. A violation means the oracle search is not searching what it claims to.
  S5  DEGENERATE LADDER CONTROL. If every year's current part is the same device, then
      all policies must hold identical fleets and record identical energy. This isolates
      the vintage ladder as the only thing driving any difference between policies.
  S6  MONTE CARLO. Every fleet is evaluated over six paired seeds and reported with its
      spread; the paper's facility simulations carry about 1.4 per cent seed-to-seed
      variation, so differences smaller than that are not read as real.

Three defects found on the first run, each root-caused rather than tuned away
-----------------------------------------------------------------------------
1. THE LOAD WAS OUTSIDE EVERY FLEET'S OPERATING RANGE. The first run used the arrival
   rate of Section 8, 0.5 jobs per second. A ten-slot T4 fleet saturates at 0.109, so
   the 2018 fleet was being asked to carry five times its capacity and every policy
   recorded mean queueing delays of thousands of seconds. Energy comparisons taken deep
   in saturation measure the queue, not the fleet. The experiment now runs at TWO rates,
   0.06 and 0.10, because inspecting the results made clear that the answer is regime
   dependent rather than universal: at 0.06 idle power dominates and the lowest-idle part
   wins outright, while at 0.10 the oldest fleet is at its capacity limit and refreshing
   is the only way to keep up. Reporting one rate would have reported half the answer.

4. THE SEED SPREAD WAS LARGER THAN THE EFFECTS, which S6 caught: 9.4 per cent against
   policy differences of a few per cent. The cause is not the simulator but the sample
   size, since a one-hour run at 0.06 admits only about 216 jobs and the Poisson
   variation in that count alone accounts for 6.8 per cent. Runs are now six hours. This
   reduces Monte Carlo error without changing any modelled quantity, so it is not a
   parameter tuned to make a check pass.

2. THE MEASUREMENT WINDOW WAS WRONG, the same defect found and fixed in G6. Arrivals are
   drawn over one hour, but jobs admitted near the end finish after it. Charging idle
   power over the horizon while work continued past it under-counted the fixed cost of
   holding the fleet. Idle is now charged over max(horizon, makespan).

3. THE ORACLE AND THE POLICIES WERE NOT CONSTRUCT-MATCHED, which is what S4's sixty
   violations were really reporting. The oracle was filtered to compositions meeting the
   60-second SLA while the policies were not, so a policy holding an SLA-violating fleet
   could legitimately record less energy than the SLA-respecting oracle. That is not an
   oracle failure, it is two different questions being compared. The oracle is now the
   pure energy minimum over the compositions buildable from the parts released so far,
   which makes S4 an exact invariant, and the SLA-feasible optimum is reported as a
   separate column alongside its delay.

Free: reuses the measured 24 x 5 grid. No compute cost.
"""
import io, json, math, sys, warnings
from collections import defaultdict
from itertools import product
import numpy as np

sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH, HW

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, HORIZON, SLA_S = 20000, 21600.0, 60.0
# Two load regimes, because the answer depends on which cost dominates. At 0.06 the
# fleet is idle-dominated and the 2018 T4 fleet runs near 55 per cent; at 0.10 it is
# at the edge of its capacity, since ten T4 slots saturate at 0.109. The horizon is six
# hours rather than one: at these rates a one-hour run admits only about 216 jobs, and
# the Poisson variation in that count alone put the seed spread at 9.4 per cent, larger
# than the differences between policies. Lengthening the run reduces Monte Carlo error
# without changing any modelled quantity.
RATES = [(0.06, 'LIGHT LOAD, idle-dominated'), (0.10, 'HEAVY LOAD, at T4 fleet capacity')]
SLOTS = 10
SEEDS = list(range(6))
YEARS = list(range(2018, 2027))
SANITY, OUT = [], {}


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


keys, Ylog, Tput = load_grid()
NK = len(keys)
E_J = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(len(MACH))} for i, k in enumerate(keys)}
T_S = {k: {MACH[j]: Tput[i, j] for j in range(len(MACH))} for i, k in enumerate(keys)}
YEAR_OF = {m: HW[m]["year"] for m in MACH}


def sim(pool, lam, seed=0):
    rng = np.random.default_rng(seed)
    slots = [mm for mm, c in pool.items() for _ in range(c)]
    ns = len(slots)
    free_at = np.zeros(ns)
    t, arr = 0.0, []
    while t < HORIZON:
        t += rng.exponential(1.0 / lam)
        if t < HORIZON:
            arr.append((t, keys[rng.integers(NK)]))
    dyn, delays, busy_time = 0.0, [], 0.0
    for at, jk in arr:
        idx = [i for i in range(ns) if free_at[i] <= at]
        start = at if idx else float(np.min(free_at))
        if not idx:
            idx = [i for i in range(ns) if free_at[i] <= start]
        cand = sorted({slots[i] for i in idx})
        mm = min(cand, key=lambda x: E_J[jk][x])
        i = next(i for i in idx if slots[i] == mm)
        rt = JOB / T_S[jk][mm]
        free_at[i] = start + rt
        delays.append(start - at)
        busy_time += rt
        dyn += JOB * E_J[jk][mm] - IDLE[mm] * rt
    # idle is charged over the interval the fleet is actually powered, which is the
    # makespan when work spills past the arrival horizon; see the docstring note
    window = max(HORIZON, float(free_at.max()) if ns else HORIZON)
    static = sum(IDLE[s] for s in slots) * window
    return ((static + dyn) / max(len(arr), 1),
            float(np.mean(delays)) if delays else 0.0,
            busy_time / (ns * window))


_CACHE = {}


def evaluate(pool, lam):
    """Paired-seed evaluation, cached: identical fleets always get identical numbers."""
    key = (lam, tuple(sorted(pool.items())))
    if key not in _CACHE:
        rs = [sim(dict(pool), lam, seed=s) for s in SEEDS]
        e = np.array([r[0] for r in rs]); d = np.array([r[1] for r in rs])
        u = np.array([r[2] for r in rs])
        _CACHE[key] = dict(per_job=float(e.mean()), per_job_sd=float(e.std(ddof=1)),
                           delay=float(d.mean()), delay_sd=float(d.std(ddof=1)),
                           utilisation=float(u.mean()),
                           per_seed=[float(x) for x in e])
    return _CACHE[key]


# ------------------------------------------------------------------ vintage ladder
def ladder(ada="L4"):
    """The part a buyer would purchase in each year, given what has been released."""
    lad = {}
    for y in YEARS:
        avail = [m for m in MACH if YEAR_OF[m] <= y] or ["T4"]
        newest_year = max(YEAR_OF[m] for m in avail)
        cands = [m for m in avail if YEAR_OF[m] == newest_year]
        lad[y] = ada if (ada in cands) else cands[0]
    return lad


def available(y):
    return [m for m in MACH if YEAR_OF[m] <= y] or ["T4"]


# ------------------------------------------------------------------ refresh policies
def run_policy(kind, R, lad):
    """Returns {year: (fleet dict, per-slot vintage list)}. Slots carry the year bought."""
    slots = [(lad[YEARS[0]], YEARS[0]) for _ in range(SLOTS)]
    out = {}
    for idx, y in enumerate(YEARS):
        if idx > 0:
            if kind == "never":
                pass
            elif kind == "bigbang":
                if idx % R == 0:
                    slots = [(lad[y], y) for _ in range(SLOTS)]
            elif kind == "rolling":
                # replace floor(SLOTS*(i+1)/R) - floor(SLOTS*i/R) oldest slots this year,
                # so a full cycle completes in exactly R years and slot count is conserved
                i = idx % R
                k = (SLOTS * (i + 1)) // R - (SLOTS * i) // R
                order = sorted(range(SLOTS), key=lambda s: (slots[s][1], s))
                for s in order[:k]:
                    slots[s] = (lad[y], y)
            else:
                raise ValueError(kind)
        pool = defaultdict(int)
        for m, _ in slots:
            pool[m] += 1
        out[y] = (dict(pool), [v for _, v in slots])
    return out


POLICIES = ([("never refresh", "never", 0)] +
            [(f"big bang every {R} years", "bigbang", R) for R in (2, 3, 4)] +
            [(f"rolling, {R}-year cycle", "rolling", R) for R in (2, 3, 4)])


# ------------------------------------------------------------------ oracle per year
def oracle(y, lam):
    """Pure energy minimum over the compositions buildable from the parts released by y.

    Deliberately NOT filtered by the SLA: the policies are not filtered either, and S4 is
    an exact invariant only when the oracle minimises over a set containing every policy
    fleet. The SLA-feasible optimum is computed here too and reported separately.
    """
    av = available(y)
    best, bn = float("inf"), None
    sla_best, sla_n = float("inf"), None
    for c in product(range(SLOTS + 1), repeat=len(av)):
        if sum(c) != SLOTS:
            continue
        pool = {av[i]: c[i] for i in range(len(av)) if c[i] > 0}
        r = evaluate(pool, lam)
        if r["per_job"] < best:
            best, bn = r["per_job"], pool
        if r["delay"] <= SLA_S and r["per_job"] < sla_best:
            sla_best, sla_n = r["per_job"], pool
    return bn, best, sla_n, sla_best


print("sanity checks stated in advance:")
LAD = ladder("L4")
print(f"  vintage ladder (the part bought in each year): "
      f"{{{', '.join(f'{y}:{LAD[y]}' for y in YEARS)}}}")

r_roll1 = run_policy("rolling", 1, LAD)
r_bb1 = run_policy("bigbang", 1, LAD)
sane("S1 a one-year rolling cycle is identical to annual big-bang replacement",
     all(r_roll1[y][0] == r_bb1[y][0] for y in YEARS),
     "fleets agree in every year of the horizon")

r_never = run_policy("never", 0, LAD)
sane("S2 the never-refresh policy holds its 2018 fleet unchanged",
     all(r_never[y][0] == r_never[YEARS[0]][0] for y in YEARS),
     f"holds {r_never[YEARS[0]][0]} throughout")

allruns = {name: run_policy(kind, R, LAD) for name, kind, R in POLICIES}
cons = all(sum(allruns[name][y][0].values()) == SLOTS for name in allruns for y in YEARS)
sane("S3 slot conservation: every policy holds exactly ten slots in every year", cons,
     f"{len(allruns)} policies x {len(YEARS)} years all at {SLOTS} slots")

# ------------------------------------------------------------------ evaluate, per load regime
def analyse(lam, tag):
    print(f"\n{'='*112}")
    print(f"{tag}: energy per job by year, ten slots, arrival rate {lam}/s, "
          f"{len(SEEDS)} paired seeds, {HORIZON/3600:.0f}-hour runs")
    print(f"{'='*112}")
    orc_full = {y: oracle(y, lam) for y in YEARS}
    ORACLE_ = {y: (orc_full[y][0], orc_full[y][1]) for y in YEARS}
    SLA_ = {y: (orc_full[y][2], orc_full[y][3]) for y in YEARS}
    orc = np.array([ORACLE_[y][1] for y in YEARS])
    print(f"{'policy':26}" + "".join(f"{y:>9}" for y in YEARS) + f"{'mean':>10}{'vs oracle':>11}")
    tbl = {}
    for name in allruns:
        rs = [evaluate(allruns[name][y][0], lam) for y in YEARS]
        vals = np.array([r["per_job"] for r in rs])
        gap = 100 * (vals - orc) / orc
        tbl[name] = dict(per_job=[float(v) for v in vals],
                         per_job_sd_pct=[float(100 * r["per_job_sd"] / r["per_job"]) for r in rs],
                         delay=[float(r["delay"]) for r in rs],
                         utilisation=[float(r["utilisation"]) for r in rs],
                         n_types=[len(allruns[name][y][0]) for y in YEARS],
                         gap_vs_oracle_pct=[float(g) for g in gap],
                         mean_per_job=float(vals.mean()), mean_gap_pct=float(gap.mean()))
        print(f"{name:26}" + "".join(f"{v:9.0f}" for v in vals) +
              f"{vals.mean():10.0f}{gap.mean():10.1f}%")
    print(f"{'oracle (energy minimum)':26}" + "".join(f"{ORACLE_[y][1]:9.0f}" for y in YEARS) +
          f"{orc.mean():10.0f}{0.0:10.1f}%")

    print(f"\n{'policy':26}{'distinct types held, by year':>40}{'mean util %':>13}{'mean delay s':>14}")
    for name in allruns:
        print(f"{name:26}{str(tbl[name]['n_types']):>40}"
              f"{100*np.mean(tbl[name]['utilisation']):13.1f}{np.mean(tbl[name]['delay']):14.1f}")
    print(f"\noracle fleet by year: " +
          ", ".join(f"{y}:{'+'.join(f'{v}x{k}' for k, v in sorted(ORACLE_[y][0].items()))}"
                    for y in YEARS))
    print(f"SLA-feasible optimum (mean delay at most {SLA_S:.0f} s):")
    for y in YEARS:
        n_, v_ = SLA_[y]
        if n_ is None:
            print(f"  {y}: no ten-slot composition from the parts released by then meets the SLA")
        else:
            print(f"  {y}: {'+'.join(f'{v}x{k}' for k, v in sorted(n_.items())):28} {v_:8.0f} J/job, "
                  f"delay {evaluate(n_, lam)['delay']:6.1f} s, "
                  f"{100*(v_-ORACLE_[y][1])/ORACLE_[y][1]:+5.1f}% above the unconstrained optimum")

    def mg(name, years):
        idx = [YEARS.index(y) for y in years]
        return float(np.mean([tbl[name]["gap_vs_oracle_pct"][i] for i in idx]))

    print(f"\nover {LIVE[0]}-{LIVE[-1]}, the years in which the vintage ladder still advances:")
    matched_ = []
    for R in (2, 3, 4):
        a, b = f"rolling, {R}-year cycle", f"big bang every {R} years"
        matched_.append(dict(cycle=R, rolling=mg(a, LIVE), bigbang=mg(b, LIVE),
                             staggering_worth_points=mg(b, LIVE) - mg(a, LIVE)))
        print(f"  cycle length {R}: rolling {mg(a, LIVE):+6.1f}% vs big bang {mg(b, LIVE):+6.1f}%"
              f", staggering is worth {mg(b, LIVE)-mg(a, LIVE):+5.1f} points")
    print(f"  never refreshing: {mg('never refresh', LIVE):+6.1f}% above the oracle")

    ALT_ = ladder("L40S")
    alt_runs_ = {name: run_policy(kind, R, ALT_) for name, kind, R in POLICIES}
    alt_ = {}
    print(f"\n  robustness, if the 2023 purchase is the L40S rather than the L4:")
    print(f"  {'policy':26}{'gap, L4 buy':>14}{'gap, L40S buy':>16}")
    for name in alt_runs_:
        v = np.array([evaluate(alt_runs_[name][y][0], lam)["per_job"] for y in YEARS])
        g = 100 * (v - orc) / orc
        alt_[name] = dict(per_job=[float(x) for x in v], gap_vs_oracle_pct=[float(x) for x in g],
                          mean_gap_pct=float(g.mean()))
        idx = [YEARS.index(y) for y in LIVE]
        print(f"  {name:26}{np.mean([tbl[name]['gap_vs_oracle_pct'][i] for i in idx]):13.1f}%"
              f"{np.mean([alt_[name]['gap_vs_oracle_pct'][i] for i in idx]):15.1f}%")

    sd = float(np.mean([x for n in tbl for x in tbl[n]["per_job_sd_pct"]]))
    return dict(lam=lam, policies=tbl, policies_L40S=alt_,
                oracle={str(y): dict(fleet=ORACLE_[y][0], per_job=ORACLE_[y][1],
                                     delay=evaluate(ORACLE_[y][0], lam)["delay"],
                                     utilisation=evaluate(ORACLE_[y][0], lam)["utilisation"],
                                     available_types=available(y)) for y in YEARS},
                sla_feasible_optimum={str(y): (None if SLA_[y][0] is None else
                                      dict(fleet=SLA_[y][0], per_job=SLA_[y][1],
                                           delay=evaluate(SLA_[y][0], lam)["delay"]))
                                      for y in YEARS},
                matched_cycle_comparison=matched_, mean_seed_sd_pct=sd)


LIVE = [y for y in YEARS if y <= 2023]        # the years in which the ladder still moves
BY_RATE = {}
for lam_, tag_ in RATES:
    BY_RATE[tag_] = analyse(lam_, tag_)

# ------------------------------------------------------------------ S4, S6
viol, checked = [], 0
for tag_, res in BY_RATE.items():
    for name in allruns:
        for iy, y in enumerate(YEARS):
            checked += 1
            if res["policies"][name]["per_job"][iy] < res["oracle"][str(y)]["per_job"] - 1e-9:
                viol.append((tag_, name, y))
sane("S4 the oracle is at least as good as every policy in every year, at every load", not viol,
     f"{len(viol)} violations across {checked} policy-years")

sd_all = {t: BY_RATE[t]["mean_seed_sd_pct"] for t in BY_RATE}
sane("S6 seed spread is smaller than the effects discussed",
     max(sd_all.values()) < 3.0,
     "; ".join(f"{t}: mean seed sd {v:.2f}% of the mean per-job energy" for t, v in sd_all.items()))

# ------------------------------------------------------------------ S5 degenerate control
FLAT = {y: "L4" for y in YEARS}
flat_runs = {name: run_policy(kind, R, FLAT) for name, kind, R in POLICIES}
LAM_MAIN = RATES[-1][0]
flat_e = {name: [evaluate(flat_runs[name][y][0], LAM_MAIN)["per_job"] for y in YEARS]
          for name in flat_runs}
same_fleet = all(flat_runs[n][y][0] == flat_runs["never refresh"][y][0]
                 for n in flat_runs for y in YEARS)
same_e = all(abs(flat_e[n][i] - flat_e["never refresh"][i]) < 1e-9
             for n in flat_e for i in range(len(YEARS)))
sane("S5 degenerate ladder control: with one part on the ladder, all policies coincide",
     same_fleet and same_e,
     "identical fleets and identical energy in every year for all seven policies")

OUT["vintage_ladder"] = {"L4_buy": LAD, "L40S_buy": ladder("L40S")}
OUT["years"] = YEARS
OUT["ladder_live_years"] = LIVE
OUT["by_load_regime"] = BY_RATE
OUT["fleets_by_year"] = {n: {str(y): allruns[n][y][0] for y in YEARS} for n in allruns}
OUT["vintages_by_year"] = {n: {str(y): allruns[n][y][1] for y in YEARS} for n in allruns}
OUT["config"] = dict(slots=SLOTS, seeds=SEEDS, rates=[dict(lam=l, label=t) for l, t in RATES],
                     horizon_s=HORIZON, sla_s=SLA_S,
                     scope="operational energy only; embodied carbon of the replacement "
                           "hardware is not modelled")

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/g2_refresh.json", "w", encoding="utf-8"), indent=1)
print(f"\nsanity: {sum(1 for s in SANITY if s['passed'])}/{len(SANITY)} passed")
for s in SANITY:
    if not s["passed"]:
        print(f"  FAILED: {s['check']}: {s['detail']}")
print("saved -> experiments/results/g2_refresh.json")
