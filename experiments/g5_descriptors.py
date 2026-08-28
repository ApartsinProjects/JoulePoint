# -*- coding: utf-8 -*-
"""
G5.  The minimum sufficient descriptor set, and the accuracy/acquisition-cost frontier.

Section 7 (Table 5) withholds descriptor groups ONE AT A TIME. That answers "does this
group matter" but not "what is the smallest set that suffices", because a one-at-a-time
ablation never visits most of the lattice. Here every subset of the descriptor pool is
enumerated, each is scored under the same leave-one-workload-family-out protocol on the
same evaluation panel, and the frontier of (descriptors acquired, ranking accuracy) is
reported together with what each descriptor costs to acquire.

Descriptor atoms and their acquisition cost
-------------------------------------------
  BATCH   batch size                       already in every job spec            cost 0
  FAMILY  workload family / script identity already in every job spec            cost 0
  MEM     peak memory footprint            already requested by the job          cost 0
  PREC    numerical precision              known at submission, ONE new field    cost 1
  PROF    reference-device profile         one instrumented run per workload     cost 10
          (log10 seconds per sample and mean-power fraction measured on T4)

Feature expansion follows Section 7 exactly: BATCH contributes log2(b) and log2(b)^2,
PREC contributes an fp32 indicator, FAMILY a one-hot, MEM log10 of the median peak
footprint, PROF the two measured reference quantities; and whenever PREC and BATCH are
both present the model also receives their product, as it does in Table 5's "precision
and batch size" row. Subsets are therefore nested information sets, not arbitrary
column bags.

Two panels
----------
  PANEL-5   all five accelerators ranked; descriptor pool is the four free atoms only.
            This panel exists to ANCHOR the reimplementation against Table 5.
  PANEL-4   T4 excluded from the ranking and used instead as the profiling reference,
            so PROF can enter the pool without being scored on the device it came from.
            Operationally: profile on the cheap box, then rank the four you would
            actually place on. All PANEL-4 subsets are scored on identical columns, so
            every number in the frontier is construct-matched.

Uncertainty
-----------
Accuracy is a mean over (row, accelerator-pair) comparisons that are not independent:
the six comparisons inside one workload row share that row's residual. A cluster
bootstrap over the 24 workload rows is therefore used, with the SAME 4000 resample
draws applied to every subset so that differences between subsets are paired. A
within-row permutation null is deliberately NOT used: the column main effect on this
grid is large, which is exactly the regime in which such a null is invalid.

Sanity checks, stated in advance
--------------------------------
  S1  The empty descriptor set must score EXACTLY the additive model's accuracy.
      Reason, stated before running: with a constant feature the fitted interaction
      loading is constant, so base + outer(const, v1) shifts every column by a fixed
      amount and the within-row ordering is unchanged. Any discrepancy is a bug in the
      fitting path, not a finding.
  S2  On PANEL-5, the full free set must reproduce Table 5's 87.9 and PREC-only must
      reproduce 83.3, each within 0.2 points. This validates the reimplementation.
  S3  Every subset must score strictly above chance (50) and at or below 100.
  S4  NEGATIVE CONTROL. A pure-noise descriptor appended to the best subset must not
      improve accuracy beyond bootstrap noise. Stated expectation: the paired bootstrap
      interval for (best + noise) minus best must contain zero.
  S5  Monotonicity is NOT asserted for individual subsets: more descriptors can hurt
      through variance. What IS asserted is that the FRONTIER, defined as the best
      accuracy over all subsets of size at most c, is non-decreasing in c, which is
      true by construction and checks the frontier code rather than the science.
  S6  A descriptor whose value is identical for every workload row must behave exactly
      like the empty set. This catches an expansion bug that would let a degenerate
      column carry signal.
  S7  The best subset must beat the empty set by a paired bootstrap interval that
      excludes zero. If it does not, no descriptor carries usable signal and nothing
      below is worth reading.

Check that was MIS-SPECIFIED and replaced, recorded rather than quietly deleted
---------------------------------------------------------------------------------
The first version of this script defined a subset as "sufficient" when the paired
bootstrap interval for (FULL SET minus subset) contained zero. On PANEL-4 that returned
the empty set as sufficient, which is impossible: S1 shows the empty set is exactly the
additive model, and Proposition 1 shows an additive model can carry no interaction at
all. Inspecting the lattice gave the cause immediately, and it is a property of the
data, not a coding error: the full set is NOT the best subset here. Adding descriptors
to only 24 training rows costs variance, so the full five-atom set scores 84.7 while
BATCH+FAMILY+PREC scores 88.2. Measuring every subset against a reference that is itself
3.5 points below the ceiling makes weak subsets look adequate. The reference point is
therefore the EMPIRICAL CEILING, the best-scoring subset in the lattice, and both the
point gap and the paired interval are reported against it. No parameter was tuned; only
the comparison target was corrected.

Free: reuses the measured 24 x 5 grid. No compute cost.
"""
import io, json, math, sys, warnings
from collections import defaultdict
from itertools import combinations
import numpy as np

sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from e4_e5_models import load_grid, MACH

SANITY, OUT = [], {}
NBOOT = 4000
REFERENCE = "T4"           # profiling box, excluded from PANEL-4 ranking


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ------------------------------------------------------------------ measured inputs
keys, Ylog, Tput = load_grid()
n, m = Ylog.shape
FAM = sorted({k[0] for k in keys})
raw = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
rows_raw = [r for b in raw for r in b["rows"] if r.get("status") == "ok"]

memd, prof_t, prof_p = defaultdict(list), {}, {}
for r in rows_raw:
    k = (r["load"], r["precision"], r["batch"])
    memd[k].append(r["peak_mem_gb"])
    if r["machine"] == REFERENCE:
        prof_t[k] = 1.0 / r["throughput_sps"]              # seconds per sample on the box
        prof_p[k] = r["mean_power_w"] / r["power_cap_w"] if r.get("power_cap_w") else np.nan
CAPW = {b["machine"]: b["power_cap_w"] for b in raw}
for k in keys:
    if not np.isfinite(prof_p.get(k, np.nan)):
        prof_p[k] = [r["mean_power_w"] for r in rows_raw
                     if r["machine"] == REFERENCE and (r["load"], r["precision"], r["batch"]) == k][0] / CAPW[REFERENCE]

memv = np.array([np.median(memd[k]) for k in keys])
p32 = np.array([1.0 if k[1] == "fp32" else 0.0 for k in keys])
lb = np.array([math.log2(k[2]) for k in keys])
onehot = np.array([[1.0 if k[0] == f else 0.0 for f in FAM] for k in keys])
proft = np.array([math.log10(prof_t[k]) for k in keys])
profp = np.array([prof_p[k] for k in keys])

ATOM_COLS = {
    "BATCH":  np.column_stack([lb, lb ** 2]),
    "FAMILY": onehot,
    "MEM":    np.log10(memv)[:, None],
    "PREC":   p32[:, None],
    "PROF":   np.column_stack([proft, profp]),
}
COST = {"BATCH": 0, "FAMILY": 0, "MEM": 0, "PREC": 1, "PROF": 10}
COST_WORDS = {
    "BATCH":  "free, already in the job spec",
    "FAMILY": "free, already in the job spec",
    "MEM":    "free, already requested by the job",
    "PREC":   "free at submission, needs one new field in the scheduling interface",
    "PROF":   "one instrumented run per new workload on a reference device",
}


def build(atoms):
    """Nested information set -> design matrix, with the PREC x BATCH product when both."""
    if not atoms:
        return np.ones((n, 1))
    cols = [ATOM_COLS[a] for a in sorted(atoms)]
    if "PREC" in atoms and "BATCH" in atoms:
        cols.append((p32 * lb)[:, None])
    return np.column_stack(cols)


# ------------------------------------------------------------------ model + scoring
def pair_correct(P, panel):
    """Per (row, accelerator-pair) correctness on the given column panel."""
    rowsi, correct = [], []
    for i in range(n):
        for a in range(len(panel)):
            for b in range(a + 1, len(panel)):
                ja, jb = panel[a], panel[b]
                t = Ylog[i, ja] - Ylog[i, jb]
                if t == 0:
                    continue
                p = P[i, ja] - P[i, jb]
                rowsi.append(i)
                correct.append(1.0 if (t > 0) == (p > 0) else 0.0)
    return np.array(rowsi), np.array(correct)


def fit_lofo(Xf, interaction=True):
    """Leave-one-workload-family-out bilinear rank-1, the Section 7 model.

    Training always uses all five measured columns; only SCORING is restricted to the
    panel. Using the reference device's measurements as a descriptor is therefore not
    leakage on PANEL-4, because the reference column is never ranked.
    """
    P = np.zeros_like(Ylog)
    for L in FAM:
        tr = [i for i, k in enumerate(keys) if k[0] != L]
        te = [i for i, k in enumerate(keys) if k[0] == L]
        mu = Ylog[tr].mean()
        r = Ylog[tr].mean(1) - mu
        c = Ylog[tr].mean(0) - mu
        rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[tr], r)
        P[te] = mu + rr.predict(Xf[te])[:, None] + c[None, :]
        if interaction:
            R = Ylog[tr] - (mu + r[:, None] + c[None, :])
            v1 = np.linalg.svd(R, full_matrices=False)[2][0]
            sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[tr], R @ v1)
            P[te] = P[te] + np.outer(sm.predict(Xf[te]), v1)
    return P


def evaluate(atoms, panel, interaction=True, Xf=None):
    Xf = build(atoms) if Xf is None else Xf
    P = fit_lofo(Xf, interaction=interaction)
    ri, cc = pair_correct(P, panel)
    return 100.0 * cc.mean(), ri, cc, P


# cluster bootstrap: resample WORKLOAD ROWS, keep every pair inside a row
RNG = np.random.default_rng(20260819)
BOOT_ROWS = RNG.integers(0, n, size=(NBOOT, n))
ROW_IDX = None  # filled per panel


def boot_dist(ri, cc, row_slices):
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        sel = np.concatenate([row_slices[i] for i in BOOT_ROWS[b]])
        out[b] = 100.0 * cc[sel].mean()
    return out


def slices_for(ri):
    return [np.where(ri == i)[0] for i in range(n)]


PANEL5 = list(range(m))
PANEL4 = [j for j in range(m) if MACH[j] != REFERENCE]
print(f"grid {n} workload rows x {m} accelerators; families {FAM}")
print(f"PANEL-5 ranks {[MACH[j] for j in PANEL5]}")
print(f"PANEL-4 ranks {[MACH[j] for j in PANEL4]} (reference {REFERENCE} used for PROF)\n")

# ================================================================== sanity S1, S2, S6
print("sanity checks stated in advance:")

acc_empty5, ri_e5, cc_e5, _ = evaluate(frozenset(), PANEL5)
P_add = fit_lofo(build(frozenset()), interaction=False)
_, cc_add = pair_correct(P_add, PANEL5)
sane("S1 empty descriptor set scores exactly the additive model",
     abs(acc_empty5 - 100.0 * cc_add.mean()) < 1e-9,
     f"empty {acc_empty5:.4f}% vs additive {100.0*cc_add.mean():.4f}%")

acc_full5, _, _, _ = evaluate(frozenset({"PREC", "BATCH", "FAMILY"}), PANEL5)
acc_prec5, _, _, _ = evaluate(frozenset({"PREC"}), PANEL5)
sane("S2 PANEL-5 reproduces Table 5",
     abs(acc_full5 - 87.9) < 0.2 and abs(acc_prec5 - 83.3) < 0.2 and abs(acc_empty5 - 81.7) < 0.2,
     f"full {acc_full5:.1f} (paper 87.9), precision-only {acc_prec5:.1f} (83.3), "
     f"empty {acc_empty5:.1f} (81.7)")

Xdeg = np.column_stack([np.ones(n), np.full(n, 3.7)])
acc_deg, _, _, _ = evaluate(None, PANEL5, Xf=Xdeg)
sane("S6 a constant-valued descriptor behaves exactly like the empty set",
     abs(acc_deg - acc_empty5) < 1e-9, f"{acc_deg:.4f}% vs {acc_empty5:.4f}%")

# ================================================================== full lattice
ATOMS4 = ["BATCH", "FAMILY", "MEM", "PREC"]              # free atoms, PANEL-5
ATOMS5 = ["BATCH", "FAMILY", "MEM", "PREC", "PROF"]      # full pool, PANEL-4

records = []


def run_lattice(atoms_pool, panel, tag):
    res = {}
    slices = None
    for r in range(len(atoms_pool) + 1):
        for S in combinations(atoms_pool, r):
            Sf = frozenset(S)
            acc, ri, cc, _ = evaluate(Sf, panel)
            if slices is None:
                slices = slices_for(ri)
            res[Sf] = dict(acc=acc, cc=cc, cost=sum(COST[a] for a in S), size=len(S))
    for Sf, d in res.items():
        d["boot"] = boot_dist(None, d["cc"], slices)
        d["ci"] = [float(np.percentile(d["boot"], 2.5)), float(np.percentile(d["boot"], 97.5))]
    # reference point is the EMPIRICAL CEILING, not the full set; see the docstring note
    ceil_key = max(res, key=lambda s: res[s]["acc"])
    ceil = res[ceil_key]
    for Sf, d in res.items():
        diff = ceil["boot"] - d["boot"]
        d["gap_vs_ceiling"] = float(ceil["acc"] - d["acc"])
        d["gap_ci"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        d["gap_vs_full"] = float(res[frozenset(atoms_pool)]["acc"] - d["acc"])
        # statistically indistinguishable from the ceiling
        d["sufficient"] = bool(d["gap_ci"][0] <= 0.0)
        # and the stricter operational reading: within one point of the ceiling
        d["sufficient_strict"] = bool(d["sufficient"] and d["gap_vs_ceiling"] <= 1.0)
        records.append(dict(panel=tag, descriptors=sorted(Sf), size=d["size"], cost=d["cost"],
                            accuracy=d["acc"], ci95=d["ci"], gap_vs_ceiling=d["gap_vs_ceiling"],
                            gap_vs_full=d["gap_vs_full"], gap_ci95=d["gap_ci"],
                            sufficient=d["sufficient"], sufficient_strict=d["sufficient_strict"]))
    return res, slices, ceil_key


def show(res, ceil_key):
    print(f"{'descriptors':32}{'cost':>6}{'acc %':>9}{'95% CI':>16}{'gap to ceiling':>17}"
          f"{'paired 95% CI':>18}{'suff':>7}{'strict':>8}")
    for Sf in sorted(res, key=lambda s: -res[s]["acc"]):
        d = res[Sf]
        lab = ("+".join(sorted(Sf)) if Sf else "(nothing)") + (" <-ceil" if Sf == ceil_key else "")
        print(f"{lab:32}{d['cost']:6}{d['acc']:9.1f}  [{d['ci'][0]:5.1f},{d['ci'][1]:5.1f}]"
              f"{d['gap_vs_ceiling']:17.1f}  [{d['gap_ci'][0]:+5.1f},{d['gap_ci'][1]:+5.1f}]"
              f"{'yes' if d['sufficient'] else '-':>7}{'yes' if d['sufficient_strict'] else '-':>8}")


print(f"\n{'='*94}\nPANEL-5, free descriptors only, all {2**len(ATOMS4)} subsets\n{'='*94}")
res5, sl5, ceil5 = run_lattice(ATOMS4, PANEL5, "PANEL-5")
show(res5, ceil5)

print(f"\n{'='*94}\nPANEL-4, full pool including the profiled descriptor, all {2**len(ATOMS5)} subsets\n{'='*94}")
res4, sl4, ceil4 = run_lattice(ATOMS5, PANEL4, "PANEL-4")
show(res4, ceil4)

# ================================================================== frontiers
def frontier(res, key):
    """Best accuracy attainable at each budget level of `key` (size or cost)."""
    levels = sorted({d[key] for d in res.values()})
    fr = []
    for L in levels:
        cands = [(d["acc"], Sf) for Sf, d in res.items() if d[key] <= L]
        acc, Sf = max(cands)
        fr.append(dict(budget=L, accuracy=acc, descriptors=sorted(Sf),
                       ci95=res[Sf]["ci"]))
    return fr


fr5_size, fr5_cost = frontier(res5, "size"), frontier(res5, "cost")
fr4_size, fr4_cost = frontier(res4, "size"), frontier(res4, "cost")

print("\nacquisition-cost frontier (PANEL-4, best subset attainable at each cost budget)")
print(f"{'budget':>8}{'acc %':>9}   descriptors")
for f in fr4_cost:
    print(f"{f['budget']:8}{f['accuracy']:9.1f}   {'+'.join(f['descriptors']) or '(nothing)'}")

mono = all(fr4_cost[i]["accuracy"] <= fr4_cost[i + 1]["accuracy"] + 1e-9 for i in range(len(fr4_cost) - 1)) \
    and all(fr4_size[i]["accuracy"] <= fr4_size[i + 1]["accuracy"] + 1e-9 for i in range(len(fr4_size) - 1))
sane("S5 the frontier is non-decreasing in budget (checks the frontier code)", mono,
     "size and cost frontiers both non-decreasing")

allacc = [d["acc"] for d in list(res5.values()) + list(res4.values())]
sane("S3 every subset scores above chance and at or below 100",
     min(allacc) > 50.0 and max(allacc) <= 100.0,
     f"min {min(allacc):.1f}%, max {max(allacc):.1f}%")

# ================================================================== minimum sufficient set
def min_sufficient(res, flag):
    ok = sorted((d["cost"], d["size"], sorted(Sf), Sf) for Sf, d in res.items() if d[flag])
    return ok[0] if ok else None


MINSUF = {}
for tag, res_, ceil_key in (("PANEL-5", res5, ceil5), ("PANEL-4", res4, ceil4)):
    print(f"\n{tag}: empirical ceiling is {'+'.join(sorted(ceil_key))} at {res_[ceil_key]['acc']:.1f}%")
    MINSUF[tag] = dict(ceiling=dict(descriptors=sorted(ceil_key), accuracy=res_[ceil_key]["acc"],
                                    cost=res_[ceil_key]["cost"]))
    for flag, word in (("sufficient_strict", "minimum sufficient (within 1 point of the ceiling)"),
                       ("sufficient", "cheapest not significantly below the ceiling")):
        r_ = min_sufficient(res_, flag)
        if r_ is None:
            print(f"  {word}: none")
            continue
        c_, sz_, lab_, Sf_ = r_
        print(f"  {word}: {'+'.join(lab_) or '(nothing)'} at {res_[Sf_]['acc']:.1f}%, cost {c_}")
        MINSUF[tag][flag] = dict(descriptors=lab_, cost=c_, size=sz_,
                                 accuracy=res_[Sf_]["acc"],
                                 gap_vs_ceiling=res_[Sf_]["gap_vs_ceiling"])
    empty_ = res_[frozenset()]
    diff_ = res_[ceil_key]["boot"] - empty_["boot"]
    lo_, hi_ = float(np.percentile(diff_, 2.5)), float(np.percentile(diff_, 97.5))
    MINSUF[tag]["ceiling_vs_empty"] = dict(gap=res_[ceil_key]["acc"] - empty_["acc"], ci95=[lo_, hi_])
    sane(f"S7 {tag} best subset beats the empty set, paired interval excluding zero", lo_ > 0.0,
         f"ceiling {res_[ceil_key]['acc']:.1f}% vs empty {empty_['acc']:.1f}%, gap "
         f"{res_[ceil_key]['acc']-empty_['acc']:.1f} pp, 95% CI [{lo_:+.1f},{hi_:+.1f}]")

# ================================================================== S4 negative control
best4 = max(res4, key=lambda s: res4[s]["acc"])
rngn = np.random.default_rng(7)
Xbest = build(best4)
noise_gaps = []
for trial in range(5):
    Xn = np.column_stack([Xbest, rngn.normal(size=(n, 1))])
    a_n, ri_n, cc_n, _ = evaluate(None, PANEL4, Xf=Xn)
    bd = boot_dist(None, cc_n, sl4)
    diff = bd - res4[best4]["boot"]
    noise_gaps.append(dict(trial=trial, acc=a_n, delta=float(a_n - res4[best4]["acc"]),
                           delta_ci95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]))
zero_in = all(g["delta_ci95"][0] <= 0.0 <= g["delta_ci95"][1] for g in noise_gaps)
sane("S4 negative control: a pure-noise descriptor does not improve the best subset",
     zero_in,
     "best {} = {:.1f}%; noise deltas {}".format("+".join(sorted(best4)), res4[best4]["acc"],
        ", ".join("{:+.1f} [{:+.1f},{:+.1f}]".format(g["delta"], *g["delta_ci95"]) for g in noise_gaps)))

# ================================================================== marginal value of each atom
print("\nmarginal value of each descriptor, averaged over every subset it can be added to (PANEL-4)")
print(f"{'descriptor':10}{'mean gain pp':>14}{'max gain pp':>13}{'acquisition cost':>50}")
marg = {}
for a in ATOMS5:
    gains = [res4[Sf | {a}]["acc"] - res4[Sf]["acc"] for Sf in res4 if a not in Sf]
    marg[a] = dict(mean=float(np.mean(gains)), max=float(np.max(gains)),
                   min=float(np.min(gains)), cost=COST[a], cost_words=COST_WORDS[a])
    print(f"{a:10}{np.mean(gains):14.2f}{np.max(gains):13.2f}{COST_WORDS[a]:>50}")

OUT["panel5"] = {"+".join(sorted(k)) or "(nothing)": dict(acc=v["acc"], ci=v["ci"], cost=v["cost"],
                 size=v["size"], gap_vs_ceiling=v["gap_vs_ceiling"], gap_vs_full=v["gap_vs_full"],
                 gap_ci=v["gap_ci"], sufficient=v["sufficient"],
                 sufficient_strict=v["sufficient_strict"]) for k, v in res5.items()}
OUT["panel4"] = {"+".join(sorted(k)) or "(nothing)": dict(acc=v["acc"], ci=v["ci"], cost=v["cost"],
                 size=v["size"], gap_vs_ceiling=v["gap_vs_ceiling"], gap_vs_full=v["gap_vs_full"],
                 gap_ci=v["gap_ci"], sufficient=v["sufficient"],
                 sufficient_strict=v["sufficient_strict"]) for k, v in res4.items()}
OUT["frontier_panel5_by_size"] = fr5_size
OUT["frontier_panel5_by_cost"] = fr5_cost
OUT["frontier_panel4_by_size"] = fr4_size
OUT["frontier_panel4_by_cost"] = fr4_cost
OUT["minimum_sufficient"] = MINSUF
OUT["marginal_value"] = marg
OUT["negative_control"] = noise_gaps
OUT["per_subset_records"] = records
OUT["descriptor_costs"] = {a: dict(ordinal=COST[a], description=COST_WORDS[a]) for a in ATOMS5}
OUT["config"] = dict(nboot=NBOOT, reference_device=REFERENCE, panel5=[MACH[j] for j in PANEL5],
                     panel4=[MACH[j] for j in PANEL4], protocol="leave-one-workload-family-out",
                     bootstrap="cluster over the 24 workload rows, paired across subsets")

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/g5_descriptors.json", "w", encoding="utf-8"), indent=1)
print(f"\nsanity: {sum(1 for s in SANITY if s['passed'])}/{len(SANITY)} passed")
for s in SANITY:
    if not s["passed"]:
        print(f"  FAILED: {s['check']}: {s['detail']}")
print("saved -> experiments/results/g5_descriptors.json")
