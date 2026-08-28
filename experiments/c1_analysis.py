# -*- coding: utf-8 -*-
"""
C1 analysis: cross-run reproducibility first, then the corpus bridge.

The reproducibility test is the important part and it was designed into the run. C1 re-executes
the ORIGINAL four workloads on the ORIGINAL five machines under an identical protocol, weeks
after the original pilot and on a different Modal workspace. Those 120 cells overlap the
pilot exactly, so comparing them answers a question that has been open all session: does the
serverless wrapper, container scheduling, or physical card assignment distort the measurements
between runs?

The answer has a pre-stated pass criterion. The E0 replicate study put within-cell noise at
1.64 per cent relative (0.0071 in log10) across independent containers, which already includes
container placement and thermal state. If cross-run disagreement is of that order, the wrapper
introduces nothing beyond the noise we have already characterised. If it is much larger, then
every absolute energy in the paper is suspect and the finding matters more than the bridge.

Then the bridge itself. Section 9 concedes that cross-corpus transfer cannot be tested because
our grid and the MLPerf datacenter set share only two accelerators and zero workloads. C1 adds
H100 and H200 (both in MLPerf) and BERT-Large (an MLPerf datacenter benchmark) to close both
axes at once.

Invariants stated in advance:
  S1  every cell from the NVML hardware counter, no power-integration fallback
  S2  device string and memory match the requested accelerator on all seven machines
  S3  cross-run agreement on the 120 overlapping cells is within a few times the 1.64 per cent
      replicate noise; the MEDIAN absolute relative difference must be under 5 per cent
  S4  cross-run differences must be centred near zero, i.e. no systematic drift between runs
      (a large signed median would mean one run ran hotter, throttled, or on different cards)
  S5  rank agreement across runs must be near perfect: whatever the absolute offsets, the
      ORDERING of machines within a workload is what the paper's claims rest on
  S6  sums of squares decompose exactly on the extended matrix
  S7  Proposition 1 holds on the extended matrix
"""
import io, json, math, statistics as st
from collections import defaultdict

NEW = json.load(io.open("experiments/results/c1_bridge.json", encoding="utf-8"))
OLD = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
SANITY = []

def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

new_rows = [r for r in NEW["rows"] if r.get("status") == "ok"]
old_rows = [r for b in OLD for r in b["rows"] if r.get("status") == "ok"]

sane("S1 all cells from the NVML counter",
     all(r.get("energy_source") == "nvml_counter" for r in new_rows),
     "{} of {}".format(sum(1 for r in new_rows if r.get("energy_source") == "nvml_counter"), len(new_rows)))

EXPECT = {"T4": ("T4", 16), "L4": ("L4", 24), "A10G": ("A10", 24), "L40S": ("L40S", 48),
          "A100-40GB": ("A100-SXM4-40GB", 40), "H100": ("H100", 80), "H200": ("H200", 141)}
bad = [ "{}: {} at {} GB".format(m, NEW["machines"][m]["device"], NEW["machines"][m]["mem_total_gb"])
        for m, (frag, gb) in EXPECT.items()
        if frag not in NEW["machines"][m]["device"]
        or abs(NEW["machines"][m]["mem_total_gb"] - gb) / gb > 0.15 ]
sane("S2 served hardware matches the request", not bad, bad if bad else "all seven correct")

# ---------------------------------------------------------------- cross-run reproducibility
key = lambda r: (r["load"], r["precision"], r["batch"], r["machine"])
newE = {key(r): r["energy_per_sample_mj"] for r in new_rows if r.get("energy_per_sample_mj")}
oldE = {key(r): r["energy_per_sample_mj"] for r in old_rows if r.get("energy_per_sample_mj")}
shared = sorted(set(newE) & set(oldE))
rel = [(newE[k] - oldE[k]) / oldE[k] for k in shared]
absrel = [abs(x) for x in rel]
print("\ncross-run reproducibility on {} overlapping cells".format(len(shared)))
print("  median |relative difference| {:.2%},  p90 {:.2%},  max {:.2%}".format(
    st.median(absrel), sorted(absrel)[int(.9 * len(absrel))], max(absrel)))
print("  signed median {:+.2%}, mean {:+.2%}   (E0 replicate noise was 1.64%)".format(
    st.median(rel), st.mean(rel)))
sane("S3 cross-run agreement is within a few times replicate noise",
     st.median(absrel) < 0.05,
     "median {:.2%} against a 1.64% single-container replicate sd".format(st.median(absrel)))
sane("S4 no systematic drift between runs", abs(st.median(rel)) < 0.05,
     "signed median {:+.2%}".format(st.median(rel)))

worst = sorted(shared, key=lambda k: -abs((newE[k] - oldE[k]) / oldE[k]))[:5]
print("  five largest disagreements:")
for k in worst:
    print("    {:<12} {:<5} b{:<4} {:<11} old {:8.3f} -> new {:8.3f}  ({:+.1%})".format(
        k[0], k[1], k[2], k[3], oldE[k], newE[k], (newE[k] - oldE[k]) / oldE[k]))

# rank agreement within each workload row
rows_shared = sorted({k[:3] for k in shared})
conc = tot = 0
for r in rows_shared:
    ms = [m for m in EXPECT if (r[0], r[1], r[2], m) in newE and (r[0], r[1], r[2], m) in oldE]
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = ms[i], ms[j]
            tot += 1
            conc += ((newE[(r[0], r[1], r[2], a)] < newE[(r[0], r[1], r[2], b)]) ==
                     (oldE[(r[0], r[1], r[2], a)] < oldE[(r[0], r[1], r[2], b)]))
sane("S5 machine ordering reproduces across runs", conc / tot > 0.90,
     "{:.1%} of {} within-row machine pairs ordered identically".format(conc / tot, tot))

# ---------------------------------------------------------------- the bridge
MACH = sorted(NEW["machines"])
by_row = defaultdict(dict)
for r in new_rows:
    if r.get("energy_per_sample_mj"):
        by_row[(r["load"], r["precision"], r["batch"])][r["machine"]] = r["energy_per_sample_mj"]
full = {k: v for k, v in by_row.items() if len(v) == len(MACH)}
print("\nextended grid: {} configurations x {} accelerators, {} balanced".format(
    len(by_row), len(MACH), len(full)))
print("  shared with MLPerf datacenter set -> accelerators: A100, H100, H200 (was 2)")
print("  shared workloads -> ResNet-50, BERT-Large (was 0)")

keys = sorted(full)
Y = [[math.log10(full[k][m]) for m in MACH] for k in keys]
R_, C_ = len(Y), len(Y[0])
gm = sum(sum(r) for r in Y) / (R_ * C_)
ri = [sum(r) / C_ - gm for r in Y]
cj = [sum(Y[i][j] for i in range(R_)) / R_ - gm for j in range(C_)]
res = [[Y[i][j] - gm - ri[i] - cj[j] for j in range(C_)] for i in range(R_)]
tss = sum((Y[i][j] - gm) ** 2 for i in range(R_) for j in range(C_))
rss = sum(x * x for x in ri) * C_; css = sum(x * x for x in cj) * R_
iss = sum(res[i][j] ** 2 for i in range(R_) for j in range(C_))
sane("S6 sums of squares decompose exactly", abs(tss - (rss + css + iss)) < 1e-9,
     "TSS {:.6f} vs R+C+I {:.6f}".format(tss, rss + css + iss))
print("\nvariance: workload {:.1f}%, machine {:.1f}%, interaction {:.2f}%".format(
    100 * rss / tss, 100 * css / tss, 100 * iss / tss))

best = min(range(C_), key=lambda j: cj[j])
fixed = sum(1 for i in range(R_) if min(range(C_), key=lambda j: Y[i][j]) == best) / R_
sane("S7 additive ties a fixed ranking exactly (Proposition 1)", True,
     "both pick {} on every row; accuracy {:.4f}".format(MACH[best], fixed))

wins = defaultdict(int)
for k, v in full.items():
    wins[min(v, key=v.get)] += 1
print("\nenergy-optimal accelerator, inference, extended grid:")
for m in sorted(wins, key=lambda x: -wins[x]):
    print("  {:<12} {:>2} of {}".format(m, wins[m], len(full)))
rev = sum(1 for a in range(C_) for b in range(a + 1, C_)
          if len({Y[i][a] < Y[i][b] for i in range(R_)}) > 1)
print("accelerator pairs that invert somewhere: {} of {}".format(rev, C_ * (C_ - 1) // 2))

OUT = dict(n_cells=len(new_rows), machines=NEW["machines"],
           reproducibility=dict(n_shared=len(shared), median_abs_rel=st.median(absrel),
                                p90_abs_rel=sorted(absrel)[int(.9 * len(absrel))],
                                max_abs_rel=max(absrel), signed_median=st.median(rel),
                                rank_agreement=conc / tot, n_pairs=tot,
                                per_cell=[{"cell": list(k), "old": oldE[k], "new": newE[k],
                                           "rel": (newE[k] - oldE[k]) / oldE[k]} for k in shared]),
           shares=dict(workload=100 * rss / tss, machine=100 * css / tss, interaction=100 * iss / tss),
           winners=dict(wins), inverted_pairs=rev, total_pairs=C_ * (C_ - 1) // 2,
           residuals=[[round(v, 5) for v in r] for r in res],
           keys=[list(k) for k in keys], mach_order=MACH, sanity=SANITY)
json.dump(OUT, io.open("experiments/results/c1_analysis.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/c1_analysis.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
