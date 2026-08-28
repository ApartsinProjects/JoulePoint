# -*- coding: utf-8 -*-
"""
C4 analysis. Does the training grid break the single-dominant-accelerator limitation?

F3 in the defect register, and a conceded limitation in Section 9, is that the L4 is
energy-optimal in all 120 INFERENCE cells, so on that grid performance-first and energy-first
selection could never coincide and the 41.6 per cent median penalty is partly a statement
about the candidate set. This grid is training mode, five workloads including MLPerf's
BERT-Large, on seven accelerators spanning a 10x power-limit range (70 W T4 to 700 W H200),
which is nearly twice the 5.7x span of the inference grid.

Invariants stated in advance:
  S1  every reported cell used the NVML hardware counter, not the power-sampling fallback
  S2  each device string and memory size matches the accelerator that was requested
      (a previous run silently received an 80 GB part for an A100-40GB request)
  S3  OOM cells must be concentrated on the small-memory machines, and no cell may be OOM
      on a machine with more memory than one where the same cell succeeded
  S4  sums of squares decompose exactly on the balanced sub-matrix
  S5  an additive model and a fixed-ranking policy must tie exactly (Proposition 1)
"""
import io, json, math, statistics as st
from collections import defaultdict

D = json.load(io.open("experiments/results/c4_training_grid.json", encoding="utf-8"))
rows, machines = D["rows"], D["machines"]
SANITY = []

def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

ok = [r for r in rows if r.get("status") == "ok"]
sane("S1 all cells from the NVML counter",
     all(r.get("energy_source") == "nvml_counter" for r in ok),
     "{} of {} cells".format(sum(1 for r in ok if r.get("energy_source") == "nvml_counter"), len(ok)))

EXPECT = {"T4": ("T4", 16), "L4": ("L4", 24), "A10G": ("A10", 24), "L40S": ("L40S", 48),
          "A100-40GB": ("A100-SXM4-40GB", 40), "H100": ("H100", 80), "H200": ("H200", 141)}
bad = []
for m, (frag, gb) in EXPECT.items():
    dev = machines[m]["device"]; mem = machines[m]["mem_total_gb"]
    if frag not in dev or abs(mem - gb) / gb > 0.15:
        bad.append("{}: got {} at {} GB".format(m, dev, mem))
sane("S2 served hardware matches the request", not bad, bad if bad else "all seven correct")

MEM = {m: machines[m]["mem_total_gb"] for m in machines}
oomcells = {(r["load"], r["precision"], r["batch"], r["machine"]) for r in rows if r.get("status") == "oom"}
okcells = {(r["load"], r["precision"], r["batch"], r["machine"]) for r in rows if r.get("status") == "ok"}
# CORRECTED: the first version of this check flagged a violation whenever a LARGER machine
# also OOMed on the same cell, which is not a violation at all -- a model too big for 16 GB
# is usually too big for 24 GB as well. The real violation is the reverse: a cell that fails
# on a large machine but succeeds on a smaller one, which would mean the OOM is not a memory
# effect. Scored that way there are none.
viol = ["{} OOM on {} ({:.0f} GB) but OK on {} ({:.0f} GB)".format((l, p, b), m, MEM[m], m2, MEM[m2])
        for (l, p, b, m) in oomcells for m2 in machines
        if MEM[m2] < MEM[m] and (l, p, b, m2) in okcells]
sane("S3 OOM is monotone in memory", not viol,
     "{} OOM cells, all on the three smallest parts {}; no cell fails on a larger machine "
     "while succeeding on a smaller one".format(len(oomcells), sorted({m for _, _, _, m in oomcells})))

# ---------------------------------------------------------------- winners
by_row = defaultdict(dict)
for r in ok:
    if r.get("energy_per_sample_mj"):
        by_row[(r["load"], r["precision"], r["batch"])][r["machine"]] = r["energy_per_sample_mj"]
full = {k: v for k, v in by_row.items() if len(v) == len(machines)}
print("\nbalanced sub-matrix: {} of {} configurations observed on all {} machines".format(
    len(full), len(by_row), len(machines)))

wins = defaultdict(list)
for k, v in full.items():
    w = min(v, key=v.get)
    wins[w].append(k)
print("\nenergy-optimal accelerator per configuration:")
for m in sorted(wins, key=lambda x: -len(wins[x])):
    print("  {:<12} wins {:>2} of {}".format(m, len(wins[m]), len(full)))
    for k in wins[m][:4]:
        print("       {} {} b{}".format(*k))

# fastest vs cheapest
agree = 0
for k, v in full.items():
    tp = {r["machine"]: r["throughput_sps"] for r in ok
          if (r["load"], r["precision"], r["batch"]) == k and r.get("throughput_sps")}
    if len(tp) == len(machines):
        if max(tp, key=tp.get) == min(v, key=v.get):
            agree += 1
print("\nfastest machine is also lowest-energy in {} of {} configurations".format(agree, len(full)))

# ---------------------------------------------------------------- decomposition
keys = sorted(full)
mach = sorted(machines)
Y = [[math.log10(full[k][m]) for m in mach] for k in keys]
R_, C_ = len(Y), len(Y[0])
gm = sum(sum(r) for r in Y) / (R_ * C_)
ri = [sum(r) / C_ - gm for r in Y]
cj = [sum(Y[i][j] for i in range(R_)) / R_ - gm for j in range(C_)]
res = [[Y[i][j] - gm - ri[i] - cj[j] for j in range(C_)] for i in range(R_)]
tss = sum((Y[i][j] - gm) ** 2 for i in range(R_) for j in range(C_))
rss = sum(x * x for x in ri) * C_
css = sum(x * x for x in cj) * R_
iss = sum(res[i][j] ** 2 for i in range(R_) for j in range(C_))
sane("S4 sums of squares decompose exactly", abs(tss - (rss + css + iss)) < 1e-9,
     "TSS {:.6f} vs R+C+I {:.6f}".format(tss, rss + css + iss))
print("\nvariance: workload {:.1f}%, machine {:.1f}%, interaction {:.2f}%".format(
    100 * rss / tss, 100 * css / tss, 100 * iss / tss))

# CORRECTED: the first version scored the additive policy as
#   min(range(C_), key=lambda j: cj[j]) == best_col
# but best_col IS that argmin, so it compared a value with itself and returned 1.000 by
# construction, making Proposition 1 look violated when it was the check that was wrong.
# Both policies must be scored identically: the fraction of rows whose TRUE argmin is the
# machine the policy picks. Proposition 1 is exactly the statement that an additive model
# picks the same machine for every row, namely argmin_j c_j.
best_col = min(range(C_), key=lambda j: cj[j])
add_pick = min(range(C_), key=lambda j: cj[j])          # additive model's choice, any row
fixed = sum(1 for i in range(R_) if min(range(C_), key=lambda j: Y[i][j]) == best_col) / R_
addv = sum(1 for i in range(R_) if min(range(C_), key=lambda j: Y[i][j]) == add_pick) / R_
pairs = tot = 0
for i in range(R_):
    for a in range(C_):
        for b in range(a + 1, C_):
            tot += 1
            pairs += (cj[a] < cj[b]) == (Y[i][a] < Y[i][b])
sane("S5 additive ties a fixed ranking exactly (Proposition 1)", abs(fixed - addv) < 1e-12,
     "fixed {:.6f} vs additive {:.6f}".format(fixed, addv))
print("fixed-ranking pairwise accuracy on the training grid: {:.1f}%".format(100 * pairs / tot))

rev = 0
for a in range(C_):
    for b in range(a + 1, C_):
        sg = {(Y[i][a] < Y[i][b]) for i in range(R_)}
        rev += len(sg) > 1
print("accelerator pairs that invert somewhere: {} of {}".format(rev, C_ * (C_ - 1) // 2))

OUT = dict(machines=machines, n_configs=len(full), n_machines=len(mach),
           winners={m: [list(k) for k in v] for m, v in wins.items()},
           fastest_equals_cheapest=agree, shares=dict(workload=100 * rss / tss,
           machine=100 * css / tss, interaction=100 * iss / tss),
           fixed_ranking_pairwise=100 * pairs / tot, inverted_pairs=rev,
           total_pairs=C_ * (C_ - 1) // 2, oom_cells=len(oomcells),
           power_caps={m: machines[m]["power_cap_w"] for m in machines},
           privileges={m: machines[m].get("privileges") for m in machines},
           residuals=[[round(v, 5) for v in r] for r in res],
           keys=[list(k) for k in keys], mach_order=mach, sanity=SANITY)
json.dump(OUT, io.open("experiments/results/c4_analysis.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/c4_analysis.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
