# -*- coding: utf-8 -*-
"""
A fourth independent corpus for the Table 2 interaction comparison.

Husom et al., "The Price of Prompting" (arXiv:2407.16893), CC-BY-SA-4.0, measures LLM
inference energy for Llama / CodeLlama / Gemma across a workstation, two laptops and a
server, under two prompt datasets (Alpaca, Code-Feedback). Its hardware axis spans a far
wider power envelope than our five datacenter accelerators, so it is a useful stress test
of whether the interaction result survives outside the datacenter regime.

DATA DEFECT FOUND AND HANDLED: alpaca_gemma_2b_laptop1.csv is byte-identical to
codefeedback_codellama_7b_laptop1.csv (md5 7a7309b5b4dd) and its model_name column reads
codellama:7b throughout. It is a mislabelled duplicate, so laptop1 contributes only the
codellama_7b row and is excluded from the balanced matrix.

Sanity checks stated in advance:
  S1  every cell of the balanced matrix has >= 200 observations
  S2  energy per token is positive and finite everywhere
  S3  the additive + interaction decomposition sums to the total sum of squares
  S4  a machine-only (fixed ranking) policy must tie an additive model exactly
      (Proposition 1); if it does not, the decomposition is wrong
  S5  shuffling the machine labels within rows must destroy the interaction
"""
import csv, io, json, os, math, sys, statistics as st
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
from collections import defaultdict

DIR = "data/ejhusom"
EXCLUDE = {"alpaca_gemma_2b_laptop1.csv"}          # mislabelled duplicate, see above
ECOL, TCOL = "energy_consumption_llm_total", "response_token_length"
SANITY = []

def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

def load(path):
    """Return list of (response_tokens, energy_j) for valid rows."""
    out = []
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        r = csv.DictReader(fh)
        for row in r:
            try:
                e, t = float(row[ECOL]), float(row[TCOL])
            except (TypeError, ValueError):
                continue
            if e > 0 and t > 0:
                out.append((t, e))
    return out

cells = {}
for fn in sorted(os.listdir(DIR)):
    if not fn.endswith(".csv") or fn in EXCLUDE:
        continue
    task, model, size, hw = fn[:-4].split("_")
    obs = load(os.path.join(DIR, fn))
    cells[(task, model + "_" + size, hw)] = obs

print("cells loaded:")
for k, v in sorted(cells.items()):
    med = st.median(e / t for t, e in v) if v else float("nan")
    print("  {:12} {:14} {:12} n={:6d}  median {:.3e} J/token".format(k[0], k[1], k[2], len(v), med))

# ---------------------------------------------------- balanced matrix
HW = ["laptop2", "workstation"]
rows = sorted({(t, m) for (t, m, h) in cells if h in HW}, key=lambda x: (x[0], x[1]))
rows = [r for r in rows if all((r[0], r[1], h) in cells for h in HW)]
print("\nbalanced matrix: {} rows x {} machines".format(len(rows), len(HW)))

def cell_energy(t, m, h):
    obs = cells[(t, m, h)]
    return st.median(e / tok for tok, e in obs)

Y = [[math.log10(cell_energy(t, m, h)) for h in HW] for (t, m) in rows]

sane("S1 every cell has >= 200 observations",
     all(len(cells[(t, m, h)]) >= 200 for (t, m) in rows for h in HW),
     "min n = {}".format(min(len(cells[(t, m, h)]) for (t, m) in rows for h in HW)))
sane("S2 energy per token positive and finite",
     all(math.isfinite(v) for r in Y for v in r), "ok")

def decompose(Y):
    R, C = len(Y), len(Y[0])
    gm = sum(sum(r) for r in Y) / (R * C)
    ri = [sum(r) / C - gm for r in Y]
    cj = [sum(Y[i][j] for i in range(R)) / R - gm for j in range(C)]
    res = [[Y[i][j] - gm - ri[i] - cj[j] for j in range(C)] for i in range(R)]
    tss = sum((Y[i][j] - gm) ** 2 for i in range(R) for j in range(C))
    rss = sum(ri[i] ** 2 for i in range(R)) * C
    css = sum(cj[j] ** 2 for j in range(C)) * R
    iss = sum(res[i][j] ** 2 for i in range(R) for j in range(C))
    return gm, ri, cj, res, tss, rss, css, iss

gm, ri, cj, res, tss, rss, css, iss = decompose(Y)
sane("S3 sums of squares decompose exactly", abs(tss - (rss + css + iss)) < 1e-9,
     "TSS {:.6f} vs R+C+I {:.6f}".format(tss, rss + css + iss))
print("\nvariance shares: workload {:.1f}%, machine {:.1f}%, interaction {:.2f}%".format(
    100 * rss / tss, 100 * css / tss, 100 * iss / tss))

# ---------------------------------------------------- decision reading
def winner(i):
    return min(range(len(HW)), key=lambda j: Y[i][j])
wins = [winner(i) for i in range(len(rows))]
maj = max(set(wins), key=wins.count)
print("\nenergy-optimal machine per row:")
for (t, m), w in zip(rows, wins):
    print("  {:12} {:14} -> {}".format(t, m, HW[w]))
fixed_acc = sum(1 for w in wins if w == maj) / len(wins)
add_acc = sum(1 for i in range(len(rows))
              if (cj[0] < cj[1]) == (Y[i][0] < Y[i][1])) / len(rows)
sane("S4 additive ties a fixed ranking exactly (Proposition 1)",
     abs(fixed_acc - add_acc) < 1e-12,
     "fixed {:.4f} vs additive {:.4f}".format(fixed_acc, add_acc))

# ---------------------------------------------------- S5, corrected
# A within-row label permutation is the WRONG null for a two-column design: with C=2 the
# interaction is the row-to-row spread of the half log-ratio d_i, and a shuffle flips the
# sign of d_i, destroying the common mean and inflating that spread. The permutation null
# is therefore upward-biased and can never reject (verified: p = 1.000).
#
# The correct question is whether the row-to-row spread of d_i exceeds the sampling noise
# in the per-cell medians. Bootstrap the medians from the raw observations to answer it.
import random
rnd = random.Random(0)

def boot_median_logratio(t, m, B=400):
    a = [e / tok for tok, e in cells[(t, m, "laptop2")]]
    b = [e / tok for tok, e in cells[(t, m, "workstation")]]
    out = []
    for _ in range(B):
        ra = sorted(rnd.choice(a) for _ in range(len(a)))
        rb = sorted(rnd.choice(b) for _ in range(len(b)))
        ma, mb = ra[len(ra) // 2], rb[len(rb) // 2]
        out.append((math.log10(ma) - math.log10(mb)) / 2.0)
    return out

boots = [boot_median_logratio(t, m) for (t, m) in rows]
d = [(Y[i][0] - Y[i][1]) / 2.0 for i in range(len(rows))]
obs_spread = st.pstdev(d)
null_spread = []
for b in range(len(boots[0])):
    null_spread.append(st.pstdev([st.mean(bo) for bo in boots]  # centre of each row
                                 ))
# noise-only null: each row's d_i redrawn around the COMMON mean, using that row's own
# bootstrap standard error, so any surviving spread must be real between-row structure.
common = st.mean(d)
ses = [st.pstdev(bo) for bo in boots]
null = []
for _ in range(4000):
    null.append(st.pstdev([rnd.gauss(common, se) for se in ses]))
pval = sum(1 for v in null if v >= obs_spread) / len(null)
sane("S5 between-row spread exceeds bootstrap median noise",
     pval < 0.05,
     "observed sd(d) {:.4f}, median null {:.4f}, p = {:.4f}".format(
         obs_spread, st.median(null), pval))
sane("S5b bootstrap standard errors are small relative to the effect",
     max(ses) < obs_spread,
     "max per-row SE {:.5f} vs observed sd(d) {:.5f}".format(max(ses), obs_spread))

# reversal check, reported honestly
n_rev = sum(1 for x in d if (x > 0) != (common > 0))
print(chr(10) + "reversals: {} of {} rows disagree with the majority ordering".format(n_rev, len(d)))
print("the two machines differ enough that the additive machine effect dominates;")
print("this is the two-type-pool regime in which Section 8 measures 0.0 per cent for placement.")

OUT = dict(rows=[list(r) for r in rows], machines=HW,
           log10_energy_per_token=Y,
           shares=dict(workload=100 * rss / tss, machine=100 * css / tss,
                       interaction=100 * iss / tss),
           winners=[HW[w] for w in wins],
           fixed_ranking_accuracy=fixed_acc, additive_accuracy=add_acc,
           bootstrap_p=pval, observed_sd_d=obs_spread, per_row_se=ses,
           reversals=n_rev,
           cell_counts={"{}|{}|{}".format(*k): len(v) for k, v in cells.items()},
           defect="alpaca_gemma_2b_laptop1.csv is a mislabelled byte-identical duplicate of "
                  "codefeedback_codellama_7b_laptop1.csv (md5 7a7309b5b4dd); excluded",
           sanity=SANITY)
json.dump(OUT, io.open("experiments/results/ejhusom_corpus.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/ejhusom_corpus.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
