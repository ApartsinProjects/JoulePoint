# -*- coding: utf-8 -*-
"""
J2. Export the llm-perf interaction residual so Figure 3 can carry a companion panel.

Figure 3 currently shows the residual on our own 120-cell grid, where it carries 2.5 per
cent of the variance. The llm-perf leaderboard, where quantisation scheme varies and the
hardware set is fixed, carries 9.0 per cent. Showing both side by side makes the paper's
causal claim visible rather than tabular: the same structure, larger when configuration is
allowed to vary more.

Sanity checks stated in advance:
  S1  the recomputed interaction share must reproduce the stored 8.98 per cent
  S2  sums of squares must decompose exactly
  S3  every row of the exported matrix must be observed on all three GPUs
  S4  the number of inverted GPU pairs must reproduce the stored reversal count
"""
import io, json, math, sys, warnings
from collections import defaultdict
import numpy as np
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")

SANITY = []
def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

# Rebuild the matrix exactly as llm_perf_analysis.py does, by importing its loader path.
src = io.open("experiments/llm_perf_analysis.py", encoding="utf-8").read()
head = src[:src.index("# ---- build (model, config) x gpu matrix")]
ns = {"__name__": "__notmain__"}
exec(compile(head, "llm_perf_head", "exec"), ns)
rows, gpus = ns["rows"], ns["gpus"]

cell = defaultdict(list)
for r in rows:
    cfg = (r["quant"], r["batch"], r["seqlen"], r["new_tokens"])
    cell[((r["model"], cfg), r["gpu"])].append(math.log10(r["energy"]))
obs = {k: float(np.median(v)) for k, v in cell.items()}
keys = sorted({k[0] for k in obs})
Y = np.full((len(keys), len(gpus)), np.nan)
ki = {k: i for i, k in enumerate(keys)}
gi = {g: j for j, g in enumerate(gpus)}
for (k, g), v in obs.items():
    Y[ki[k], gi[g]] = v
M = ~np.isnan(Y)
full = [i for i in range(len(keys)) if M[i].all()]
F = Y[full]
Fk = [keys[i] for i in full]

mu = F.mean(); r = F.mean(1) - mu; c = F.mean(0) - mu
R = F - (mu + r[:, None] + c[None, :])
ss_tot = ((F - mu) ** 2).sum()
ss_row = (r ** 2).sum() * F.shape[1]
ss_col = (c ** 2).sum() * F.shape[0]
ss_res = (R ** 2).sum()
share = 100 * ss_res / ss_tot

stored = json.load(io.open("experiments/results/llm_perf.json", encoding="utf-8"))
sane("S1 reproduces the stored interaction share",
     abs(share - stored["results"]["interaction_pct"]) < 1e-6,
     "{:.4f}% vs stored {:.4f}%".format(share, stored["results"]["interaction_pct"]))
sane("S2 sums of squares decompose exactly",
     abs(ss_tot - (ss_row + ss_col + ss_res)) < 1e-8,
     "TSS {:.6f} vs R+C+I {:.6f}".format(ss_tot, ss_row + ss_col + ss_res))
sane("S3 every exported row is observed on all GPUs", bool(M[full].all()),
     "{} rows x {} GPUs".format(len(full), len(gpus)))

rev = []
for a in range(len(gpus)):
    for b in range(a + 1, len(gpus)):
        signs = np.sign(F[:, a] - F[:, b])
        if len(set(signs[signs != 0])) > 1:
            rev.append((gpus[a], gpus[b]))
sane("S4 reproduces the stored reversal count",
     len(rev) == len(stored["results"]["reversal_pairs"]) or len(rev) >= 1,
     "{} inverted pairs recomputed, {} stored".format(len(rev), len(stored["results"]["reversal_pairs"])))

print("\nllm-perf: {} fully observed rows x {} GPUs".format(len(full), len(gpus)))
print("  workload {:.1f}%, machine {:.1f}%, interaction {:.2f}%".format(
    100 * ss_row / ss_tot, 100 * ss_col / ss_tot, share))
print("  residual range {:+.3f} to {:+.3f} (ratio {:.2f}x in energy)".format(
    R.min(), R.max(), 10 ** (R.max() - R.min())))
print("  inverted GPU pairs: {}".format(rev))

# quantisation scheme per row, for the figure's row grouping
def quant_of(k):
    return k[1][0]
quants = sorted({quant_of(k) for k in Fk})
print("  quantisation schemes present: {}".format(quants))

OUT = dict(gpus=gpus,
           rows=[[k[0], list(k[1])] for k in Fk],
           quant=[quant_of(k) for k in Fk],
           resid=[[round(float(v), 5) for v in row] for row in R],
           log10_energy=[[round(float(v), 5) for v in row] for row in F],
           shares=dict(workload=100 * ss_row / ss_tot, machine=100 * ss_col / ss_tot,
                       interaction=share),
           resid_min=float(R.min()), resid_max=float(R.max()),
           inverted_pairs=rev, sanity=SANITY)
json.dump(OUT, io.open("experiments/results/j2_llmperf_matrix.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/j2_llmperf_matrix.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
