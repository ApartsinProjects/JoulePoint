# -*- coding: utf-8 -*-
"""
A fifth independent corpus for the Table 2 interaction comparison.

Wilkins, Keshav and Mortier, "Hybrid Heterogeneous Clusters Can Lower the Energy
Consumption of LLM Inference Workloads" (arXiv:2407.00010), repository
github.com/grantwilkins/energy-inference. NO LICENCE FILE: the repo is publicly
readable but unlicensed, so we compute derived statistics and cite the paper
rather than redistribute the data.

WHAT THE DATA IS. The aggregate files plots/independence-test/all-{input,output}-
stats.csv hold one row per instrumented phase of one inference call, with columns
System, Model, Number of Input Tokens, Number of Output Tokens, Runtime (s),
CPU Energy (J), GPU Energy (J). Energy is MEASURED, not TDP-estimated: pyJoules
integrates RAPL package counters (Intel hosts) and the NVML/nvidia-smi power
series (all GPUs); the Apple host integrates the powermetrics GPU power series.

MATRIX. Rows are workloads = (model, sweep, nominal token count):
  - out sweep: prompt fixed, max_new_tokens in {8,16,32,64,128,256,512}
  - in  sweep: max_new_tokens fixed at 32, prompt padded to {8..2048} tokens
Columns are the four hardware platforms: Apple M1 Pro, Palmetto Intel+V100,
Palmetto Intel+A100, Swing AMD+A100. Three models are present (Llama-2 7B,
Mistral 7B, Falcon 7B) but Falcon was never run on the M1, so the four-platform
balanced matrix keeps Llama-2 and Mistral only.

UNIT OF WORK. log10 of median accelerator energy per nominal generated token.
Accelerator (GPU) energy is the primary metric because it is directly measured on
all four platforms and is 94-99% of the total; host CPU energy is used only as a
robustness check, because on the AMD host it is NOT measured per run (see DEFECT).
For the input sweep the generated length is constant at 32, so dividing by it is a
per-row constant and is absorbed exactly by the workload main effect.

DEFECT FOUND. plots/input-test/amd-cpu-energy-calculation.py does not measure the
AMD host's CPU energy per run. It integrates two core-power columns from ONE
AMDuProf timechart and then prorates that single total across every run by
Runtime (s) / total_time, so Swing "CPU Energy (J)" is an estimate that is exactly
proportional to runtime by construction (check S7 confirms r = 1.000000). The same
script also computes Mistral's energy from `falcon_cpu_cores` (indices 73, 74)
instead of `mistral_cpu_cores` -- a copy-paste bug in the released pipeline.
Separately, three of the aggregate CSVs are byte-identical copies of each other
under the same basename in different directories (benign, unlike the mislabelled
duplicate found in the ejhusom corpus); check S6 verifies no file's contents
contradict its name.

Sanity checks stated in advance:
  S1  every cell of the balanced matrix has >= 4 inference observations
  S2  every energy-per-token value is positive and finite
  S3  the additive + interaction decomposition sums to the total sum of squares
  S4  a fixed-hardware-ranking policy must tie an additive model EXACTLY
      (Proposition 1); if it does not, the decomposition is wrong
  S5  with C = 4 columns a within-row label permutation IS a valid null (the
      C = 2 degeneracy that forced a bootstrap on the ejhusom corpus does not
      apply here), so permute machine labels within rows and require the observed
      interaction sum of squares to exceed the null
  S6  duplicate-file audit: md5 every downloaded file and require that each
      file's System and Model columns are consistent with its provenance
  S7  the AMD host's CPU energy is a runtime prorated estimate, so its
      correlation with runtime must be 1.000 while the RAPL-measured hosts'
      must not be; this is a POSITIVE check that the defect is what we claim
  S8  the GPU-only and CPU+GPU variants must give the same qualitative verdict
      (same sign of interaction share ordering, same reversal count)

CORRECTIONS MADE AFTER RUNNING THE CHECKS AS FIRST STATED. Three of the eight
checks above are reported here in corrected form; the original wordings and the
numbers that broke them are preserved in the results JSON.

  S5 was MIS-SPECIFIED and is replaced by S5b. The caution carried over from the
  ejhusom corpus was that a within-row label permutation is invalid at C = 2. That
  was too narrow a statement of the problem. Shuffling machine labels within a row
  randomises the COLUMN assignment, so it destroys the hardware main effect at any
  C and re-attributes that variance to the interaction term. Here the hardware main
  effect is 46.6% of the total sum of squares, so the permutation null sits at a
  median interaction SS of 19.46 against an observed 1.08, and p = 1.0000 by
  construction: 18.88 (hardware SS) + 1.08 (interaction SS) = 19.96, which is
  where the null lands. The permutation test can never reject when the column
  effect is large. S5b instead asks the question that matters: does the observed
  interaction exceed what per-cell median SAMPLING NOISE alone would produce?
  Bootstrap each cell's median from its own observations to get its standard
  error, generate null matrices from the fitted ADDITIVE model plus that noise,
  and compare interaction sums of squares.

  S7 was MIS-SPECIFIED and is replaced by S7b. Pooling all Swing records gave
  r = 0.858, not 1.000, and the check failed. Inspecting the data rather than
  guessing: the prorating constant differs per model AND per measurement campaign
  (input sweep 2.71 / 2.023 / 3.04 W-equivalent per second for Falcon / Llama-2 /
  Mistral, output sweep 3.235 / 1.592 / 7.635), so pooling six different constants
  breaks the linearity. Within each (system, model, campaign) group the Swing ratio
  CPU energy / runtime has EXACTLY zero spread (min == max to full precision) while
  every RAPL-measured group has real spread. S7b tests that, and the defect is
  confirmed more strongly than the original wording claimed.

  S8 FAILED AS STATED and is reported as failed, not weakened away. The two
  variants agree on the ordering and magnitude of all three variance shares
  (interaction 2.663% GPU-only against 2.405% with host CPU added) but the
  reversal count differs by one, 9 against 8. The responsible workload is
  Llama-2 7B, input sweep, 32 prompt tokens, where the M1 Pro and the A100 are
  a near-tie in accelerator energy (log10 0.6618 against 0.6638, a 0.5% gap) and
  adding host CPU energy, which is proportionally larger on the Apple part,
  tips it to the A100. S8a records the share agreement that does hold; S8b
  records the exact-count disagreement and its cause.
"""
import csv, io, json, math, os, random, statistics as st, sys, hashlib
from collections import defaultdict

DIR = "data/wilkins"
IN_F = os.path.join(DIR, "independence-test", "all-input-stats.csv")
OUT_F = os.path.join(DIR, "independence-test", "all-output-stats.csv")
HW = ["M1-Pro", "Palmetto Intel+V100", "Palmetto Intel+A100", "Swing AMD+A100"]
SANITY = []


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))


# ------------------------------------------------------------------ load
def load(path, sweep):
    """Return list of dicts, one per inference call (startup phases dropped)."""
    rows = []
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if "inference" not in r["Phase"]:
                continue
            try:
                cpu, gpu = float(r["CPU Energy (J)"]), float(r["GPU Energy (J)"])
                rt = float(r["Runtime (s)"])
                nin = int(float(r["Number of Input Tokens"]))
                nout = int(float(r["Number of Output Tokens"]))
            except (TypeError, ValueError, KeyError):
                continue
            if gpu <= 0 or rt <= 0:
                continue
            level = nin if sweep == "in" else nout
            work = 32 if sweep == "in" else nout   # nominal generated tokens
            rows.append(dict(system=r["System"], model=r["Model"], sweep=sweep,
                             level=level, gpu=gpu, cpu=cpu, runtime=rt, work=work))
    return rows


records = load(IN_F, "in") + load(OUT_F, "out")
print("inference records loaded: {}".format(len(records)))

# ------------------------------------------------------------------ S6 file audit
digests = {}
for root, _, files in os.walk(DIR):
    for fn in files:
        if fn.endswith(".csv"):
            p = os.path.join(root, fn)
            digests[os.path.relpath(p, DIR).replace("\\", "/")] = \
                hashlib.md5(open(p, "rb").read()).hexdigest()
dupe_groups = defaultdict(list)
for k, v in digests.items():
    dupe_groups[v].append(k)
dupes = {v: sorted(ks) for v, ks in dupe_groups.items() if len(ks) > 1}
# a duplicate is BENIGN only if every copy shares the same basename
mislabelled = [ks for ks in dupes.values()
               if len({os.path.basename(k) for k in ks}) > 1]
sane("S6 no mislabelled byte-identical duplicate", not mislabelled,
     "{} duplicate group(s), all same-basename copies: {}".format(
         len(dupes), "; ".join(",".join(v) for v in dupes.values()) or "none"))

# ------------------------------------------------------------------ S7 AMD defect
def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")

cpu_rt_corr = {}
for s in HW:
    z = [r for r in records if r["system"] == s and r["cpu"] > 0]
    cpu_rt_corr[s] = pearson([r["runtime"] for r in z], [r["cpu"] for r in z]) if len(z) > 3 else float("nan")
sane("S7 (as first stated, POOLED) AMD CPU energy correlates 1.000 with runtime",
     cpu_rt_corr["Swing AMD+A100"] > 0.999999,
     "pooled Swing r = {:.6f}; mis-specified, six different prorating constants "
     "are pooled, see S7b".format(cpu_rt_corr["Swing AMD+A100"]))

# S7b: the correct form -- constancy of CPU energy / runtime WITHIN each group
groups = defaultdict(list)
for r in records:
    if r["cpu"] > 0:
        groups[(r["system"], r["model"], r["sweep"])].append(r["cpu"] / r["runtime"])
ratio_spread = {}
for k, v in sorted(groups.items()):
    if len(v) < 4:
        continue
    ratio_spread["|".join(k)] = dict(n=len(v), lo=min(v), hi=max(v),
                                     rel_spread=(max(v) - min(v)) / st.median(v))
    print("    {:22} {:14} {:3}  n={:3d}  J/s in [{:.4g}, {:.4g}]  rel spread {:.3e}".format(
        k[0], k[1], k[2], len(v), min(v), max(v), ratio_spread["|".join(k)]["rel_spread"]))
swing = [v for k, v in ratio_spread.items() if k.startswith("Swing")]
rapl = [v for k, v in ratio_spread.items() if not k.startswith("Swing")]
# threshold is 1e-6, not exact zero: the ratio is constant only to the ~10 significant
# figures the aggregate CSV preserves, so the residual spread is 1e-9 float round-trip
sane("S7b AMD host CPU energy is runtime prorated (zero within-group spread to CSV "
     "precision), RAPL hosts are not",
     all(g["rel_spread"] < 1e-6 for g in swing) and
     all(g["rel_spread"] > 1e-3 for g in rapl),
     "max Swing rel spread {:.2e} over {} groups, min other {:.3f} over {} groups".format(
         max(g["rel_spread"] for g in swing), len(swing),
         min(g["rel_spread"] for g in rapl), len(rapl)))

# ------------------------------------------------------------------ cells
def build_cells(metric):
    cells = defaultdict(list)
    for r in records:
        e = r["gpu"] if metric == "gpu" else r["gpu"] + r["cpu"]
        if e <= 0:
            continue
        cells[(r["model"], r["sweep"], r["level"], r["system"])].append(e / r["work"])
    return cells


def balanced_rows(cells, hw, min_n):
    rows = sorted({k[:3] for k in cells},
                  key=lambda w: (w[1], w[0], w[2]))
    return [w for w in rows
            if all(len(cells.get(w + (h,), [])) >= min_n for h in hw)]


MIN_N = 4
cells = build_cells("gpu")
all_rows = sorted({k[:3] for k in cells}, key=lambda w: (w[1], w[0], w[2]))
rows = balanced_rows(cells, HW, MIN_N)
dens = sum(1 for w in all_rows for h in HW if len(cells.get(w + (h,), [])) >= MIN_N)
print("\nfull grid: {} candidate workloads x {} platforms, density {}/{} = {:.1%}".format(
    len(all_rows), len(HW), dens, len(all_rows) * len(HW), dens / (len(all_rows) * len(HW))))
print("balanced 4-platform matrix: {} workloads".format(len(rows)))


def matrix(cells, rows, hw):
    return [[math.log10(st.median(cells[w + (h,)])) for h in hw] for w in rows]


Y = matrix(cells, rows, HW)
counts = {"{}|{}|{}|{}".format(*(w + (h,))): len(cells[w + (h,)]) for w in rows for h in HW}
sane("S1 every cell has >= {} observations".format(MIN_N),
     min(counts.values()) >= MIN_N, "min n = {}, median n = {}".format(
         min(counts.values()), int(st.median(counts.values()))))
sane("S2 energy per token positive and finite",
     all(math.isfinite(v) for r in Y for v in r) and
     all(st.median(cells[w + (h,)]) > 0 for w in rows for h in HW), "ok")


# ------------------------------------------------------------------ decomposition
def decompose(Y):
    R, C = len(Y), len(Y[0])
    gm = sum(sum(r) for r in Y) / (R * C)
    ri = [sum(r) / C - gm for r in Y]
    cj = [sum(Y[i][j] for i in range(R)) / R - gm for j in range(C)]
    res = [[Y[i][j] - gm - ri[i] - cj[j] for j in range(C)] for i in range(R)]
    tss = sum((Y[i][j] - gm) ** 2 for i in range(R) for j in range(C))
    rss = sum(x * x for x in ri) * C
    css = sum(x * x for x in cj) * R
    iss = sum(v * v for r in res for v in r)
    return gm, ri, cj, res, tss, rss, css, iss


gm, ri, cj, res, tss, rss, css, iss = decompose(Y)
sane("S3 sums of squares decompose exactly", abs(tss - (rss + css + iss)) < 1e-9,
     "TSS {:.8f} vs R+C+I {:.8f}".format(tss, rss + css + iss))
shares = dict(workload=100 * rss / tss, hardware=100 * css / tss,
              interaction=100 * iss / tss)
print("\nvariance shares: workload {:.2f}%, hardware {:.2f}%, interaction {:.3f}%".format(
    shares["workload"], shares["hardware"], shares["interaction"]))
print("hardware main effects (log10 J/token, centred):")
for j, h in enumerate(HW):
    print("   {:22} {:+.4f}".format(h, cj[j]))


# ------------------------------------------------------------------ decisions
def winner(row):
    return min(range(len(row)), key=lambda j: row[j])


wins = [winner(r) for r in Y]
best_fixed = min(range(len(HW)), key=lambda j: cj[j])
maj = max(set(wins), key=wins.count)
print("\nenergy-optimal platform per workload:")
for w, k in zip(rows, wins):
    print("   {:14} {:4} {:5d} -> {}".format(w[0], w[1], w[2], HW[k]))
n_rev = sum(1 for k in wins if k != maj)
print("ranking reversals: {} of {} workloads deviate from the majority winner {}".format(
    n_rev, len(rows), HW[maj]))

fixed_acc = sum(1 for k in wins if k == best_fixed) / len(wins)
add_pred = [winner(cj) for _ in rows]           # additive model: row effect is common
add_acc = sum(1 for k, p in zip(wins, add_pred) if k == p) / len(wins)
sane("S4 additive ties a fixed ranking exactly (Proposition 1)",
     abs(fixed_acc - add_acc) < 1e-15,
     "fixed {:.10f} vs additive {:.10f}".format(fixed_acc, add_acc))

# ------------------------------------------------------------------ S5 permutation
# Reported for completeness and then discarded: see the CORRECTIONS block. Shuffling
# machine labels within a row destroys the hardware main effect at ANY number of
# columns, not only at C = 2, so the null absorbs hardware SS into interaction SS.
rnd = random.Random(0)
null_perm = []
for _ in range(20000):
    Yp = []
    for r in Y:
        rr = list(r)
        rnd.shuffle(rr)
        Yp.append(rr)
    null_perm.append(decompose(Yp)[7])
p_perm = sum(1 for v in null_perm if v >= iss) / len(null_perm)
sane("S5 (as first stated) interaction exceeds the within-row permutation null",
     p_perm < 0.05,
     "observed ISS {:.5f}, median null {:.5f}, p = {:.4f}; mis-specified, the null "
     "median {:.2f} matches hardware SS {:.2f} + interaction SS {:.2f} = {:.2f}, "
     "so the test cannot reject. Replaced by S5b.".format(
         iss, st.median(null_perm), p_perm, st.median(null_perm), css, iss, css + iss))

# S5b: bootstrap noise null. Is the interaction bigger than per-cell median noise?
def boot_se(vals, B=600):
    n = len(vals)
    out = []
    for _ in range(B):
        s = sorted(rnd.choice(vals) for _ in range(n))
        m = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        out.append(math.log10(m))
    return st.pstdev(out)


ses = [[boot_se(cells[rows[i] + (HW[j],)]) for j in range(len(HW))]
       for i in range(len(rows))]
null_boot = []
for _ in range(4000):
    Yn = [[gm + ri[i] + cj[j] + rnd.gauss(0.0, ses[i][j]) for j in range(len(HW))]
          for i in range(len(rows))]
    null_boot.append(decompose(Yn)[7])
p_boot = sum(1 for v in null_boot if v >= iss) / len(null_boot)
sane("S5b interaction exceeds the per-cell bootstrap noise null",
     p_boot < 0.05,
     "observed ISS {:.5f}, median null {:.5f}, p = {:.4f}, max cell SE {:.4f}".format(
         iss, st.median(null_boot), p_boot, max(max(r) for r in ses)))
pval = p_boot

# ------------------------------------------------------------------ routing gain
def routing_gain(Y, subset):
    """Per-workload oracle routing vs best single platform, on linear energy."""
    lin = [[10 ** Y[i][j] for j in subset] for i in range(len(Y))]
    tot_fixed = min(sum(col) for col in zip(*lin))
    tot_oracle = sum(min(r) for r in lin)
    return 100 * (1 - tot_oracle / tot_fixed)


gain_all = routing_gain(Y, list(range(len(HW))))
dc = [HW.index("Palmetto Intel+V100"), HW.index("Palmetto Intel+A100"),
      HW.index("Swing AMD+A100")]
gain_dc = routing_gain(Y, dc)
pair = [HW.index("Palmetto Intel+V100"), HW.index("Palmetto Intel+A100")]
gain_a100v100 = routing_gain(Y, pair)


def reversals_in(subset):
    w = [min(subset, key=lambda j: Y[i][j]) for i in range(len(Y))]
    m = max(set(w), key=w.count)
    return sum(1 for k in w if k != m), HW[m]


rev_dc, maj_dc = reversals_in(dc)
rev_pair, maj_pair = reversals_in(pair)
print("\nreversals restricted to the datacenter platforms: {}/{} (majority {})".format(
    rev_dc, len(Y), maj_dc))
print("reversals restricted to the V100+A100 pair:        {}/{} (majority {})".format(
    rev_pair, len(Y), maj_pair))
print("\noracle routing gain over the best single platform:")
print("   all 4 platforms          {:.2f}%".format(gain_all))
print("   3 datacenter platforms   {:.2f}%".format(gain_dc))
print("   V100 + A100 pair         {:.2f}%".format(gain_a100v100))

# ------------------------------------------------------------------ S8 robustness
cells2 = build_cells("gpu+cpu")
rows2 = balanced_rows(cells2, HW, MIN_N)
Y2 = matrix(cells2, rows2, HW)
d2 = decompose(Y2)
shares2 = dict(workload=100 * d2[5] / d2[4], hardware=100 * d2[6] / d2[4],
               interaction=100 * d2[7] / d2[4])
wins2 = [winner(r) for r in Y2]
maj2 = max(set(wins2), key=wins2.count)
n_rev2 = sum(1 for k in wins2 if k != maj2)
print("\nCPU+GPU variant: workload {:.2f}%, hardware {:.2f}%, interaction {:.3f}%, "
      "reversals {}/{}".format(shares2["workload"], shares2["hardware"],
                               shares2["interaction"], n_rev2, len(rows2)))
flips = [(rows[i], HW[wins[i]], HW[wins2[i]]) for i in range(len(rows))
         if rows2 == rows and wins[i] != wins2[i]]
sane("S8a GPU-only and CPU+GPU variants agree on the variance shares",
     rows2 == rows and
     sorted(shares, key=shares.get) == sorted(shares2, key=shares2.get) and
     abs(shares2["interaction"] - shares["interaction"]) < 1.0,
     "interaction {:.3f}% vs {:.3f}%, same share ordering".format(
         shares["interaction"], shares2["interaction"]))
sane("S8b GPU-only and CPU+GPU variants agree on the exact reversal count",
     n_rev2 == n_rev,
     "reversals {} vs {}; the one workload that flips is {} where the two winners "
     "are a near-tie in accelerator energy".format(
         n_rev, n_rev2, flips[0][0] if flips else "none"))
for w, a, b in flips:
    i = rows.index(w)
    print("    flip: {} {} {} -> {}  log10 J/tok {} vs {}".format(
        w[0], w[1], w[2], "{} -> {}".format(a, b),
        [round(x, 4) for x in Y[i]], [round(x, 4) for x in Y2[i]]))

# ------------------------------------------------------------------ extremes
resid_flat = sorted(((abs(res[i][j]), i, j) for i in range(len(rows))
                     for j in range(len(HW))), reverse=True)
print("\nlargest interaction residuals:")
for a, i, j in resid_flat[:6]:
    print("   {:14} {:4} {:5d} on {:22} resid {:+.4f} (n={})".format(
        rows[i][0], rows[i][1], rows[i][2], HW[j], res[i][j],
        counts["{}|{}|{}|{}".format(*(rows[i] + (HW[j],)))]))

# ------------------------------------------------------------------ output
OUT = dict(
    corpus="wilkins-2407.00010",
    licence="NONE: repository has no LICENSE file; GitHub default is all rights "
            "reserved. Derived statistics computed here are ours; the raw CSVs are "
            "not redistributed.",
    energy_is_measured=True,
    measurement="pyJoules RAPL package counters (Intel hosts) + NVML/nvidia-smi power "
                "integration (all GPUs); powermetrics GPU power integration (Apple M1 Pro)",
    hardware=HW,
    unit="log10 median accelerator energy (J) per nominal generated token",
    min_obs_per_cell=MIN_N,
    grid=dict(candidate_workloads=len(all_rows), platforms=len(HW),
              filled_cells=dens, density=dens / (len(all_rows) * len(HW))),
    rows=[list(w) for w in rows],
    log10_energy_per_token=Y,
    cell_counts=counts,
    cell_medians={"{}|{}|{}|{}".format(*(w + (h,))): st.median(cells[w + (h,)])
                  for w in rows for h in HW},
    cell_values={"{}|{}|{}|{}".format(*(w + (h,))): cells[w + (h,)]
                 for w in rows for h in HW},
    grand_mean=gm, workload_effects=ri, hardware_effects=cj, residuals=res,
    sums_of_squares=dict(total=tss, workload=rss, hardware=css, interaction=iss),
    shares=shares,
    winners=[HW[k] for k in wins],
    majority_winner=HW[maj], best_fixed_platform=HW[best_fixed],
    reversals=n_rev, n_workloads=len(rows),
    reversals_datacenter_only=rev_dc, reversals_v100_a100_pair=rev_pair,
    fixed_ranking_accuracy=fixed_acc, additive_accuracy=add_acc,
    bootstrap_noise_p=p_boot, bootstrap_null_median_iss=st.median(null_boot),
    per_cell_bootstrap_se=ses,
    discarded_permutation_test=dict(
        p=p_perm, null_median_iss=st.median(null_perm),
        why="within-row label permutation destroys the hardware main effect at any C, "
            "not only at C=2, so the null absorbs hardware SS into interaction SS"),
    routing_gain_percent=dict(all_four=gain_all, datacenter_three=gain_dc,
                              v100_a100_pair=gain_a100v100),
    paper_claim_routing_gain_percent=7.5,
    cpu_runtime_correlation=cpu_rt_corr,
    cpu_runtime_ratio_by_group=ratio_spread,
    cpu_plus_gpu_variant=dict(shares=shares2, reversals=n_rev2,
                              log10_energy_per_token=Y2,
                              winner_flips=[[list(w), a, b] for w, a, b in flips]),
    file_md5=digests,
    duplicate_groups={k: v for k, v in dupes.items()},
    defects=[
        "Swing AMD+A100 'CPU Energy (J)' is not measured per run: "
        "plots/input-test/amd-cpu-energy-calculation.py prorates a single AMDuProf "
        "two-core integral across all runs by runtime, giving r = {:.6f} with runtime."
        .format(cpu_rt_corr["Swing AMD+A100"]),
        "The same script computes Mistral's CPU energy from falcon_cpu_cores "
        "(indices 73, 74) rather than mistral_cpu_cores: a copy-paste bug.",
        "Falcon 7B was never run on the Apple M1 Pro, so the four-platform matrix "
        "is restricted to Llama-2 7B and Mistral 7B.",
        "all-input-stats.csv and all-output-stats.csv are each duplicated verbatim "
        "in two or three directories; all copies share a basename, so unlike the "
        "ejhusom corpus none is mislabelled.",
    ],
    sanity=SANITY,
)
os.makedirs("experiments/results", exist_ok=True)
json.dump(OUT, io.open("experiments/results/wilkins_corpus.json", "w", encoding="utf-8"),
          indent=1)
print("\nsaved -> experiments/results/wilkins_corpus.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
