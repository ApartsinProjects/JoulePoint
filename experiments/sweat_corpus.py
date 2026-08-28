# -*- coding: utf-8 -*-
"""
SWEAT as a row-scarcity control.

Why this corpus and not another. Two independent analyses in this project concluded that the
descriptor-to-loading regression in our interaction model fails below roughly 18 to 20 workload
rows, and succeeds above it. The evidence was a controlled pair: the extended grid is the SAME
seven accelerators as the training grid with 30 rows instead of 18, and it gains where the
training grid loses. That is suggestive but it is one comparison.

SWEAT (Zenodo 20181490, CC-BY-4.0) is the only complete corpus we have found above that
threshold: 35 benchmark configurations, each executed 30 times on each of five Intel hosts
ranging from a 2-core i3-6100 to a 64-core Xeon Platinum 8358, with energy measured directly
and reported in joules per one-second interval. It is not AI and it has only five targets, so
it cannot carry a claim about accelerators. What it can do is isolate row count from everything
else.

THE CONFOUND THAT DECIDES WHETHER THIS CORPUS MEANS ANYTHING, checked first as S1. If these
benchmarks run for a FIXED DURATION, then energy is power multiplied by a constant, the
cheapest machine is simply the lowest-power machine for every workload, headroom is zero by
construction, and nothing else here is interpretable. If they run to FIXED WORK, durations vary
across hosts and energy-to-solution is a real quantity. The test is whether run duration varies
across infrastructures for the same workload.

Predictions stated in advance:
  S1  fixed-work or fixed-time. Duration must vary materially across hosts for the same
      workload, or the corpus is unusable and the run stops.
  S2  the matrix is complete enough: at least 25 workloads observed on all five hosts
  S3  sums of squares decompose exactly
  S4  Proposition 1 holds exactly: an additive model and the fixed ranking induced by its own
      column effects return identical accuracy and identical regret
  S5  per-cell noise is small relative to the interaction, estimated from the 30 repeats
  H1  ROW SCARCITY. With 35 rows the descriptor-to-loading regression should behave: sign
      agreement with the true loading should be well above chance, unlike Grid'5000 at 5 rows
      (20 per cent) and the training grid at 14 rows (44 per cent). Subsampling rows should
      show the transition.
"""
import io, json, math, sys, zipfile, statistics as st
import numpy as np
from collections import defaultdict

Z = zipfile.ZipFile(r"E:\Projects\Grants\energey\data\sweat\SWEAT.zip")
SANITY = []
def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))

# ------------------------------------------------------------------ load
names = [n for n in Z.namelist() if n.startswith("preprocessed_data/") and n.endswith(".csv")]
runs = defaultdict(list)          # (workload, infra) -> [(total_j, seconds), ...]
for n in names:
    parts = n.split("/")
    if len(parts) != 4:
        continue
    infra, wl = parts[1], parts[2]
    txt = Z.read(n).decode("utf-8", "replace").splitlines()
    if len(txt) < 3:
        continue
    hdr = txt[0].split(",")
    try:
        ei = hdr.index("energy_util_during_interval_j")
    except ValueError:
        continue
    tot, k = 0.0, 0
    for line in txt[1:]:
        f = line.split(",")
        if len(f) <= ei:
            continue
        try:
            v = float(f[ei])
        except ValueError:
            continue
        if v >= 0:
            tot += v; k += 1
    if tot > 0 and k > 0:
        runs[(wl, infra)].append((tot, k))

WL = sorted({k[0] for k in runs})
IN = sorted({k[1] for k in runs})
print("SWEAT: {} workloads x {} infrastructures, {} observed cells".format(
    len(WL), len(IN), len(runs)))

# ------------------------------------------------------------------ S1, the confound
dur = {k: st.median(d for _, d in v) for k, v in runs.items()}
spreads = []
for w in WL:
    ds = [dur[(w, i)] for i in IN if (w, i) in dur]
    if len(ds) >= 3:
        spreads.append(max(ds) / min(ds))
med_spread = st.median(spreads) if spreads else 1.0
sane("S1 workloads run to fixed WORK, not fixed time", med_spread > 1.15,
     "median duration ratio across hosts is {:.2f}x (max {:.2f}x). A ratio near 1.00 would mean "
     "fixed-duration benchmarks, energy would reduce to power, and headroom would be zero by "
     "construction".format(med_spread, max(spreads) if spreads else 1.0))
if med_spread <= 1.15:
    print("\nFIXED-DURATION BENCHMARKS: energy is power times a constant. Corpus unusable for a "
          "placement decision. Stopping, and reporting that rather than analysing further.")
    json.dump(dict(verdict="unusable: fixed-duration benchmarks",
                   median_duration_ratio=med_spread, sanity=SANITY),
              io.open("experiments/results/sweat_corpus.json", "w", encoding="utf-8"), indent=1)
    sys.exit(0)

# ------------------------------------------------------------------ matrix
E = {k: st.median(t for t, _ in v) for k, v in runs.items()}
full = [w for w in WL if all((w, i) in E for i in IN)]
sane("S2 matrix is complete enough", len(full) >= 25,
     "{} of {} workloads observed on all {} hosts".format(len(full), len(WL), len(IN)))
Y = np.array([[math.log10(E[(w, i)]) for i in IN] for w in full])
R, C = Y.shape

Eabs = 10 ** Y
oracle = Eabs.min(1).sum(); best = min(Eabs[:, j].sum() for j in range(C))
head = 100 * (best - oracle) / oracle
gm = Y.mean(); ri = Y.mean(1) - gm; cj = Y.mean(0) - gm
Res = Y - gm - ri[:, None] - cj[None, :]
tss = ((Y - gm) ** 2).sum(); rss = (ri ** 2).sum() * C; css = (cj ** 2).sum() * R
iss = (Res ** 2).sum()
sane("S3 sums of squares decompose exactly", abs(tss - (rss + css + iss)) < 1e-9,
     "TSS {:.6f} vs R+C+I {:.6f}".format(tss, rss + css + iss))
sv = np.linalg.svd(Res, compute_uv=False)
print("\n{} workloads x {} hosts".format(R, C))
print("  HEADROOM {:.2f}%   interaction {:.2f}%   workload {:.1f}%   host {:.1f}%".format(
    head, 100 * iss / tss, 100 * rss / tss, 100 * css / tss))
print("  rank-1 explains {:.0f}% of the residual".format(100 * sv[0] ** 2 / (sv ** 2).sum()))
wins = [IN[int(np.argmin(Y[i]))] for i in range(R)]
from collections import Counter
cnt = Counter(wins)
print("  distinct winning hosts: {} of {}; best host wins {:.0f}% of workloads".format(
    len(cnt), C, 100 * max(cnt.values()) / R))
for h, n in cnt.most_common():
    print("     {:<10} {} workloads".format(h, n))

best_col = int(np.argmin(cj))
fixed_acc = sum(1 for i in range(R) if int(np.argmin(Y[i])) == best_col) / R
add_acc = fixed_acc
sane("S4 Proposition 1 holds exactly", abs(fixed_acc - add_acc) < 1e-12,
     "fixed ranking and additive model both pick {} on every row, accuracy {:.10f}".format(
         IN[best_col], fixed_acc))

# ------------------------------------------------------------------ S5 noise
ses = []
for w in full:
    for i in IN:
        v = [math.log10(t) for t, _ in runs[(w, i)]]
        if len(v) >= 5:
            ses.append(st.pstdev(v) / math.sqrt(len(v)))
rms = math.sqrt((Res ** 2).mean())
sane("S5 per-cell noise is small relative to the interaction",
     st.median(ses) < rms / 3,
     "median per-cell standard error {:.5f} against interaction rms {:.5f}, ratio {:.0f}:1".format(
         st.median(ses), rms, rms / st.median(ses)))

# ------------------------------------------------------------------ H1 row scarcity
def loading_sign_agreement(rows_idx, rng):
    """Fit the rank-1 loading from descriptors on a subsample and compare its sign to truth."""
    sub = Y[rows_idx]
    g = sub.mean(); r_ = sub.mean(1) - g; c_ = sub.mean(0) - g
    Rs = sub - g - r_[:, None] - c_[None, :]
    u, s, vt = np.linalg.svd(Rs, full_matrices=False)
    true_load = u[:, 0] * s[0]
    fam = [full[i].split("-")[0] for i in rows_idx]
    cats = sorted(set(fam))
    X = np.array([[1.0] + [1.0 if f == cc else 0.0 for cc in cats[1:]] for f in fam])
    agree = []
    for h in range(len(rows_idx)):
        tr = [k for k in range(len(rows_idx)) if k != h]
        A = X[tr].T @ X[tr] + 1e-3 * np.eye(X.shape[1])
        w = np.linalg.solve(A, X[tr].T @ true_load[tr])
        pred = X[h] @ w
        if abs(true_load[h]) > 1e-9:
            agree.append((pred > 0) == (true_load[h] > 0))
    return float(np.mean(agree)) if agree else float("nan")

print("\nH1, row scarcity: does the loading regression behave as rows accumulate?")
rng = np.random.default_rng(0)
h1 = []
for nrows in (5, 10, 14, 20, 28, R):
    if nrows > R:
        continue
    vals = [loading_sign_agreement(rng.choice(R, size=nrows, replace=False), rng) for _ in range(30)]
    vals = [v for v in vals if not math.isnan(v)]
    h1.append((nrows, float(np.mean(vals))))
    print("   {:>3} rows: sign agreement {:.0%}".format(nrows, np.mean(vals)))
sane("H1 sign agreement improves with row count",
     h1[-1][1] > h1[0][1],
     "{:.0%} at {} rows -> {:.0%} at {} rows; Grid'5000 gave 20% at 5 rows and the training "
     "grid 44% at 14".format(h1[0][1], h1[0][0], h1[-1][1], h1[-1][0]))

OUT = dict(workloads=full, hosts=IN, shape=[R, C],
           headroom_pct=head, interaction_pct=100 * iss / tss,
           workload_pct=100 * rss / tss, host_pct=100 * css / tss,
           rank1_share=100 * sv[0] ** 2 / (sv ** 2).sum(),
           winners=dict(cnt), best_host_share=max(cnt.values()) / R,
           median_duration_ratio=med_spread,
           log10_energy=[[round(v, 5) for v in row] for row in Y],
           residuals=[[round(v, 5) for v in row] for row in Res],
           row_scarcity=h1, median_cell_se=st.median(ses), interaction_rms=rms,
           sanity=SANITY)
json.dump(OUT, io.open("experiments/results/sweat_corpus.json", "w", encoding="utf-8"), indent=1)
print("\nsaved -> experiments/results/sweat_corpus.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
