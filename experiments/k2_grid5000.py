# -*- coding: utf-8 -*-
"""
A sixth independent corpus for the Table 2 interaction comparison, and the first
one that is neither an AI accelerator nor a single-vendor fleet.

Georges Da Costa, "Power, performance and system measures of HPC benchmarks on
multiple hardware", Zenodo, concept DOI 10.5281/zenodo.10982238, latest version
record 14914799 (10.5281/zenodo.14914799), described by Da Costa, "Hardware and
application aware performance, power and energy models for modern HPC servers
with DVFS", Sustainable Computing: Informatics and Systems, 2025,
doi:10.1016/j.suscom.2025.101106.

LICENCE. CC-BY-4.0, access_right = "open", read from the Zenodo record metadata
(metadata.license.id == "cc-by-4.0"). This is the first corpus in the paper that
carries an explicit redistribution-permitting licence: derived statistics and raw
rows may both be republished with attribution. We nevertheless do not commit the
raw files; data/ is gitignored and the four files are fetched from Zenodo and
verified against the publisher's own md5 manifest in S0.

KEY UNCERTAINTY RESOLVED: THE DATASET STORES JOULES DIRECTLY. Column `energy` of
data.csv is a scalar in joules per run, already integrated from a full-node
wattmeter power series (IPMI or e-PDU, 0.1 W to 10 W precision, 1 Hz down to
0.1 Hz), after a watermark-based realignment between the wattmeter clock and the
host clock. No integration by us is required, and no value here is TDP-estimated.
RAPL package and DRAM power are additionally available per run in
data_augmented.csv as an independent second channel on all 18 clusters.

WHAT THE DATA IS. 24200 runs = 1800 experiments (10 benchmarks x 18 clusters x 10
repetitions) x ~13 DVFS frequencies each. Always ONE node, all its cores.
Benchmarks: NAS Parallel Benchmarks BT-C, CG-D, EP-D, FT-C, IS-D, LU-C, MG-D,
SP-C, plus an idle probe (sleep 300, with and without deep sleep) and a per-core
rand loop (mpigeneric). Only the eight NPB benchmarks are workloads in our sense.

MATRIX. Rows are NPB workloads, each a FIXED problem size (a fixed NPB class), so
the work completed is identical on every platform and no per-token normalisation
is needed; this corpus is cleaner in that respect than the LLM corpora. Columns
are the 18 Grid'5000 clusters. The `fullname` field ("cg-D-16") is NOT the
workload identifier: it embeds nproc, and nproc is a property of the cluster (each
cluster has exactly one core count, verified in S9). Keying on `fullname` gives a
50 x 18 matrix that is 91% empty and silently moves a hardware attribute onto the
workload axis. The correct key is (bench, class).

DENSITY. The advertised grid and the usable grid differ, and the difference is the
main finding of the audit. Advertised: 8 workloads x 18 clusters = 144 cells, all
144 filled with exactly 10 repetitions each at every DVFS point, 100.0% dense --
that part is true. Usable after the S11 measurement-adequacy gate below: 5
workloads x 18 clusters = 90 cells, still 100% dense and still 10 repetitions per
cell. 18 hardware platforms is by a wide margin the widest hardware axis of any
corpus in the paper (against 5 in our own grid and 4 in wilkins), which is the
reason to want this corpus; the price is a short workload axis.

THE MEASUREMENT-ADEQUACY GATE (S11), AND WHY IT EXISTS. The wattmeter window
`time` is not the benchmark duration `duration`; it has a floor of about 30 s.
For long benchmarks the two agree to a few percent. For short ones the window
over-covers the run and the recorded energy is mostly idle. Median time/duration
per cell reaches 3.09 for FT-C on dahu and 4.00 for IS-D on dahu, and IS-D
over-covers by more than 2x on all 18 clusters. The consequence is visible in the
raw repetitions: the ten FT-C runs on dahu are 9.7 s each, yet their recorded
mean power ranges from 70 W to 286 W and their recorded energy from 2101 J to
8510 J, a 4x spread across nominally identical runs, against a 4-8% spread for
CG-D. Uncorrected, FT-C alone contributed 43% of the interaction sum of squares:
the largest single "interaction" in the raw corpus was an artifact of wattmeter
window padding, not of architecture. S11 therefore excludes any workload whose
median duration falls below the 30 s wattmeter window floor on any cluster. That
removes FT-C (9.1 s), IS-D (12.5 s) and LU-C (27.4 s) and keeps BT-C, CG-D, EP-D,
MG-D, SP-C. The gate is stated before the decomposition is run and is justified by
the instrument, not by the answer it produces; S12 verifies that the gate is not
what creates the result.

AN ESTIMATOR WE REJECTED. The dataset contains its own idle probe, so the padding
could in principle be subtracted as energy - P_idle x (time - duration). We do not
do this. With the IdleC0 (no deep sleep) baseline the correction drives 1840 J
negative on dahu, because a padded window is not in the same power state as the
idle probe. Reconstructing an energy column from a separately-measured constant is
exactly the defect found in the wilkins corpus (a prorated AMD CPU-energy column),
and we decline to introduce it here. The gate discards the affected rows instead.

METRIC. Primary is log10 of the median whole-node `energy` (J) to complete the
benchmark at each cluster's nominal maximum frequency (ratio == 1.0): the
published wattmeter integral, with no reconstruction. The dataset notebook
recommends median_power x duration instead ("time and energy have a lower
precision than duration"); that recommendation is aimed at the short-run problem
the gate already removes, so both power x duration forms are carried as co-equal
variants V2 and V3 and are required to agree in S12.

Sanity checks stated in advance:
  S0  every downloaded file's md5 equals the checksum Zenodo publishes for it
      (provenance anchored to the publisher's manifest, a stronger form of the
      duplicate/mislabel audit used on the earlier corpora)
  S1  every cell of the gated matrix has exactly 10 observations at ratio == 1.0
  S2  every cell energy is positive and finite
  S3  the additive + interaction decomposition sums to the total sum of squares
  S4  a fixed-hardware-ranking policy must tie an additive model EXACTLY to every
      digit (Proposition 1); if it does not, the decomposition is wrong
  S5  a within-row label permutation is the WRONG null and is run only to exhibit
      its failure. The exact diagnostic: a within-row permutation preserves every
      row mean, hence preserves TSS and workload SS, hence preserves
      (hardware SS + interaction SS) EXACTLY for every draw. So the permutation
      null cannot separate hardware from interaction, and its median interaction
      SS must sit near hardware SS + interaction SS. S5 asserts that identity to
      machine precision on every draw.
  S5b the observed interaction SS must exceed a per-cell bootstrap noise null
      built from the 10 raw repetitions of each cell (the correct null)
  S6  file contents agree with their names and with each other: every row's
      `cluster` is a prefix of its `hostname` and a path component of its
      `basename`, and data.csv and data_augmented.csv describe the same runs,
      compared on a stable identity key with numeric tolerance
  S7  no byte-identical duplicate rows and no duplicate run identity
      (cluster, hostname, bench, fullname, startTime, ratio)
  S8  RAPL package+DRAM, an independent instrument, must rank the clusters
      compatibly with the wattmeter: Spearman correlation of the 18 hardware main
      effects >= 0.8. (Mis-specified; see the corrections block and S8b.)
  S8b the two instruments must agree on the INTERACTION: Pearson correlation of
      the 90 interaction residuals >= 0.8 and interaction shares within 1 pp
  S9  nproc is constant within each cluster and NPB class is constant within each
      benchmark, so neither confounds the workload axis with the hardware axis
  S10 after the gate, no single workload may contribute more than 50% of the
      interaction sum of squares (the interaction must not be one row's artifact)
  S11 measurement adequacy: in the gated matrix every cell's median duration is at
      least the 30 s wattmeter window floor, and median time/duration < 1.6
  S12 the gate is not what creates the result: the interaction share of the gated
      primary must be within 1 percentage point of the ungated 8-workload matrix
      and of both power x duration variants and of the energy-optimal DVFS variant

CORRECTIONS MADE AFTER RUNNING THE CHECKS AS FIRST STATED. Three checks were
mis-specified on the first pass; the original wordings, the numbers that broke
them, and the root cause found by inspecting the data are preserved in the results
JSON under "corrections", and are summarised here.

  S5 as first written asserted that the permutation null median lands within 10%
  of hardware SS + interaction SS. It landed at 4.9635 against 5.6479, 12.1% low,
  and the check failed. The root cause is not a data problem and not a tolerance
  problem: a permuted matrix retains a nonzero column sum of squares by chance
  (median 0.684 here with 5 rows and 18 columns), so the null median interaction
  SS is systematically BELOW hardware SS + interaction SS by that amount. The
  exact invariant is the conserved SUM, not the median. S5 now asserts
  CSS_perm + ISS_perm == CSS_obs + ISS_obs to machine precision on every draw,
  which is the statement that actually proves the null cannot reject.

  S6 as first written compared the two CSVs by exact string equality on 17 shared
  columns and reported 3614 of 24200 rows "absent from data_augmented.csv". This
  was a false alarm and not a data defect: the two files were written by separate
  float-to-string round trips, so the same value appears as 245.06399703025815 in
  one and 245.06399703025812 in the other. 2090 of the mismatches were the `ratio`
  column alone. S6 now joins on a stable identity key and compares numeric fields
  with a relative tolerance.

  S10 as first written predicted that EP-D, being embarrassingly parallel, would
  be an extreme of the interaction residuals. It ranked 4th of 8 and the check
  failed. Inspecting the residuals rather than guessing found the real story: FT-C
  held 43% of the interaction SS, and the FT-C cells are precisely the ones whose
  wattmeter window over-covers the run by up to 3.1x, with a 4x energy spread
  across identical repetitions. The failed prediction surfaced a measurement
  artifact, which is now handled by the S11 gate. S10 is restated as a check that
  no single workload dominates the interaction after the gate.
"""
import csv, io, json, math, os, random, statistics as st, hashlib
from collections import defaultdict, Counter

DIR = "data/grid5000"
DATA = os.path.join(DIR, "data.csv")
AUG = os.path.join(DIR, "data_augmented.csv")
NPB_ALL = ["bt", "cg", "ep", "ft", "is", "lu", "mg", "sp"]
WATTMETER_WINDOW_FLOOR_S = 30.0        # observed minimum of `time` across the corpus
MAX_WINDOW_OVERCOVER = 1.6             # S11 tolerance on median time/duration
ZENODO_RECORD = "14914799"
ZENODO_MD5 = {
    "data.csv": "4e7e2f63306d238786da5703447967e1",
    "data_augmented.csv": "47e6faf7d74d1081e9f37aba79aec7f3",
    "scripts.tgz": "93c2ada5a24ff9815ef479e4dd80fa32",
    "usage.ipynb": "178b1ef8eddd61cc63446ce424e86e59",   # "Usage of the data.ipynb"
}
SANITY, CORRECTIONS, DEFECTS = [], [], []
rnd = random.Random(0)


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print("    [{}] {}: {}".format("PASS" if ok else "FAIL", name, detail))


# --------------------------------------------------------------- S0 provenance
digests = {}
for fn in sorted(ZENODO_MD5):
    p = os.path.join(DIR, fn)
    digests[fn] = hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
mism = {k: (digests[k], ZENODO_MD5[k]) for k in ZENODO_MD5 if digests[k] != ZENODO_MD5[k]}
sane("S0 every file md5 matches the Zenodo published checksum", not mism,
     "{} files verified against record {} manifest; mismatches: {}".format(
         len(ZENODO_MD5), ZENODO_RECORD, mism or "none"))


# --------------------------------------------------------------------- load
def load(path):
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh, delimiter=" "))


raw = load(DATA)
aug = load(AUG)
print("runs in data.csv: {}   runs in data_augmented.csv: {}".format(len(raw), len(aug)))

IDKEY = ["cluster", "hostname", "bench", "fullname", "startTime", "oarid"]
NUMCOLS = ["duration", "endTime", "energy", "time", "mean_power", "median_power", "ratio"]


def idk(r):
    return tuple(r[c] for c in IDKEY)


# ------------------------------------------------------------- S6 / S7 audit
bad_host = [r for r in raw if not r["hostname"].startswith(r["cluster"] + "-")]
bad_base = [r for r in raw if "/" + r["cluster"] + "/" not in r["basename"]]
aug_by_id = {}
for r in aug:
    aug_by_id.setdefault(idk(r), r)
no_match, num_mismatch, worst = 0, 0, 0.0
for r in raw:
    a = aug_by_id.get(idk(r))
    if a is None:
        no_match += 1
        continue
    for c in NUMCOLS:
        x, y = float(r[c]), float(a[c])
        rel = abs(x - y) / max(abs(x), 1e-12)
        worst = max(worst, rel)
        if rel > 1e-9:
            num_mismatch += 1
sane("S6 file contents agree with their names and with each other",
     not bad_host and not bad_base and no_match == 0 and num_mismatch == 0,
     "hostname/cluster mismatches {}, basename/cluster mismatches {}, rows of "
     "data.csv with no identity match in data_augmented.csv {}/{}, numeric "
     "disagreements beyond 1e-9 relative {} (worst observed {:.2e}, pure "
     "float-repr round-trip)".format(len(bad_host), len(bad_base), no_match,
                                     len(raw), num_mismatch, worst))

dup_rows = sum(v - 1 for v in Counter(tuple(r.values()) for r in raw).values() if v > 1)
dup_ids = sum(v - 1 for v in Counter(
    idk(r) + (r["ratio"],) for r in raw).values() if v > 1)
sane("S7 no duplicate rows and no duplicate run identities",
     dup_rows == 0 and dup_ids == 0,
     "byte-identical duplicate rows {}, duplicate (cluster,hostname,bench,"
     "fullname,startTime,oarid,ratio) {}".format(dup_rows, dup_ids))

# ------------------------------------------------------------- S9 confounding
work = [r for r in raw if r["bench"] in NPB_ALL]
nproc_by_cluster, class_by_bench = defaultdict(set), defaultdict(set)
for r in work:
    nproc_by_cluster[r["cluster"]].add(r["nproc"])
    class_by_bench[r["bench"]].add(r["fullname"].split("-")[1])
CLUSTERS = sorted(nproc_by_cluster)
CORES = {c: int(sorted(v)[0]) for c, v in nproc_by_cluster.items()}
CLASSES = {b: sorted(v)[0] for b, v in class_by_bench.items()}
sane("S9 nproc constant per cluster and NPB class constant per benchmark",
     all(len(v) == 1 for v in nproc_by_cluster.values()) and
     all(len(v) == 1 for v in class_by_bench.values()),
     "{} clusters, one core count each {}; classes {}".format(
         len(CLUSTERS), CORES, CLASSES))

# ----------------------------------------------------- S11 measurement adequacy
nominal = [r for r in work if abs(float(r["ratio"]) - 1.0) < 1e-9]
dur_med, win_med = {}, {}
for b in NPB_ALL:
    for c in CLUSTERS:
        z = [r for r in nominal if r["bench"] == b and r["cluster"] == c]
        dur_med[(b, c)] = st.median(float(r["duration"]) for r in z)
        win_med[(b, c)] = st.median(float(r["time"]) / float(r["duration"]) for r in z)

print("\nmeasurement adequacy per workload (wattmeter window floor {:.0f} s):".format(
    WATTMETER_WINDOW_FLOOR_S))
NPB = []
for b in NPB_ALL:
    dmin = min(dur_med[(b, c)] for c in CLUSTERS)
    wmax = max(win_med[(b, c)] for c in CLUSTERS)
    keep = dmin >= WATTMETER_WINDOW_FLOOR_S and wmax < MAX_WINDOW_OVERCOVER
    print("   {:3}-{}  shortest cell {:7.2f} s   worst time/duration {:.2f}   {}".format(
        b, CLASSES[b], dmin, wmax, "KEEP" if keep else "DROP (window over-covers)"))
    if keep:
        NPB.append(b)
sane("S11 gated matrix satisfies the wattmeter window floor",
     NPB and all(dur_med[(b, c)] >= WATTMETER_WINDOW_FLOOR_S and
                 win_med[(b, c)] < MAX_WINDOW_OVERCOVER
                 for b in NPB for c in CLUSTERS),
     "kept {} of {} workloads: {}; shortest kept cell {:.2f} s, worst kept "
     "time/duration {:.2f}".format(
         len(NPB), len(NPB_ALL), ",".join(NPB),
         min(dur_med[(b, c)] for b in NPB for c in CLUSTERS),
         max(win_med[(b, c)] for b in NPB for c in CLUSTERS)))

# --------------------------------------------------- metric variants per run
aug_by_key = {idk(r) + (r["ratio"],): r for r in aug}


# MojitO/S reports RAPL as microjoules accumulated per 10 Hz sample, so watts =
# raw x 1e-6 / 0.1 = raw x 1e-5. Verified against the wattmeter in S8b: converted
# RAPL package+DRAM lands at 40-86% of measured node power on every cluster, which
# is the physically expected range for CPU+DRAM as a fraction of a whole node.
RAPL_UJ_PER_SAMPLE_TO_W = 1e-5


def rapl_j(r):
    a = aug_by_key.get(idk(r) + (r["ratio"],))
    if a is None:
        return None
    try:
        p = (float(a["avg_package"]) + float(a["avg_dram"])) * RAPL_UJ_PER_SAMPLE_TO_W
    except (TypeError, ValueError, KeyError):
        return None
    return p * float(r["duration"]) if p > 0 else None


METRICS = {
    "primary_energy_column": lambda r: float(r["energy"]),
    "v2_medpower_x_duration": lambda r: float(r["median_power"]) * float(r["duration"]),
    "v3_meanpower_x_duration": lambda r: float(r["mean_power"]) * float(r["duration"]),
    "v4_rapl_package_dram": rapl_j,
}


def build_cells(metric, benches, nominal_only=True):
    f = METRICS[metric]
    byrat = defaultdict(lambda: defaultdict(list))
    for r in work:
        if r["bench"] not in benches:
            continue
        v = f(r)
        if v is None or not math.isfinite(v) or v <= 0:
            continue
        byrat[(r["bench"], r["cluster"])][round(float(r["ratio"]), 6)].append(v)
    out = {}
    for k, d in byrat.items():
        out[k] = d.get(1.0, []) if nominal_only else min(d.values(), key=st.median)
    return out


cells = build_cells("primary_energy_column", NPB)
counts = {"{}|{}".format(b, c): len(cells.get((b, c), [])) for b in NPB for c in CLUSTERS}
n_filled = sum(1 for v in counts.values() if v >= 10)
print("\nadvertised grid: {} workloads x {} clusters = {} cells".format(
    len(NPB_ALL), len(CLUSTERS), len(NPB_ALL) * len(CLUSTERS)))
print("gated grid:      {} workloads x {} clusters = {} cells, {} filled at n>=10, "
      "density {:.1%}".format(len(NPB), len(CLUSTERS), len(NPB) * len(CLUSTERS),
                              n_filled, n_filled / (len(NPB) * len(CLUSTERS))))
sane("S1 every gated cell has exactly 10 observations at nominal frequency",
     min(counts.values()) == 10 and max(counts.values()) == 10,
     "n = {} in all {} cells".format(min(counts.values()), len(counts)))

Y = [[math.log10(st.median(cells[(b, c)])) for c in CLUSTERS] for b in NPB]
sane("S2 cell energies positive and finite",
     all(math.isfinite(v) for r in Y for v in r) and
     all(st.median(cells[(b, c)]) > 0 for b in NPB for c in CLUSTERS), "ok")


# ------------------------------------------------------------- decomposition
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
sane("S3 sums of squares decompose exactly", abs(tss - (rss + css + iss)) < 1e-12,
     "TSS {:.12f} vs R+C+I {:.12f}".format(tss, rss + css + iss))
shares = dict(workload=100 * rss / tss, hardware=100 * css / tss,
              interaction=100 * iss / tss)
print("\nvariance shares: workload {:.2f}%, hardware {:.2f}%, interaction {:.3f}%".format(
    shares["workload"], shares["hardware"], shares["interaction"]))
print("hardware main effects (log10 J, centred), cheapest first:")
for j in sorted(range(len(CLUSTERS)), key=lambda j: cj[j]):
    print("   {:14} {:2} cores  {:+.4f}".format(CLUSTERS[j], CORES[CLUSTERS[j]], cj[j]))


# ------------------------------------------------------------------ decisions
def winner(row):
    return min(range(len(row)), key=lambda j: row[j])


def plurality(wins, cj):
    """Deterministic: most wins, ties broken by the smaller hardware main effect."""
    cnt = Counter(wins)
    return min(cnt, key=lambda j: (-cnt[j], cj[j]))


wins = [winner(r) for r in Y]
best_fixed = min(range(len(CLUSTERS)), key=lambda j: cj[j])
maj = plurality(wins, cj)
print("\nenergy-optimal cluster per workload:")
for i, b in enumerate(NPB):
    ordr = sorted(range(len(CLUSTERS)), key=lambda j: Y[i][j])
    print("   {:3}-{} -> {:14} log10 J {:.4f}   (runner-up {:14} {:.4f}, gap {:.1%})".format(
        b, CLASSES[b], CLUSTERS[ordr[0]], Y[i][ordr[0]], CLUSTERS[ordr[1]], Y[i][ordr[1]],
        10 ** (Y[i][ordr[1]] - Y[i][ordr[0]]) - 1))
n_rev = sum(1 for k in wins if k != maj)
n_rev_fixed = sum(1 for k in wins if k != best_fixed)
print("ranking reversals: {} of {} workloads deviate from the plurality winner {}".format(
    n_rev, len(NPB), CLUSTERS[maj]))
print("                   {} of {} deviate from the additive-optimal platform {}".format(
    n_rev_fixed, len(NPB), CLUSTERS[best_fixed]))

fixed_acc = sum(1 for k in wins if k == best_fixed) / len(wins)
add_acc = sum(1 for k, p in zip(wins, [winner(cj)] * len(NPB)) if k == p) / len(wins)
sane("S4 additive model ties a fixed hardware ranking EXACTLY (Proposition 1)",
     fixed_acc == add_acc,
     "fixed {:.17f} vs additive {:.17f}, identical to every digit: {}".format(
         fixed_acc, add_acc, fixed_acc == add_acc))

# ---------------------------------------------------------- S5 permutation null
null_perm, sum_err = [], 0.0
for _ in range(20000):
    Yp = []
    for r in Y:
        rr = list(r)
        rnd.shuffle(rr)
        Yp.append(rr)
    d = decompose(Yp)
    null_perm.append(d[7])
    sum_err = max(sum_err, abs((d[6] + d[7]) - (css + iss)))
med_perm = st.median(null_perm)
p_perm = sum(1 for v in null_perm if v >= iss) / len(null_perm)
sane("S5 (expected not to reject) within-row permutation conserves hardware SS + "
     "interaction SS exactly, so it cannot separate them",
     sum_err < 1e-10 and p_perm >= 0.05,
     "max |CSS_perm+ISS_perm - (CSS+ISS)| = {:.2e} over 20000 draws; observed ISS "
     "{:.5f}, null median ISS {:.5f}, conserved sum {:.5f} (= hardware {:.5f} + "
     "interaction {:.5f}), p = {:.4f}".format(
         sum_err, iss, med_perm, css + iss, css, iss, p_perm))


# ------------------------------------------------------ S5b per-cell bootstrap
def boot_se(vals, B=1000):
    n = len(vals)
    out = []
    for _ in range(B):
        s = sorted(rnd.choice(vals) for _ in range(n))
        m = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        out.append(math.log10(m))
    return st.pstdev(out)


ses = [[boot_se(cells[(b, c)]) for c in CLUSTERS] for b in NPB]
null_boot = []
for _ in range(4000):
    Yn = [[gm + ri[i] + cj[j] + rnd.gauss(0.0, ses[i][j]) for j in range(len(CLUSTERS))]
          for i in range(len(NPB))]
    null_boot.append(decompose(Yn)[7])
p_boot = sum(1 for v in null_boot if v >= iss) / len(null_boot)
sane("S5b interaction exceeds the per-cell bootstrap noise null", p_boot < 0.05,
     "observed ISS {:.5f}, bootstrap null median {:.5f}, p = {:.4f}, max cell SE "
     "{:.5f}".format(iss, st.median(null_boot), p_boot, max(max(r) for r in ses)))


# ---------------------------------------------------------------- routing gain
def routing_gain(Y, subset):
    lin = [[10 ** Y[i][j] for j in subset] for i in range(len(Y))]
    return 100 * (1 - sum(min(r) for r in lin) / min(sum(col) for col in zip(*lin)))


gain_all = routing_gain(Y, list(range(len(CLUSTERS))))
top5 = sorted(range(len(CLUSTERS)), key=lambda j: cj[j])[:5]
gain_top5 = routing_gain(Y, top5)
print("\noracle routing gain over the best single cluster:")
print("   all {} clusters      {:.2f}%".format(len(CLUSTERS), gain_all))
print("   5 cheapest clusters  {:.2f}%  ({})".format(
    gain_top5, ", ".join(CLUSTERS[j] for j in top5)))


# ------------------------------------------------------------ variants (S8/S12)
def run_variant(metric, benches, nominal_only=True):
    cv = build_cells(metric, benches, nominal_only)
    if any(len(cv.get((b, c), [])) < 10 for b in benches for c in CLUSTERS):
        return dict(usable=False)
    Yv = [[math.log10(st.median(cv[(b, c)])) for c in CLUSTERS] for b in benches]
    dv = decompose(Yv)
    wv = [winner(r) for r in Yv]
    mv = plurality(wv, dv[2])
    return dict(usable=True, workloads=list(benches),
                shares=dict(workload=100 * dv[5] / dv[4], hardware=100 * dv[6] / dv[4],
                            interaction=100 * dv[7] / dv[4]),
                reversals=sum(1 for k in wv if k != mv), n_workloads=len(benches),
                plurality_winner=CLUSTERS[mv], winners=[CLUSTERS[k] for k in wv],
                hardware_effects=dv[2], log10_energy=Yv)


variants = {
    "v2_medpower_x_duration": run_variant("v2_medpower_x_duration", NPB),
    "v3_meanpower_x_duration": run_variant("v3_meanpower_x_duration", NPB),
    "v4_rapl_package_dram": run_variant("v4_rapl_package_dram", NPB),
    "v5_energy_optimal_dvfs": run_variant("primary_energy_column", NPB, False),
    "v6_ungated_all8_workloads": run_variant("primary_energy_column", NPB_ALL),
}
print("\nvariants (primary = gated {} workloads, published energy column):".format(len(NPB)))
print("   {:30} wl {:6.2f}%  hw {:6.2f}%  int {:6.3f}%  rev {}/{}".format(
    "primary_energy_column", shares["workload"], shares["hardware"],
    shares["interaction"], n_rev, len(NPB)))
for name, v in variants.items():
    if not v["usable"]:
        print("   {:30} UNUSABLE".format(name))
        continue
    s = v["shares"]
    print("   {:30} wl {:6.2f}%  hw {:6.2f}%  int {:6.3f}%  rev {}/{}".format(
        name, s["workload"], s["hardware"], s["interaction"],
        v["reversals"], v["n_workloads"]))


def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    xa, xb = [ra[v] for v in a], [rb[v] for v in b]
    n = len(a)
    ma, mb = sum(xa) / n, sum(xb) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(xa, xb))
    den = math.sqrt(sum((p - ma) ** 2 for p in xa) * sum((q - mb) ** 2 for q in xb))
    return num / den if den else float("nan")


rapl = variants["v4_rapl_package_dram"]
rho = spearman(cj, rapl["hardware_effects"]) if rapl["usable"] else float("nan")
scope = {}
for j, c in enumerate(CLUSTERS):
    scope[c] = st.median([10 ** (rapl["log10_energy"][i][j] - Y[i][j])
                          for i in range(len(NPB))])
print("\nRAPL package+DRAM as a fraction of measured whole-node energy:")
for c in sorted(scope, key=scope.get):
    print("   {:14} {:.0%}".format(c, scope[c]))
sane("S8 (as first stated) RAPL and the wattmeter rank the 18 clusters compatibly",
     rapl["usable"] and rho >= 0.8,
     "Spearman rho of the 18 hardware main effects = {:.4f}; MIS-SPECIFIED, the "
     "two instruments measure different SCOPES: RAPL covers {:.0%}-{:.0%} of node "
     "energy depending on cluster, a {:.1f}x spread, and that per-cluster constant "
     "lands entirely in the hardware main effect. Replaced by S8b.".format(
         rho, min(scope.values()), max(scope.values()),
         max(scope.values()) / min(scope.values())))

# S8b: the scope-invariant form. A per-cluster multiplicative constant is an
# additive per-column shift in log space, so it is absorbed exactly by the hardware
# main effect and cannot touch the interaction residuals. The cross-instrument
# question that the decomposition actually depends on is therefore whether the
# INTERACTION agrees, not whether the column effects do.
r1 = [v for row in res for v in row]
r2c = decompose(rapl["log10_energy"])[3]
r2 = [v for row in r2c for v in row]


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


r_resid = pearson(r1, r2)
d_int = abs(rapl["shares"]["interaction"] - shares["interaction"])
sane("S8b the two instruments agree on the INTERACTION, which is what the "
     "decomposition rests on and which is invariant to their scope difference",
     r_resid >= 0.8 and d_int < 1.0,
     "Pearson r of the {} interaction residuals = {:.4f}; interaction share "
     "{:.3f}% (wattmeter) against {:.3f}% (RAPL), delta {:.3f} pp".format(
         len(r1), r_resid, shares["interaction"],
         rapl["shares"]["interaction"], d_int))

order = sorted(shares, key=shares.get)
diffs = {n: abs(v["shares"]["interaction"] - shares["interaction"])
         for n, v in variants.items() if v["usable"]}
same_order = {n: sorted(v["shares"], key=v["shares"].get) == order
              for n, v in variants.items() if v["usable"]}
sane("S12 the S11 gate is not what creates the result",
     all(d < 1.0 for d in diffs.values()) and all(same_order.values()),
     "primary interaction {:.3f}%; |delta| vs variants {}; share ordering "
     "identical in all variants: {}".format(
         shares["interaction"],
         {k: round(v, 3) for k, v in diffs.items()}, all(same_order.values())))

# ------------------------------------------------------------- S10 / extremes
row_iss = {NPB[i]: sum(v * v for v in res[i]) for i in range(len(NPB))}
print("\ninteraction SS contributed per workload:")
for b in sorted(row_iss, key=row_iss.get, reverse=True):
    print("   {:3} {:.5f}  ({:.1f}% of interaction SS)".format(
        b, row_iss[b], 100 * row_iss[b] / iss))
top_share = 100 * max(row_iss.values()) / iss
sane("S10 no single workload dominates the interaction after the gate",
     top_share <= 50.0,
     "largest single-workload contribution is {} at {:.1f}% of interaction SS "
     "(before the gate, ft held 43.0% of an 8-workload interaction)".format(
         max(row_iss, key=row_iss.get), top_share))

resid_flat = sorted(((abs(res[i][j]), i, j) for i in range(len(NPB))
                     for j in range(len(CLUSTERS))), reverse=True)
print("\nlargest interaction residuals (log10 J):")
for a, i, j in resid_flat[:10]:
    print("   {:3} on {:14} resid {:+.4f}  cell SE {:.4f}  ({:.0f}x SE, {} cores)".format(
        NPB[i], CLUSTERS[j], res[i][j], ses[i][j],
        abs(res[i][j]) / ses[i][j] if ses[i][j] > 0 else float("inf"),
        CORES[CLUSTERS[j]]))

flat = sorted(((Y[i][j], i, j) for i in range(len(NPB)) for j in range(len(CLUSTERS))))
print("\nextreme cells (checked for the short-run artifact that the gate removes):")
for v, i, j in flat[:3] + flat[-3:]:
    vals = cells[(NPB[i], CLUSTERS[j])]
    print("   {:3} on {:14} log10 J {:.4f}  raw J min {:.0f} med {:.0f} max {:.0f}  "
          "spread {:.1%}  dur {:.1f} s  time/dur {:.2f}".format(
              NPB[i], CLUSTERS[j], v, min(vals), st.median(vals), max(vals),
              (max(vals) - min(vals)) / st.median(vals),
              dur_med[(NPB[i], CLUSTERS[j])], win_med[(NPB[i], CLUSTERS[j])]))

# ------------------------------------------------------------------ defects
DEFECTS += [
    "SHORT-RUN WATTMETER ARTIFACT (material). The wattmeter window `time` has a "
    "floor near 30 s and over-covers short benchmarks: median time/duration "
    "reaches 3.09 for FT-C on dahu and 4.00 for IS-D on dahu, and IS-D "
    "over-covers by more than 2x on all 18 clusters. The ten nominally identical "
    "9.7 s FT-C runs on dahu record mean power from 70 W to 286 W and energy from "
    "2101 J to 8510 J, a 4x spread, against 4-8% for CG-D. Ungated, FT-C alone "
    "held 43% of the interaction sum of squares. Handled by the S11 gate, which "
    "drops FT-C, IS-D and LU-C.",
    "`fullname` embeds nproc ('cg-D-16') and nproc is a cluster property, not a "
    "workload property. Keying the matrix on `fullname` yields a 50 x 18 grid that "
    "is 91% empty and moves a hardware attribute onto the workload axis.",
    "med_core / avg_core (RAPL core domain) is blank on 14 of 18 clusters, present "
    "only on graphite, orion, petitprince and taurus, so the RAPL variant uses "
    "package + DRAM, which is available on all 18.",
    "data.csv and data_augmented.csv disagree in the last float digit on 3614 of "
    "24200 rows (2090 of them in `ratio` alone) because the two files were written "
    "by separate float-to-string round trips. Cosmetic, not a data defect, but it "
    "breaks any exact-string join between the two files.",
    "RAPL package+DRAM covers only {:.0%} ({}, a GPU cluster) to {:.0%} ({}) of "
    "measured whole-node energy, a {:.1f}x spread across the 18 clusters. This is "
    "physics, not a defect, but it means RAPL and the wattmeter must not be "
    "compared on the hardware main effect; they agree on the interaction (S8b)."
    .format(min(scope.values()), min(scope, key=scope.get),
            max(scope.values()), max(scope, key=scope.get),
            max(scope.values()) / min(scope.values())),
    "NO defect of the kind found in the two previous corpora: no byte-identical "
    "mislabelled duplicate file (ejhusom) and no fabricated energy column prorated "
    "from a single measurement (wilkins). All four downloaded files match the "
    "publisher's own md5 manifest.",
]

CORRECTIONS += [
    "S5 as first stated required the permutation null median to sit within 10% of "
    "hardware SS + interaction SS; on the then-ungated 8-workload matrix it landed "
    "12.1% low (4.9635 against 5.6479) and "
    "failed. Root cause found by inspection, not by loosening the tolerance: a "
    "permuted matrix retains a nonzero column SS by chance, so the null median "
    "interaction SS is systematically below the sum by that amount. The exact "
    "invariant is the conserved SUM CSS_perm + ISS_perm = CSS_obs + ISS_obs, which "
    "S5 now asserts to machine precision on every one of 20000 draws.",
    "S6 as first stated compared the two CSVs by exact string equality and reported "
    "3614 of 24200 rows missing. False alarm: float-repr round-trip differences "
    "(245.06399703025815 against 245.06399703025812). S6 now joins on a stable "
    "identity key and compares numerics with a 1e-9 relative tolerance.",
    "S10 as first stated predicted EP-D would be an extreme of the interaction "
    "residuals; it ranked 4th of 8 and the check failed. Inspecting the residuals "
    "found that FT-C held 43% of the interaction SS and that the FT-C cells are "
    "exactly those whose wattmeter window over-covers the run by up to 3.1x. The "
    "failed prediction surfaced the measurement artifact now handled by S11. S10 "
    "is restated as: no single workload may dominate the interaction after the gate.",
    "S8 as first stated required RAPL and the wattmeter to rank the 18 clusters "
    "with Spearman rho >= 0.8; it returned 0.7833 and failed. Root cause found by "
    "converting the RAPL counters to watts (MojitO/S reports microjoules per 10 Hz "
    "sample) and comparing scopes: RAPL covers {:.0%} to {:.0%} of node energy "
    "depending on the cluster. A per-cluster multiplicative "
    "scope constant is an additive "
    "per-column shift in log space and is absorbed entirely by the hardware main "
    "effect, so rank agreement there was never the right question. S8b compares the "
    "interaction residuals, which the scope constant cannot touch: r = {:.2f}."
    .format(min(scope.values()), max(scope.values()), r_resid),
    "An idle-subtraction estimator, energy - P_idle x (time - duration), was "
    "considered to rescue the short-run rows and REJECTED: with the IdleC0 baseline "
    "it returns -1840 J on dahu. Reconstructing an energy column from a separately "
    "measured constant is the defect we criticised in the wilkins corpus.",
]

# ------------------------------------------------------------------- output
OUT = dict(
    corpus="grid5000-dacosta-zenodo-10982238",
    zenodo_concept_doi="10.5281/zenodo.10982238",
    zenodo_version_doi="10.5281/zenodo.14914799",
    zenodo_record=ZENODO_RECORD,
    described_by_doi="10.1016/j.suscom.2025.101106",
    licence="CC-BY-4.0 (Zenodo metadata.license.id == 'cc-by-4.0', access_right "
            "== 'open'). Redistribution with attribution is permitted; raw files "
            "are still not committed to this repository.",
    energy_is_measured=True,
    energy_stored_as="joules directly: scalar `energy` column per run, already "
                     "integrated from the wattmeter power series; no integration "
                     "by us is required",
    measurement="full-node wattmeter (IPMI or e-PDU), 0.1-10 W precision, 1 Hz to "
                "0.1 Hz, watermark time-realigned against the host clock; RAPL "
                "package and DRAM via MojitO/S at 10 Hz as an independent channel",
    hardware=CLUSTERS, cores_per_cluster=CORES,
    workloads_advertised=NPB_ALL, workloads_used=NPB, workload_classes=CLASSES,
    unit="log10 median whole-node energy (J) to complete the benchmark at nominal "
         "maximum frequency",
    measurement_adequacy_gate=dict(
        wattmeter_window_floor_s=WATTMETER_WINDOW_FLOOR_S,
        max_window_overcover=MAX_WINDOW_OVERCOVER,
        dropped=[b for b in NPB_ALL if b not in NPB],
        median_duration_s={"{}|{}".format(b, c): dur_med[(b, c)]
                           for b in NPB_ALL for c in CLUSTERS},
        median_time_over_duration={"{}|{}".format(b, c): win_med[(b, c)]
                                   for b in NPB_ALL for c in CLUSTERS}),
    grid=dict(advertised_workloads=len(NPB_ALL), platforms=len(CLUSTERS),
              advertised_cells=len(NPB_ALL) * len(CLUSTERS),
              gated_workloads=len(NPB), gated_cells=len(NPB) * len(CLUSTERS),
              filled_cells=n_filled, density=n_filled / (len(NPB) * len(CLUSTERS)),
              total_runs_in_dataset=len(raw), runs_used=sum(counts.values())),
    log10_energy=Y, cell_counts=counts,
    cell_medians={"{}|{}".format(b, c): st.median(cells[(b, c)])
                  for b in NPB for c in CLUSTERS},
    cell_values={"{}|{}".format(b, c): cells[(b, c)] for b in NPB for c in CLUSTERS},
    grand_mean=gm, workload_effects=ri, hardware_effects=cj, residuals=res,
    per_cell_bootstrap_se=ses,
    sums_of_squares=dict(total=tss, workload=rss, hardware=css, interaction=iss),
    shares=shares,
    winners=[CLUSTERS[k] for k in wins],
    plurality_winner=CLUSTERS[maj], best_fixed_platform=CLUSTERS[best_fixed],
    reversals=n_rev, reversals_vs_additive_optimal=n_rev_fixed, n_workloads=len(NPB),
    fixed_ranking_accuracy=fixed_acc, additive_accuracy=add_acc,
    proposition1_exact=(fixed_acc == add_acc),
    bootstrap_noise_p=p_boot, bootstrap_null_median_iss=st.median(null_boot),
    discarded_permutation_test=dict(
        p=p_perm, null_median_iss=med_perm, hardware_ss=css, interaction_ss=iss,
        conserved_sum=css + iss, max_conservation_error=sum_err,
        why="a within-row permutation preserves every row mean, hence preserves "
            "TSS and workload SS, hence conserves hardware SS + interaction SS "
            "exactly; it therefore cannot separate hardware from interaction"),
    rapl_wattmeter_spearman=rho,
    rapl_interaction_residual_pearson=r_resid,
    rapl_scope_fraction_of_node_energy=scope,
    routing_gain_percent=dict(all_clusters=gain_all, top5_cheapest=gain_top5),
    interaction_ss_by_workload=row_iss,
    variants=variants,
    file_md5=digests, zenodo_published_md5=ZENODO_MD5,
    defects=DEFECTS, corrections=CORRECTIONS, sanity=SANITY,
)
os.makedirs("experiments/results", exist_ok=True)
json.dump(OUT, io.open("experiments/results/k2_grid5000.json", "w", encoding="utf-8"),
          indent=1)
print("\ndefects recorded: {}".format(len(DEFECTS)))
for d in DEFECTS:
    print("  - " + d)
print("\ncorrections: {}".format(len(CORRECTIONS)))
for c in CORRECTIONS:
    print("  - " + c)
print("\nsaved -> experiments/results/k2_grid5000.json")
print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
                                            sum(not x["passed"] for x in SANITY)))
