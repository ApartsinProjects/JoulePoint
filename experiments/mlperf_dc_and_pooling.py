# -*- coding: utf-8 -*-
"""
Two corrections and one new analysis.

(1) MLPerf restricted to DATACENTER-class accelerators. Mixing Jetson and M.2 edge
    parts with H100s in one matrix was a design error of mine: an edge device is
    never a candidate for a datacenter job, and those sparsely-observed rows were
    what drove the unexplained Task C regret of 4.24x.

(2) Cross-corpus transfer. BUTTER-E is dense on the load axis with two machine
    classes; MLPerf is dense on the machine axis with few loads. Neither alone
    supports both cold-start regimes. Test whether machine effects estimated on one
    corpus transfer to the other.
"""
import io, json, math, sys, warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from accelerator_specs import SPECS
from mlperf_experiment import build, acc_features, fit_additive, fit_hybrid, coldstart_metrics

SANITY, OUT = [], {}


def sane(n, ok, d):
    SANITY.append(dict(check=n, passed=bool(ok), detail=d))
    print(f"    [{'PASS' if ok else 'FAIL'}] {n}: {d}")


# ---------------------------------------------------------------- (1)
print("=" * 84)
print("(1) MLPerf restricted to datacenter-class accelerators")
print("=" * 84)
models, accs, Y, NPN = build("Offline", confident_only=False)
M = ~np.isnan(Y)
dc = [j for j, a in enumerate(accs) if SPECS[a]["klass"] == "datacenter"]
edge = [j for j, a in enumerate(accs) if SPECS[a]["klass"] != "datacenter"]
print(f"  full matrix       : {Y.shape}, {M.sum()} observed")
print(f"  datacenter columns: {len(dc)}   non-datacenter: {len(edge)}")

Yd = Y[:, dc]
Md = ~np.isnan(Yd)
keep = [i for i in range(Yd.shape[0]) if Md[i].sum() >= 2]
Yd, Md = Yd[keep], Md[keep]
accs_d = [accs[j] for j in dc]
Fd = acc_features(accs_d)
print(f"  datacenter-only   : {Yd.shape}, {Md.sum()} observed ({100*Md.sum()/Yd.size:.1f}% dense)")

# was the previous regret driven by edge parts winning rows?
edge_wins = 0
tot_rows = 0
for i in range(Y.shape[0]):
    obs = [j for j in range(Y.shape[1]) if M[i, j]]
    if len(obs) < 4:
        continue
    tot_rows += 1
    b = max(obs, key=lambda j: Y[i, j])
    if SPECS[accs[b]]["klass"] != "datacenter":
        edge_wins += 1
print(f"  rows whose true optimum was a NON-datacenter part: {edge_wins}/{tot_rows}")
sane("edge devices were winning datacenter rows in the original matrix",
     edge_wins > 0, f"{edge_wins} of {tot_rows} rows")

# rerun machine cold start on the corrected matrix
cand = [j for j in range(Yd.shape[1]) if Md[:, j].sum() >= 4]
res = defaultdict(list)
for j0 in cand:
    tm = Md.copy(); tm[:, j0] = False
    mu, r, c = fit_additive(Yd, tm)
    preds = {
        "no column information": (lambda i, j, mu=mu, r=r, c=c: mu + r[i] + (c[j] if j != j0 else 0.0)),
        "hybrid with specifications": fit_hybrid(Yd, tm, Fd, rank=2),
    }
    for nm, p in preds.items():
        a_, rg, tot = coldstart_metrics(Yd, Md, p, j0)
        if not math.isnan(a_):
            res[nm].append((a_, rg))
print(f"\n  machine cold start, datacenter-only ({len(cand)} held-out accelerators)")
print(f"  {'method':30}{'pair acc %':>12}{'regret':>10}")
for nm in ["no column information", "hybrid with specifications"]:
    a = np.array(res[nm], dtype=float)
    print(f"  {nm:30}{a[:,0].mean():12.1f}{a[:,1].mean():10.3f}")
OUT["mlperf_datacenter_only"] = {nm: dict(pair_acc=float(np.array(v)[:, 0].mean()),
                                          regret=float(np.array(v)[:, 1].mean()))
                                 for nm, v in res.items()}
sane("hardware specifications still help after removing edge parts",
     np.array(res["hybrid with specifications"])[:, 0].mean() >
     np.array(res["no column information"])[:, 0].mean(),
     f"hybrid {np.array(res['hybrid with specifications'])[:,0].mean():.1f}% vs "
     f"none {np.array(res['no column information'])[:,0].mean():.1f}%")

# ---------------------------------------------------------------- (2)
print("\n" + "=" * 84)
print("(2) Cross-corpus transfer: do machine effects estimated on one corpus")
print("    predict the ordering observed on another?")
print("=" * 84)

# our grid
sys.path.insert(0, "experiments")
from e4_e5_models import load_grid, MACH as OURM
keys, Yo, To = load_grid()
mu_o = Yo.mean(); c_o = Yo.mean(0) - mu_o          # machine effects on our grid
print(f"  our grid machine effects (log10 energy, lower is better):")
for j, mm in enumerate(OURM):
    print(f"    {mm:12} {c_o[j]:+.3f}")

# MLPerf machine effects, restricted to the accelerators we can name-match
NAME = {"NVIDIA L4": "L4", "NVIDIA L40S": "L40S", "NVIDIA A100-SXM4-40GB": "A100-40GB",
        "NVIDIA A100-PCIe-80GB": "A100-40GB", "NVIDIA A100-SXM-80GB": "A100-40GB"}
mu_m, r_m, c_m = fit_additive(Yd, Md)
shared = [(NAME[a], c_m[j]) for j, a in enumerate(accs_d) if a in NAME]
print(f"\n  accelerators appearing in BOTH corpora: {sorted({s for s, _ in shared})}")
if len(shared) >= 3:
    agg = defaultdict(list)
    for nm, v in shared:
        agg[nm].append(v)
    ours, theirs = [], []
    for nm, vals in agg.items():
        if nm in OURM:
            ours.append(c_o[OURM.index(nm)])
            theirs.append(float(np.mean(vals)))
    # MLPerf is efficiency (higher better), ours is energy (lower better): expect NEGATIVE correlation
    rho = float(np.corrcoef(ours, theirs)[0, 1]) if len(ours) >= 3 else float("nan")
    print(f"  correlation of machine effects across corpora: {rho:+.3f}")
    print(f"  (MLPerf measures efficiency, ours measures energy, so agreement means NEGATIVE)")
    OUT["cross_corpus_machine_effect_corr"] = rho
    sane("machine effects agree in sign across independently collected corpora",
         (not math.isnan(rho)) and rho < 0,
         f"correlation {rho:+.3f} over {len(ours)} shared accelerators")
else:
    print("  too few shared accelerators for a transfer test")
    sane("shared accelerators available for transfer", False,
         f"only {len(shared)} name-matched")

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/mlperf_dc_pooling.json", "w", encoding="utf-8"), indent=1)
print(f"\nsanity: {sum(1 for s in SANITY if s['passed'])}/{len(SANITY)} passed")
print("saved -> experiments/results/mlperf_dc_pooling.json")
