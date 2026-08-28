# -*- coding: utf-8 -*-
"""
Root-cause analysis of three negative results on the MLPerf matrix:
  N1  additive beats biased MF and hybrid on pair completion
  N2  machine cold start shows no benefit from hardware features
  N3  hybrid is worse than accelerator-only on load cold start

Hypothesis under test: the interaction signal on MLPerf is small relative to the
OBSERVATIONAL noise (vendor-tuned submissions, varying node counts, frameworks),
so extra model capacity fits noise. The control is our own pilot grid, where the
same decomposition can be computed under controlled conditions.
"""
import io, json, sys, math
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
from accelerator_specs import SPECS


def variance_decomposition(Y, mask, label):
    """How much of the observed variance is additive vs interaction?"""
    obs = Y[mask]
    mu = obs.mean()
    r = np.zeros(Y.shape[0]); c = np.zeros(Y.shape[1])
    for _ in range(300):
        for i in range(Y.shape[0]):
            o = mask[i]
            if o.any(): r[i] = np.mean(Y[i, o] - mu - c[o])
        for j in range(Y.shape[1]):
            o = mask[:, j]
            if o.any(): c[j] = np.mean(Y[o, j] - mu - r[o])
    pred = mu + r[:, None] + c[None, :]
    resid = np.where(mask, Y - pred, 0.0)
    ss_tot = float(((np.where(mask, Y, mu) - mu) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    print(f"  {label}: additive explains {100*(1-ss_res/ss_tot):5.1f}%   "
          f"interaction residual {100*ss_res/ss_tot:5.1f}%   "
          f"resid sd {resid[mask].std():.3f} (log10)")
    return resid, mask


def replicate_noise(scenario="Offline"):
    """How noisy is a single MLPerf cell? Use spread across repeated submissions."""
    recs = json.load(io.open("data/mlperf-power/records.json", encoding="utf-8"))
    cells = defaultdict(list)
    for r in recs:
        if not r.get("has_power") or r.get("Scenario") != scenario: continue
        a, m, ipj = r.get("accelerator_model_name"), r.get("MlperfModel"), r.get("Inference_per_Joule")
        if not a or a in ("-", "N/A") or not m or not ipj: continue
        if a not in SPECS: continue
        try: v = float(ipj)
        except (TypeError, ValueError): continue
        if v > 0: cells[(m, a)].append(math.log10(v))
    multi = {k: v for k, v in cells.items() if len(v) >= 3}
    sds = [float(np.std(v)) for v in multi.values()]
    print(f"\n  cells with >=3 submissions: {len(multi)}")
    print(f"  within-cell sd of log10 efficiency: median {np.median(sds):.3f}, "
          f"mean {np.mean(sds):.3f}, p90 {np.percentile(sds,90):.3f}")
    return float(np.median(sds))


def build_mlperf(scenario="Offline", fixed_npn=None):
    recs = json.load(io.open("data/mlperf-power/records.json", encoding="utf-8"))
    cells = defaultdict(list)
    for r in recs:
        if not r.get("has_power") or r.get("Scenario") != scenario: continue
        a, m, ipj = r.get("accelerator_model_name"), r.get("MlperfModel"), r.get("Inference_per_Joule")
        if not a or a in ("-", "N/A") or not m or not ipj or a not in SPECS: continue
        npn = r.get("accelerators_per_node") or 1
        if fixed_npn is not None and float(npn) != fixed_npn: continue
        try: v = float(ipj)
        except (TypeError, ValueError): continue
        if v > 0: cells[(m, a)].append(math.log10(v))
    obs = {k: float(np.median(v)) for k, v in cells.items()}
    models = sorted({k[0] for k in obs}); accs = sorted({k[1] for k in obs})
    Y = np.full((len(models), len(accs)), np.nan)
    mi = {m: i for i, m in enumerate(models)}; ai = {a: j for j, a in enumerate(accs)}
    for (m, a), v in obs.items(): Y[mi[m], ai[a]] = v
    return models, accs, Y, ~np.isnan(Y)


def pilot_matrix():
    d = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
    rows = [r for b in d for r in b["rows"] if r.get("status") == "ok"]
    MACH = ["T4", "L4", "A10G", "L40S", "A100-40GB"]
    keys = sorted({(r["load"], r["precision"], r["batch"]) for r in rows})
    Y = np.zeros((len(keys), len(MACH)))
    ki = {k: i for i, k in enumerate(keys)}; mi = {m: j for j, m in enumerate(MACH)}
    for r in rows:
        Y[ki[(r["load"], r["precision"], r["batch"])], mi[r["machine"]]] = math.log10(r["energy_per_sample_mj"])
    return Y, np.ones_like(Y, dtype=bool)


print("=" * 78)
print("DIAGNOSIS 1: how big is the interaction signal, and how big is the noise?")
print("=" * 78)
print("\nMLPerf Offline (all node counts pooled):")
models, accs, Y, M = build_mlperf("Offline")
print(f"  matrix {Y.shape}, {M.sum()} observed")
resid_mlperf, _ = variance_decomposition(Y, M, "MLPerf Offline")
noise_sd = replicate_noise("Offline")

print("\nOur controlled pilot grid (fully observed, one protocol, one session):")
Yp, Mp = pilot_matrix()
print(f"  matrix {Yp.shape}, {Mp.sum()} observed")
resid_pilot, _ = variance_decomposition(Yp, Mp, "pilot grid")

print("\n" + "=" * 78)
print("DIAGNOSIS 2: is the interaction bigger than the measurement noise?")
print("=" * 78)
r_ml = resid_mlperf[M].std()
print(f"  MLPerf : interaction residual sd = {r_ml:.3f} log10")
print(f"           within-cell submission noise sd = {noise_sd:.3f} log10")
print(f"           signal-to-noise ratio = {r_ml/max(noise_sd,1e-9):.2f}")
if r_ml / max(noise_sd, 1e-9) < 2:
    print("           --> interaction is NOT separable from submission noise on this data")
print(f"\n  pilot  : interaction residual sd = {resid_pilot[Mp].std():.3f} log10")
print(f"           (controlled protocol; replicate noise not yet measured, see workaround W5)")

print("\n" + "=" * 78)
print("DIAGNOSIS 3: does restricting node count reduce the nuisance variance?")
print("=" * 78)
for npn in [1.0, 8.0]:
    try:
        m2, a2, Y2, M2 = build_mlperf("Offline", fixed_npn=npn)
        if M2.sum() < 20:
            print(f"  accelerators_per_node={npn:g}: only {M2.sum()} cells, skipped")
            continue
        print(f"  accelerators_per_node={npn:g}: matrix {Y2.shape}, {M2.sum()} observed "
              f"({100*M2.sum()/Y2.size:.1f}% dense)")
        variance_decomposition(Y2, M2, f"    npn={npn:g}")
    except Exception as e:
        print(f"  npn={npn}: {e}")

print("\n" + "=" * 78)
print("DIAGNOSIS 4: parameter count vs data (why MF overfits)")
print("=" * 78)
n, m = Y.shape
obs = int(M.sum())
for rank in (1, 2, 3):
    p_add = 1 + n + m
    p_mf = p_add + rank * (n + m)
    print(f"  rank-{rank}: additive params {p_add}, +interaction params {rank*(n+m)}, "
          f"total {p_mf} vs {obs} observations  (obs/param = {obs/p_mf:.2f})")
print(f"  a ratio below ~3 is generally hopeless for recovering a {100*(resid_mlperf[M]**2).sum()/((np.where(M,Y,Y[M].mean())-Y[M].mean())**2).sum():.1f}% signal")
