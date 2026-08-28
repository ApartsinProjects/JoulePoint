# -*- coding: utf-8 -*-
"""
Test the central reframing on a THIRD public dataset.

Claim under test: the decision-relevant interaction is carried by CONFIGURATION,
not by hardware identity. Our grid says so (precision r=-0.515, batch r=+0.394);
MLPerf could not test it because vendors tune configuration away per cell.

llm-perf-leaderboard is the first public source we have found with BOTH a hardware
axis (T4 / A10 / A100) and a real configuration axis (quantization scheme, dtype,
batch size, sequence length), with measured energy per phase.
"""
import csv, io, glob, json, math, os, sys, warnings
csv.field_size_limit(10_000_000)
import numpy as np
from collections import defaultdict
warnings.filterwarnings("ignore")

SANITY, OUT = [], {}


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def num(x):
    try:
        v = float(x)
        return v if v > 0 and math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


rows = []
for f in sorted(glob.glob("data/llm-perf/*.csv")):
    base = os.path.basename(f)
    quant = base.split("cuda-")[1].split("-1x")[0]
    gpu = base.split("-1x")[1].replace(".csv", "")
    with io.open(f, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            e = num(r.get("report.decode.energy.gpu")) or num(r.get("report.decode.energy.total"))
            if e is None:
                continue
            rows.append(dict(
                gpu=gpu, quant=quant,
                model=r.get("config.backend.model") or r.get("config.name") or "",
                dtype=(r.get("config.backend.torch_dtype") or "").strip(),
                batch=num(r.get("config.scenario.input_shapes.batch_size")) or 1.0,
                seqlen=num(r.get("config.scenario.input_shapes.sequence_length")) or 0.0,
                new_tokens=num(r.get("config.scenario.generate_kwargs.max_new_tokens")) or 0.0,
                energy=e,
                params=num(r.get("config.backend.model_kwargs.num_parameters")) or None,
                tput=num(r.get("report.decode.throughput.value")),
            ))
print(f"rows with decode-phase GPU energy: {len(rows)}")
gpus = sorted({r["gpu"] for r in rows})
print(f"GPUs: {gpus}")
print(f"quantization schemes: {sorted({r['quant'] for r in rows})}")
print(f"distinct models: {len({r['model'] for r in rows})}")
for c in ("batch", "seqlen", "new_tokens", "dtype"):
    print(f"  distinct {c}: {sorted({r[c] for r in rows})[:8]}")

sane("the dataset has a real hardware axis", len(gpus) >= 3, f"{len(gpus)} GPUs: {gpus}")
sane("the dataset has a real configuration axis",
     len({(r['quant'], r['batch'], r['seqlen']) for r in rows}) > 5,
     f"{len({(r['quant'], r['batch'], r['seqlen']) for r in rows})} distinct (quant,batch,seqlen) combos")

# ---- build (model, config) x gpu matrix -------------------------------------
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
print(f"\nmatrix {Y.shape}, {M.sum()} observed ({100*M.sum()/Y.size:.1f}% dense)")
print(f"rows observed on ALL {len(gpus)} GPUs: {len(full)}")

if len(full) < 8:
    print("\nnot enough fully-observed rows for a reversal test; reporting coverage only")
    json.dump({"sanity": SANITY, "coverage": {"rows": len(keys), "gpus": gpus,
                                              "full_rows": len(full)}},
              io.open("experiments/results/llm_perf.json", "w", encoding="utf-8"), indent=1)
    sys.exit(0)

F = Y[full]
Fk = [keys[i] for i in full]

# ---- 1. additive vs interaction --------------------------------------------
mu = F.mean(); r = F.mean(1) - mu; c = F.mean(0) - mu
R = F - (mu + r[:, None] + c[None, :])
ss_tot = ((F - mu) ** 2).sum(); ss_res = (R ** 2).sum()
print(f"\n=== variance decomposition ===")
print(f"  additive explains {100*(1-ss_res/ss_tot):.1f}%, interaction residual {100*ss_res/ss_tot:.1f}%")
print(f"  (our controlled grid: 2.5%   MLPerf, config tuned away: 0.3%)")
S = np.linalg.svd(R, full_matrices=False)[1]
print(f"  residual spectrum: " + " ".join(f"{100*x:.1f}%" for x in (S**2)/(S**2).sum()))
OUT["interaction_pct"] = float(100 * ss_res / ss_tot)

# ---- 2. rank reversals ------------------------------------------------------
rev = 0; tot = 0
for a in range(len(gpus)):
    for b in range(a + 1, len(gpus)):
        signs = {F[i, a] < F[i, b] for i in range(len(F))}
        tot += 1
        if len(signs) > 1:
            rev += 1
            n_a = sum(1 for i in range(len(F)) if F[i, a] < F[i, b])
            print(f"  REVERSAL {gpus[a]} vs {gpus[b]}: {gpus[a]} wins {n_a}/{len(F)} rows")
print(f"\n=== reversals: {rev}/{tot} GPU pairs reverse across (model, config) rows ===")
OUT["reversal_pairs"] = [rev, tot]

# ---- 3. does CONFIGURATION drive the interaction? ---------------------------
v1 = np.linalg.svd(R, full_matrices=False)[2][0]
inter = R @ v1
qs = sorted({k[1][0] for k in Fk})
lb = np.array([math.log2(max(k[1][1], 1)) for k in Fk])
ls = np.array([math.log10(max(k[1][2], 1)) for k in Fk])
print(f"\n=== what correlates with the interaction score? ===")
corrs = {}
for name, v in [("log batch size", lb), ("log sequence length", ls)]:
    if np.std(v) > 0:
        corrs[name] = float(np.corrcoef(v, inter)[0, 1])
        print(f"  {name:26} r = {corrs[name]:+.3f}")
for q in qs:
    d = np.array([1.0 if k[1][0] == q else 0.0 for k in Fk])
    if 0 < d.sum() < len(d):
        corrs[f"quant={q}"] = float(np.corrcoef(d, inter)[0, 1])
        print(f"  {'quant='+q:26} r = {corrs[f'quant={q}']:+.3f}  (n={int(d.sum())})")
OUT["interaction_corr"] = corrs
if corrs:
    best = max(corrs, key=lambda k: abs(corrs[k]))
    sane("a configuration variable is the strongest correlate of the interaction",
         abs(corrs[best]) > 0.2, f"strongest is {best} at r={corrs[best]:+.3f}")

json.dump({"results": OUT, "sanity": SANITY,
           "matrix": {"rows": len(keys), "full_rows": len(full), "gpus": gpus}},
          io.open("experiments/results/llm_perf.json", "w", encoding="utf-8"), indent=1)
print(f"\nsanity: {sum(1 for s in SANITY if s['passed'])}/{len(SANITY)} passed")
print("saved -> experiments/results/llm_perf.json")
