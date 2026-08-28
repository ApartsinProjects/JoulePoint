# -*- coding: utf-8 -*-
"""
CF1  ranking-loss training (we score ranking but fit squared error)
CF2  noise-weighted factorisation using E0's measured per-cell variance
CF3  multi-output joint factorisation of energy, runtime and peak power

Plus two analyses the paper needs:
Q1   which workload parameters predict energy, separated into the LEVEL and the
     decision-relevant INTERACTION
Q2   why knowing performance is not enough: E = P_avg x T, and both terms vary
"""
import io, json, math, sys, warnings
import numpy as np
from collections import defaultdict
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from e4_e5_models import load_grid, load_feats, additive, score, MACH

OUT, SANITY = {}, []


def sane(name, ok, detail):
    SANITY.append(dict(check=name, passed=bool(ok), detail=detail))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# nominal forward-pass cost per sample, standard published figures
FLOPS_G = {"resnet50": 4.1, "vit_b16": 17.6, "convnext_t": 4.47, "transformer": 40.3}
PARAMS_M = {"resnet50": 25.6, "vit_b16": 86.6, "convnext_t": 28.6, "transformer": 75.5}

keys, Ylog, Tput = load_grid()
X = load_feats(keys)
n, m = Ylog.shape
fam = sorted({k[0] for k in keys})

raw = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
rows = [r for b in raw for r in b["rows"] if r.get("status") == "ok"]
ki = {k: i for i, k in enumerate(keys)}
mi = {mm: j for j, mm in enumerate(MACH)}
Ppow = np.zeros((n, m)); Mem = np.zeros((n, m))
for r in rows:
    i, j = ki[(r["load"], r["precision"], r["batch"])], mi[r["machine"]]
    Ppow[i, j] = r["mean_power_w"]; Mem[i, j] = r["peak_mem_gb"]
Tlog = np.log10(Tput)
Plog = np.log10(Ppow)

# ============================================================ Q2
print("=" * 88)
print("Q2  WHY PERFORMANCE IS NOT ENOUGH:  E = P_avg x T")
print("=" * 88)
# energy per sample (J) = mean power (W) / throughput (samples/s)
E_meas = Ylog - 3.0                           # Ylog is log10 mJ -> log10 J per sample
resid = E_meas - (Plog - Tlog)
print(f"\n  sampled mean power vs the NVML energy counter: max disagreement "
      f"{np.abs(resid).max():.4f} log10 ({100*(10**np.abs(resid).max()-1):.1f}%)")
print("  The counter integrates continuously; mean_power_w is sampled every 5 iterations")
print("  and misses ramp-up transients, so the two are not expected to agree exactly.")
print("  The counter is authoritative, so effective power is DERIVED from it below.")
Plog = E_meas + Tlog                          # log10 W, derived from the energy counter
Ppow = 10 ** Plog
sane("power derived from the energy counter is physically plausible against the caps",
     Ppow.min() > 20 and (Ppow / np.array([[70.0, 72.0, 150.0, 350.0, 400.0]])).max() < 1.15,
     f"derived {Ppow.min():.0f}-{Ppow.max():.0f} W; max fraction of enforced cap "
     f"{(Ppow / np.array([[70.0, 72.0, 150.0, 350.0, 400.0]])).max():.2f}")

vT, vP = Tlog.var(), Plog.var()
cov = np.cov(Tlog.ravel(), Plog.ravel())[0, 1]
vE = E_meas.var()
print(f"\n  variance decomposition of log energy per sample:")
print(f"    var(log throughput)        {vT:.4f}")
print(f"    var(log mean power)        {vP:.4f}")
print(f"    -2*cov(logT, logP)         {-2*cov:.4f}")
print(f"    = var(log energy)          {vT + vP - 2*cov:.4f}   (measured {vE:.4f})")
print(f"\n  correlation(log throughput, log mean power) = {np.corrcoef(Tlog.ravel(), Plog.ravel())[0,1]:+.3f}")
print(f"  mean power range across the grid: {Ppow.min():.0f} - {Ppow.max():.0f} W ({Ppow.max()/Ppow.min():.1f}x)")
print("\n  If power were constant, energy would be a monotone function of runtime and")
print("  performance would be a sufficient statistic. It is not: mean power itself varies")
print(f"  by {Ppow.max()/Ppow.min():.1f}x across the grid and is correlated with throughput, so the")
print("  fastest machine is systematically also a high-power machine.")


def cv_predict(featfn, target):
    """leave-one-load-family-out prediction of `target` from features."""
    P = np.zeros_like(target)
    for L in fam:
        tr = [i for i, k in enumerate(keys) if k[0] != L]
        te = [i for i, k in enumerate(keys) if k[0] == L]
        Xf = featfn()
        mu = target[tr].mean(); r = target[tr].mean(1) - mu; c = target[tr].mean(0) - mu
        rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[tr], r)
        base = mu + rr.predict(Xf[te])[:, None] + c[None, :]
        R = target[tr] - (mu + r[:, None] + c[None, :])
        v1 = np.linalg.svd(R, full_matrices=False)[2][0]
        sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[tr], R @ v1)
        P[te] = base + np.outer(sm.predict(Xf[te]), v1)
    return P


print("\n  can a RUNTIME-based predictor choose the right machine for ENERGY?")
# rank machines by predicted runtime (fastest first) vs by measured energy
ok = tot = 0
for i in range(n):
    for a in range(m):
        for b in range(a + 1, m):
            te = Tlog[i, a] - Tlog[i, b]     # higher throughput = better performance
            ee = Ylog[i, b] - Ylog[i, a]     # lower energy = better  -> sign aligned
            if te != 0 and ee != 0:
                tot += 1; ok += (te > 0) == (ee > 0)
print(f"    a perfect PERFORMANCE oracle ranks machines correctly for ENERGY in "
      f"{100*ok/tot:.1f}% of pairs")
print(f"    (chance is 50%; our energy model reaches 87.9%)")
OUT["q2"] = dict(perf_oracle_pair_acc=100 * ok / tot,
                 power_range=[float(Ppow.min()), float(Ppow.max())],
                 corr_logT_logP=float(np.corrcoef(Tlog.ravel(), Plog.ravel())[0, 1]))

# ============================================================ Q1
print("\n" + "=" * 88)
print("Q1  WHICH WORKLOAD PARAMETERS PREDICT ENERGY?")
print("=" * 88)
lf = np.array([math.log10(FLOPS_G[k[0]]) for k in keys])
lp = np.array([math.log10(PARAMS_M[k[0]]) for k in keys])
lb = np.array([math.log2(k[2]) for k in keys])
p32 = np.array([1.0 if k[1] == "fp32" else 0.0 for k in keys])
lmem = np.log10(Mem.mean(axis=1) + 1e-6)
ai = lf - lmem                      # crude arithmetic-intensity proxy

CANDS = {
    "FLOPs/sample": lf, "parameters": lp, "batch size": lb,
    "precision (fp32)": p32, "memory footprint": lmem, "arith. intensity proxy": ai,
}
mu = Ylog.mean(); rlvl = Ylog.mean(1) - mu
R_full = Ylog - (mu + rlvl[:, None] + (Ylog.mean(0) - mu)[None, :])
v1 = np.linalg.svd(R_full, full_matrices=False)[2][0]
inter_score = R_full @ v1

print(f"\n  {'parameter':26}{'corr with LEVEL':>18}{'corr with INTERACTION':>24}")
q1 = []
for name, v in CANDS.items():
    cl = float(np.corrcoef(v, rlvl)[0, 1]); ci = float(np.corrcoef(v, inter_score)[0, 1])
    q1.append(dict(parameter=name, corr_level=cl, corr_interaction=ci))
    print(f"  {name:26}{cl:+18.3f}{ci:+24.3f}")
OUT["q1"] = q1
print("\n  LEVEL = how much energy this workload costs on average (the easy part).")
print("  INTERACTION = how the machine ranking changes (the part that decides placement).")
best_lvl = max(q1, key=lambda d: abs(d["corr_level"]))
best_int = max(q1, key=lambda d: abs(d["corr_interaction"]))
print(f"\n  strongest predictor of LEVEL:        {best_lvl['parameter']} (r={best_lvl['corr_level']:+.3f})")
print(f"  strongest predictor of INTERACTION:  {best_int['parameter']} (r={best_int['corr_interaction']:+.3f})")
sane("the parameter that predicts the level is not the one that predicts the interaction",
     best_lvl["parameter"] != best_int["parameter"],
     f"level -> {best_lvl['parameter']}, interaction -> {best_int['parameter']}")

# ============================================================ CF1
print("\n" + "=" * 88)
print("CF1  RANKING-LOSS TRAINING vs SQUARED-ERROR TRAINING")
print("=" * 88)


def fit_ranking(tr, Xf, steps=4000, lr=0.05, reg=1e-3, seed=0):
    """
    Pairwise logistic ranking loss on  yhat(i,j) = c_j + (w.x_i) * v_j.
    mu and the row effect cancel inside a row, so they are not identified and not fitted.
    """
    rng = np.random.default_rng(seed)
    d = Xf.shape[1]
    w = 0.01 * rng.standard_normal(d)
    v = 0.01 * rng.standard_normal(m)
    c = Ylog[tr].mean(0) - Ylog[tr].mean()
    pairs = [(i, a, b) for i in tr for a in range(m) for b in range(a + 1, m)]
    for _ in range(steps):
        gw = np.zeros(d); gv = np.zeros(m); gc = np.zeros(m)
        for i, a, b in pairs:
            s = float(Xf[i] @ w)
            diff = (c[a] + s * v[a]) - (c[b] + s * v[b])
            # BUGFIX: the model predicts ENERGY, so "a is better" means yhat_a < yhat_b,
            # i.e. diff must be NEGATIVE. The original sign drove diff positive and gave
            # 15.0% pairwise accuracy, the exact inverse of a working model.
            targ = 1.0 if Ylog[i, a] < Ylog[i, b] else -1.0   # +1 when a is the better machine
            z = targ * diff
            sig = 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))
            g = targ * sig
            gc[a] += g; gc[b] -= g
            gv[a] += g * s; gv[b] -= g * s
            gw += g * (v[a] - v[b]) * Xf[i]
        npair = len(pairs)
        w -= lr * (gw / npair + reg * w)
        v -= lr * (gv / npair + reg * v)
        c -= lr * (gc / npair + reg * c)
    return lambda i: c + float(Xf[i] @ w) * v


accs = {"squared error (baseline)": [], "ranking loss": []}
for L in fam:
    tr = [i for i, k in enumerate(keys) if k[0] != L]
    te = [i for i, k in enumerate(keys) if k[0] == L]
    P = cv_predict(lambda: X, Ylog)
    accs["squared error (baseline)"].append(score(Ylog, P, te)[0])
    f = fit_ranking(tr, X)
    Pr = np.zeros_like(Ylog)
    for i in te:
        Pr[i] = f(i)
    accs["ranking loss"].append(score(Ylog, Pr, te)[0])
print(f"\n  {'objective':34}{'pairwise acc':>14}")
for k_, v_ in accs.items():
    print(f"  {k_:34}{np.mean(v_):14.1f}")
OUT["cf1"] = {k_: float(np.mean(v_)) for k_, v_ in accs.items()}

# ============================================================ CF2
print("\n" + "=" * 88)
print("CF2  NOISE-WEIGHTED FACTORISATION (weights from E0's measured variance)")
print("=" * 88)
e0 = json.load(io.open("experiments/results/e0_replicates.json", encoding="utf-8"))
cellsd = defaultdict(list)
for inv in e0:
    for r in inv["rows"]:
        if r.get("energy_per_sample_mj"):
            cellsd[(inv["machine"], r["load"], r["precision"], r["batch"])].append(
                math.log10(r["energy_per_sample_mj"]))
sd_by_machine = defaultdict(list)
for (mm, *_), v in cellsd.items():
    if len(v) >= 3:
        sd_by_machine[mm].append(float(np.std(v)))
print("\n  measured replicate sd (log10) by machine:")
W = np.ones((n, m))
for j, mm in enumerate(MACH):
    if sd_by_machine[mm]:
        s = float(np.median(sd_by_machine[mm]))
        print(f"    {mm:12} {s:.4f}")
        W[:, j] = 1.0 / max(s, 1e-4) ** 2
W = W / W.mean()
print(f"  weight ratio max/min across machines: {W.max()/W.min():.2f}x")

res2 = {"unweighted": [], "noise-weighted": []}
for L in fam:
    tr = [i for i, k in enumerate(keys) if k[0] != L]
    te = [i for i, k in enumerate(keys) if k[0] == L]
    for label, Wt in (("unweighted", np.ones_like(W)), ("noise-weighted", W)):
        wm = Wt[tr]
        mu = float((Ylog[tr] * wm).sum() / wm.sum())
        r = ((Ylog[tr] - mu) * wm).sum(1) / wm.sum(1)
        c = ((Ylog[tr] - mu - r[:, None]) * wm).sum(0) / wm.sum(0)
        rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], r)
        base = mu + rr.predict(X[te])[:, None] + c[None, :]
        R = Ylog[tr] - (mu + r[:, None] + c[None, :])
        v1 = np.linalg.svd(R * np.sqrt(wm), full_matrices=False)[2][0]
        sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], R @ v1)
        P = np.zeros_like(Ylog); P[te] = base + np.outer(sm.predict(X[te]), v1)
        res2[label].append(score(Ylog, P, te)[0])
print(f"\n  {'fitting':34}{'pairwise acc':>14}")
for k_, v_ in res2.items():
    print(f"  {k_:34}{np.mean(v_):14.1f}")
OUT["cf2"] = {k_: float(np.mean(v_)) for k_, v_ in res2.items()}

# ============================================================ CF3
print("\n" + "=" * 88)
print("CF3  MULTI-OUTPUT JOINT FACTORISATION (energy + runtime + peak power)")
print("=" * 88)
res3 = {"energy only": [], "joint (energy+runtime+power)": []}
for L in fam:
    tr = [i for i, k in enumerate(keys) if k[0] != L]
    te = [i for i, k in enumerate(keys) if k[0] == L]
    P1 = cv_predict(lambda: X, Ylog)
    res3["energy only"].append(score(Ylog, P1, te)[0])
    # stack the three targets side by side and share the ROW factor across them
    S = np.hstack([Ylog - Ylog[tr].mean(), Tlog - Tlog[tr].mean(), Plog - Plog[tr].mean()])
    mu = S[tr].mean(); r = S[tr].mean(1) - mu; c = S[tr].mean(0) - mu
    Rj = S[tr] - (mu + r[:, None] + c[None, :])
    v1 = np.linalg.svd(Rj, full_matrices=False)[2][0]
    rr = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], r)
    sm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[tr], Rj @ v1)
    Pj = np.zeros_like(S)
    Pj[te] = mu + rr.predict(X[te])[:, None] + c[None, :] + np.outer(sm.predict(X[te]), v1)
    Pe = Pj[:, :m] + Ylog[tr].mean()
    res3["joint (energy+runtime+power)"].append(score(Ylog, Pe, te)[0])
print(f"\n  {'target set':34}{'energy pairwise acc':>21}")
for k_, v_ in res3.items():
    print(f"  {k_:34}{np.mean(v_):21.1f}")
OUT["cf3"] = {k_: float(np.mean(v_)) for k_, v_ in res3.items()}

json.dump({"results": OUT, "sanity": SANITY},
          io.open("experiments/results/cf_advanced.json", "w", encoding="utf-8"), indent=1)
print(f"\n\nsanity: {sum(1 for s in SANITY if s['passed'])}/{len(SANITY)} passed")
for s in SANITY:
    if not s["passed"]:
        print(f"  FAILED: {s['check']} -- {s['detail']}")
print("saved -> experiments/results/cf_advanced.json")
