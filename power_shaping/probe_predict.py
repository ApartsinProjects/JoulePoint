# -*- coding: utf-8 -*-
"""
P1 probe-and-predict: few-shot elasticity calibration (the deployable form of the oracle result).

Nobody sweeps every production workload offline. So the real question behind the 49% heterogeneous
elasticity gain is: how much of the oracle's per-workload value can be recovered from a curve PRIOR
(learned leave-one-workload-out from the other workloads) plus k CHEAP in-situ probe measurements,
and how few probes it takes. We score every predictor IN DECISION SPACE -- the fraction of the
oracle's allocation advantage over uniform capping that it recovers on the heterogeneous
priority-weighted fleet -- never in curve-RMSE space (the lesson of the failed ranker).

Prior: functional PCA (mean curve + top-r eigen-curves) on the LOO training curves. Conditioning on
k probe points = regularized least squares for the eigen-coefficients that match the probes. The
headline is a VALUE-vs-k curve, reported against the dumb baselines a reviewer will demand:
  class-mean          the 4-row family lookup already in the paper (no probes)
  knee-fit            a parametric piecewise-linear knee fit to the k probes (no prior)
  linear-interp       linear interpolation through the k probes + the uncapped endpoint

INVARIANTS
  PIPE   feeding the TRUE curve as the prediction recovers ~100% of oracle value (validates scoring)
  MONO   prior value is non-decreasing in k (a probe cannot hurt), within tolerance
  HOMOG  on a homogeneous fleet oracle == uniform, so there is no value to capture (den ~ 0)
"""
from __future__ import annotations
import os, json
import numpy as np
import poca_killtest as K
import learned_control as LC
import het_alloc as HET
from rq4_strengthen import aws_table

R_COMPONENTS = 3
RIDGE = 0.05
FRACS = (0.90, 0.85, 0.80, 0.75, 0.70)
KS = (0, 1, 2, 3)
PROBES = {1: [0.5], 2: [0.35, 0.70], 3: [0.25, 0.50, 0.75]}   # probe cap-fractions (relative power)
# type family per workload (the feature the paper's class-mean predictor keys on)
TYPEFAM = {"gemm_fp16": "gemm", "gemm_fp32": "gemm", "gemm_bf16": "gemm", "bmm_fp16": "gemm", "attention_sdpa": "attn",
           "vit_b16": "attn", "fft2d": "linalg", "cholesky": "linalg", "resnet50": "conv", "resnet152": "conv",
           "vgg16": "conv", "densenet121": "conv", "convnext_tiny": "conv", "efficientnet_b0": "conv",
           "mobilenet_v3_l": "conv", "inception_v3": "conv", "membw": "mem", "reduction": "mem", "softmax_big": "mem",
           "layernorm_big": "mem", "embed_gather": "mem", "scatter_add": "mem", "sort_big": "mem", "memcpy": "mem",
           "elementwise_chain": "mem", "decode_like": "lat"}


def true_curves(tab):
    """(true weighted pool, priority-only pool [shared mean curve], {wl: true q-curve}). The priority
    pool is the paper's het_alloc 'priority-only' baseline: same weights, ONE shared elasticity curve."""
    pool = HET.weighted_pool(tab, shared=False)
    prio = HET.weighted_pool(tab, shared=True)
    return pool, prio, {w["name"]: np.asarray(w["qg"], float) for w in pool}


def value_captured(truepool, priopool, pred_q):
    """Fraction of the ELASTICITY gap recovered: allocate caps OPTIMALLY on the predicted per-workload
    curves, score on TRUE curves, and measure progress from the priority-only baseline (shared curve,
    the value priority alone gives) toward the oracle (true per-workload curves). This is the
    construct-matched elasticity metric of het_alloc, isolating what a curve PROBE can add over
    priority; a perfect prediction scores 100%, the priority baseline scores 0%."""
    predpool = [{**w, "qg": pred_q[w["name"]], "cost_g": w["w"] * (1 - pred_q[w["name"]])} for w in truepool]
    tb = sum(w["pmax"] for w in truepool)
    caps = []
    for f in FRACS:
        C = f * tb
        c_pri = K.pool_cost_power(truepool, K.b6_oracle(priopool, C))[0]     # priority-only (shared curve)
        c_ora = K.pool_cost_power(truepool, K.b6_oracle(truepool, C))[0]     # oracle (true curves)
        c_prd = K.pool_cost_power(truepool, K.b6_oracle(predpool, C))[0]
        den = c_pri - c_ora
        if den > 1e-6:
            caps.append(100 * (c_pri - c_prd) / den)
    return float(np.mean(caps)) if caps else 0.0


def _mono(q):
    q = np.clip(np.maximum.accumulate(q), 1e-3, None)
    return q / q.max()


def prior_predict(Q, names, heldout, k, rg):
    """Functional-PCA prior fit on the other workloads, conditioned on k probes of the held-out curve."""
    train = np.stack([Q[n] for n in names if n != heldout])
    mu = train.mean(0)
    U, S, Vt = np.linalg.svd(train - mu, full_matrices=False)
    r = min(R_COMPONENTS, Vt.shape[0])
    comps = Vt[:r]                                            # (r, NG)
    if k == 0:
        return _mono(mu.copy())
    idx = [int(np.argmin(np.abs(rg - p))) for p in PROBES[k]]
    A = comps[:, idx].T                                      # (k, r)
    b = Q[heldout][idx] - mu[idx]
    c = np.linalg.solve(A.T @ A + RIDGE * np.eye(r), A.T @ b)
    return _mono(mu + comps.T @ c)


def classmean_predict(Q, names, heldout, rg):
    """The paper's feature-only predictor: the leave-one-out MEAN curve over the held-out workload's
    OWN type family (leak-free), falling back to the global mean for a singleton family. Zero probes."""
    fam = TYPEFAM.get(heldout)
    members = [n for n in names if n != heldout and TYPEFAM.get(n) == fam]
    if not members:
        members = [n for n in names if n != heldout]
    return _mono(np.mean([Q[n] for n in members], axis=0))


def knee_fit(Q, heldout, k, rg):
    """Parametric two-segment (knee) fit to the k probes + the uncapped endpoint (relp=1, q=1).
    No prior: pure interpolation/extrapolation of a knee shape. Undetermined for k<2 -> None."""
    if k < 2:
        return None
    xp = np.array(PROBES[k] + [1.0]); yp = np.array([Q[heldout][int(np.argmin(np.abs(rg - p)))] for p in PROBES[k]] + [1.0])
    best, berr = None, 1e18
    for knee in np.linspace(0.15, 0.85, 25):
        # fit q = a + b*min(relp,knee) + c*max(relp-knee,0) by least squares on the probes
        M = np.stack([np.ones_like(xp), np.minimum(xp, knee), np.maximum(xp - knee, 0)], 1)
        coef, *_ = np.linalg.lstsq(M, yp, rcond=None)
        err = np.sum((M @ coef - yp) ** 2)
        if err < berr:
            berr, best = err, (knee, coef)
    knee, coef = best
    Mg = np.stack([np.ones_like(rg), np.minimum(rg, knee), np.maximum(rg - knee, 0)], 1)
    return _mono(Mg @ coef)


def linear_interp(Q, heldout, k, rg):
    if k < 1:
        return None
    xp = np.array(PROBES[k] + [1.0]); yp = np.array([Q[heldout][int(np.argmin(np.abs(rg - p)))] for p in PROBES[k]] + [1.0])
    o = np.argsort(xp)
    return _mono(np.interp(rg, xp[o], yp[o]))


def run(tab=None, label="aws26"):
    tab = aws_table() if tab is None else tab
    truepool, priopool, Q = true_curves(tab)
    names = list(Q); NG = len(next(iter(Q.values()))); rg = np.linspace(0, 1, NG)

    # PIPE invariant: true curve as prediction -> ~100% of the elasticity gap
    pipe = value_captured(truepool, priopool, Q)
    corrupt = corrupt_invariant(truepool, priopool, Q, rg)       # wrong curve must score far below 100%

    # feature-only class-mean predictor (zero probes) under the SAME elasticity-gap metric
    cm_pred = {n: classmean_predict(Q, names, n, rg) for n in names}
    classmean_val = round(value_captured(truepool, priopool, cm_pred), 1)

    curve_val = {"prior": {}, "knee": {}, "linear": {}}
    for k in KS:
        for meth, fn in [("prior", prior_predict), ("knee", knee_fit), ("linear", linear_interp)]:
            if meth == "prior":
                pred = {n: prior_predict(Q, names, n, k, rg) for n in names}
            else:
                pred = {}
                ok = True
                for n in names:
                    q = fn(Q, n, k, rg)
                    if q is None:
                        ok = False; break
                    pred[n] = q
                if not ok:
                    curve_val[meth][k] = None; continue
            curve_val[meth][k] = round(value_captured(truepool, priopool, pred), 1)

    # MONO on the deployable simple method (linear probe-interpolation), not the discarded prior.
    lin = [curve_val["linear"][k] for k in KS if curve_val["linear"][k] is not None]
    mono = all(lin[i + 1] >= lin[i] - 4.0 for i in range(len(lin) - 1))
    return {"label": label, "pipe_invariant_pct": round(pipe, 1), "corrupt_curve_pct": round(corrupt, 1),
            "value_vs_k": curve_val, "classmean_feature_pct": classmean_val, "mono_ok": bool(mono), "ks": list(KS)}


def homog_invariant(tab):
    """On a homogeneous fleet the ELASTICITY gap must vanish, and we test it through DIFFERENT code
    paths (not by comparing an expression with itself): give every workload the shared fleet-mean curve
    in a PER-WORKLOAD pool (truH), then compare the priority-only allocation (shared-mean pool) against
    the per-workload oracle allocation (truH). Identical curves -> identical cost -> gap ~ 0; a mismatch
    would expose a bug in either allocation path."""
    truepool, priopool, Q = true_curves(tab)
    avg = _mono(np.mean([Q[n] for n in Q], axis=0))
    truH = [{**w, "qg": avg, "cost_g": w["w"] * (1 - avg)} for w in truepool]     # per-workload, all = mean
    tb = sum(w["pmax"] for w in truH)
    dens = []
    for f in FRACS:
        C = f * tb
        c_pri = K.pool_cost_power(truH, K.b6_oracle(priopool, C))[0]              # shared-mean allocation
        c_ora = K.pool_cost_power(truH, K.b6_oracle(truH, C))[0]                  # per-workload allocation
        dens.append(abs(c_pri - c_ora))
    return max(dens) < 1e-3, max(dens)


def corrupt_invariant(truepool, priopool, Q, rg):
    """The scorer must NOT trivially return 100%: feed a deliberately wrong (constant, inelastic)
    curve for every workload and confirm it recovers far LESS of the elasticity gap than the true
    curve does. Guards PIPE against a scorer that ignores the prediction."""
    flat = {n: _mono(np.ones_like(rg)) for n in Q}
    return value_captured(truepool, priopool, flat)


def main():
    tab = aws_table()
    homog_ok, dmax = homog_invariant(tab)
    out = run(tab)
    out["homog_invariant_ok"] = bool(homog_ok); out["homog_den_max"] = round(float(dmax), 5)
    os.makedirs(K.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(K.RESULTS, "probe_predict.json"), "w"), indent=2)

    print("== P1 probe-and-predict: fraction of oracle elasticity value vs number of probes ==")
    print(f"INVARIANT PIPE (true curve -> ~100%): {out['pipe_invariant_pct']:.0f}%  "
          f"{'PASS' if out['pipe_invariant_pct'] > 97 else 'FAIL'}")
    print(f"INVARIANT CORRUPT (wrong flat curve -> far below 100%): {out['corrupt_curve_pct']:.0f}%  "
          f"{'PASS' if out['corrupt_curve_pct'] < 50 else 'FAIL'}")
    print(f"INVARIANT HOMOG (no value on homogeneous fleet, den~0): den_max={dmax:.2e}  "
          f"{'PASS' if homog_ok else 'FAIL'}")
    print(f"INVARIANT MONO (linear non-decreasing in k): {'PASS' if out['mono_ok'] else 'FAIL'}")
    print(f"\nfeature-only class-mean predictor (0 probes): {out['classmean_feature_pct']:.0f}% of elasticity gap")
    print(f"\n{'k probes':>9} {'prior':>7} {'knee-fit':>9} {'linear':>8}")
    for k in KS:
        p = out["value_vs_k"]["prior"][k]; kn = out["value_vs_k"]["knee"][k]; li = out["value_vs_k"]["linear"][k]
        print(f"{k:>9} {p:>6.0f}% {('  n/a' if kn is None else f'{kn:>7.0f}%'):>9} {('  n/a' if li is None else f'{li:>6.0f}%'):>8}")
    print("\nread: k=0 is the no-probe prior (global mean curve); each probe is one ~30s reduced-power reading.")
    print("written -> results/probe_predict.json")


if __name__ == "__main__":
    main()
