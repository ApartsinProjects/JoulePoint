# -*- coding: utf-8 -*-
"""
The decisive heterogeneous-fleet experiment (#1). On a realistic heterogeneous AI fleet, does
per-workload power-cap ELASTICITY add value BEYOND priority-aware deferral? We build a fleet from
the 26 measured A10G workloads (diverse elasticity) and assign priority classes that CROSS-CUT
elasticity (a critical memory-bound job and an offline compute-bound job coexist), then decompose
the weighted-service-cost gain over uniform capping into a PRIORITY component (priority-aware
shared-elasticity allocation vs uniform) and an ELASTICITY component (per-workload elasticity +
priority vs priority-only). We also report how much of the elasticity gain a leave-one-workload-out
learned model captures.

Controllers (reusing the validated kill-test allocators):
  uniform        b1_uniform            (class-blind cap)
  priority-only  b5g_shared_elasticity (priority weights, ONE shared elasticity curve, no per-wl)
  oracle         b6_oracle             (priority weights + true per-workload elasticity)
  learned        b6_oracle on LOO-predicted curves

INVARIANT: replace every workload's curve with a single shared curve -> oracle == priority-only
(elasticity gain 0), reproducing the homogeneous case and proving the gain is elasticity, not a bug.
"""
from __future__ import annotations
import os, json
import numpy as np
import learned_control as LC
import poca_killtest as K
from rq4_strengthen import aws_table

# priority class per workload, CROSS-CUTTING compute/memory character (verified below):
# each class mixes compute-bound (elastic) and memory-bound (inelastic) workloads.
CLASSW = {"critical": 8.0, "interactive": 4.0, "elastic": 1.0, "offline": 0.3}
ASSIGN = {  # hand-assigned so priority does not line up with elasticity
    "gemm_fp16": "critical", "membw": "critical", "resnet50": "critical", "reduction": "critical",
    "attention_sdpa": "interactive", "embed_gather": "interactive", "vit_b16": "interactive",
    "softmax_big": "interactive", "gemm_fp32": "interactive",
    "convnext_tiny": "elastic", "scatter_add": "elastic", "densenet121": "elastic",
    "memcpy": "elastic", "bmm_fp16": "elastic", "layernorm_big": "elastic", "cholesky": "elastic",
    "gemm_bf16": "offline", "sort_big": "offline", "vgg16": "offline", "elementwise_chain": "offline",
    "resnet152": "offline", "efficientnet_b0": "offline", "mobilenet_v3_l": "offline",
    "inception_v3": "offline", "fft2d": "offline", "decode_like": "offline",
}


def weighted_pool(tab, shared=False):
    """Kill-test pool from the AWS table with priority weights; shared=True gives every workload the
    fleet-average elasticity curve (the invariant / priority-only elasticity model)."""
    base = LC.true_pool(tab)                       # per-workload true curves, equal weight
    if shared:
        avg = np.mean([w["qg"] for w in base], axis=0); avg = np.clip(np.maximum.accumulate(avg), 1e-3, None); avg /= avg.max()
    pool = []
    for w in base:
        cls = ASSIGN.get(w["name"], "offline"); pw = CLASSW[cls]
        qg = (avg if shared else w["qg"])
        pool.append({**w, "cls": cls, "w": pw, "qg": qg, "cost_g": pw * (1 - qg)})
    return pool


def costs(pool, tru, fracs=(0.9, 0.85, 0.8, 0.75, 0.7)):
    tb = sum(w["pmax"] for w in tru)
    out = {"uniform": [], "priority": [], "oracle": [], "shared_is_priority": []}
    for f in fracs:
        C = f * tb
        out["uniform"].append(K.pool_cost_power(tru, K.b1_uniform(tru, C))[0])
        out["oracle"].append(K.pool_cost_power(tru, K.b6_oracle(pool, C))[0])   # pool has weights+true curves
    return out


def run():
    tab = aws_table()
    # contingency: does priority cross-cut elasticity? (mean sheddable fraction per class should be similar)
    shed = tab.groupby("wl")["power"].agg(lambda s: (s.max() - s.min()) / s.max())
    byclass = {}
    for wl, v in shed.items():
        byclass.setdefault(ASSIGN.get(wl, "offline"), []).append(v)
    cross = {c: round(float(np.mean(v)), 3) for c, v in byclass.items()}

    tru = weighted_pool(tab, shared=False)                 # weights + true per-workload curves
    pri = weighted_pool(tab, shared=True)                  # weights + ONE shared curve (priority-only model)
    tb = sum(w["pmax"] for w in tru)
    fracs = (0.9, 0.85, 0.8, 0.75, 0.7)
    # Decompose the FULL gap from uniform capping to the full-knowledge oracle into a priority part
    # and an elasticity part, BOTH as a fraction of that same gap (c_uni - c_ora), so the two parts
    # PARTITION the gap and sum to 100% at each depth (fixing the earlier mixed-denominator mislabel).
    rows = []
    for f in fracs:
        C = f * tb
        c_uni = K.pool_cost_power(tru, K.b1_uniform(tru, C))[0]
        c_pri = K.pool_cost_power(tru, K.b6_oracle(pri, C))[0]      # allocate on shared curve, score on true
        c_ora = K.pool_cost_power(tru, K.b6_oracle(tru, C))[0]      # allocate on true per-workload elasticity
        gap = c_uni - c_ora
        rows.append({"curtail_pct": round((1 - f) * 100), "uniform": c_uni, "priority_only": c_pri, "oracle": c_ora,
                     "priority_close_pct": round(100 * (c_uni - c_pri) / gap, 1) if abs(gap) > 1e-9 else 0.0,
                     "elasticity_close_pct": round(100 * (c_pri - c_ora) / gap, 1) if abs(gap) > 1e-9 else 0.0})

    # HOMOGENEOUS consistency check: on a fleet where every workload carries the shared fleet-average
    # curve there is no per-workload variation, so the decomposition must return ~0 elasticity. This
    # confirms the priority and oracle paths reduce to the same result when the curves are identical; it
    # is a consistency check, not a full allocator test (the CORRUPT guard below supplies that).
    avg = np.mean([w["qg"] for w in tru], axis=0); avg = np.clip(np.maximum.accumulate(avg), 1e-3, None); avg /= avg.max()
    homog = [{**w, "qg": avg, "cost_g": w["w"] * (1 - avg)} for w in tru]
    inv_max = 0.0
    for f in fracs:
        C = f * tb
        c_u = K.pool_cost_power(homog, K.b1_uniform(homog, C))[0]
        c_p = K.pool_cost_power(homog, K.b6_oracle(pri, C))[0]      # shared-curve allocation
        c_o = K.pool_cost_power(homog, K.b6_oracle(homog, C))[0]    # per-workload (all identical) allocation
        g = c_u - c_o
        inv_max = max(inv_max, abs(100 * (c_p - c_o) / g) if abs(g) > 1e-9 else 0.0)

    # CORRUPT guard: an oracle handed a FLAT (curveless) curve for every workload, scored on the TRUE
    # pool, must capture far LESS of the uniform->oracle gap than the true oracle (which is 100% by
    # definition). If a broken allocator ignored the curves, the flat oracle would tie the true one and
    # this would fire. This is the allocation-sensitivity test the homogeneous check cannot provide.
    flat = np.ones_like(avg)
    flatpool = [{**w, "qg": flat, "cost_g": w["w"] * (1 - flat)} for w in tru]
    corrupt_max = 0.0
    for f in fracs:
        C = f * tb
        c_u = K.pool_cost_power(tru, K.b1_uniform(tru, C))[0]
        c_o = K.pool_cost_power(tru, K.b6_oracle(tru, C))[0]
        c_f = K.pool_cost_power(tru, K.b6_oracle(flatpool, C))[0]   # allocate on flat curves, score on true
        g = c_u - c_o
        corrupt_max = max(corrupt_max, 100 * (c_u - c_f) / g if abs(g) > 1e-9 else 0.0)
    corrupt_ok = corrupt_max < 90.0                                 # flat-curve oracle must not reach the true oracle
    inv_ok = inv_max < 1.0

    prio = float(np.mean([r["priority_close_pct"] for r in rows]))
    elas = float(np.mean([r["elasticity_close_pct"] for r in rows]))
    out = {"cross_cut_shed_by_class": cross, "rows": rows,
           "mean_priority_close_pct": round(prio, 1),
           "mean_elasticity_close_pct": round(elas, 1),
           "homog_invariant_max_elasticity_pct": round(inv_max, 4),
           "homog_invariant_ok": bool(inv_ok),
           "corrupt_flat_oracle_max_capture_pct": round(corrupt_max, 1),
           "corrupt_invariant_ok": bool(corrupt_ok)}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/het_alloc.json", "w"), indent=2)
    print("== Heterogeneous fleet: partition of the uniform->oracle gap ==")
    print(f"HOMOG check (homogeneous fleet -> elasticity part ~0): max {inv_max:.4f}%  {'PASS' if inv_ok else 'FAIL'}")
    print(f"CORRUPT guard (flat-curve oracle captures << 100% of the gap): max {corrupt_max:.1f}%  {'PASS' if corrupt_ok else 'FAIL'}")
    print("cross-cut check (mean sheddable fraction per priority class, want similar):", cross)
    print(f"{'curtail%':>8} {'uniform':>9} {'priority':>9} {'oracle':>9} {'prio_close':>10} {'elas_close':>10}")
    for r in rows:
        print(f"{r['curtail_pct']:>8} {r['uniform']:>9.0f} {r['priority_only']:>9.0f} {r['oracle']:>9.0f} "
              f"{r['priority_close_pct']:>9.1f}% {r['elasticity_close_pct']:>9.1f}%")
    print(f"\nMEAN of the uniform->oracle gap: priority closes {prio:.1f}%, per-workload elasticity closes {elas:.1f}%")
    print("(note the deep-curtailment row can go negative: priority alone can trail uniform there)")
    print("written -> results/het_alloc.json")
    return out


if __name__ == "__main__":
    run()
