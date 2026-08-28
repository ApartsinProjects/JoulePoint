# -*- coding: utf-8 -*-
"""
Sensitivity of the priority/elasticity gap-partition to the CONSTRUCTED class assignment and weights.
het_alloc reports one partition (priority ~53%, elasticity ~47% of the uniform->oracle gap) under one
hand-chosen cross-cutting assignment. A reviewer will ask whether that split is an artifact of the
assignment. Here we sweep (a) many RANDOM class assignments and (b) a grid of class-weight ratios, and
report the DISTRIBUTION of the elasticity share, so the headline is a defended range, not a single number.

Reuses het_alloc.weighted_pool (via its module globals) and the kill-test allocators unchanged.

INVARIANT: with EQUAL class weights, priority ordering carries no information, so the priority component
must collapse toward 0 and elasticity should account for essentially the whole gap.
"""
from __future__ import annotations
import os, json
import numpy as np
import het_alloc as H
import learned_control as LC
import poca_killtest as K
from rq4_strengthen import aws_table

FRACS = (0.90, 0.85, 0.80, 0.75, 0.70)
CLASSES = ["critical", "interactive", "elastic", "offline"]
ORIG_ASSIGN = dict(H.ASSIGN)
ORIG_CLASSW = dict(H.CLASSW)


def elas_share(tab, assign, classw):
    """Mean elasticity share (%) of the uniform->oracle gap, using het_alloc's own pool builder."""
    H.ASSIGN = assign
    H.CLASSW = classw
    tru = H.weighted_pool(tab, shared=False)
    pri = H.weighted_pool(tab, shared=True)
    tb = sum(w["pmax"] for w in tru)
    es, ps = [], []
    for f in FRACS:
        C = f * tb
        c_uni = K.pool_cost_power(tru, K.b1_uniform(tru, C))[0]
        c_pri = K.pool_cost_power(tru, K.b6_oracle(pri, C))[0]
        c_ora = K.pool_cost_power(tru, K.b6_oracle(tru, C))[0]
        g = c_uni - c_ora
        if abs(g) > 1e-9:
            es.append(100 * (c_pri - c_ora) / g)
            ps.append(100 * (c_uni - c_pri) / g)
    return (float(np.mean(es)) if es else 0.0), (float(np.mean(ps)) if ps else 0.0)


def main():
    tab = aws_table()
    wls = list(tab["wl"].unique())
    rng = np.random.default_rng(0)

    base_e, base_p = elas_share(tab, ORIG_ASSIGN, ORIG_CLASSW)

    # (a) random cross-cutting assignments, fixed paper weights
    e_rand = []
    for _ in range(300):
        assign = {w: CLASSES[int(rng.integers(4))] for w in wls}
        e, _ = elas_share(tab, assign, ORIG_CLASSW)
        e_rand.append(e)
    e_rand = np.array(e_rand)

    # (b) class-weight-ratio sweep (critical:interactive:elastic:offline), fixed paper assignment
    weight_grid = [(4, 2, 1, 0.3), (8, 4, 1, 0.3), (16, 4, 1, 0.1), (8, 4, 2, 1), (2, 2, 1, 1), (4, 4, 4, 4)]
    e_w = []
    for cw in weight_grid:
        e, p = elas_share(tab, ORIG_ASSIGN, dict(zip(CLASSES, cw)))
        e_w.append({"weights": list(cw), "elasticity_close_pct": round(e, 1), "priority_close_pct": round(p, 1)})

    # INVARIANT: equal weights -> priority component ~0
    e_eq, p_eq = elas_share(tab, ORIG_ASSIGN, dict(zip(CLASSES, (1, 1, 1, 1))))
    inv_ok = abs(p_eq) < 10.0

    out = {
        "paper_assignment": {"elasticity_close_pct": round(base_e, 1), "priority_close_pct": round(base_p, 1)},
        "random_assignments": {
            "n": int(len(e_rand)), "mean": round(float(e_rand.mean()), 1), "std": round(float(e_rand.std()), 1),
            "p10": round(float(np.percentile(e_rand, 10)), 1), "p50": round(float(np.percentile(e_rand, 50)), 1),
            "p90": round(float(np.percentile(e_rand, 90)), 1), "min": round(float(e_rand.min()), 1),
            "max": round(float(e_rand.max()), 1)},
        "weight_sweep": e_w,
        "equal_weight_invariant": {"priority_close_pct": round(p_eq, 1), "elasticity_close_pct": round(e_eq, 1),
                                   "ok": bool(inv_ok)},
    }
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/het_sensitivity.json", "w"), indent=2)

    print("== Sensitivity of the priority/elasticity partition ==")
    print(f"paper assignment: elasticity closes {base_e:.1f}% (priority {base_p:.1f}%) of the gap")
    r = out["random_assignments"]
    print(f"random assignments (n={r['n']}): elasticity share mean {r['mean']:.1f}% "
          f"[p10 {r['p10']:.1f}, p50 {r['p50']:.1f}, p90 {r['p90']:.1f}], range [{r['min']:.1f}, {r['max']:.1f}]")
    print("weight sweep (elasticity close %):")
    for w in e_w:
        print(f"  weights {w['weights']}: elasticity {w['elasticity_close_pct']:.1f}%  priority {w['priority_close_pct']:.1f}%")
    print(f"INVARIANT equal-weight -> priority ~0: priority {p_eq:.1f}%  {'PASS' if inv_ok else 'FAIL'}")
    print("written -> results/het_sensitivity.json")


if __name__ == "__main__":
    main()
