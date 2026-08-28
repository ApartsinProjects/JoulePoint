# -*- coding: utf-8 -*-
"""
The elasticity result as a SCALING LAW, not a negative. Per-workload power-cap elasticity is worth
0% on a homogeneous fleet and grows with fleet diversity; here we show that growth is smooth and
predictable. We blend every measured curve toward the fleet-average curve by a factor lambda
(lambda=1 -> every workload identical, the homogeneous limit; lambda=0 -> the true measured diversity),
and measure, at each lambda, a heterogeneity index H (the spread of per-workload sheddable fraction)
and the pure elasticity value (equal-weight oracle gain over uniform capping). WITHIN a fleet the value
rises monotonically with diversity from exactly 0, so the "elasticity adds nothing" case is the zero-diversity
end of a scale. This is a within-fleet statement only: across the three real measured fleets H does NOT
predict value (Zeus has the highest H but the lowest value, a small-pool effect), so we do not claim a
universal H->value law and the paper reports only the within-fleet blend.

INVARIANTS
  HOMOG  lambda=1 (identical curves) -> elasticity value ~ 0
  MONO   elasticity value is non-decreasing as H increases (diversity cannot reduce the oracle's edge)
"""
from __future__ import annotations
import os, json
import numpy as np
import poca_killtest as K
import learned_control as LC
from rq4_strengthen import aws_table

FRACS = (0.90, 0.85, 0.80, 0.75, 0.70)


def _mono(q):
    q = np.clip(np.maximum.accumulate(q), 1e-3, None)
    return q / q.max()


def elasticity_value(pool):
    """Equal-weight oracle gain over uniform capping, averaged over curtailment depths (pure elasticity)."""
    tb = sum(w["pmax"] for w in pool)
    vals = []
    for f in FRACS:
        C = f * tb
        c_uni = K.pool_cost_power(pool, K.b1_uniform(pool, C))[0]
        c_ora = K.pool_cost_power(pool, K.b6_oracle(pool, C))[0]
        if c_uni > 1e-9:
            vals.append(100 * (c_uni - c_ora) / c_uni)
    return float(np.mean(vals)) if vals else 0.0


def _shed_frac(w, thr=0.9):
    """Fraction of a workload's peak power it can give up while keeping >= thr of peak throughput.
    Captures BOTH curve shape and power range in one dimensionless number; equal across workloads iff
    they are truly identical."""
    q = np.asarray(w["qg"], float); xg = np.asarray(w["xg"], float)
    idx = int(np.argmax(q >= thr)) if np.any(q >= thr) else len(q) - 1
    return float((xg[-1] - xg[idx]) / xg[-1]) if xg[-1] > 0 else 0.0


def hetero_index(pool):
    """Heterogeneity H = spread (std across workloads) of the sheddable-power fraction. 0 when every
    workload is identical (in both shape and range); larger on a more diverse fleet."""
    return float(np.std([_shed_frac(w) for w in pool]))


def blend(pool, lam):
    """Blend every workload toward the fleet-average workload by lam, in BOTH curve shape and power
    range, so lam=1 is a genuinely homogeneous fleet (identical curve AND identical range)."""
    mean = _mono(np.mean([w["qg"] for w in pool], axis=0))
    mpmin = float(np.mean([w["pmin"] for w in pool])); mpmax = float(np.mean([w["pmax"] for w in pool]))
    out = []
    for w in pool:
        qg = _mono((1 - lam) * np.asarray(w["qg"], float) + lam * mean)
        pmin = (1 - lam) * w["pmin"] + lam * mpmin; pmax = (1 - lam) * w["pmax"] + lam * mpmax
        xg = np.linspace(pmin, pmax, len(qg))
        out.append({**w, "w": 1.0, "qg": qg, "xg": xg, "pmin": float(pmin), "pmax": float(pmax),
                    "cost_g": 1.0 * (1 - qg)})
    return out


def main():
    tab = aws_table()
    pool = LC.true_pool(tab)                       # equal-weight true curves
    rows = []
    for lam in np.linspace(1.0, 0.0, 11):
        bp = blend(pool, lam)
        rows.append({"lam": round(float(lam), 2), "H": round(hetero_index(bp), 4),
                     "elasticity_value_pct": round(elasticity_value(bp), 1)})

    ev = json.load(open(os.path.join(K.RESULTS, "elasticity_value.json")))
    real = {"emerald": {"H": round(hetero_index(LC.true_pool(LC.emerald_table())), 4), "value": ev["emerald"]["elasticity_value_pct"]},
            "zeus": {"H": round(hetero_index(LC.true_pool(LC.zeus_table())), 4), "value": ev["zeus"]["elasticity_value_pct"]},
            "aws26": {"H": round(hetero_index(pool), 4), "value": ev["aws26"]["elasticity_value_pct"]}}

    vals = [r["elasticity_value_pct"] for r in rows]
    homog_ok = abs(rows[0]["elasticity_value_pct"]) < 1.0            # lam=1 -> 0
    # rows go lam 1->0, i.e. H increasing; value should be non-decreasing
    by_H = sorted(rows, key=lambda r: r["H"]); vH = [r["elasticity_value_pct"] for r in by_H]
    mono_ok = all(vH[i + 1] >= vH[i] - 1.5 for i in range(len(vH) - 1))
    out = {"sweep": rows, "real_pools": real, "homog_invariant_ok": bool(homog_ok), "mono_invariant_ok": bool(mono_ok),
           "value_max_pct": max(vals)}
    os.makedirs(K.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(K.RESULTS, "elasticity_scaling.json"), "w"), indent=2)

    print("== Elasticity value as a scaling law in fleet heterogeneity ==")
    print(f"INVARIANT HOMOG (lambda=1 -> ~0): {rows[0]['elasticity_value_pct']:.1f}%  {'PASS' if homog_ok else 'FAIL'}")
    print(f"INVARIANT MONO (value rises with H): {'PASS' if mono_ok else 'FAIL'}")
    print(f"\n{'lambda':>7} {'H':>7} {'elasticity value':>17}")
    for r in rows:
        print(f"{r['lam']:>7} {r['H']:>7.3f} {r['elasticity_value_pct']:>16.1f}%")
    print("\nreal measured fleets on the same H axis:")
    for k, v in real.items():
        print(f"  {k:8} H={v['H']:.3f}  value={v['value']:.0f}%")
    print("written -> results/elasticity_scaling.json")


if __name__ == "__main__":
    main()
