# -*- coding: utf-8 -*-
"""
PoC-A ablations: WHERE does the workload-aware gain come from, and is the small
per-workload-elasticity value on the real (homogeneous LLM) pool a property of the data
rather than of the method?

Three scenarios, same continuous kill-test machinery:
  (1) real-weighted   : 8 measured emerald LLM curves, priority weights on
  (2) real-equalweight: same curves, all weights = 1  (isolates PURE curve-shape value)
  (3) surrogate-hetero : compute-bound vs memory-bound archetypes (LABELLED SURROGATE),
                         the heterogeneity the emerald LLM-only data does not contain

For each we report the oracle gain over uniform, decomposed into the share captured by a
priority+AVERAGE-elasticity controller (shared) versus the residual that requires
PER-WORKLOAD elasticity knowledge. This tests the hypothesis: per-workload elasticity
value scales with workload heterogeneity.
"""
from __future__ import annotations
import os, json
import numpy as np
import poca_killtest as K
from elasticity import synthetic_curves

RESULTS = K.RESULTS


def _finegrid_from_points(p_abs, q_rel):
    xg = np.linspace(p_abs.min(), p_abs.max(), K.NGRID)
    qg = np.interp(xg, p_abs, q_rel)
    return xg, qg


def build_surrogate_pool():
    """Heterogeneous pool from the compute-vs-memory-bound surrogate (clearly labelled)."""
    curves = synthetic_curves(seed=0)
    pool = []
    for c in curves:
        wc = c["weight_class"]
        tdp = K.CLASS_TDP.get(wc, 350.0) if hasattr(K, "CLASS_TDP") else \
              {"online": 350., "training": 400., "offline": 300., "dev": 300.}.get(wc, 350.)
        draw = c.get("natural_draw", 0.92)
        base = tdp * draw
        p_abs = np.array(c["power"]) * base          # relative power -> absolute W
        q_rel = np.array(c["throughput"])
        # points are high->low power (CAP_SETTINGS descending); sort ascending for interp
        order = np.argsort(p_abs)
        xg, qg = _finegrid_from_points(p_abs[order], q_rel[order])
        w = {"online": 4.0, "training": 2.0, "offline": 1.0, "dev": 0.5}.get(wc, 1.0)
        pool.append(dict(name=c["workload_type"], cls=wc, w=w, k=c["instance"],
                         xg=xg, qg=qg, cost_g=w * (1.0 - qg),
                         pmin=float(xg[0]), pmax=float(xg[-1])))
    return pool


def set_equal_weight(pool):
    out = []
    for wl in pool:
        w = 1.0
        out.append({**wl, "w": w, "cost_g": 1.0 * (1.0 - wl["qg"])})
    return out


def heterogeneity_spread(pool):
    """Spread of throughput at 50% of each workload's power range."""
    qs = []
    for wl in pool:
        x = wl["pmin"] + 0.5 * (wl["pmax"] - wl["pmin"])
        qs.append(K.q_at(wl, x))
    qs = np.array(qs)
    return float(qs.max() - qs.min())


def run_scenario(pool, fracs=(0.90, 0.80, 0.70)):
    tb = sum(wl["pmax"] for wl in pool)
    rows = []
    for frac in fracs:
        C = frac * tb
        uni = K.pool_cost_power(pool, K.b1_uniform(pool, C))[0]
        ora = K.pool_cost_power(pool, K.b6_oracle(pool, C))[0]
        shared = K.pool_cost_power(pool, K.b5g_shared_elasticity(pool, C))[0]
        denom = uni - ora
        rows.append(dict(
            curt=round((1 - frac) * 100),
            oracle_gain_vs_uniform_pct=100 * (uni - ora) / uni if uni > 1e-9 else 0.0,
            shared_capture_pct=100 * (uni - shared) / denom if denom > 1e-6 else float("nan"),
            per_workload_elasticity_gain_pct=100 * (shared - ora) / shared if shared > 1e-9 else 0.0,
        ))
    return rows


def main():
    real = K.build_pool(replicate=5)
    scenarios = {
        "real_weighted": real,
        "real_equalweight": set_equal_weight(real),
        "surrogate_hetero": build_surrogate_pool(),
    }
    out = {}
    print(f"{'scenario':>18} {'het_spread':>10} {'curt%':>6} {'orac↑vsUni%':>11} "
          f"{'shared_cap%':>11} {'perWL_elast%':>12}")
    for name, pool in scenarios.items():
        spread = heterogeneity_spread(pool)
        rows = run_scenario(pool)
        out[name] = {"heterogeneity_spread": spread, "rows": rows}
        for r in rows:
            print(f"{name:>18} {spread:>10.3f} {r['curt']:>6} "
                  f"{r['oracle_gain_vs_uniform_pct']:>11.1f} {r['shared_capture_pct']:>11.1f} "
                  f"{r['per_workload_elasticity_gain_pct']:>12.1f}")
    with open(os.path.join(RESULTS, "poca_ablations.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwritten -> results/poca_ablations.json")


if __name__ == "__main__":
    main()
