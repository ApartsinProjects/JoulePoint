# -*- coding: utf-8 -*-
"""
RQ1 strengthened: measured power-cap elasticity of REAL GPU-inference workloads (own AWS
A10G sweep) -- the E1 experiment the paper's characterization section needs.

Loads data/raw/own_aws_sweep.csv (7 heterogeneous inference workloads x 5 power caps x 3
reps, measured NVML power + throughput on an A10G) and reports:
  - the heterogeneity spread (strong, unlike the LLM-only emerald data);
  - the compute- vs memory-bound split (a real, measured finding: LLM-decode and
    recommendation/embedding are memory-bound and keep throughput when power-capped);
  - the kill-test decomposition (oracle gain over uniform + per-workload elasticity value).
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import poca_killtest as K

RESULTS = K.RESULTS


def load_pool(replicate=3):
    d = pd.read_csv(os.path.join(K.RAW, "own_aws_sweep.csv"))
    g = d.groupby(["workload", "cap_w"], as_index=False).agg(power=("power_w", "mean"),
                                                             thr=("throughput", "mean"))
    pool = []
    stats = {}
    for wl, x in g.groupby("workload"):
        x = x.sort_values("power")
        p = x["power"].to_numpy(float)
        thr = np.maximum.accumulate(x["thr"].to_numpy(float)); thr = thr / thr.max()
        rng = float(p.max() - p.min())
        stats[wl] = {"draw_range_w": rng, "thr_at_min_cap": float(thr[0]),
                     "kind": "memory-bound" if rng < 40 else "compute-bound"}
        for k in range(replicate):
            xg = np.linspace(p.min(), p.max(), K.NGRID)
            qg = np.interp(xg, p, thr)
            pool.append(dict(name=wl, cls=wl, w=1.0, k=k, xg=xg, qg=qg,
                             cost_g=1.0 * (1 - qg), pmin=float(xg[0]), pmax=float(xg[-1])))
    return pool, stats


def het_spread(pool):
    qs = [K.q_at(wl, wl["pmin"] + 0.5 * (wl["pmax"] - wl["pmin"])) for wl in pool]
    return float(max(qs) - min(qs))


def decompose(pool, fracs=(0.90, 0.85, 0.80, 0.75, 0.70)):
    tb = sum(wl["pmax"] for wl in pool)
    rows = []
    for frac in fracs:
        C = frac * tb
        uni = K.pool_cost_power(pool, K.b1_uniform(pool, C))[0]
        ora = K.pool_cost_power(pool, K.b6_oracle(pool, C))[0]
        shared = K.pool_cost_power(pool, K.b5g_shared_elasticity(pool, C))[0]
        den = uni - ora
        rows.append(dict(curt=round((1 - frac) * 100),
                         oracle_gain_vs_uniform_pct=100 * (uni - ora) / uni if uni > 1e-9 else 0.0,
                         per_workload_elasticity_gain_pct=100 * (shared - ora) / shared if shared > 1e-9 else 0.0))
    return rows


def main():
    pool, stats = load_pool()
    spread = het_spread(pool)
    rows = decompose(pool)
    n_mem = sum(1 for s in stats.values() if s["kind"] == "memory-bound")
    out = {"platform": "AWS EC2 g5.xlarge (NVIDIA A10G), measured power-cap sweep",
           "n_workloads": len(stats), "heterogeneity_spread": spread,
           "n_memory_bound": n_mem, "n_compute_bound": len(stats) - n_mem,
           "per_workload": stats, "decomposition": rows}
    with open(os.path.join(RESULTS, "aws_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("== RQ1: measured elasticity of REAL GPU-inference workloads (AWS A10G) ==")
    print(f"heterogeneity spread {spread:.3f} across {len(stats)} inference workloads "
          f"({n_mem} memory-bound, {len(stats)-n_mem} compute-bound)")
    print(f"\n{'workload':>14} {'draw range W':>12} {'thr @ deepest cap':>18} {'kind':>13}")
    for wl, s in sorted(stats.items(), key=lambda kv: kv[1]["draw_range_w"]):
        print(f"{wl:>14} {s['draw_range_w']:>12.0f} {s['thr_at_min_cap']:>18.2f} {s['kind']:>13}")
    print(f"\n{'curt%':>5} {'oracle↑vsUni%':>13} {'per-workload elast%':>19}")
    for r in rows:
        print(f"{r['curt']:>5} {r['oracle_gain_vs_uniform_pct']:>13.1f} {r['per_workload_elasticity_gain_pct']:>19.1f}")
    print("\nwritten -> results/aws_analysis.json")


if __name__ == "__main__":
    main()
