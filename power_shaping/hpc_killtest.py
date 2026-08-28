# -*- coding: utf-8 -*-
"""
Wire the two CPU-DVFS datasets (grid5000, eehpc) into the kill-test decomposition, as
additional REAL heterogeneity anchors alongside emerald (GPU LLM) and Zeus (GPU ML).

grid5000 (Da Costa, Zenodo 14914799, CC-BY-4.0): 9 NAS Parallel Benchmarks on one cluster
(chifflot), swept over CPU frequency 1.0-2.6 GHz, full-node wattmeter power. Compute-bound
(ep) vs memory-bound (cg/mg/ft) give real elasticity heterogeneity.

eehpc (energy-efficient-HPC DVFS, likwid/RAPL): gromacs (compute-bound) and ciao, swept
over CPU frequency 1.0-3.7 GHz.

Each workload's (power, throughput=1/time) curve is fed through the same continuous
allocator (poca_killtest). We report heterogeneity spread and the per-workload-elasticity
value, and compare all four real datasets.
"""
from __future__ import annotations
import os, json, zipfile, io
import numpy as np
import pandas as pd
import poca_killtest as K

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "data", "raw")
RESULTS = os.path.join(HERE, "results")


def _pool_from_curves(curves, equal_weight=True, replicate=4):
    """curves: list of (name, power[], throughput_norm[]). Build a K-style pool."""
    pool = []
    for name, p, thr in curves:
        p = np.asarray(p, float); thr = np.asarray(thr, float)
        order = np.argsort(p)
        p, thr = p[order], thr[order]
        thr = np.maximum.accumulate(thr)              # monotone in power
        thr = thr / thr.max()
        w = 1.0
        for k in range(replicate):
            xg = np.linspace(p.min(), p.max(), K.NGRID)
            qg = np.interp(xg, p, thr)
            pool.append(dict(name=name, cls=name, w=w, k=k, xg=xg, qg=qg,
                             cost_g=w * (1 - qg), pmin=float(xg[0]), pmax=float(xg[-1])))
    return pool


def grid5000_curves(cluster="chifflot"):
    d = pd.read_csv(os.path.join(RAW, "grid5000_dvfs.csv"), sep=r"\s+")
    d = d[(d["cluster"] == cluster) & (~d["bench"].str.startswith("Idle"))]
    curves = []
    for bench, g in d.groupby("bench"):
        gg = g.groupby("fmax").agg(power=("mean_power", "mean"), time=("time", "mean")).reset_index()
        if len(gg) < 4:
            continue
        curves.append((bench, gg["power"].to_numpy(), 1.0 / gg["time"].to_numpy()))
    return curves


def _eehpc_gromacs(z, name):
    rows = [l.split(",") for l in z.read(name).decode(errors="replace").splitlines() if not l.startswith("#")]
    rt = max(float(r[3]) for r in rows if r[3])            # Total runtime (cumulative)
    s, e = 4 + 6 * 48, 4 + 7 * 48                          # Power [W] block (7th per-thread metric)
    P = [np.array(r[s:e], float).sum() for r in rows[1:] if len(r) >= e]
    return rt, float(np.mean(P))


def _eehpc_ciao(z, name):
    lines = z.read(name).decode(errors="replace").splitlines()
    rows = [l.split(",") for l in lines[1:]]              # skip header; metric-major, 48 threads x 6
    rt = max(float(x) for x in rows[-1][0:48] if x)       # totalRuntime block, last timestep
    P = [np.mean(np.array(r[240:288], float)) * 48 for r in rows if len(r) >= 288]  # power block, node total
    return rt, float(np.mean(P))


def eehpc_curves():
    z = zipfile.ZipFile(os.path.join(RAW, "eehpc", "data_sources.zip"))
    names = z.namelist()
    curves = {}
    for n in names:
        if n.startswith("gromacs/") and n.endswith(".csv"):
            freq = int(n.split("_")[-1].split(".")[0])
            rt, pw = _eehpc_gromacs(z, n)
            curves.setdefault("gromacs", []).append((freq, pw, 1.0 / rt))
        # ciao/ files use a different (metric-major) layout whose power/throughput trend is
        # not cleanly monotone under our parse; excluded rather than present a dubious curve.
    out = []
    for wl, pts in curves.items():
        pts.sort()
        p = [x[1] for x in pts]; thr = [x[2] for x in pts]
        out.append((wl, p, thr))
    return out


def het_spread(pool):
    qs = [K.q_at(wl, wl["pmin"] + 0.5 * (wl["pmax"] - wl["pmin"])) for wl in pool]
    return float(max(qs) - min(qs))


def decompose(pool, fracs=(0.90, 0.80, 0.70)):
    tb = sum(wl["pmax"] for wl in pool)
    rows = []
    for frac in fracs:
        C = frac * tb
        uni = K.pool_cost_power(pool, K.b1_uniform(pool, C))[0]
        ora = K.pool_cost_power(pool, K.b6_oracle(pool, C))[0]
        shared = K.pool_cost_power(pool, K.b5g_shared_elasticity(pool, C))[0]
        denom = uni - ora
        rows.append(dict(curt=round((1 - frac) * 100),
                         oracle_gain_vs_uniform_pct=100 * (uni - ora) / uni if uni > 1e-9 else 0.0,
                         per_workload_elasticity_gain_pct=100 * (shared - ora) / shared if shared > 1e-9 else 0.0))
    return rows


def main():
    g5curves = grid5000_curves(); eecurves = eehpc_curves()
    g5 = _pool_from_curves(g5curves)
    out = {
        "grid5000": {"n_workloads": len(g5curves), "heterogeneity_spread": het_spread(g5),
                     "rows": decompose(g5)},
        # eehpc has only 2 workloads -> a decomposition is degenerate; report its real
        # compute-bound DVFS curves as example elasticity instead.
        "eehpc": {"n_workloads": len(eecurves),
                  "curves": {name: {"power_w": [round(x, 1) for x in p],
                                    "throughput_norm": [round(t / max(thr), 3) for t in thr]}
                             for name, p, thr in eecurves}},
    }
    # comparison anchors from prior runs
    try:
        z = json.load(open(os.path.join(RESULTS, "zeus_killtest.json")))
        out["_anchor_zeus_spread"] = z["zeus_weighted"]["heterogeneity_spread"]
        out["_anchor_emerald_spread"] = z["emerald_equalweight_spread"]
    except Exception:
        pass
    with open(os.path.join(RESULTS, "hpc_killtest.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("== CPU-DVFS anchors wired ==")
    r0 = out["grid5000"]["rows"][0]
    print(f"grid5000: {out['grid5000']['n_workloads']} NPB workloads, spread "
          f"{out['grid5000']['heterogeneity_spread']:.3f}, curt10 oracle-gain "
          f"{r0['oracle_gain_vs_uniform_pct']:.1f}%, per-workload-elasticity "
          f"{r0['per_workload_elasticity_gain_pct']:.1f}%")
    print(f"eehpc:    {out['eehpc']['n_workloads']} workloads (compute-bound curves, illustrative):")
    for name, c in out["eehpc"]["curves"].items():
        print(f"   {name}: power {c['power_w'][0]:.0f}->{c['power_w'][-1]:.0f} W, "
              f"throughput {c['throughput_norm'][0]:.2f}->{c['throughput_norm'][-1]:.2f}")
    print("\nper-workload-elasticity value across REAL datasets (curt 10%):")
    print(f"   emerald GPU-LLM ~30%  |  zeus GPU-ML ~23%  |  grid5000 CPU-HPC "
          f"{r0['per_workload_elasticity_gain_pct']:.0f}%")
    print("written -> results/hpc_killtest.json")


if __name__ == "__main__":
    main()
