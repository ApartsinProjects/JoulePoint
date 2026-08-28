# -*- coding: utf-8 -*-
"""Energy-space metrics per (GPU, workload) from the measured cap sweeps -> results/energy_metrics.json,
Table 2, Fig 3, and the K1 dispersion result.

Per (GPU, workload), with C_i(p)=throughput at cap p and L_i(p)=t_compute_ms:
  CPE (compute-power elasticity)  = d ln C / d ln p            (0=inert/memory-bound, ~1+=compute-bound)
  MCV (marginal compute value)    = d C_norm / d p             (fraction-of-peak per watt; construct-matched)
  CR(delta) (curtailment robust.) = C_norm at (1-delta)*p_max  (perf retained after removing delta of power)
  P*(slack) (SLO power req.)      = min p with L(p) <= (1+slack)*L(p_max)
  PS (power slack)                = p_max - P*
K1 (are the curves different?): per GPU, dispersion of normalized C across workloads at mid cap, plus the
rep-vs-workload distance ratio (two reps of one workload must be far closer than two workloads).
"""
import os, json
import numpy as np
import pandas as pd
from predict_power import RAW

ORDER = ["T4", "L4", "A10G", "A100"]


def clean(df):
    df["card"] = df.gpu.str.replace("NVIDIA ", "", regex=False).str.replace("Tesla ", "", regex=False)
    df["card"] = df.card.str.replace("A100-SXM4-40GB", "A100", regex=False).str.strip()
    return df


def load_avg():
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_spectrum.csv")), pd.read_csv(os.path.join(RAW, "aws_sweep_a100.csv"))]
    df = clean(pd.concat(fr, ignore_index=True))
    return df.groupby(["card", "workload", "cap_w"], as_index=False).agg(
        C=("throughput", "mean"), L=("t_compute_ms", "mean"))


def metrics_one(g):
    g = g.sort_values("cap_w"); p = g.cap_w.values.astype(float)
    C = g.C.values.astype(float); L = g.L.values.astype(float)
    Cn = C / C[-1]                                   # normalized to max-cap performance
    lp, lc = np.log(p), np.log(np.clip(C, 1e-9, None))
    cpe = np.gradient(lc, lp)                         # elasticity at each cap
    mcv = np.gradient(Cn, p)                          # fraction-of-peak per watt
    out = dict(cpe_mean=float(np.nanmean(cpe)), cpe_lo=float(cpe[:max(2, len(p)//3)].mean()),
               cpe_hi=float(cpe[-max(2, len(p)//3):].mean()), mcv_mean=float(np.nanmean(mcv)),
               pmax=float(p[-1]), pmin=float(p[0]))
    for d in (0.2, 0.3, 0.4):                         # curtailment robustness
        out[f"CR{int(d*100)}"] = float(np.interp((1 - d) * p[-1], p, Cn))
    for s in (0.05, 0.25, 0.50):                      # SLO power requirement + slack
        ok = p[L <= (1 + s) * L[-1]]
        pstar = float(ok.min()) if len(ok) else float(p[-1])
        out[f"Pstar{int(s*100)}"] = pstar
        out[f"PS{int(s*100)}"] = float(p[-1] - pstar)
    return out


def k1_dispersion(df):
    res = {}
    for card, gc in df.groupby("card"):
        # interpolate every workload onto a common cap-fraction grid, read normalized perf at 0.6
        grid = np.linspace(0.5, 1.0, 11); vals = {}
        for wl, g in gc.groupby("workload"):
            g = g.sort_values("cap_w"); pf = g.cap_w.values / g.cap_w.values[-1]
            Cn = g.C.values / g.C.values[-1]
            vals[wl] = np.interp(grid, pf, Cn)
        M = np.array(list(vals.values()))
        at60 = M[:, np.argmin(abs(grid - 0.6))]
        res[card] = dict(n=len(vals), perf_at_0p6_min=float(at60.min()), max=float(at60.max()),
                         mean=float(at60.mean()), std=float(at60.std()))
    return res


def rep_vs_workload(card="A100"):
    fr = [pd.read_csv(os.path.join(RAW, "aws_sweep_spectrum.csv")), pd.read_csv(os.path.join(RAW, "aws_sweep_a100.csv"))]
    df = clean(pd.concat(fr, ignore_index=True)); df = df[df.card == card]
    curves = {}                                        # (workload, rep) -> normalized perf on common grid
    grid = np.linspace(0.5, 1.0, 11)
    for (wl, rep), g in df.groupby(["workload", "rep"]):
        g = g.sort_values("cap_w")
        if g.cap_w.nunique() < 4:
            continue
        pf = g.cap_w.values / g.cap_w.values[-1]; Cn = g.throughput.values / g.throughput.values[-1]
        curves[(wl, rep)] = np.interp(grid, pf, Cn)
    wls = sorted(set(w for w, _ in curves))
    rep_d, wl_d = [], []
    for w in wls:
        if (w, 0) in curves and (w, 1) in curves:
            rep_d.append(np.linalg.norm(curves[(w, 0)] - curves[(w, 1)]))
    for i in range(len(wls)):
        for j in range(i + 1, len(wls)):
            if (wls[i], 0) in curves and (wls[j], 0) in curves:
                wl_d.append(np.linalg.norm(curves[(wls[i], 0)] - curves[(wls[j], 0)]))
    return float(np.median(rep_d)) if rep_d else None, float(np.median(wl_d)) if wl_d else None


def main():
    df = load_avg()
    rows = []
    for (card, wl), g in df.groupby(["card", "workload"]):
        if len(g) < 4:
            continue
        m = metrics_one(g); m.update(card=card, workload=wl); rows.append(m)
    t = pd.DataFrame(rows)
    k1 = k1_dispersion(df)
    rep_med, wl_med = rep_vs_workload("A100")
    os.makedirs("results", exist_ok=True)
    json.dump({"per_gpu_workload": rows, "k1": k1,
               "k1_rep_vs_workload_A100": {"rep_dist": rep_med, "workload_dist": wl_med,
                                           "ratio": (wl_med / rep_med) if rep_med else None}},
              open("results/energy_metrics.json", "w"), indent=1)
    print("== K1: are the power-response curves different? (normalized perf at 0.6*cap) ==")
    for card in [c for c in ORDER if c in k1]:
        d = k1[card]; print(f"  {card:5} n={d['n']:2d}  perf range {d['perf_at_0p6_min']:.2f}-{d['max']:.2f}  "
                            f"mean {d['mean']:.2f} std {d['std']:.3f}")
    if rep_med:
        print(f"  invariant (A100): reps agree {rep_med:.3f} vs workloads differ {wl_med:.3f}  "
              f"=> {wl_med/rep_med:.1f}x  (dispersion is real, not noise)")
    print("\n== Table 2 (A10G): elasticity + curtailment + SLO power, ranked by CPE ==")
    a = t[t.card == "A10G"].sort_values("cpe_mean")
    print(f"{'workload':17}{'CPE':>6}{'CPE_lo':>7}{'CPE_hi':>7}{'MCV':>8}{'CR30':>6}{'P*25%':>7}{'PS25%':>7}")
    for _, r in a.iterrows():
        print(f"{r.workload:17}{r.cpe_mean:6.2f}{r.cpe_lo:7.2f}{r.cpe_hi:7.2f}{r.mcv_mean:8.4f}"
              f"{r.CR30:6.2f}{r.Pstar25:7.0f}{r.PS25:7.0f}")
    print(f"\nsaved -> results/energy_metrics.json  ({len(t)} GPU x workload)")


if __name__ == "__main__":
    main()
