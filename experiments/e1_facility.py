# -*- coding: utf-8 -*-
"""E1. Facility-level re-analysis using E0's MEASURED per-machine idle power."""
import io, json, numpy as np, sys
sys.path.insert(0, "experiments")
from e6_contention import (load_tables, simulate, build_predictor, IDLE_MEASURED, CAP,
                           p_random, p_fastest, p_lowtdp, p_static, p_oracle, p_model,
                           MACH, RUN_LOG, JOB_LOG)
import e6_contention as E6

keys, E, T = load_tables()
E6.EE, E6.TT = E, T
E6.STATIC_RANK = sorted(MACH, key=lambda m: np.median([E[k][m] for k in keys]))
pred = build_predictor(keys, E)

print("E0-measured idle power vs the flat 38% proxy used earlier:")
print(f"{'machine':12}{'cap W':>7}{'measured idle':>15}{'% of cap':>10}{'38% proxy':>12}")
for m in MACH:
    print(f"{m:12}{CAP[m]:7.0f}{IDLE_MEASURED[m]:15.1f}{100*IDLE_MEASURED[m]/CAP[m]:9.0f}%{0.38*CAP[m]:12.1f}")

POOLS = {"full heterogeneity (2 each)": {m: 2 for m in MACH},
         "L4+A100 (5+5)": {"L4": 5, "A100-40GB": 5}}
policies = [("random-free", p_random), ("fastest-free", p_fastest),
            ("lowest-TDP-free", p_lowtdp), ("static ranking lookup", p_static),
            ("predicted energy", p_model), ("oracle (measured)", p_oracle)]
lams = [0.05, 0.10, 0.20]

for pname, pool in POOLS.items():
    print(f"\n{'='*92}\nFACILITY ENERGY PER JOB (J), MEASURED idle -- pool: {pname}\n{'='*92}")
    print(f"{'policy':28}" + "".join(f"{'lam='+str(l):>14}" for l in lams))
    base = {}
    for name, pol in policies:
        line = f"{name:28}"; vals = []
        for lam in lams:
            rs = [simulate(keys, E, T, pool, pol, lam, None, seed=s, predictor=pred,
                           tag=f"E1|{pname}|{name}|lam{lam}") for s in range(3)]
            v = float(np.mean([r["per_job_j"] for r in rs])); vals.append(v); line += f"{v:14.1f}"
        base[name] = vals; print(line)
    print("  saving of predicted-energy vs:")
    for b in ["fastest-free", "static ranking lookup"]:
        print(f"    {b:26}" + "".join(
            f"{100*(base[b][i]-base['predicted energy'][i])/base[b][i]:13.1f}%" for i in range(len(lams))))
    print(f"    {'oracle headroom remaining':26}" + "".join(
        f"{100*(base['predicted energy'][i]-base['oracle (measured)'][i])/base['oracle (measured)'][i]:13.1f}%" for i in range(len(lams))))

json.dump(E6.RUN_LOG, io.open("experiments/results/e1_facility_runs.json", "w", encoding="utf-8"))
print(f"\nsaved {len(E6.RUN_LOG)} runs -> experiments/results/e1_facility_runs.json")
