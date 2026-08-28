# -*- coding: utf-8 -*-
"""Full analysis backing the pilot technical report."""
import io, json, math
from collections import defaultdict

d = json.load(io.open("experiments/pilot_results.json", encoding="utf-8"))
rows = [r for b in d for r in b["rows"] if r.get("status") == "ok"]
MACH = ["T4", "L4", "A10G", "L40S", "A100-40GB"]
LOADS = ["resnet50", "vit_b16", "convnext_t", "transformer"]
caps = {b["machine"]: b.get("power_cap_w") for b in d}
devs = {b["machine"]: b.get("device") for b in d}

E = defaultdict(dict)   # (load,prec,bs) -> machine -> energy/sample mJ
T = defaultdict(dict)   # throughput
P = defaultdict(dict)   # peak power
for r in rows:
    k = (r["load"], r["precision"], r["batch"])
    E[k][r["machine"]] = r["energy_per_sample_mj"]
    T[k][r["machine"]] = r["throughput_sps"]
    P[k][r["machine"]] = r["peak_power_w"]

print("=" * 70)
print("MACHINES")
for m in MACH:
    print(f"  {m:11} {devs[m]:26} cap={caps[m]:.0f} W")
print(f"\ncells: {len(rows)} measured, {len(E)} (load,precision,batch) configurations")
print(f"energy source: {set(r['energy_source'] for r in rows)}")

# ---------- 1. spread ----------
sp = sorted((max(v.values()) / min(v.values()), k) for k, v in E.items())
print("\n" + "=" * 70)
print("1. SPREAD (worst/best machine, same load+config)")
print(f"   min {sp[0][0]:.2f}x  median {sp[len(sp)//2][0]:.2f}x  max {sp[-1][0]:.2f}x")
print(f"   max case: {sp[-1][1]}")

# ---------- 2. reversals ----------
print("\n" + "=" * 70)
print("2. RANK REVERSALS (same load, ordering flips across configurations)")
rev = []
for load in LOADS:
    cfgs = [k for k in E if k[0] == load]
    for i, a in enumerate(MACH):
        for b in MACH[i+1:]:
            signs = {(E[c][a] < E[c][b]) for c in cfgs}
            if len(signs) > 1:
                # what drives the flip: precision or batch?
                byprec = {p: {E[c][a] < E[c][b] for c in cfgs if c[1] == p} for p in ("fp16", "fp32")}
                bybs = {n: {E[c][a] < E[c][b] for c in cfgs if c[2] == n} for n in (8, 32, 128)}
                prec_flip = all(len(v) == 1 for v in byprec.values()) and byprec["fp16"] != byprec["fp32"]
                bs_flip = any(len(v) > 1 for v in bybs.values())
                rev.append((load, a, b, "precision" if prec_flip else ("batch" if bs_flip else "mixed")))
npairs = len(LOADS) * len(MACH) * (len(MACH) - 1) // 2
print(f"   {len(rev)} of {npairs} load-machine-pairs reverse ({100*len(rev)/npairs:.0f}%)")
drivers = defaultdict(int)
for _, _, _, dr in rev:
    drivers[dr] += 1
print(f"   flip driver: {dict(drivers)}")
for load, a, b, dr in rev:
    print(f"     {load:12} {a:10} vs {b:10}  driver={dr}")

# ---------- 3. is the fastest machine the greenest? ----------
print("\n" + "=" * 70)
print("3. FASTEST vs MOST ENERGY-EFFICIENT")
dis = 0
for k in sorted(E):
    fast = max(T[k], key=T[k].get)
    green = min(E[k], key=E[k].get)
    if fast != green:
        dis += 1
        pen = 100 * (E[k][fast] - E[k][green]) / E[k][green]
        print(f"   {str(k):34} fastest={fast:10} greenest={green:10} energy penalty {pen:5.1f}%")
print(f"   performance-first picks a sub-optimal machine in {dis}/{len(E)} configurations "
      f"({100*dis/len(E):.0f}%)")
pens = []
for k in E:
    fast = max(T[k], key=T[k].get); green = min(E[k], key=E[k].get)
    pens.append(100 * (E[k][fast] - E[k][green]) / E[k][green])
pens.sort()
print(f"   energy penalty of performance-first: median {pens[len(pens)//2]:.1f}%, max {pens[-1]:.1f}%")

# ---------- 4. low-rank structure of log-energy ----------
print("\n" + "=" * 70)
print("4. LOW-RANK STRUCTURE (SVD of log-energy, loads x machines, per configuration)")
def svd_energy(M):
    # M: list of rows; power iteration free -> use numpy
    import numpy as np
    A = np.array(M, dtype=float)
    A = A - A.mean()
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    tot = (S**2).sum()
    return [float((S[:r]**2).sum()/tot) for r in range(1, len(S)+1)], S
import numpy as np
for prec in ("fp16", "fp32"):
    for bs in (8, 32, 128):
        M = [[math.log(E[(l, prec, bs)][m]) for m in MACH] for l in LOADS]
        ev, S = svd_energy(M)
        print(f"   {prec} bs{bs:<4} variance explained: rank1={ev[0]*100:5.1f}%  rank2={ev[1]*100:5.1f}%  rank3={ev[2]*100:5.1f}%")

# pooled across all configs: rows = (load,prec,bs), cols = machines
M = [[math.log(E[k][m]) for m in MACH] for k in sorted(E)]
A = np.array(M); A = A - A.mean()
S = np.linalg.svd(A, compute_uv=False)
tot = (S**2).sum()
print(f"\n   pooled matrix {A.shape[0]}x{A.shape[1]}: "
      + "  ".join(f"rank{r+1}={100*(S[:r+1]**2).sum()/tot:.1f}%" for r in range(min(4, len(S)))))

# ---------- 5. additive model residual (the interaction term) ----------
print("\n" + "=" * 70)
print("5. IS AN ADDITIVE 'load difficulty + machine efficiency' MODEL ENOUGH?")
L = np.array(M)
rm = L.mean(axis=1, keepdims=True); cm = L.mean(axis=0, keepdims=True); gm = L.mean()
add = rm + cm - gm                      # best additive (rank-1 in log space + offsets)
resid = L - add
ss_tot = ((L - gm)**2).sum(); ss_res = (resid**2).sum()
print(f"   additive model explains {100*(1-ss_res/ss_tot):.1f}% of log-energy variance")
print(f"   interaction residual: {100*ss_res/ss_tot:.1f}% of variance, "
      f"max |residual| = {np.abs(resid).max():.3f} in log space "
      f"(= {math.exp(np.abs(resid).max()):.2f}x in energy)")

# ---------- 6. peak power ----------
print("\n" + "=" * 70)
print("6. PEAK POWER vs CAP")
for m in MACH:
    obs = [P[k][m] for k in P]
    print(f"   {m:11} cap={caps[m]:5.0f} W   observed peak {min(obs):5.1f}-{max(obs):5.1f} W "
          f"({100*max(obs)/caps[m]:4.0f}% of cap)")
