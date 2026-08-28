# -*- coding: utf-8 -*-
"""
Does BUTTER-E independently reproduce the pilot's central claim?

Claim: the energy-optimal hardware choice is a property of the workload, not a
fixed ranking. BUTTER-E offers a binary machine axis (CPU node vs 2xV100 node)
over ~21k distinct workloads, so it can test the claim at a scale no private
grid could reach.
"""
import io, zipfile, csv, math, statistics as st
from collections import defaultdict

Z = "data/butter-e/runs_with_standardized_energy.csv.zip"

def num(x):
    try:
        return float(x)
    except Exception:
        return None

rows = []
with zipfile.ZipFile(Z).open("runs_with_standardized_energy.csv") as f:
    for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace")):
        e = num(r.get("std_energy")) or num(r.get("energy"))
        t = num(r.get("run_time"))
        if e is None or e <= 0:
            continue
        rows.append({
            "load": (r.get("dataset"), r.get("shape"), r.get("size"), r.get("depth")),
            "gpu": r.get("is_gpu") == "1",
            "energy": e, "time": t,
            "size": num(r.get("size")), "depth": num(r.get("depth")),
            "dataset": r.get("dataset"), "shape": r.get("shape"),
            "nonoh": num(r.get("non_overhead_energy")),
        })
print(f"usable runs: {len(rows)}")

# aggregate per (load, machine-class)
agg = defaultdict(list)
for r in rows:
    agg[(r["load"], r["gpu"])].append(r["energy"])

loads = defaultdict(dict)
for (load, gpu), v in agg.items():
    loads[load][gpu] = st.median(v)

both = {l: d for l, d in loads.items() if True in d and False in d}
print(f"distinct workloads: {len(loads)}")
print(f"workloads measured on BOTH machine classes: {len(both)}")

gpu_win = sum(1 for d in both.values() if d[True] < d[False])
cpu_win = len(both) - gpu_win
print(f"\n=== which machine class is energy-optimal? ===")
print(f"  GPU node wins: {gpu_win:6d}  ({100*gpu_win/len(both):.1f}%)")
print(f"  CPU node wins: {cpu_win:6d}  ({100*cpu_win/len(both):.1f}%)")
print("  --> the optimal machine is workload-dependent" if min(gpu_win, cpu_win) > 0
      else "  --> one machine class dominates")

ratios = sorted(d[False] / d[True] for d in both.values())
print(f"\n=== energy ratio CPU/GPU across workloads ===")
for q, name in [(0.01, "p1"), (0.10, "p10"), (0.25, "p25"), (0.50, "median"),
                (0.75, "p75"), (0.90, "p90"), (0.99, "p99")]:
    print(f"  {name:6}: {ratios[int(q*(len(ratios)-1))]:8.2f}x")
print(f"  full range: {ratios[0]:.2f}x to {ratios[-1]:.2f}x")

# does the winner depend on model size? (the compatibility signal)
print(f"\n=== does the winner depend on workload scale? ===")
bysize = defaultdict(lambda: [0, 0])
for load, d in both.items():
    s = num(load[2])
    if s is None:
        continue
    b = int(math.log10(max(s, 1)))
    bysize[b][0 if d[True] < d[False] else 1] += 1
print(f"  {'size (log10 params)':22} {'GPU wins':>10} {'CPU wins':>10} {'GPU win rate':>14}")
for b in sorted(bysize):
    g, c = bysize[b]
    if g + c < 20:
        continue
    print(f"  10^{b:<19} {g:10d} {c:10d} {100*g/(g+c):13.1f}%")

# by network shape
print(f"\n=== does the winner depend on network shape? ===")
byshape = defaultdict(lambda: [0, 0])
for load, d in both.items():
    byshape[load[1]][0 if d[True] < d[False] else 1] += 1
for sh in sorted(byshape, key=lambda k: -sum(byshape[k])):
    g, c = byshape[sh]
    if g + c < 50:
        continue
    print(f"  {sh:22} GPU {g:6d}  CPU {c:6d}  GPU win rate {100*g/(g+c):5.1f}%")

# by dataset
print(f"\n=== does the winner depend on dataset? ===")
byds = defaultdict(lambda: [0, 0])
for load, d in both.items():
    byds[load[0]][0 if d[True] < d[False] else 1] += 1
for ds in sorted(byds, key=lambda k: -sum(byds[k]))[:10]:
    g, c = byds[ds]
    print(f"  {ds:24} GPU {g:6d}  CPU {c:6d}  GPU win rate {100*g/(g+c):5.1f}%")
