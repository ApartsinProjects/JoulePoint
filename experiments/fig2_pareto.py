# -*- coding: utf-8 -*-
"""
Figure 2: the fleet design space.

Every composition of 10 slots over 5 accelerator types, plotted as facility energy
per job against the service level it delivers. The Pareto frontier is the set of
fleets for which no other fleet is both cheaper in energy and faster; everything
above it is dominated and should never be purchased.
"""
import io, json, math, sys, warnings
from itertools import product
import numpy as np
sys.path.insert(0, "experiments")
warnings.filterwarnings("ignore")
from e4_e5_models import load_grid, MACH

IDLE = {"T4": 27.9, "L4": 29.7, "A10G": 61.2, "L40S": 88.7, "A100-40GB": 71.2}
JOB, HORIZON, LAM, SLA = 20000, 3600.0, 0.5, 60.0
SHORT = {"T4": "T4", "L4": "L4", "A10G": "A10", "L40S": "L40S", "A100-40GB": "A100"}

keys, Ylog, Tput = load_grid()
E = {k: {MACH[j]: 10 ** Ylog[i, j] / 1000.0 for j in range(len(MACH))} for i, k in enumerate(keys)}
T = {k: {MACH[j]: Tput[i, j] for j in range(len(MACH))} for i, k in enumerate(keys)}


def facility(pool, policy="energy", seed=0):
    rng = np.random.default_rng(seed)
    slots = [m for m, c in pool.items() for _ in range(c)]
    ns = len(slots); free_at = np.zeros(ns)
    t, arr = 0.0, []
    while t < HORIZON:
        t += rng.exponential(1.0 / LAM)
        if t < HORIZON:
            arr.append((t, keys[rng.integers(len(keys))]))
    dyn, delays = 0.0, []
    for at, jk in arr:
        free = [i for i in range(ns) if free_at[i] <= at]
        start = at if free else float(np.min(free_at))
        if not free:
            free = [i for i in range(ns) if free_at[i] <= start]
        cand = sorted({slots[i] for i in free})
        m = (min(cand, key=lambda mm: E[jk][mm]) if policy == "energy"
             else max(cand, key=lambda mm: T[jk][mm]))
        i = next(i for i in free if slots[i] == m)
        rt = JOB / T[jk][m]
        free_at[i] = start + rt; delays.append(start - at)
        dyn += JOB * E[jk][m] - IDLE[m] * rt
    static = sum(IDLE[s] for s in slots) * HORIZON
    return (static + dyn) / max(len(arr), 1), float(np.mean(delays)) if delays else 0.0


COMPS = [c for c in product(range(0, 11), repeat=5) if sum(c) == 10]
pts = []
for c in COMPS:
    pool = {MACH[i]: c[i] for i in range(5) if c[i] > 0}
    e, d = facility(pool, "energy", 0)
    pts.append(dict(fleet=pool, e=e, d=max(d, 0.05), ntypes=len(pool)))
ref_e, ref_d = facility({"A100-40GB": 10}, "fastest", 0)

# Pareto frontier: minimise both energy and delay
srt = sorted(pts, key=lambda p: p["d"])
front, best_e = [], float("inf")
for p in srt:
    if p["e"] < best_e:
        best_e = p["e"]; front.append(p)
feas = [p for p in pts if p["d"] <= SLA]
opt = min(feas, key=lambda p: p["e"])
print(f"{len(pts)} fleets; frontier {len(front)}; feasible at {SLA:.0f}s: {len(feas)}")
print(f"reference all-A100 fastest-first: {ref_e:.0f} J/job, {ref_d:.1f}s")
print(f"constrained optimum: {opt['fleet']} at {opt['e']:.0f} J/job, {opt['d']:.1f}s")
json.dump(dict(points=[{**p, "fleet": {k: int(v) for k, v in p["fleet"].items()}} for p in pts],
               reference=dict(e=ref_e, d=ref_d), optimum={**opt, "fleet": {k: int(v) for k, v in opt["fleet"].items()}}),
          io.open("experiments/results/fig2_pareto.json", "w", encoding="utf-8"))

# ---------------- SVG ----------------
W, H = 700, 330
PX, PY, PW, PH = 74, 30, 508, 226
xs = [p["d"] for p in pts] + [ref_d]; ys = [p["e"] for p in pts] + [ref_e]
xmin, xmax = math.log10(max(min(xs), .05) * .7), math.log10(max(xs) * 1.5)
ymin, ymax = math.log10(min(ys) * .93), math.log10(max(ys) * 1.06)
def X(v): return PX + (math.log10(max(v, .05)) - xmin) / (xmax - xmin) * PW
def Y(v): return PY + PH - (math.log10(v) - ymin) / (ymax - ymin) * PH
COL = {1: "#b9c0c7", 2: "#7f9aab", 3: "#3f7690", 4: "#1d5a78", 5: "#14385c"}

s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
     'aria-label="Fleet design space: facility energy per job against delivered service level">']
s.append('<defs><style>.ax{stroke:#111418;stroke-width:.9;fill:none}.gr{stroke:#e8eaec;stroke-width:.7}'
         '.lb{font:9.5px Georgia,serif;fill:#2c3138}.sm{font:8.5px Georgia,serif;fill:#5a626c}'
         '.ti{font:bold 9.5px Georgia,serif;fill:#111418}</style></defs>')
for v in (0.1, 1, 10, 60, 600, 6000):
    if xmin <= math.log10(v) <= xmax:
        s.append(f'<line class="gr" x1="{X(v):.1f}" y1="{PY}" x2="{X(v):.1f}" y2="{PY+PH}"/>')
        s.append(f'<text class="sm" x="{X(v):.1f}" y="{PY+PH+13}" text-anchor="middle">{v:g}</text>')
for v in (2000, 3000, 5000, 8000, 12000):
    if ymin <= math.log10(v) <= ymax:
        s.append(f'<line class="gr" x1="{PX}" y1="{Y(v):.1f}" x2="{PX+PW}" y2="{Y(v):.1f}"/>')
        s.append(f'<text class="sm" x="{PX-6}" y="{Y(v)+3:.1f}" text-anchor="end">{v//1000}k</text>')
# SLA line
s.append(f'<line x1="{X(SLA):.1f}" y1="{PY}" x2="{X(SLA):.1f}" y2="{PY+PH}" stroke="#b0543a" '
         'stroke-width="1.1" stroke-dasharray="5,3"/>')
s.append(f'<text class="sm" x="{X(SLA)+4:.1f}" y="{PY+11}" fill="#b0543a">SLA 60 s</text>')
for p in pts:
    s.append(f'<circle cx="{X(p["d"]):.1f}" cy="{Y(p["e"]):.1f}" r="2.1" fill="{COL[p["ntypes"]]}" '
             'opacity=".62"/>')
path = " ".join(("M" if i == 0 else "L") + f"{X(p['d']):.1f},{Y(p['e']):.1f}" for i, p in enumerate(front))
s.append(f'<path d="{path}" stroke="#14385c" stroke-width="1.6" fill="none"/>')
s.append(f'<circle cx="{X(ref_d):.1f}" cy="{Y(ref_e):.1f}" r="5.5" fill="#b0543a" stroke="#fff" stroke-width="1.2"/>')
s.append(f'<text class="lb" x="{X(ref_d):.1f}" y="{Y(ref_e)-10:.1f}" text-anchor="middle" fill="#b0543a">all-A100, throughput-first</text>')
s.append(f'<circle cx="{X(opt["d"]):.1f}" cy="{Y(opt["e"]):.1f}" r="5.5" fill="#1a6b33" stroke="#fff" stroke-width="1.2"/>')
lab = " + ".join(f"{v}x{SHORT[k]}" for k, v in sorted(opt["fleet"].items(), key=lambda kv: -kv[1]))
s.append(f'<text class="lb" x="{X(opt["d"])-8:.1f}" y="{Y(opt["e"])+14:.1f}" text-anchor="end" fill="#1a6b33">{lab}</text>')
s.append(f'<line class="ax" x1="{PX}" y1="{PY+PH}" x2="{PX+PW}" y2="{PY+PH}"/>')
s.append(f'<line class="ax" x1="{PX}" y1="{PY}" x2="{PX}" y2="{PY+PH}"/>')
s.append(f'<text class="lb" x="{PX+PW/2}" y="{PY+PH+28}" text-anchor="middle">mean queueing delay (s, log)</text>')
s.append(f'<text class="lb" x="17" y="{PY+PH/2}" text-anchor="middle" transform="rotate(-90 17 {PY+PH/2})">facility energy per job (kJ, log)</text>')
ly = PY + 10
s.append(f'<text class="ti" x="{PX+PW+14}" y="{ly}">accelerator</text>'); ly += 11
s.append(f'<text class="ti" x="{PX+PW+14}" y="{ly}">types in fleet</text>'); ly += 13
for k in sorted(COL):
    s.append(f'<circle cx="{PX+PW+22}" cy="{ly-3}" r="3.4" fill="{COL[k]}"/>')
    s.append(f'<text class="sm" x="{PX+PW+31}" y="{ly}">{k}</text>'); ly += 12
s.append(f'<text class="sm" x="{PX+PW+14}" y="{ly+8}">line = Pareto</text>')
s.append(f'<text class="sm" x="{PX+PW+14}" y="{ly+19}">frontier</text>')
s.append('</svg>')
io.open("paper/fig2_fleet_pareto.svg", "w", encoding="utf-8").write("\n".join(s))
print("wrote paper/fig2_fleet_pareto.svg")
print(f"saving of constrained optimum vs reference: {100*(ref_e-opt['e'])/ref_e:.1f}%")
