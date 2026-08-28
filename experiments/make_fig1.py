# -*- coding: utf-8 -*-
"""Figure 1: peak performance does not predict measured energy efficiency."""
import io, json, math
import numpy as np

d = json.load(io.open("experiments/results/fig_perf_vs_energy.json", encoding="utf-8"))
MACH = d["machines"]; TF16 = d["tf16"]; CAP = d["cap"]
pts = d["points"]
SHORT = {"T4": "T4", "L4": "L4", "A10G": "A10", "L40S": "L40S", "A100-40GB": "A100"}

med16 = {m: float(np.median([p["e"][m] for p in pts if p["prec"] == "fp16"])) for m in MACH}
x = np.array([math.log10(TF16[m]) for m in MACH])
y = np.array([math.log10(med16[m]) for m in MACH])
r = float(np.corrcoef(x, y)[0, 1])

# ---- panel A geometry: peak TFLOPS vs median energy per sample -------------
AX, AY, AW, AH = 62, 34, 258, 190
xmin, xmax = math.log10(55), math.log10(420)
ymin, ymax = math.log10(78), math.log10(155)
def ax_(v): return AX + (math.log10(v) - xmin) / (xmax - xmin) * AW
def ay_(v): return AY + AH - (math.log10(v) - ymin) / (ymax - ymin) * AH

# ---- panel B geometry: throughput vs energy, per workload ------------------
BX, BY, BW, BH = 400, 34, 258, 190
sel = [p for p in pts if p["prec"] == "fp16" and p["batch"] == 32]
allt = [p["t"][m] for p in sel for m in MACH]
alle = [p["e"][m] for p in sel for m in MACH]
tmin, tmax = math.log10(min(allt) * .85), math.log10(max(allt) * 1.2)
emin, emax = math.log10(min(alle) * .88), math.log10(max(alle) * 1.15)
def bx_(v): return BX + (math.log10(v) - tmin) / (tmax - tmin) * BW
def by_(v): return BY + BH - (math.log10(v) - emin) / (emax - emin) * BH

COL = {"resnet50": "#14385c", "vit_b16": "#8a5a1e", "convnext_t": "#3d6b4a", "transformer": "#7a2f3f"}
LBL = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "convnext_t": "ConvNeXt-T", "transformer": "Transformer"}

s = []
s.append('<svg viewBox="0 0 700 268" xmlns="http://www.w3.org/2000/svg" role="img" '
         'aria-label="Peak performance does not predict measured energy efficiency">')
s.append('<defs><style>'
         '.ax{stroke:#111418;stroke-width:.9;fill:none}'
         '.gr{stroke:#e6e8ea;stroke-width:.7}'
         '.lb{font:9.5px Georgia,serif;fill:#2c3138}'
         '.sm{font:8.5px Georgia,serif;fill:#5a626c}'
         '.ti{font:bold 9.5px Georgia,serif;fill:#111418}'
         '.pt{stroke:#fff;stroke-width:1.1}'
         '</style></defs>')

# ---------------- panel A ----------------
s.append(f'<text class="ti" x="{AX}" y="16">(a) Vendor peak throughput vs measured energy</text>')
for tv in (60, 100, 200, 400):
    if 55 <= tv <= 420:
        s.append(f'<line class="gr" x1="{ax_(tv):.1f}" y1="{AY}" x2="{ax_(tv):.1f}" y2="{AY+AH}"/>')
        s.append(f'<text class="sm" x="{ax_(tv):.1f}" y="{AY+AH+13}" text-anchor="middle">{tv}</text>')
for ev in (80, 100, 120, 150):
    if 78 <= ev <= 155:
        s.append(f'<line class="gr" x1="{AX}" y1="{ay_(ev):.1f}" x2="{AX+AW}" y2="{ay_(ev):.1f}"/>')
        s.append(f'<text class="sm" x="{AX-6}" y="{ay_(ev)+3:.1f}" text-anchor="end">{ev}</text>')
s.append(f'<line class="ax" x1="{AX}" y1="{AY+AH}" x2="{AX+AW}" y2="{AY+AH}"/>')
s.append(f'<line class="ax" x1="{AX}" y1="{AY}" x2="{AX}" y2="{AY+AH}"/>')
s.append(f'<text class="lb" x="{AX+AW/2}" y="{AY+AH+27}" text-anchor="middle">peak fp16 throughput (TFLOP/s, log)</text>')
s.append(f'<text class="lb" x="16" y="{AY+AH/2}" text-anchor="middle" transform="rotate(-90 16 {AY+AH/2})">energy per sample (mJ, log)</text>')
# least-squares line to show the absence of trend
b1 = np.polyfit(x, y, 1)
xs = np.array([xmin, xmax])
s.append(f'<line x1="{AX}" y1="{AY+AH-(np.polyval(b1,xs)[0]-ymin)/(ymax-ymin)*AH:.1f}" '
         f'x2="{AX+AW}" y2="{AY+AH-(np.polyval(b1,xs)[1]-ymin)/(ymax-ymin)*AH:.1f}" '
         'stroke="#9aa2ab" stroke-width="1" stroke-dasharray="4,3"/>')
for m in MACH:
    cx, cy = ax_(TF16[m]), ay_(med16[m])
    s.append(f'<circle class="pt" cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#14385c"/>')
    dy = -10 if m != "A10G" else 15
    s.append(f'<text class="lb" x="{cx:.1f}" y="{cy+dy:.1f}" text-anchor="middle">{SHORT[m]}</text>')
    s.append(f'<text class="sm" x="{cx:.1f}" y="{cy+dy+10:.1f}" text-anchor="middle">{CAP[m]:.0f} W</text>')
s.append(f'<text class="lb" x="{AX+AW-4}" y="{AY+16}" text-anchor="end">r = {r:+.3f}</text>')
s.append(f'<text class="sm" x="{AX+AW-4}" y="{AY+28}" text-anchor="end">5.6x range in peak, no trend in energy</text>')

# ---------------- panel B ----------------
s.append(f'<text class="ti" x="{BX}" y="16">(b) Measured throughput vs energy, per workload</text>')
for tv in (10, 30, 100, 300, 1000, 3000):
    if tmin <= math.log10(tv) <= tmax:
        s.append(f'<line class="gr" x1="{bx_(tv):.1f}" y1="{BY}" x2="{bx_(tv):.1f}" y2="{BY+BH}"/>')
        s.append(f'<text class="sm" x="{bx_(tv):.1f}" y="{BY+BH+13}" text-anchor="middle">{tv}</text>')
for ev in (50, 100, 200, 400):
    if emin <= math.log10(ev) <= emax:
        s.append(f'<line class="gr" x1="{BX}" y1="{by_(ev):.1f}" x2="{BX+BW}" y2="{by_(ev):.1f}"/>')
        s.append(f'<text class="sm" x="{BX-6}" y="{by_(ev)+3:.1f}" text-anchor="end">{ev}</text>')
s.append(f'<line class="ax" x1="{BX}" y1="{BY+BH}" x2="{BX+BW}" y2="{BY+BH}"/>')
s.append(f'<line class="ax" x1="{BX}" y1="{BY}" x2="{BX}" y2="{BY+BH}"/>')
s.append(f'<text class="lb" x="{BX+BW/2}" y="{BY+BH+27}" text-anchor="middle">measured throughput (samples/s, log)</text>')
for p in sel:
    c = COL[p["load"]]
    order = sorted(MACH, key=lambda m: p["t"][m])
    path = " ".join(("M" if i == 0 else "L") + f"{bx_(p['t'][m]):.1f},{by_(p['e'][m]):.1f}"
                    for i, m in enumerate(order))
    s.append(f'<path d="{path}" stroke="{c}" stroke-width="1.3" fill="none" opacity=".75"/>')
    for m in order:
        s.append(f'<circle cx="{bx_(p["t"][m]):.1f}" cy="{by_(p["e"][m]):.1f}" r="3" fill="{c}" '
                 f'stroke="#fff" stroke-width=".8"/>')
    best = min(MACH, key=lambda m: p["e"][m]); fast = max(MACH, key=lambda m: p["t"][m])
    s.append(f'<circle cx="{bx_(p["t"][best]):.1f}" cy="{by_(p["e"][best]):.1f}" r="6" fill="none" '
             f'stroke="{c}" stroke-width="1.4"/>')
ly = BY + 8
for k, lab in LBL.items():
    s.append(f'<line x1="{BX+8}" y1="{ly}" x2="{BX+24}" y2="{ly}" stroke="{COL[k]}" stroke-width="1.6"/>')
    s.append(f'<text class="sm" x="{BX+28}" y="{ly+3}">{lab}</text>')
    ly += 12
s.append(f'<text class="sm" x="{BX+BW-4}" y="{BY+BH-8}" text-anchor="end">circled = lowest energy</text>')
s.append(f'<text class="sm" x="{BX+BW-4}" y="{BY+BH+2}" text-anchor="end">rightmost = fastest</text>')
s.append('</svg>')

io.open("paper/fig1_perf_vs_energy.svg", "w", encoding="utf-8").write("\n".join(s))
print("wrote paper/fig1_perf_vs_energy.svg")
print(f"panel A correlation r = {r:+.3f}")
print("median mJ/sample (fp16):", {SHORT[m]: round(med16[m], 1) for m in MACH})
