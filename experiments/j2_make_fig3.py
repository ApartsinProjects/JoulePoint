# -*- coding: utf-8 -*-
"""
J2. Rebuild Figure 3 with a companion panel on llm-perf.

Panel (a) is our controlled grid: 24 workload-configurations by 5 accelerators, grouped by
numerical precision, interaction 2.5 per cent. Panel (b) is the llm-perf leaderboard, where
quantisation scheme varies and the hardware set is fixed, interaction 9.0 per cent. Showing
them together turns Table 2 into a picture: the same structure, larger where configuration
is freer to vary, and in panel (b) every accelerator column changes sign across schemes.
"""
import io, json, statistics as st
from collections import defaultdict

A = json.load(io.open("experiments/results/interaction_matrix.json", encoding="utf-8"))
B = json.load(io.open("experiments/results/j2_llmperf_matrix.json", encoding="utf-8"))

rows, mach, res = A["rows"], A["machines"], A["resid"]
SHORT = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "convnext_t": "ConvNeXt-T",
         "transformer": "Transformer"}
MLAB = {"A100-40GB": "A100", "A10G": "A10", "L4": "L4", "L40S": "L40S", "T4": "T4"}
order = sorted(range(len(rows)), key=lambda i: (rows[i][1], rows[i][0], rows[i][2]))

gpus, quant, bres = B["gpus"], B["quant"], B["resid"]
agg = defaultdict(list)
for qi, row in zip(quant, bres):
    for j, v in enumerate(row):
        agg[(qi, gpus[j])].append(v)
QLAB = {"unquantized": "none", "bnb": "bitsandbytes 8-bit", "gptq": "GPTQ 4-bit"}
qs = ["unquantized", "bnb", "gptq"]
qn = {q: sum(1 for x in quant if x == q) for q in qs}

def ramp(v, hi):
    t = max(-1.0, min(1.0, v / hi))
    if t >= 0:
        a = t ** 0.75
        return "#{:02x}{:02x}{:02x}".format(int(255 - 133 * a), int(255 - 208 * a), int(255 - 192 * a))
    a = (-t) ** 0.75
    return "#{:02x}{:02x}{:02x}".format(int(255 - 235 * a), int(255 - 199 * a), int(255 - 163 * a))

X0, Y0, CW, CH = 172, 60, 88, 9.9
PA_H = Y0 + len(order) * CH
PB_Y = PA_H + 62
H = PB_Y + 3 * 20 + 60

p = ['<svg viewBox="0 0 700 {}" xmlns="http://www.w3.org/2000/svg" role="img" '
     'aria-label="Interaction residual heat map">'.format(int(H)),
     '<defs><style>.lb{font:9px Georgia,serif;fill:#2c3138}.sm{font:7.6px Georgia,serif;fill:#5a626c}'
     '.ti{font:bold 9.5px Georgia,serif;fill:#111418}.cw{font:7.4px Georgia,serif}'
     '.cw2{font:8.4px Georgia,serif}</style></defs>',
     '<text class="ti" x="8" y="15">(a) This work: 24 workload-configurations x 5 accelerators, '
     'interaction 2.5% of variance</text>',
     '<text class="sm" x="8" y="28">Warm = the pair costs more than the additive model predicts; '
     'cool = less. Rows grouped by numerical precision.</text>',
     '<text class="sm" x="8" y="40">Extremes differ by 1.94x in energy; 18 of 40 accelerator pairs '
     'invert somewhere in the grid.</text>']
for c, m in enumerate(mach):
    p.append('<text class="lb" x="{:.1f}" y="{}" text-anchor="middle">{}</text>'.format(
        X0 + c * CW + CW / 2, Y0 - 6, MLAB[m]))
prev = None
for k, i in enumerate(order):
    load, prec, batch = rows[i]
    y = Y0 + k * CH
    if prec != prev:
        p.append('<text class="ti" x="8" y="{:.1f}">{}</text>'.format(y + 7.4, prec))
        prev = prec
    p.append('<text class="sm" x="{}" y="{:.1f}" text-anchor="end">{} b{}</text>'.format(
        X0 - 6, y + 7.4, SHORT[load], batch))
    for c in range(len(mach)):
        v = res[i][c]
        p.append('<rect x="{:.1f}" y="{:.1f}" width="{}" height="{:.1f}" fill="{}" stroke="#fff" '
                 'stroke-width=".6"/>'.format(X0 + c * CW, y, CW, CH, ramp(v, 0.15)))
        p.append('<text class="cw" x="{:.1f}" y="{:.1f}" text-anchor="middle" fill="{}">{:+.2f}</text>'.format(
            X0 + c * CW + CW / 2, y + 7.2, "#111418" if abs(v) < .085 else "#fff", v))

# ---------------------------------------------------------------- panel (b)
p += ['<text class="ti" x="8" y="{:.1f}">(b) llm-perf leaderboard: mean residual by quantisation '
      'scheme, interaction 9.0% of variance</text>'.format(PB_Y - 30),
      '<text class="sm" x="8" y="{:.1f}">Same three GPUs throughout; only the software changes. '
      'Every column changes sign across schemes, and all three GPU pairs invert.</text>'.format(PB_Y - 18)]
BX, BCW, BCH = 214, 96, 20
for c, g in enumerate(gpus):
    p.append('<text class="lb" x="{:.1f}" y="{:.1f}" text-anchor="middle">{}</text>'.format(
        BX + c * BCW + BCW / 2, PB_Y - 4, g))
for rIdx, q in enumerate(qs):
    y = PB_Y + rIdx * BCH
    p.append('<text class="sm" x="{}" y="{:.1f}" text-anchor="end">{} (n={})</text>'.format(
        BX - 6, y + 13.5, QLAB[q], qn[q]))
    for c, g in enumerate(gpus):
        v = st.mean(agg[(q, g)])
        p.append('<rect x="{:.1f}" y="{:.1f}" width="{}" height="{}" fill="{}" stroke="#fff" '
                 'stroke-width=".7"/>'.format(BX + c * BCW, y, BCW, BCH, ramp(v, 0.23)))
        p.append('<text class="cw2" x="{:.1f}" y="{:.1f}" text-anchor="middle" fill="{}">{:+.3f}</text>'.format(
            BX + c * BCW + BCW / 2, y + 13.5, "#111418" if abs(v) < .13 else "#fff", v))
p.append('<text class="sm" x="{}" y="{:.1f}">under GPTQ the A100 and T4 swap by 2.74x in energy</text>'.format(
    BX, PB_Y + 3 * BCH + 14))

LY = PB_Y + 3 * BCH + 30
for t in range(21):
    v = -1.0 + 2.0 * t / 20.0
    p.append('<rect x="{:.1f}" y="{:.1f}" width="6.2" height="8" fill="{}"/>'.format(
        BX + t * 6, LY, ramp(v, 1.0)))
p.append('<text class="sm" x="{}" y="{:.1f}" text-anchor="end">residual (log10 J)</text>'.format(BX - 6, LY + 7))
p.append('<text class="sm" x="{:.1f}" y="{:.1f}" text-anchor="middle">cheaper than additive</text>'.format(BX + 26, LY + 18))
p.append('<text class="sm" x="{:.1f}" y="{:.1f}" text-anchor="middle">0</text>'.format(BX + 63, LY + 18))
p.append('<text class="sm" x="{:.1f}" y="{:.1f}" text-anchor="middle">dearer than additive</text>'.format(BX + 100, LY + 18))
p.append("</svg>")

io.open("paper/fig3_interaction_heatmap.svg", "w", encoding="utf-8", newline="\n").write("\n".join(p))
print("fig3 rebuilt with two panels, height {}".format(int(H)))
for q in qs:
    print("  {:<14}".format(q) + "".join("{:>10.3f}".format(st.mean(agg[(q, g)])) for g in gpus))
