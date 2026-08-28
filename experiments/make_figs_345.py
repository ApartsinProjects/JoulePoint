# -*- coding: utf-8 -*-
"""
Three figures the paper's argument needs and does not have.

Fig 3  the interaction residual as a heat map. Shows directly that the 2.5 per cent
       residual is structured, not noise, and that its sign flips with precision.
Fig 4  the same residual read as a decision. Left, where the variance lives; right,
       the A100/T4 ordering inverting between precisions.
Fig 5  the three levers, to scale, which is the paper's headline claim.
"""
import io, json

D = json.load(io.open("experiments/results/interaction_matrix.json", encoding="utf-8"))
rows, mach, res = D["rows"], D["machines"], D["resid"]

SHORT = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "convnext_t": "ConvNeXt-T",
         "transformer": "Transformer"}
order = sorted(range(len(rows)), key=lambda i: (rows[i][1], rows[i][0], rows[i][2]))
MLAB = {"A100-40GB": "A100", "A10G": "A10", "L4": "L4", "L40S": "L40S", "T4": "T4"}

def ramp(v, lo=-0.15, hi=0.15):
    t = max(-1.0, min(1.0, v / hi))
    if t >= 0:                       # positive residual = worse than additive = warm
        a = t ** 0.75
        return "#{:02x}{:02x}{:02x}".format(int(255 - 133 * a), int(255 - 208 * a), int(255 - 192 * a))
    a = (-t) ** 0.75
    return "#{:02x}{:02x}{:02x}".format(int(255 - 235 * a), int(255 - 199 * a), int(255 - 163 * a))

# ---------------------------------------------------------------- Figure 3
X0, Y0, CW, CH = 172, 46, 88, 9.9
W, H = 700, Y0 + len(order) * CH + 74
p = ['<svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg" role="img" '
     'aria-label="Interaction residual heat map">'.format(W, int(H)),
     '<defs><style>.lb{font:9px Georgia,serif;fill:#2c3138}.sm{font:7.6px Georgia,serif;fill:#5a626c}'
     '.ti{font:bold 9.5px Georgia,serif;fill:#111418}.cw{font:7.4px Georgia,serif}</style></defs>',
     '<text class="ti" x="8" y="15">Interaction residual after removing workload and machine effects</text>',
     '<text class="sm" x="8" y="28">Warm = the pair costs more than the additive model predicts; cool = less. '
     'The residual carries 2.5 per cent of variance and inverts 18 of 40 accelerator pairs.</text>']
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
                 'stroke-width=".6"/>'.format(X0 + c * CW, y, CW, CH, ramp(v)))
        p.append('<text class="cw" x="{:.1f}" y="{:.1f}" text-anchor="middle" fill="{}">{:+.2f}</text>'.format(
            X0 + c * CW + CW / 2, y + 7.2, "#111418" if abs(v) < .085 else "#fff", v))
LY = Y0 + len(order) * CH + 22
for t in range(21):
    v = -0.15 + 0.3 * t / 20.0
    p.append('<rect x="{:.1f}" y="{:.1f}" width="6.2" height="8" fill="{}"/>'.format(X0 + t * 6, LY, ramp(v)))
p.append('<text class="sm" x="{}" y="{:.1f}" text-anchor="end">residual (log10 J)</text>'.format(X0 - 6, LY + 7))
p.append('<text class="sm" x="{:.1f}" y="{:.1f}" text-anchor="middle">-0.15</text>'.format(X0, LY + 18))
p.append('<text class="sm" x="{:.1f}" y="{:.1f}" text-anchor="middle">0</text>'.format(X0 + 60, LY + 18))
p.append('<text class="sm" x="{:.1f}" y="{:.1f}" text-anchor="middle">+0.15</text>'.format(X0 + 120, LY + 18))
p.append('<text class="sm" x="{:.1f}" y="{:.1f}">extremes differ by 1.94x in energy</text>'.format(X0 + 148, LY + 7))
p.append("</svg>")
io.open("paper/fig3_interaction_heatmap.svg", "w", encoding="utf-8", newline="\n").write("\n".join(p))
print("fig3 ok, {} rows".format(len(order)))
