# -*- coding: utf-8 -*-
"""Figure 4: where the variance lives versus where the decision lives.
   Figure 5: the three levers, drawn to scale."""
import io

# ---------------------------------------------------------------- Figure 4
p = ['<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" role="img" '
     'aria-label="Variance share versus decision relevance">',
     '<defs><style>.lb{font:9.5px Georgia,serif;fill:#2c3138}.sm{font:8.5px Georgia,serif;fill:#5a626c}'
     '.ti{font:bold 9.5px Georgia,serif;fill:#111418}.bg{font:bold 11px Georgia,serif;fill:#111418}</style></defs>',
     '<text class="ti" x="10" y="16">(a) Where the variance is</text>',
     '<text class="ti" x="372" y="16">(b) Where the decision is</text>']

BX, BY, BW, BH = 20, 44, 300, 30
p += ['<text class="sm" x="{:.1f}" y="{}" text-anchor="middle">additive: workload effect + machine effect</text>'.format(BX + BW * .49, BY - 8),
      '<rect x="{}" y="{}" width="{:.1f}" height="{}" fill="#c9d2da" stroke="#8e99a4" stroke-width=".7"/>'.format(BX, BY, BW * .975, BH),
      '<rect x="{:.1f}" y="{}" width="{:.1f}" height="{}" fill="#7a2f3f" stroke="#5d232f" stroke-width=".7"/>'.format(BX + BW * .975, BY, BW * .025, BH),
      '<text class="bg" x="{:.1f}" y="{}" text-anchor="middle" fill="#3c454e">97.5%</text>'.format(BX + BW * .49, BY + 20),
      '<line x1="{:.1f}" y1="{}" x2="{:.1f}" y2="{}" stroke="#7a2f3f" stroke-width=".9"/>'.format(BX + BW * .9875, BY + BH, BX + BW * .9875, BY + BH + 12),
      '<text class="bg" x="{:.1f}" y="{}" text-anchor="middle" fill="#7a2f3f">2.5%</text>'.format(BX + BW * .9875 - 2, BY + BH + 25),
      '<text class="sm" x="{:.1f}" y="{}" text-anchor="end">interaction residual</text>'.format(BX + BW * .9875 - 34, BY + BH + 25)]
p += ['<text class="lb" x="20" y="140">By Proposition 1 the shaded 97.5 per cent is exactly a fixed</text>',
      '<text class="lb" x="20" y="154">ranking over accelerators, identical for every workload, and</text>',
      '<text class="lb" x="20" y="168">so cannot change any placement decision. All of the decision</text>',
      '<text class="lb" x="20" y="182">content lies in the 2.5 per cent.</text>',
      '<rect x="20" y="196" width="300" height="42" fill="#f4f2ee" stroke="#ddd8cf" stroke-width=".7"/>',
      '<text class="sm" x="30" y="213">pairwise ranking accuracy, additive ceiling  81.7%</text>',
      '<text class="sm" x="30" y="229">pairwise ranking accuracy, interaction modelled  87.9%</text>']

LX, RX, TOP, BOT = 445, 605, 52, 172
def yy(v, lo=0.62, hi=1.46):
    return BOT - (v - lo) / (hi - lo) * (BOT - TOP)
p += ['<text class="sm" x="{}" y="34" text-anchor="middle">energy per sample relative to T4 (Transformer, batch 32)</text>'.format((LX + RX) / 2),
      '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#c8cdd2" stroke-width=".8"/>'.format(LX, TOP - 8, LX, BOT + 6),
      '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#c8cdd2" stroke-width=".8"/>'.format(RX, TOP - 8, RX, BOT + 6),
      '<text class="lb" x="{}" y="{}" text-anchor="middle">single precision</text>'.format(LX, BOT + 20),
      '<text class="lb" x="{}" y="{}" text-anchor="middle">half precision</text>'.format(RX, BOT + 20)]
A32, A16 = 1.353, 1.0 / 1.410
p += ['<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#e0e3e6" stroke-width=".8" stroke-dasharray="3,3"/>'.format(LX, yy(1.0), RX, yy(1.0)),
      '<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#7a2f3f" stroke-width="2"/>'.format(LX, yy(A32), RX, yy(A16)),
      '<circle cx="{}" cy="{:.1f}" r="4.5" fill="#7a2f3f" stroke="#fff" stroke-width="1.1"/>'.format(LX, yy(A32)),
      '<circle cx="{}" cy="{:.1f}" r="4.5" fill="#7a2f3f" stroke="#fff" stroke-width="1.1"/>'.format(RX, yy(A16)),
      '<text class="lb" x="{}" y="{:.1f}" text-anchor="end" fill="#7a2f3f">A100, 1.35x worse</text>'.format(LX - 8, yy(A32) + 3),
      '<text class="lb" x="{}" y="{:.1f}" fill="#7a2f3f">A100, 1.41x better</text>'.format(RX + 8, yy(A16) + 3),
      '<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#14385c" stroke-width="2"/>'.format(LX, yy(1.0), RX, yy(1.0)),
      '<circle cx="{}" cy="{:.1f}" r="4.5" fill="#14385c" stroke="#fff" stroke-width="1.1"/>'.format(LX, yy(1.0)),
      '<circle cx="{}" cy="{:.1f}" r="4.5" fill="#14385c" stroke="#fff" stroke-width="1.1"/>'.format(RX, yy(1.0)),
      '<text class="sm" x="{}" y="{:.1f}" fill="#14385c">T4, reference</text>'.format(RX + 8, yy(1.0) + 3)]
p += ['<text class="lb" x="372" y="204">The hardware does not change; a single software flag moves,</text>',
      '<text class="lb" x="372" y="218">and the cheaper device swaps. Eighteen of forty accelerator</text>',
      '<text class="lb" x="372" y="232">pairs reverse somewhere in the grid.</text>',
      "</svg>"]
io.open("paper/fig4_variance_vs_decision.svg", "w", encoding="utf-8", newline="\n").write("\n".join(p))
print("fig4 ok")

# ---------------------------------------------------------------- Figure 5
LEV = [("Fleet composition", "what the facility buys", 34.3, "#14385c",
        "best feasible mix against a throughput-first all-A100 purchase"),
       ("Consolidation", "letting idle machines sleep", 13.2, "#3d6b4a",
        "300 s sleep threshold, 0.7 s of added delay"),
       ("Placement, mixed fleet", "where each job runs", 9.5, "#8a5a1e",
        "five accelerator types; upper end of a 6.9 to 9.5 per cent range"),
       ("Placement, uniform fleet", "where each job runs", 0.0, "#9aa2ab",
        "two types, no ranking can invert; exact optimal assignment adds 0.00")]
H = 62 + len(LEV) * 46
q = ['<svg viewBox="0 0 700 {}" xmlns="http://www.w3.org/2000/svg" role="img" '
     'aria-label="Facility energy saving available from each lever">'.format(H),
     '<defs><style>.lb{font:10px Georgia,serif;fill:#111418}.sm{font:8.5px Georgia,serif;fill:#5a626c}'
     '.ti{font:bold 9.5px Georgia,serif;fill:#111418}.vn{font:bold 11px Georgia,serif}</style></defs>',
     '<text class="ti" x="10" y="16">Facility energy saved by each lever, on identical replayed arrivals</text>']
X0, SC = 214, 11.0
for t in (0, 10, 20, 30):
    q.append('<line x1="{:.1f}" y1="30" x2="{:.1f}" y2="{}" stroke="#e8eaec" stroke-width=".7"/>'.format(X0 + t * SC, X0 + t * SC, H - 24))
    q.append('<text class="sm" x="{:.1f}" y="{}" text-anchor="middle">{}%</text>'.format(X0 + t * SC, H - 12, t))
for k, (name, what, val, col, note) in enumerate(LEV):
    y = 44 + k * 46
    q += ['<text class="lb" x="{}" y="{}" text-anchor="end">{}</text>'.format(X0 - 10, y + 12, name),
          '<text class="sm" x="{}" y="{}" text-anchor="end">{}</text>'.format(X0 - 10, y + 24, what),
          '<rect x="{}" y="{}" width="{:.1f}" height="19" fill="{}" opacity=".9"/>'.format(X0, y, max(val * SC, .8), col),
          '<text class="vn" x="{:.1f}" y="{}" fill="{}">{:.1f}%</text>'.format(X0 + max(val * SC, .8) + 7, y + 14, col, val),
          '<text class="sm" x="{}" y="{}">{}</text>'.format(X0 + 2, y + 33, note)]
q.append('<line x1="{}" y1="30" x2="{}" y2="{}" stroke="#111418" stroke-width=".9"/>'.format(X0, X0, H - 24))
q.append("</svg>")
io.open("paper/fig5_three_levers.svg", "w", encoding="utf-8", newline="\n").write("\n".join(q))
print("fig5 ok")
