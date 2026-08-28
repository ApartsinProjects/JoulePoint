# -*- coding: utf-8 -*-
"""
Figure 6: the decision ceiling as the action space widens.

Every headroom number in this paper is an expected value of perfect information: the cost of
the best single fixed choice minus the cost of an oracle that chooses per workload. Measured
that way, the interesting quantity is not any one figure but how it MOVES as the facility is
allowed to decide more.

  fix everything                    0 by definition
  choose hardware per workload      eight corpora, 0.00 to 8.33 per cent, median about 2
  choose hardware AND configuration 9.05 per cent
  choose the fleet itself           34.5 per cent

The paper's argument is that reading down this list is the whole recommendation, and it is
easier to see than to say.
"""
import io, json

CORPORA = [("our grid", 0.00), ("SWEAT", 0.00), ("Watt Counts", 2.35), ("ML.ENERGY", 1.57),
           ("llm-perf", 1.81), ("extended", 1.97), ("training", 2.00), ("Grid'5000", 8.33)]
LEVELS = [
    ("Fix one accelerator for everything", 0.0, None, "#9aa2ab",
     "the baseline every comparison is against"),
    ("Choose the accelerator per workload", 2.0, (0.0, 8.33), "#8a5a1e",
     "eight corpora, median 2.0, range 0.00 to 8.33"),
    ("Choose accelerator and configuration", 9.05, None, "#3d6b4a",
     "same hardware, precision and quantisation now decidable"),
    ("Choose the fleet itself", 34.5, (34.0, 35.5), "#14385c",
     "twelve seeds, sd 0.4, identical fleet in all twelve"),
]

W, H = 700, 300
X0, SC = 258, 12.6
p = ['<svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg" role="img" '
     'aria-label="Decision ceiling as the action space widens">'.format(W, H),
     '<defs><style>.lb{font:10px Georgia,serif;fill:#111418}.sm{font:8.4px Georgia,serif;fill:#5a626c}'
     '.ti{font:bold 9.5px Georgia,serif;fill:#111418}.vn{font:bold 11.5px Georgia,serif}'
     '.tick{font:8.4px Georgia,serif;fill:#5a626c}</style></defs>',
     '<text class="ti" x="10" y="16">Energy available to a perfect chooser, as the facility is '
     'allowed to decide more</text>',
     '<text class="sm" x="10" y="29">Each bar is an expected value of perfect information: the best '
     'single fixed choice against an oracle choosing per workload.</text>']

for t in (0, 10, 20, 30):
    x = X0 + t * SC
    p.append('<line x1="{:.1f}" y1="44" x2="{:.1f}" y2="{}" stroke="#e8eaec" stroke-width=".7"/>'.format(x, x, H - 46))
    p.append('<text class="tick" x="{:.1f}" y="{}" text-anchor="middle">{}%</text>'.format(x, H - 32, t))

for k, (name, val, rng, col, note) in enumerate(LEVELS):
    y = 56 + k * 52
    p.append('<text class="lb" x="{}" y="{}" text-anchor="end">{}</text>'.format(X0 - 12, y + 13, name))
    p.append('<text class="sm" x="{}" y="{}" text-anchor="end">{}</text>'.format(X0 - 12, y + 25, note))
    w = max(val * SC, 1.2)
    p.append('<rect x="{}" y="{}" width="{:.1f}" height="20" fill="{}" opacity=".92"/>'.format(X0, y, w, col))
    if rng:
        lo, hi = rng
        p.append('<line x1="{:.1f}" y1="{}" x2="{:.1f}" y2="{}" stroke="#111418" stroke-width="1.1"/>'.format(
            X0 + lo * SC, y + 10, X0 + hi * SC, y + 10))
        for e in (lo, hi):
            p.append('<line x1="{:.1f}" y1="{}" x2="{:.1f}" y2="{}" stroke="#111418" stroke-width="1.1"/>'.format(
                X0 + e * SC, y + 5, X0 + e * SC, y + 15))
    p.append('<text class="vn" x="{:.1f}" y="{}" fill="{}">{}</text>'.format(
        X0 + w + 8, y + 15, col, "0" if val == 0 else "{:.2f}%".format(val) if val < 10 else "{:.1f}%".format(val)))

# corpus dots on the second bar
yb = 56 + 1 * 52
p.append('<text class="sm" x="{}" y="{}">individual corpora:</text>'.format(X0, yb + 36))
for i, (nm, v) in enumerate(sorted(CORPORA, key=lambda t: t[1])):
    x = X0 + v * SC
    p.append('<circle cx="{:.1f}" cy="{}" r="2.6" fill="#8a5a1e" stroke="#fff" stroke-width=".7"/>'.format(x, yb + 32))
p.append('<text class="sm" x="{:.1f}" y="{}">Grid\'5000, the widest at 18 platforms</text>'.format(
    X0 + 8.33 * SC + 8, yb + 35))

p.append('<line x1="{}" y1="44" x2="{}" y2="{}" stroke="#111418" stroke-width=".9"/>'.format(X0, X0, H - 46))
p.append('<text class="sm" x="10" y="{}">Reading downward is the recommendation: the lever a facility '
         'exercises once, at purchase, dominates the two it exercises continuously.</text>'.format(H - 12))
p.append("</svg>")

io.open("paper/fig6_action_space.svg", "w", encoding="utf-8", newline="\n").write("\n".join(p))
print("fig6 written, {} corpora on the placement bar".format(len(CORPORA)))
for nm, v in sorted(CORPORA, key=lambda t: t[1]):
    print("   {:<14} {:.2f}%".format(nm, v))
