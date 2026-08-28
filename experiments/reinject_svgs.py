# -*- coding: utf-8 -*-
"""Re-inject regenerated SVGs into the inlined figures, matched by aria-label, and
sharpen the Figure 3 caption to the claim the data actually supports."""
import io, re

P = "paper/greenmatch-paper.html"
s = io.open(P, encoding="utf-8").read()

FILES = {"Variance share versus decision relevance": "paper/fig4_variance_vs_decision.svg",
         "Interaction residual heat map": "paper/fig3_interaction_heatmap.svg",
         "Facility energy saving available from each lever": "paper/fig5_three_levers.svg"}

n = 0
for label, path in FILES.items():
    pat = re.compile(r'<svg[^>]*aria-label="' + re.escape(label) + r'".*?</svg>', re.S)
    new = io.open(path, encoding="utf-8").read()
    s, k = pat.subn(lambda m: new, s, count=1)
    n += k
    if not k:
        print("MISS:", label)

OLD = ("Instead the pattern reorganises between the fp32 and fp16 blocks, with "
       "several columns changing sign, which is the configuration effect of Table 2 made visible at the "
       "level of individual cells.")
NEW = ("Instead the pattern reorganises between the two blocks: every one of the five columns changes "
       "sign, most strongly the A100 at -0.044 under half precision against +0.044 under single, a "
       "swing of 1.22x in energy. This is the configuration effect of Table 2 made visible cell by cell, "
       "and it is why a device cannot be assigned a single efficiency rating.")
if OLD in s:
    s = s.replace(OLD, NEW, 1); n += 1
else:
    print("MISS: fig3 caption")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("replaced {} blocks".format(n))
