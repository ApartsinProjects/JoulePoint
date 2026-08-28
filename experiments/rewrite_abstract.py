# -*- coding: utf-8 -*-
"""Rewrite the abstract for technical decision makers: plain sentences, few numbers.
Precise figures move to a compact key-results list underneath."""
import io, re

p = "paper/greenmatch-paper.html"
s = io.open(p, encoding="utf-8").read()

NEW = '''<div class="abstract">
  <span class="lbl">Abstract</span>
  <p>
  Data centers buy accelerators by comparing performance per watt, then run each job on whichever
  device happens to be free. We measured whether either practice actually saves energy. Neither does.
  </p>
  <p style="margin-top:.6em">
  Across 120 measured combinations of workload, hardware and settings, what a vendor advertises told
  us essentially nothing about the energy a device actually used. Even perfect knowledge of how fast
  every device runs would let a scheduler pick the most efficient one only slightly more often than
  guessing, because faster devices draw proportionally more power and the two effects cancel.
  </p>
  <p style="margin-top:.6em">
  The reason is that energy is not a property of a device. It depends on the match between the
  workload, the settings it runs under, and the hardware. This matching effect is small enough in the
  data that ordinary analysis discards it, yet it is what decides which device is cheapest to run, and
  it reverses the ranking of nearly half the device pairs we tested. We confirm on three independently
  collected datasets that the effect is created by software settings such as numerical precision and
  quantisation, not by the hardware itself.
  </p>
  <p style="margin-top:.6em">
  This changes what a facility should optimise. Measured against real idle power and realistic job
  arrivals, choosing which accelerators to buy is worth roughly a third of the energy bill, letting
  unused machines sleep is worth about an eighth, and choosing where each job runs is worth less than
  a tenth, and nothing at all when every machine is identical. The best fleet is a mixed one, the
  right mix depends on what the site actually runs, and provisioning for the wrong mix can double
  energy use. Since no buyer can benchmark hardware they do not yet own, we show the energy of an
  unmeasured accelerator can be predicted from its published specifications accurately enough to
  support the purchase decision.
  </p>
  <p style="margin-top:.6em">
  <b>Energy efficiency is a property of the fleet, not of the device. A facility cannot schedule its
  way out of a homogeneous one.</b>
  </p>
  <p class="keywords"><b>Keywords:</b> data-center energy; heterogeneous accelerators; fleet
  procurement; capacity planning; matrix completion; cold start; workload placement.</p>
</div>

<div class="tablewrap" style="margin-top:18px">
<table>
<caption><b>Key results.</b> Precise figures behind the abstract; each is derived in the section named.</caption>
<thead><tr><th>Finding</th><th>Measured</th><th>Section</th></tr></thead>
<tbody>
<tr><td>Vendor peak throughput vs measured energy per sample</td><td>r = -0.016</td><td>4</td></tr>
<tr><td>Perfect performance oracle, correct energy ranking</td><td>59.2% of pairs (chance 50%)</td><td>4</td></tr>
<tr><td>Fastest accelerator was also lowest-energy</td><td>0 of 24 configurations</td><td>4</td></tr>
<tr><td>Variance captured by an additive model, provably decision-irrelevant</td><td>97.5%</td><td>5</td></tr>
<tr><td>Accelerator pairs whose ranking inverts with settings</td><td>18 of 40</td><td>5</td></tr>
<tr><td>Interaction size vs configuration variation, three datasets</td><td>0.3% / 2.5% / 9.0%</td><td>6</td></tr>
<tr><td>Ranking accuracy, fixed ranking ceiling vs our model</td><td>81.7% -> 87.9%</td><td>7</td></tr>
<tr><td>Unmeasured accelerator predicted from specifications</td><td>91.2%, energy regret 1.04x</td><td>7.1</td></tr>
<tr><td>Fleet composition, against a throughput-first purchase</td><td>34.3%</td><td>8.1</td></tr>
<tr><td>Consolidation with power-down</td><td>13.2%</td><td>8</td></tr>
<tr><td>Placement, five-type pool / two-type pool</td><td>7-10% / 0%</td><td>8</td></tr>
<tr><td>Cost of provisioning for the wrong workload mix</td><td>up to 100.4%</td><td>8.1</td></tr>
<tr><td>Measurement noise floor, signal-to-noise ratio</td><td>7.5x</td><td>4</td></tr>
</tbody>
</table>
</div>
'''

m = re.search(r'<div class="abstract">.*?</div>\n', s, re.S)
assert m, "abstract block not found"
s = s[:m.start()] + NEW + s[m.end():]
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

import html
txt = re.sub(r"<[^>]+>", " ", NEW.split('<div class="tablewrap"')[0])
words = len(html.unescape(txt).split())
digits = sum(c.isdigit() for c in html.unescape(txt))
print(f"abstract rewritten: {words} words, {digits} digit characters (was ~270 words, ~60 digits)")
print(f"key-results table: {NEW.count('<tr><td>')} rows")
