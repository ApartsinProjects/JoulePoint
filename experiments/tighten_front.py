# -*- coding: utf-8 -*-
"""Trim the abstract to conference length without losing the decision-maker framing,
and bring the contributions list in line with the expanded sections."""
import io, re, html

P = "paper/greenmatch-paper.html"
s = io.open(P, encoding="utf-8").read()

a0 = s.index('<div class="abstract">')
a1 = s.index('</div>', s.index('class="keywords"'))
NEW = '''<div class="abstract">
  <span class="lbl">Abstract</span>
  <p>
  A data center's energy bill is decided twice: once, years in advance, when the facility chooses which
  accelerators to buy, and again thousands of times an hour when a scheduler chooses where to run each
  job. Using 120 directly instrumented combinations of workload, hardware and software configuration,
  we measure how much energy each decision actually controls.
  </p>
  <p style="margin-top:.6em">
  Energy turns out to be a property of the <i>match</i> between a workload, its settings and a device,
  rather than of the device alone. Published specifications describe peak capability, which is what they
  are designed to describe; the energy a device draws on a given job depends on which of its resources
  that job saturates, and that varies with the job. Modelling the match ranks accelerators correctly for
  87.9 per cent of comparisons, against a ceiling of 81.7 per cent that we prove no device-level ranking
  can exceed. The effect carries only 2.5 per cent of the variance, small enough that ordinary model
  fitting discards it, yet it reverses the ordering of 18 of 40 accelerator pairs. Across three
  independent corpora its magnitude tracks how much software configuration is allowed to vary, which
  identifies its source.
  </p>
  <p style="margin-top:.6em">
  The consequence is a clear ordering of levers. Against instrumented idle power and replayed arrivals,
  choosing the fleet is worth roughly a third of the facility bill, letting idle machines sleep about an
  eighth, and choosing where each job runs under a tenth, falling to nothing once every machine is
  identical. The energy-optimal fleet is mixed, its composition follows from the expected workload mix,
  and because no buyer can benchmark hardware it does not own, we show an unowned accelerator's energy
  is predictable from published specifications accurately enough to guide the purchase.
  </p>
  <p style="margin-top:.6em">
  <b>Energy efficiency is a property of the fleet rather than of any device in it. Buying for diversity
  gives a scheduler something to exploit; buying for peak capability alone does not.</b>
  </p>
  <p class="keywords"><b>Keywords:</b> data-center energy; heterogeneous accelerators; fleet
  procurement; capacity planning; matrix completion; cold start; workload placement.</p>
</div>'''
s = s[:a0] + NEW + s[a1 + len("</div>"):]

c0 = s.index("<h3>1.1 Contributions</h3>")
c1 = s.index('<div class="tablewrap" style="margin-top:18px">')
NEWC = """<h3>1.1 Contributions</h3>
<ul>
<li><b>A measured account of what performance does and does not tell a buyer.</b> The
highest-throughput accelerator was never the lowest-energy one across 24 configurations, at a median
penalty of 41.6 per cent, and peak advertised throughput relates to measured energy at r = -0.016
(Section 4, Figure 1).</li>
<li><b>A structural explanation, with a proof.</b> Additive models are exactly equivalent to a single
fixed ranking over accelerators, so the 97.5 per cent of variance they capture is decision-irrelevant
by construction and their 81.7 per cent ranking ceiling cannot be raised by more data (Section 5,
Figure 2).</li>
<li><b>Identification of execution configuration as the source of the interaction</b>, corroborated
across three independently collected corpora whose interaction magnitudes order exactly as their
configuration variation does, with a physical mechanism for the quantisation case (Section 6,
Figure 3).</li>
<li><b>A prediction method that works for hardware never measured</b>, reaching 91.2 per cent pairwise
accuracy and 1.041 energy regret from specification sheets alone, with two calibration runs delivering
99 per cent of what eight give (Section 7).</li>
<li><b>Identification of the single missing scheduler input.</b> Numerical precision alone recovers most
of the achievable accuracy, while the descriptors schedulers conventionally receive recover none of it
(Section 7, Table 5).</li>
<li><b>Three quantitative recipes for fleet design</b>: a Pareto frontier over compositions, the
marginal value of each accelerator type by Shapley decomposition, and the penalty for provisioning
against the wrong workload mix (Section 8, Figures 4 and 5).</li>
</ul>

"""
s = s[:c0] + NEWC + s[c1:]

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
a = s[s.index('<div class="abstract"'):s.index('class="keywords"')]
print("abstract words:", len(re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", a))).split()))
