# -*- coding: utf-8 -*-
"""
Pass B: Sections 6 to 10, plus the three new figures and the descriptor-ablation table.

Figure order after this pass:
  1  vendor and measured performance vs energy      (Section 4, existing)
  2  variance share vs decision relevance           (Section 5, new)
  3  interaction residual heat map                  (Section 6, new)
  4  the three levers to scale                      (Section 8, new)
  5  fleet Pareto frontier                          (Section 8, existing, renumbered)

Section 9 becomes "Scope and Limitations": each item is resolved, quantified or scoped
rather than left standing as an open threat.
"""
import io

P = "paper/greenmatch-paper.html"
s = io.open(P, encoding="utf-8").read()
applied, failed = [], []

def sub(tag, old, new):
    global s
    if old in s:
        s = s.replace(old, new, 1); applied.append(tag)
    else:
        failed.append(tag)

def figure(svg_path, caption):
    svg = io.open(svg_path, encoding="utf-8").read()
    return ('<figure style="margin:16px 0">\n<div style="border:.75px solid #d1d4d8;padding:8px;'
            'background:#fff">\n' + svg + '\n</div>\n<figcaption style="font-size:9.5pt;text-align:left;'
            'margin-top:6px;text-indent:0">' + caption + '</figcaption>\n</figure>\n')

# ---------------------------------------------------------------- Figure 2 into Section 5
FIG2 = figure("paper/fig4_variance_vs_decision.svg",
  "<b>Figure 2.</b> Variance share and decision relevance are close to unrelated. (a) An additive "
  "model of workload effect plus machine effect captures 97.5 per cent of the variance in log energy. "
  "Proposition 1 shows that share is exactly a fixed ranking over accelerators and therefore cannot "
  "influence any placement; the whole decision lives in the 2.5 per cent residual, and modelling it "
  "raises pairwise ranking accuracy from 81.7 to 87.9 per cent. (b) One instance of the residual "
  "acting as a decision: on the Transformer workload the energy ordering of the A100 and the T4 "
  "reverses between single and half precision, a change of software rather than of hardware.")
sub("fig2", '<h3>5.1 Independent validation on third-party data</h3>',
    FIG2 + '\n<h3>5.1 Independent validation on third-party data</h3>')

sub("sec5.1-lead", """<p>
Proposition 1 is a statement about model structure, so it should hold on data we did not collect. The""",
"""<p>
Proposition 1 is a statement about the structure of a model rather than about our measurements, so it
carries a falsifiable prediction on any dataset with the same shape: three specific policies must tie
exactly, not approximately. A corpus collected by other people, on other hardware, for other purposes
is the right place to test that. The""")

sub("sec5.1-close", """cent of the saving available to a measured oracle relative to the production default of always using the
accelerator. The production default itself carries 89.3 per cent regret.
</p>""",
"""cent of the saving available to a measured oracle relative to the production default of always using the
accelerator. The production default itself carries 89.3 per cent regret.
</p>
<p>
Three points carry over from this independent check. The exact tie confirms Proposition 1 as an
identity on data we did not design. The finding that independent per-machine prediction is <i>worse</i>
than a constant policy is a warning about a natural and common approach: predicting each machine's
energy separately and taking the cheapest looks like the obvious method, but it compounds two
independent errors at exactly the point where the two predictions are closest, which is the decision
boundary. Modelling the contrast directly avoids that. And the crossover itself, from CPU-optimal below
roughly ten million parameters to GPU-optimal above, is a concrete instance of the paper's general
claim in a setting where the two device classes differ far more than any two accelerators do.
</p>""")

# ---------------------------------------------------------------- Section 6
sub("sec6-lead", """<h2>6. Configuration Produces the Interaction</h2>
<div class="tablewrap">""",
"""<h2>6. Configuration Produces the Interaction</h2>
<p>
Sections 4 and 5 establish that a decision-relevant interaction exists and is large enough to reverse
orderings. Neither says where it comes from. That question is not academic: an operator can only exploit
an effect whose source they can observe, and a corpus designer can only preserve an effect they know
they are at risk of removing.
</p>
<p>
The hypothesis tested here is that the interaction is produced by <i>execution configuration</i>, the
software settings a job runs under, rather than by hardware identity. If so, the size of the interaction
in a corpus should scale with how much configuration that corpus was allowed to vary. Three
independently collected corpora happen to span this axis almost perfectly, which turns a hypothesis
into a measurement.
</p>
<div class="tablewrap">""")

sub("sec6-mid", """<p>
MLPerf submissions are vendor-tuned per cell, so taking a maximum over configurations removes the
variation that generates the interaction. On llm-perf, where quantisation varies and hardware is held
fixed, the interaction is nearly four times larger, all three accelerator pairs invert, and the
strongest correlate of the interaction is the quantisation scheme itself at +0.800 for GPTQ.
</p>""",
"""<p>
The ordering is exact and it is the ordering the hypothesis predicts. MLPerf submissions are tuned
per cell by the submitting vendor and then reported at their best, so taking a maximum over
configurations is precisely the operation that removes the variation generating the interaction; the
residual there is 0.3 per cent, effectively nothing. Our own grid holds configuration fixed across
machines but varies it across rows, and lands in the middle at 2.5 per cent. On the llm-perf
leaderboard, where quantisation scheme and data type vary while the hardware set is small and fixed,
the interaction is nearly four times larger again at 9.0 per cent, all three accelerator pairs invert,
and the strongest single correlate of the interaction is the quantisation scheme itself, at +0.800 for
GPTQ, a widely used post-training method that compresses weights to four bits.
</p>""")

sub("sec6-mech", """<p>
The mechanism is physical. Under GPTQ""",
"""<p>
Correlation across three corpora would still leave the direction of causation open, so the quantisation
case is worth following to its physical mechanism, where it can be checked against how the hardware
actually works. Under GPTQ""")

sub("sec6-mech-close", """Quantisation is a
configuration choice that alters which hardware property limits, which is exactly when rankings
invert.
</p>""",
"""Quantisation is therefore a configuration
choice that changes <i>which hardware property is the binding constraint</i>, and a device's ranking is
a statement about the resource that binds. When the binding resource changes, the ranking changes with
it. This is the general mechanism, and precision and batch size act through the same channel: each
moves a workload along the spectrum between bandwidth-limited and arithmetic-limited execution, and
devices are not ordered the same way at the two ends.
</p>""")

# ---------------------------------------------------------------- Figure 3 into Section 6
FIG3 = figure("paper/fig3_interaction_heatmap.svg",
  "<b>Figure 3.</b> The interaction residual, cell by cell, after the additive workload and machine "
  "effects of Section 5 are removed. Rows are grouped by numerical precision. If the residual were "
  "measurement noise the panel would be uniformly pale; if it were a property of hardware alone, "
  "columns would be uniform. Instead the pattern reorganises between the fp32 and fp16 blocks, with "
  "several columns changing sign, which is the configuration effect of Table 2 made visible at the "
  "level of individual cells.")
sub("fig3", """<div class="tablewrap">
<table>
<caption><b>Table 3.</b> Workload parameters correlate""",
    FIG3 + """\n<div class="tablewrap">
<table>
<caption><b>Table 3.</b> Workload parameters correlate""")

sub("sec6-close", """<p>
Model scale predicts how much energy a workload consumes; precision predicts which machine should run
it. Independent measurements report the same asymmetry, finding eight-bit floating point costs up to
56 per cent more energy than sixteen-bit at batch 8 to 16 while winning by 11 per cent above batch 65
[16].
</p>""",
"""<p>
The two columns of Table 3 answer two different questions and, importantly, they are answered by
different variables. Parameter count and operation count correlate strongly with the energy
<i>level</i>, at +0.731 and +0.696, and barely at all with the interaction. Precision, arithmetic
intensity, memory footprint and batch size are the reverse: weaker on level, far stronger on the
interaction. Model scale predicts how much energy a workload will consume; configuration predicts
<i>which machine should run it</i>. Independent measurements report the same asymmetry from a different
direction, finding eight-bit floating point costs up to 56 per cent more energy than sixteen-bit at
batch 8 to 16 while winning by 11 per cent above batch 65 [16], a sign change driven by batch alone.
</p>
<p>
This has a direct consequence for how schedulers are built, and it is uncomfortable. Cluster schedulers
routinely receive parameter count or a memory request, which Table 3 shows predicts the level, the
quantity Proposition 1 has already ruled decision-irrelevant. They do not routinely receive numerical
precision or quantisation scheme, which is what predicts the decision. Section 7 quantifies the cost of
that omission, and it is the single largest actionable gap this paper identifies.
</p>""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
