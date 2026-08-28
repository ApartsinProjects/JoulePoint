# -*- coding: utf-8 -*-
"""
Pass D: Section 7.3, Section 8 with the three-levers figure, Section 9 rewritten from open
threats into resolved scope statements, Section 10, and figure/table renumbering.
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

# ------------------------------------------------- renumbering (later labels first)
sub("renum-t6", "<b>Table 6.</b> Placement regret", "<b>Table 7.</b> Placement regret")
sub("renum-t5", "<b>Table 5.</b> Machine cold start", "<b>Table 6.</b> Machine cold start")
sub("renum-f2", "<b>Figure 2.</b> The fleet design space", "<b>Figure 5.</b> The fleet design space")
s = s.replace("Section 8.1, Figure 2", "Section 8.1, Figure 5")

# ------------------------------------------------- 7.3
sub("sec7.3", """<p>
Three standard refinements were evaluated and none improved on the rank-1 bilinear at this data scale:
training against a pairwise ranking objective rather than squared error, weighting observations by
their measured replicate variance, and factorising energy jointly with runtime and peak power. At 96
training rows the additional structure costs more in variance than it recovers in fit.
</p>""",
"""<p>
Three further refinements drawn from the recommender literature were evaluated, and none improved on
the rank-one bilinear at this data scale: training against a pairwise ranking objective rather than
squared error, weighting observations by their measured replicate variance, and factorising energy
jointly with runtime and peak power as correlated targets. Each adds parameters that must be estimated
from the same 96 training rows, and at that scale the additional structure costs more in variance than
it recovers in fit. This reinforces the reading of Table 4: the return in this problem comes from
representing the interaction, and is close to flat in how elaborately it is represented.
</p>""")

sub("sec7.3-close", """queue-length threshold rather than any-free-server [57], with the caveat that the argument is unproven
beyond two server types [58]; our measurements show that at this level of cost consistency the simple
policy already attains it. Scheduling sophistication is not where the remaining energy is.
</p>""",
"""queue-length threshold rather than any-free-server [57], with the caveat that the argument is unproven
beyond two server types [58]; our measurements show that at the level of cost consistency these fleets
exhibit, the simple policy already attains what the sophisticated one would.
</p>
<p>
The result is worth stating carefully, because it cuts in an unobvious direction. It does <i>not</i>
say placement is unimportant. It says the placement opportunity that exists is already being captured
by a rule simple enough that most facilities are running it accidentally, so further investment in
scheduling algorithms has no energy return. Section 8 shows the opportunity itself, rather than the
algorithm exploiting it, is the quantity a facility controls, and that it is controlled by procurement.
Section 5 supplies the reason the two findings are consistent: a cost matrix close to consistent makes
energy hard to predict and placement easy to schedule, and both properties trace to the same weak
interaction. <b>Scheduling sophistication is not where the remaining energy is.</b>
</p>""")

# ------------------------------------------------- Section 8
sub("sec8-lead", """<h2>8. Fleet Composition Dominates</h2>
<p>
Idle power was measured rather than assumed:""",
"""<h2>8. Fleet Composition Dominates</h2>
<p>
Every result so far concerns the energy of running a job. A facility's bill is not that quantity. Over
any horizon, facility energy is the static cost of keeping machines powered, whether or not they are
working, plus the dynamic excess consumed while they work. On a fleet that is not saturated, the static
term is the larger one, which means idle behaviour has to be measured before any placement or
composition claim can be converted into a bill. This section does that, then compares the three levers
on one footing.
</p>
<p>
Idle power was measured rather than assumed:""")

sub("sec8-idle-close", """3.3x on the L40S, while resident weights add essentially nothing: the cost attaches to holding the
device.
</p>""",
"""3.3x on the L40S, while resident weights add essentially nothing: the cost attaches to holding the
device rather than to what is loaded on it.
</p>
<p>
This substantially qualifies the classical energy-proportionality argument [44], and it qualifies it in
a direction that matters for procurement. Small, low-power parts are commonly assumed to be the
conservative choice. They are the least proportional parts measured here, so at low duty a small device
spends most of its rated power doing nothing, and the advantage it holds at full load evaporates.
Proportionality, not rated power, is the property to buy when utilisation is uncertain.
</p>""")

sub("sec8-replay", """<p>
Replaying arrivals against a finite pool with measured idle power gives placement 6.9 to 9.5 per cent
against performance-first in a five-type pool, and 0.0 per cent against a tuned fixed ranking in a
two-type pool, where no ranking can invert. The model sits within 0.1 to 1.2 per cent of oracle, so
prediction accuracy is not the binding constraint; opportunity is. Consolidation with power-down is
worth 13.2 per cent at a 300-second sleep threshold for 0.7 seconds of added delay, and becomes
harmful below 120 seconds as wake energy dominates.
</p>""",
"""<p>
With idle power measured, the three levers can be compared on identical replayed arrival streams, which
is the only comparison that yields commensurable numbers. Placement against a performance-first policy
is worth 6.9 to 9.5 per cent in a five-type pool, and <b>0.0 per cent</b> in a two-type pool, where no
ranking can invert and there is consequently nothing for any policy to exploit. Across that range our
model sits within 0.1 to 1.2 per cent of an oracle with perfect knowledge, so what limits placement is
not prediction accuracy but the opportunity the fleet presents. Consolidation with power-down is worth
13.2 per cent at a 300-second sleep threshold, bought for 0.7 seconds of added mean delay; below a
120-second threshold it becomes actively harmful, as machines are woken more often than the sleep saves.
That sign change is the two-threshold hysteretic structure predicted by the control literature [62, 63]
appearing in measurement.
</p>""")

# ------------------------------------------------- Figure 4 into Section 8
svg = io.open("paper/fig5_three_levers.svg", encoding="utf-8").read()
FIG4 = ('<figure style="margin:16px 0">\n<div style="border:.75px solid #d1d4d8;padding:8px;'
        'background:#fff">\n' + svg + '\n</div>\n<figcaption style="font-size:9.5pt;text-align:left;'
        'margin-top:6px;text-indent:0"><b>Figure 4.</b> The paper\'s central claim in one panel. All four '
        'figures are measured on the same replayed arrival stream against directly instrumented idle '
        'power, so they are directly comparable. The lever a facility exercises once, at purchase, is '
        'worth more than the two it exercises continuously, and the placement lever collapses to zero '
        'on a uniform fleet because composition is what creates the opportunity placement '
        'exploits.</figcaption>\n</figure>\n')
sub("fig4", "<figure style=\"margin:16px 0\">\n<div style=\"border:.75px solid #d1d4d8;padding:8px;background:#fff\">\n<svg viewBox=\"0 0 700 330\"",
    FIG4 + "<figure style=\"margin:16px 0\">\n<div style=\"border:.75px solid #d1d4d8;padding:8px;background:#fff\">\n<svg viewBox=\"0 0 700 330\"")

# ------------------------------------------------- 8.1 closing
sub("sec8.1-close", """<p>
These recipes correspond to three ways a facility can obtain the mix that drives them: state an
expected composition at design time, observe the composition of current production allocations, or
actively probe by sampling workloads across configurations. Only the first is implemented here.
</p>""",
"""<p>
The three recipes answer three different procurement questions and are used at different moments. The
frontier answers "what should we buy", and its most useful output is not the optimum but the shape
around it: 162 of 1001 compositions are feasible under the service constraint, so most of the design
space is ruled out by delay rather than by energy, and a buyer optimising energy alone will select an
infeasible fleet. Marginal value answers "what should we buy <i>next</i>", which is the question a
facility with an existing fleet and an incremental budget actually faces, and its answer moves with the
mix: the type worth most under a uniform mix is not the type worth most under an fp16-only mix.
Sensitivity answers "how wrong can we afford to be", and its answer, up to 100.4 per cent, is larger
than the total saving any of the other levers offers. Mix estimation is therefore not a modelling
detail but the dominant risk in the whole procedure.
</p>
<p>
Each recipe consumes a workload mix, and a facility can obtain one three ways: state an expected
composition at design time, infer it from the composition of current production allocations, or probe
actively by sampling workloads across configurations. Only the first is implemented here; the second
requires a production trace and the third requires spare capacity, and both are natural extensions
rather than obstacles.
</p>""")

# ------------------------------------------------- Section 9
old9_start = s.index("<h2>9. Threats to Validity</h2>")
old9_end = s.index("<h2>10. Conclusion</h2>")
NEW9 = """<h2>9. Scope and Limitations</h2>
<p>
Three limitations bear on how far these results travel. Each is stated with what it does and does not
affect, and two of the three can be quantified rather than left open.
</p>
<p>
<b>The inference grid contains an efficiency-optimised part, which scopes Section 4 but not its
conclusion.</b> The L4 is energy-optimal in all 120 inference cells, so on that grid performance-first
and energy-first selection could never coincide, and the 41.6 per cent median penalty is partly a
statement about the presence of such a part in the candidate set. The extension to training workloads
in Section 7.2 breaks the pattern, producing three distinct winners, and the reversal result of
Section 5 does not depend on it at all, since it concerns pairs rather than a global optimum. The
scoped claim is that where a candidate set contains an efficiency-oriented part, throughput-first
selection reliably misses it; where it does not, the interaction results still hold but the headline
penalty would be smaller.
</p>
<p>
<b>Embodied carbon shifts the optimum without overturning it, and the shift is computable.</b> A
heterogeneous fleet may hold more total silicon, so the manufacturing carbon of the extra devices is a
legitimate objection to the recommendation. It resolves numerically. One additional eight-GPU HGX H100
baseboard carries 1,312 kg CO2e embodied [53] against roughly 98,000 kg operational over five years at
400 gCO2e per kWh, so a fleet 50 per cent larger in device count needs only about 0.7 per cent
operational saving to break even, against the 34.3 per cent measured here. On a grid at 20 gCO2e per
kWh the breakeven rises to roughly 13 per cent, still comfortably inside the measured saving, and rises
further at low utilisation. The defensible statement is therefore that heterogeneity remains optimal
across the plausible range, while its degree should shrink as grid carbon intensity and utilisation
fall. Published life-cycle work points the same way: host components rather than accelerators dominate
embodied carbon in inference servers [54], so adding accelerator types without adding hosts is cheaper
than device count suggests. Papers making composition recommendations should report total device count
and aggregate embodied carbon alongside operational energy, and we do.
</p>
<p>
<b>Corpus and instrumentation limits, which bound generality rather than validity.</b> MLPerf
missingness is not at random, because vendors submit where their hardware performs well; only
between-method comparisons on identical cells are therefore drawn from it, which is invariant to that
bias. Only two accelerators appear in both our grid and MLPerf's datacenter set, so cross-corpus
transfer of a fitted model cannot be tested here, and the three-corpus comparison of Section 6 is
consequently a comparison of interaction magnitudes rather than of transferred predictions. Energy
comes from vendor counters covering the accelerator on a single vendor's hardware and a single cloud
provider, excluding host processor, memory and cooling; including them would raise the static share of
facility energy and therefore strengthen rather than weaken the consolidation result of Section 8. On
Jetson-class embedded hardware, time efficiency has been reported to imply energy efficiency [14],
which is the boundary of the regime studied here: these claims are scoped to heterogeneous
datacenter fleets whose devices differ substantially in power draw, which is the condition Section 1
identified as the one that breaks the performance premise.
</p>

"""
s = s[:old9_start] + NEW9 + s[old9_end:]
applied.append("sec9")

# ------------------------------------------------- Section 10
old10_start = s.index("<h2>10. Conclusion</h2>")
old10_end = s.index("<h2>References</h2>")
NEW10 = """<h2>10. Conclusion</h2>
<p>
Energy-optimal accelerator selection is recoverable neither from published specifications nor from
measured throughput. The reason is structural rather than incidental: the governing signal is a small
fraction of the variance that additive models cannot represent by construction, and it is produced by
execution configuration rather than by hardware identity. Because the effect is small in variance and
decisive in ranking, it is systematically discarded by the model-selection criterion the field
currently uses.
</p>
<p>
Three implications follow for research practice. Evaluation of energy models should report ranking
accuracy and placement regret rather than absolute error, because Proposition 1 shows absolute error is
dominated by a component that cannot change a decision. The descriptor that matters most, numerical
precision, is the one schedulers are not given, and adding it to the job specification is the cheapest
intervention identified here. And benchmark corpora that report each cell at its vendor-tuned optimum
remove exactly the effect that placement research needs, so corpora intended to support such research
must vary configuration and publish non-optimal cells.
</p>
<p>
The implication for practice is simpler and larger. A facility's energy bill is set primarily by what
it buys, second by whether it lets idle machines sleep, and only third by where it puts each job. The
energy-optimal fleet is heterogeneous, its composition follows from the workload mix the site expects
to run, and provisioning against the wrong mix costs more than any scheduling policy can recover.
Because a buyer cannot benchmark hardware it does not own, the prediction results of Section 7 are what
make that decision actionable from published data alone.
</p>
<p>
The broader point is that heterogeneity is a resource rather than an inconvenience. A fleet of
identical machines offers a scheduler nothing to optimise over, whatever its algorithm; a mixed fleet
creates the opportunity that placement then exploits. Under the interconnection constraints described
in Section 1, where capacity is rationed rather than merely priced, buying for diversity is the lever
with the largest measured return.
</p>

"""
s = s[:old10_start] + NEW10 + s[old10_end:]
applied.append("sec10")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
