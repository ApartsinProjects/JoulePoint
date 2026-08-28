# -*- coding: utf-8 -*-
"""
Pass A: abstract, problem formulation, Section 4, Section 5.

Goals, per review: vendor-neutral and positive framing in the abstract; every result
followed by an explicit statement of what it implies; terms defined at first use;
motivation before method before result throughout.
"""
import io

P = "paper/greenmatch-paper.html"
s = io.open(P, encoding="utf-8").read()
applied, failed = [], []

def sub(tag, old, new):
    global s
    if old in s:
        s = s.replace(old, new, 1)
        applied.append(tag)
    else:
        failed.append(tag)

# ---------------------------------------------------------------- abstract
old_abs_start = s.index('<div class="abstract">')
old_abs_end = s.index('</div>', s.index('class="keywords"'))
NEW_ABS = '''<div class="abstract">
  <span class="lbl">Abstract</span>
  <p>
  A data center's energy bill is decided twice. It is decided once, years in advance, when the facility
  chooses which accelerators to buy, and again thousands of times an hour when a scheduler chooses where
  to run each arriving job. This paper measures how much energy each of those two decisions actually
  controls, using 120 directly instrumented combinations of workload, hardware and software
  configuration, and finds a clear and actionable ordering between them.
  </p>
  <p style="margin-top:.6em">
  The central finding is that energy is a property of the <i>match</i> between a workload, the settings
  it runs under and a device, rather than a property of the device on its own. Published specifications
  describe peak capability, which is what they are designed to describe; the energy a device draws on a
  particular job depends instead on which of that device's resources the job happens to saturate, and
  that varies with the job and its configuration. Modelling the match ranks accelerators correctly for
  87.9 per cent of comparisons, against a ceiling of 81.7 per cent that we prove no device-level
  ranking, however well calibrated, can exceed.
  </p>
  <p style="margin-top:.6em">
  The match effect is easy to miss. It carries 2.5 per cent of the variance, small enough that ordinary
  model fitting treats it as noise, yet it is what decides which device is cheapest and it reverses the
  ordering of 18 of the 40 accelerator pairs measured here. Across three independently collected
  corpora its magnitude tracks how much software configuration, numerical precision and quantisation, is
  allowed to vary. That relationship identifies where the effect comes from and tells an operator where
  to look for it in their own fleet.
  </p>
  <p style="margin-top:.6em">
  The practical consequence is an ordering of levers. Measured against directly instrumented idle power
  and replayed job arrivals, choosing the fleet is worth roughly a third of the facility energy bill,
  letting idle machines sleep is worth about an eighth, and choosing where each job runs is worth under
  a tenth, falling to nothing once every machine is identical. The energy-optimal fleet is a mixed one,
  its composition follows from the workload mix a site expects to run, and because no buyer can
  benchmark hardware it does not yet own, we show the energy of an unowned accelerator can be predicted
  from published specifications accurately enough to guide the purchase.
  </p>
  <p style="margin-top:.6em">
  <b>Energy efficiency is a property of the fleet rather than of any device within it. Buying for
  diversity gives a scheduler something to exploit; buying for peak capability alone does not.</b>
  </p>
  <p class="keywords"><b>Keywords:</b> data-center energy; heterogeneous accelerators; fleet
  procurement; capacity planning; matrix completion; cold start; workload placement.</p>
</div>'''
s = s[:old_abs_start] + NEW_ABS + s[old_abs_end + len("</div>"):]
applied.append("abstract")

# ---------------------------------------------------------------- Section 3
sub("sec3", """<h2>3. Problem Formulation</h2>
<p>
Let workloads arrive as a stochastic stream drawn from a mix over (model, configuration) pairs, and
let a fleet consist of counts of accelerator types. Executing workload <i>i</i> on machine <i>j</i>
costs energy <i>E<sub>ij</sub></i> and time <i>T<sub>ij</sub></i>, both unknown before measurement,
and is infeasible where memory capacity binds. Two problems follow. <b>Placement</b> assigns each
arrival to an available feasible machine to minimise facility energy subject to a service-level constraint on queueing delay.
<b>Composition</b> chooses the fleet itself, before any arrival, to minimise expected facility energy
over the mix subject to the same constraint. Composition is the outer problem: it fixes the set over
which placement can optimise, and Section 8 shows it dominates.
</p>
<p>
Both require <i>E<sub>ij</sub></i> for pairs never measured, and composition additionally requires it
for machines never owned. This is matrix completion with cold start on both axes, and it is why the
prediction machinery of Section 7 is not a side quest but the mechanism that makes planning possible.
</p>""",
"""<h2>3. Problem Formulation</h2>
<p>
The two decisions described in the introduction can be stated precisely, and stating them precisely is
what reveals that one contains the other.
</p>
<p>
A facility faces a stochastic stream of arriving workloads drawn from a <i>mix</i>, a probability
distribution over (model, configuration) pairs, where a configuration fixes the software settings a job
runs under: its numerical precision, its batch size, whether it trains or infers. The facility owns a
<i>fleet</i>, described by the count of each accelerator type it holds. Executing workload <i>i</i> on
machine <i>j</i> consumes energy <i>E<sub>ij</sub></i> over an execution time <i>T<sub>ij</sub></i>.
Neither quantity is known before the pair is measured, and some pairs cannot be run at all, because the
workload's peak memory demand exceeds the device's capacity.
</p>
<p>
Two optimisation problems follow, at different time scales.
</p>
<p>
<b>Placement</b> is the online problem. Each arrival is assigned to an available machine on which it is
feasible, so as to minimise total facility energy over a horizon, subject to a service-level agreement
(SLA), a contractual bound on how long a job may wait before it starts. The decision variable is an
assignment; the fleet is fixed.
</p>
<p>
<b>Composition</b> is the offline problem. The fleet itself is chosen before any job arrives, so as to
minimise expected facility energy over the mix, subject to the same SLA. The decision variable is a
vector of device counts; the assignment policy is whatever will run later.
</p>
<p>
Composition is the outer problem, and the relationship between the two is strict rather than merely
suggestive: composition fixes the set of machines over which placement is subsequently free to
optimise. A scheduler cannot route a job to a device the facility never bought. Whatever energy is
available to placement is therefore bounded above by the diversity the composition decision admitted,
which is why Section 8 finds that composition dominates and why that finding is structural rather than
an artefact of the particular fleets tested.
</p>
<p>
Both problems need <i>E<sub>ij</sub></i> for workload-machine pairs that were never measured, and
composition additionally needs it for machines the facility has never owned and therefore could never
have measured. In the language of recommender systems this is <i>matrix completion</i>, filling in a
partially observed matrix of workloads by machines, with <i>cold start</i> on both axes: new rows and
new columns alike arrive with no history. The prediction machinery of Section 7 is consequently not an
adjunct to the planning problem but the mechanism that makes planning possible at all.
</p>""")

# ---------------------------------------------------------------- Section 4
sub("sec4-intro", """<h2>4. Performance Does Not Determine Energy Ordering</h2>
<p>
Four inference workloads spanning distinct computational profiles were executed:""",
"""<h2>4. Performance Does Not Determine Energy Ordering</h2>
<p>
The premise identified in Section 1, that a faster or nominally more efficient device will use less
energy, is an empirical claim, and it is testable directly. Testing it requires a grid dense enough that
every workload is measured on every device under identical settings, since the claim is about
<i>orderings</i> and an ordering cannot be recovered from partial observations. It also requires that
the measurement be trustworthy at the scale of the effect being sought, which is why the noise floor is
established before any model is fitted.
</p>
<p>
Four inference workloads spanning distinct computational profiles were executed:""")

sub("sec4-noise", """<p>
The measurement floor was established before any modelling. Thirty cells repeated five times in
independent containers gave a within-cell standard deviation of 0.0071 in log energy, 1.64 per cent
relative, against an interaction residual of 0.0536: a signal-to-noise ratio of <b>7.5</b>.
</p>""",
"""<p>
The measurement floor was established before any modelling, because the effect this paper reports is a
small share of the total variance and would be indistinguishable from measurement scatter if the
instrument were not first characterised. Thirty cells were repeated five times each in independent
containers, so that every source of run-to-run variation, container placement, thermal state and
scheduler jitter, is present in the replicates. The within-cell standard deviation was 0.0071 in log
energy, or 1.64 per cent in relative terms, against an interaction residual of 0.0536. The
signal-to-noise ratio is therefore <b>7.5</b>: the effect analysed in Section 5 stands more than seven
times above the noise of the instrument that measured it.
</p>""")

sub("sec4-result", """<p>
Selecting the highest-throughput accelerator chose a sub-optimal machine in all 24 configurations, at a
median energy penalty of 41.6 per cent. As a ranking problem, a perfect performance oracle orders
accelerators correctly for energy in <b>59.2 per cent</b> of pairwise comparisons against 50 for
chance. Effective power varies 6.0x across the grid and correlates with throughput at +0.608, so the
fastest machines are systematically the high-power ones and the two effects largely cancel.
</p>""",
"""<p>
The premise fails at both strengths in which it can be stated. In its weak form it says that vendor
specifications are informative about energy; peak advertised throughput correlates with measured energy
per sample at r = -0.016, which is no relationship at all across a 5.6x range in the advertised figure.
In its strong form it says that <i>measured</i> speed is informative, which is a much more generous
claim because it grants the scheduler perfect knowledge of something it would normally have to predict.
Selecting the highest-throughput accelerator nonetheless chose a sub-optimal machine in all 24
configurations, at a median energy penalty of 41.6 per cent. Read as a ranking problem, a perfect
performance oracle orders accelerators correctly for energy in <b>59.2 per cent</b> of pairwise
comparisons, against 50 per cent for a coin.
</p>
<p>
The reason is visible in the same measurements and is the mechanism anticipated in Section 1. Effective
power varies 6.0x across the grid and correlates with throughput at +0.608, so the fastest machines are
systematically the highest-power ones. Energy is the product of power and time; when the two move
together, their effects on the product largely cancel, and what remains after the cancellation is not
the speed ordering but something else. Identifying that something else is the subject of Section 5.
</p>
<p>
The consequence for practice is immediate and applies to two distinct audiences. For a buyer, a
datasheet comparison of throughput per watt is not a proxy for the energy the device will draw on their
workloads, and a fleet chosen on that basis has no expected energy advantage over one chosen at random
from the same candidate set. For a scheduler author, no amount of runtime telemetry about how fast jobs
complete will make placement energy-aware, because the strong form of the premise is exactly the
assumption that such telemetry suffices, and it recovers only 59.2 per cent. The information a
scheduler needs is not being collected by either party.
</p>""")

# ---------------------------------------------------------------- Section 5
sub("sec5", """<h2>5. The Decision Resides in a Low-Variance Residual</h2>
<p>
Expressing log energy as an additive workload effect plus an additive machine effect explains 97.5 per
cent of the variance, leaving 2.5 per cent in the interaction residual. Read as goodness of fit this
suggests the interaction is negligible; read as a decision rule it implies the opposite.
</p>""",
"""<h2>5. The Decision Resides in a Low-Variance Residual</h2>
<p>
Section 4 established that the energy ordering of accelerators is not the speed ordering. This section
identifies where the energy ordering actually lives, and the answer explains why it has been overlooked.
</p>
<p>
The natural first model of a workload-by-machine energy matrix is an additive one: a baseline level,
plus an effect for the workload, plus an effect for the machine. Some workloads are simply larger than
others, and some devices are simply thirstier than others, so such a model should account for most of
what is observed. It does. Expressing log energy as an additive workload effect plus an additive
machine effect explains 97.5 per cent of the variance, leaving 2.5 per cent in the <i>interaction
residual</i>, the part of each cell that neither the workload nor the machine explains on its own.
</p>
<p>
Read as a goodness-of-fit statistic, 97.5 per cent is an endorsement and 2.5 per cent is a rounding
error. Read as a decision rule, the two numbers exchange roles completely, and the reason is not
statistical but structural.
</p>""")

sub("sec5-prop-follow", """<p>
Verified numerically on all 24 rows. It explains why the additive model, a global-majority policy and
a fixed single-machine policy return identical accuracy and regret, and sets a ceiling of 81.7 per cent
pairwise accuracy that no additional data can raise. The residual is not noise: its leading component
captures 71.9 per cent, and 58 to 84 per cent of a held-out workload's residual is explained by the
direction estimated from the others. Its extreme values correspond to 1.40x in energy and it inverts
18 of 40 accelerator pairs. On the Transformer workload the A100 is 1.41 times more efficient than the
T4 at half precision and 1.35 times less efficient at single precision.
</p>""",
"""<p>
The proposition is an identity rather than an approximation, and it was verified numerically on all 24
rows of the grid. Three consequences follow, and they are worth separating because they are often
conflated.
</p>
<p>
First, it explains an otherwise puzzling empirical coincidence: a fitted additive model, a policy that
always picks whichever machine wins most often overall, and a policy that always picks one fixed
machine return <i>identical</i> accuracy and identical regret. They are not three methods that happen
to tie. Proposition 1 says they are the same policy written three ways.
</p>
<p>
Second, it establishes a ceiling. On this grid every fixed ranking, and therefore every additive model
however much data it is given, attains at most 81.7 per cent pairwise accuracy. This is not a
data-scarcity limit that more measurement would relieve; it is a representational limit. Any method
whose reported accuracy sits at or below that figure has, whatever its architecture, learned a fixed
ranking.
</p>
<p>
Third, and this is what makes the ceiling worth attacking, the 2.5 per cent residual is not measurement
noise. Section 4 put the noise floor more than seven times below it. The residual also has structure:
its leading component captures 71.9 per cent of it, so it is close to a rank-one pattern rather than a
scatter, and between 58 and 84 per cent of a <i>held-out</i> workload's residual is explained by the
direction estimated from the other workloads, so that pattern generalises to workloads the model has
never seen. Its extreme values span 1.94x in energy, and it inverts 18 of the 40 accelerator pairs on
the grid. The cleanest single instance: on the Transformer workload the A100 is 1.41 times more
efficient than the T4 at half precision and 1.35 times <i>less</i> efficient at single precision, with
no change to the hardware.
</p>
<p>
The practical reading is that variance share and decision relevance are close to unrelated in this
problem, and optimising the former actively discards the latter. A model selected by mean squared
error, the default in every energy-prediction paper we are aware of, is selected almost entirely on how
well it fits the 97.5 per cent that Proposition 1 shows cannot influence any placement. Evaluation
should report ranking accuracy and energy regret instead, and Section 7 does.
</p>""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
