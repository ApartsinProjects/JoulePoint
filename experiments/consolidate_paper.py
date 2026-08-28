# -*- coding: utf-8 -*-
"""Consolidate every finding into the paper: new results, corrected numbers,
the two literature surveys as framing, and both figures."""
import io, re

head = io.open("experiments/_paper_head.html", encoding="utf-8").read()
head = head.replace("<title>Performance Is Not a Proxy for Energy</title>",
                    "<title>Buy Diversity, Not FLOPS</title>")
fig1 = io.open("paper/fig1_perf_vs_energy.svg", encoding="utf-8").read().strip()
fig2 = io.open("paper/fig2_fleet_pareto.svg", encoding="utf-8").read().strip()


def figure(svg, num, cap):
    return (f'<figure style="margin:16px 0">\n<div style="border:.75px solid #d1d4d8;padding:8px;'
            f'background:#fff">\n{svg}\n</div>\n<figcaption style="font-size:9.5pt;text-align:left;'
            f'margin-top:6px;text-indent:0"><b>Figure {num}.</b> {cap}</figcaption>\n</figure>\n')


BODY = """<body>

<div class="titleblock">
  <h1>Buy Diversity, Not FLOPS:<br>Fleet Composition Dominates Scheduling in Data-Center AI Energy</h1>
  <p class="authors">Alexander Apartsin</p>
  <p class="affil">Department of Computer Science, Holon Institute of Technology</p>
  <p class="date">19 August 2026</p>
</div>

<div class="abstract">
  <span class="lbl">Abstract</span>
  <p>
  Data centers buy accelerators on peak throughput per watt and place work on whichever device is
  free. We show both practices rest on a premise measurement does not support, and that correcting it
  reorders the levers available for saving energy. On a fully measured grid of 120 workload, hardware
  and configuration cells, vendor peak throughput predicts measured energy per sample at r = -0.016.
  A perfect performance oracle picks the energy-optimal accelerator in 59.2 per cent of pairwise
  comparisons, barely above chance, because speed and power are positively correlated and cancel. The
  reason is structural: additive models of log energy are provably equivalent to a single fixed
  accelerator ranking, so the entire decision rests on a 2.5 per cent interaction residual, which is
  created by execution configuration rather than hardware identity. Across three independently
  collected corpora the residual tracks how much configuration each admits, at 0.3, 2.5 and 9.0 per
  cent. Modelling it as a function of configuration features predicts a never-measured accelerator
  from its specification sheet at 91.2 per cent pairwise accuracy, cutting energy regret from 1.093 to
  1.041. Quantifying three levers against measured idle power then inverts the field's implicit
  ordering: fleet composition is worth 34 per cent, consolidation 13 per cent, and placement 7 to 10
  per cent, falling to zero on a homogeneous fleet. The energy-optimal fleet is heterogeneous, it
  differs for every workload mix we tested, and provisioning for the wrong mix costs up to 100 per
  cent. Energy efficiency is a property of the fleet, not of the device.
  </p>
  <p class="keywords"><b>Keywords:</b> data-center energy; heterogeneous accelerators; fleet
  procurement; capacity planning; matrix completion; cold start; workload placement.</p>
</div>

<h2>1. Introduction</h2>

<p>
Electricity is now a binding constraint on artificial-intelligence capacity, not a line in its
operating budget. In July 2026 the Israel Electricity Authority suspended processing of new
server-farm grid-connection applications for 140 days after pending requests reached roughly
27,000 MW, about three times national average consumption [1]. Grid studies elsewhere reach the same
conclusion: EirGrid and Dominion cannot host projected load under firm-reliability assumptions, and
relaxing those guarantees raises hostable capacity by 1.6 to 4.1 times [2]. Global demand is projected
near 945 TWh by 2030 [3]. Where connection capacity is rationed rather than merely expensive, the
operative quantity is useful computation per connected megawatt.
</p>

<p>
Two decisions determine that quantity. A facility chooses which accelerators to buy, ordinarily by
comparing advertised throughput per watt, and then chooses where each arriving workload runs,
ordinarily by availability or by preferring the fastest free device. Both rest on one unstated
premise: that a device advertising more computation per watt, or finishing sooner, will consume less
energy on the work actually submitted.
</p>

<p>
<b>That premise fails at both decision points, and its failure reorders the levers for saving
energy.</b> Vendor peak throughput carries essentially no information about measured energy
efficiency. Measured throughput carries little more. Energy is not a property of a device at all: it
is a property of the match between a workload, its execution configuration and a device, expressed in
a component of the data so small that ordinary model fitting discards it.
</p>

<p>
The practical consequence is this paper's headline. If efficiency is a property of the match, it can
only be exploited where a facility holds devices that differ, and the size of the opportunity is fixed
by the fleet before any scheduler runs. Measured against replayed arrivals and real idle power,
choosing the fleet is worth 34 per cent, letting idle machines sleep is worth 13 per cent, and placing
each job well is worth 7 to 10 per cent, falling to exactly zero on a homogeneous fleet. Most work on
data-center artificial-intelligence energy optimises the last and smallest of the three.
<b>A facility cannot schedule its way out of a homogeneous fleet.</b>
</p>

<h3>1.1 Contributions</h3>
<ul>
<li><b>A measured refutation of the performance premise.</b> The fastest accelerator was never the
lowest-energy one across 24 configurations, at a median penalty of 41.6 per cent, and vendor peak
throughput predicts measured energy at r = -0.016 (Section 4, Figure 1).</li>
<li><b>A structural explanation.</b> Additive models are exactly equivalent to a fixed ranking, so
97.5 per cent of the variance is decision-irrelevant by construction and the additive ceiling of 81.7
per cent ranking accuracy cannot be raised by more data (Section 5).</li>
<li><b>Identification of configuration as the source of the interaction</b>, corroborated across three
corpora whose interaction magnitudes order exactly as their configuration variation does, with a
physical mechanism for the quantisation case (Section 6).</li>
<li><b>Prediction for hardware that has never been measured</b>, at 91.2 per cent pairwise accuracy and
1.041 energy regret from specification sheets alone, with two calibration runs delivering 99 per cent
of what eight give (Section 7).</li>
<li><b>Three quantitative recipes for fleet design</b>: a Pareto frontier over compositions, the
marginal value of each accelerator type by Shapley decomposition, and the penalty for provisioning
against the wrong workload mix (Section 8).</li>
</ul>

<h2>2. Related Work</h2>

<h3>2.1 Measurement infrastructure and public corpora</h3>
<p>
MLPerf Power [4] established certified wall-power submission across many vendor systems; the
ML.ENERGY benchmark [5] provides automated inference energy measurement, and its diagnostic successor
[6] attributes energy to latent system metrics rather than latency. BUTTER-E [7] contributes 63,527
measured runs with node-level watt-meters plus per-node hardware and idle-power tables. The llm-perf
leaderboard [8] supplies per-phase energy across a small hardware set with quantisation and dtype as
explicit axes. Each corpus is dense on one axis and thin on the others, a structural property Section
6 exploits.
</p>

<h3>2.2 Predicting performance, power and energy</h3>
<p>
NeuSight [9] forecasts latency on unseen accelerators without addressing energy. WattGPU [10] predicts
inference power from published metadata and reports that its power model transfers substantially
better than its latency model, an asymmetry consistent with our finding that the two are distinct
prediction problems. Roofline-inspired models state explicitly that neither operation counts nor
runtime is a sufficient energy proxy [11], and replication studies find operation-count corrections
underestimate execution time [12, 13]. Pagoda [14] reports the boundary condition that on Jetson-class
hardware time efficiency does imply energy efficiency, which Section 9 treats as a scope limit.
Cross-vendor comparisons find the optimal platform varies with batch size, sequence length and model
size, and that rankings invert under latency constraints [15, 16].
</p>

<h3>2.3 Recommendation machinery for resource allocation</h3>
<p>
Applying recommender methods to cluster management originates with Paragon [17], which uses singular
value decomposition to classify unseen applications for heterogeneity and interference, and Quasar
[18], which extends this from placement to allocation quantity using sparse profiling. Selecta [19]
completes a sparse application-by-instance-type runtime matrix; PARIS [20] and CherryPick [21] address
cloud configuration selection. This line is directly ancestral to our method, and the distinction is
the estimand: all optimise quality of service, runtime or cost, and none targets energy. We draw on
inductive matrix completion [22, 23], which admits prediction for rows and columns unseen at training,
and on explicit low-rank feature crossing [24]. Our prior result on communication-free collaborative
filtering under a hidden low-rank compatibility structure [25] supplies the sample-complexity argument
motivating sparse profiling.
</p>

<h3>2.4 Energy-aware scheduling and power management</h3>
<p>
Zeus [26] establishes an energy-time Pareto frontier for training; Perseus [27] reduces training energy
up to 30 per cent without throughput loss. POLCA [28] treats the utility power contract as binding and
oversubscribes it, fitting 30 per cent more servers into a fixed envelope, the closest existing work to
our connection-capacity framing. On partitioning, MISO [29] predicts favourable multi-instance
partitions, ECLIP [30] reports 25 per cent better energy efficiency, and Han et al. [31] predict the
throughput-per-watt-optimal streaming-multiprocessor split for co-located pairs, exceeding equal
partitioning by 35 per cent. Energy-efficient multi-instance scheduling [32] and agentic CPU-GPU
assignment [33] both report performance-first placement is suboptimal.
</p>

<h3>2.5 Capacity planning and fleet procurement</h3>
<p>
This is where our contribution sits, and the literature divides cleanly with our cell unoccupied.
Where purchase <i>is</i> a decision variable, the objective is money or throughput: Melange [34],
Demystifying Cost-Efficiency [35] and Hercules [36] all find the optimum is a heterogeneous mix whose
composition depends on the workload profile, but minimise deployment cost or provisioned power.
Splitwise [37] is the closest prior art, designing heterogeneous machine mixes against a power budget
for a single workload. Where energy or carbon <i>is</i> the objective, the fleet is fixed [38, 39], or
the decision is CPU-server composition and refresh timing rather than accelerator selection [40, 41,
42]. Foundational provisioning work established that facility power, not server cost, sets the limit
[43], and the energy-proportionality argument [44] has been substantially revised for modern
accelerators, as Section 8 shows. We found no work making accelerator purchase the decision variable,
energy the objective, an SLA the constraint, and workload mix the input.
</p>

<h3>2.6 Task allocation and team composition</h3>
<p>
Our problem is an instance of heterogeneous task allocation. In the Gerkey and Mataric taxonomy [45] it
is single-task-per-agent, single-agent-per-task, time-extended assignment; the Korsah et al. extension
[46] adds a dependency axis. Neither taxonomy has an axis for team composition as a decision variable
or for stochastic arrivals, which is itself evidence the problem has not been posed canonically. Fleet
design does appear: Wilde and Alonso-Mora [47] make composition the decision variable under a monetary
budget, and Stralz et al. [48] co-design robots, fleet and planner. Where energy <i>is</i> the objective
the team is taken as given [49]. Coalition formation and coalition structure generation [50, 51, 52]
partition a fixed set by capability coverage. We use the Shapley value [52] to value an accelerator
<i>type</i> rather than to partition a team, which appears to be new in this setting.
</p>

<h3>2.7 Embodied carbon</h3>
<p>
A heterogeneous fleet may hold more total silicon, so embodied carbon is a live objection. Published
life-cycle figures make the tradeoff computable: NVIDIA's cradle-to-gate assessment gives 1,312 kg
CO2e for an eight-GPU HGX H100 baseboard [53], against roughly 98,000 kg operational over five years
at 400 gCO2e per kWh. EcoServe [54] names over-provisioning rather than heterogeneity as the embodied
risk and finds host components dominate embodied carbon in inference servers, so adding accelerator
types without adding hosts carries a smaller penalty than device count suggests. CarbonSim [55] finds
mixed-generation clusters cut carbon 16 to 26 per cent at substantial runtime cost, so heterogeneity
arising from staggered refresh works in the same direction. Section 9 quantifies the breakeven.
</p>

<h2>3. Problem Formulation</h2>
<p>
Let workloads arrive as a stochastic stream drawn from a mix over (model, configuration) pairs, and
let a fleet consist of counts of accelerator types. Executing workload <i>i</i> on machine <i>j</i>
costs energy <i>E<sub>ij</sub></i> and time <i>T<sub>ij</sub></i>, both unknown before measurement,
and is infeasible where memory capacity binds. Two problems follow. <b>Placement</b> assigns each
arrival to an available feasible machine to minimise facility energy subject to a service constraint.
<b>Composition</b> chooses the fleet itself, before any arrival, to minimise expected facility energy
over the mix subject to the same constraint. Composition is the outer problem: it fixes the set over
which placement can optimise, and Section 8 shows it dominates.
</p>
<p>
Both require <i>E<sub>ij</sub></i> for pairs never measured, and composition additionally requires it
for machines never owned. This is matrix completion with cold start on both axes, and it is why the
prediction machinery of Section 7 is not a side quest but the mechanism that makes planning possible.
</p>

<h2>4. Performance Does Not Determine Energy Ordering</h2>
<p>
Four inference workloads spanning distinct computational profiles were executed on five accelerator
families across three NVIDIA generations with a 5.7x range of enforced power limits, at two precisions
and three batch sizes, giving 120 cells. Energy came from the NVML total-energy counter, with runtime
and peak power recorded alongside. Execution used serverless containers, so the grid is reproducible
on rentable hardware for a few dollars.
</p>
<p>
The measurement floor was established before any modelling. Thirty cells repeated five times in
independent containers gave a within-cell standard deviation of 0.0071 in log energy, 1.64 per cent
relative, against an interaction residual of 0.0536: a signal-to-noise ratio of <b>7.5</b>.
</p>
<p>
Selecting the highest-throughput accelerator chose a sub-optimal machine in all 24 configurations, at a
median energy penalty of 41.6 per cent. As a ranking problem, a perfect performance oracle orders
accelerators correctly for energy in <b>59.2 per cent</b> of pairwise comparisons against 50 for
chance. Effective power varies 6.0x across the grid and correlates with throughput at +0.608, so the
fastest machines are systematically the high-power ones and the two effects largely cancel.
</p>

FIG1

<h2>5. The Decision Resides in a Low-Variance Residual</h2>
<p>
Expressing log energy as an additive workload effect plus an additive machine effect explains 97.5 per
cent of the variance, leaving 2.5 per cent in the interaction residual. Read as goodness of fit this
suggests the interaction is negligible; read as a decision rule it implies the opposite.
</p>
<div class="prop">
<b>Proposition 1.</b> For an additive model <i>y</i>(<i>i</i>, <i>j</i>) = <i>mu</i> +
<i>r<sub>i</sub></i> + <i>c<sub>j</sub></i>, the predicted ordering of machines within a row is the
ordering of <i>c<sub>j</sub></i>, independent of <i>i</i>. Every additive model is therefore exactly
equivalent to a single fixed ranking over accelerators, and the 97.5 per cent of variance it captures
is decision-irrelevant by construction.
</div>
<p>
Verified numerically on all 24 rows. It explains why the additive model, a global-majority policy and
a fixed single-machine policy return identical accuracy and regret, and sets a ceiling of 81.7 per cent
pairwise accuracy that no additional data can raise. The residual is not noise: its leading component
captures 71.9 per cent, and 58 to 84 per cent of a held-out workload's residual is explained by the
direction estimated from the others. Its extreme values correspond to 1.40x in energy and it inverts
18 of 40 accelerator pairs. On the Transformer workload the A100 is 1.41 times more efficient than the
T4 at half precision and 1.35 times less efficient at single precision.
</p>

<h2>6. Configuration Produces the Interaction</h2>
<div class="tablewrap">
<table>
<caption><b>Table 1.</b> Interaction residual across three independently collected corpora.</caption>
<thead><tr><th>Corpus</th><th>Configuration variation</th><th>Interaction</th></tr></thead>
<tbody>
<tr><td>MLPerf Power [4]</td><td>none; each cell vendor-tuned</td><td>0.3%</td></tr>
<tr><td>This work</td><td>precision and batch, fixed across machines</td><td>2.5%</td></tr>
<tr><td>llm-perf-leaderboard [8]</td><td>quantisation scheme and dtype</td><td>9.0%</td></tr>
</tbody>
</table>
</div>
<p>
MLPerf submissions are vendor-tuned per cell, so taking a maximum over configurations removes the
variation that generates the interaction. On llm-perf, where quantisation varies and hardware is held
fixed, the interaction is nearly four times larger, all three accelerator pairs invert, and the
strongest correlate of the interaction is the quantisation scheme itself at +0.800 for GPTQ.
</p>
<p>
The mechanism is physical. Under GPTQ the A100's energy disadvantage against the T4 widens to 6.61x
from 1.89x unquantised. Standard four-bit pipelines dequantise in kernels separate from the matrix
multiplication, with dequantisation dominating time within quantised matmul and fused alternatives
removing that overhead [56]. At single-sequence decode, four-bit weights relieve a real bandwidth
constraint on a 320 GB/s device and relieve nothing on a 1555 GB/s device. Quantisation is a
configuration choice that alters which hardware property limits, which is exactly when rankings
invert.
</p>
<div class="tablewrap">
<table>
<caption><b>Table 2.</b> Workload parameters correlate with the energy level and the interaction differently.</caption>
<thead><tr><th>Parameter</th><th>With level</th><th>With interaction</th></tr></thead>
<tbody>
<tr><td>Parameter count</td><td>+0.731</td><td>-0.182</td></tr>
<tr><td>Operations per sample</td><td>+0.696</td><td>-0.156</td></tr>
<tr><td>Numerical precision</td><td>+0.613</td><td>-0.515</td></tr>
<tr><td>Arithmetic intensity</td><td>+0.569</td><td>-0.396</td></tr>
<tr><td>Memory footprint</td><td>+0.153</td><td>+0.375</td></tr>
<tr><td>Batch size</td><td>-0.125</td><td>+0.394</td></tr>
</tbody>
</table>
</div>
<p>
Model scale predicts how much energy a workload consumes; precision predicts which machine should run
it. Independent measurements report the same asymmetry, finding eight-bit floating point costs up to
56 per cent more energy than sixteen-bit at batch 8 to 16 while winning by 11 per cent above batch 65
[16].
</p>

<h2>7. Predicting Unmeasured Pairs and Unowned Hardware</h2>
<p>
Because the residual is rank-1 dominant and its row factor correlates with configuration, it admits
parameterisation as a linear function of configuration features rather than a free per-workload
embedding:
</p>
<p class="eq">
y(i, j) = mu + r(x<sub>i</sub>) + c<sub>j</sub> + &lt;w, x<sub>i</sub>&gt; v<sub>j</sub>
</p>
<div class="tablewrap">
<table>
<caption><b>Table 3.</b> Model comparison, leave-one-workload-family-out.</caption>
<thead><tr><th>Model</th><th>Pairwise accuracy</th></tr></thead>
<tbody>
<tr><td>Fixed ranking or additive (Proposition 1 ceiling)</td><td>81.7</td></tr>
<tr><td>Bilinear rank-1 in feature space</td><td>87.9</td></tr>
<tr><td>Low-rank explicit cross network</td><td>87.5</td></tr>
<tr><td>Multilayer perceptron over concatenated features</td><td>86.7</td></tr>
<tr><td>Inductive matrix completion, bilinear</td><td>86.2</td></tr>
</tbody>
</table>
</div>
<p>
An interaction term is worth 6.2 percentage points; the choice among interaction models is worth about
one. Parsimony rather than capacity is the operative consideration.
</p>
<h3>7.1 Accelerators with no measurement history</h3>
<p>
Procurement is the ultimate cold-start problem: a facility cannot benchmark hardware it does not own.
On the public MLPerf matrix restricted to datacenter-class accelerators, 13 models by 13 accelerators
at 51.5 per cent density, predicting a fully held-out accelerator from published specifications gives:
</p>
<div class="tablewrap">
<table>
<caption><b>Table 4.</b> Machine cold start; the held-out accelerator is ranked against observed ones in the same row.</caption>
<thead><tr><th>Method</th><th>Pairwise accuracy</th><th>Energy regret</th></tr></thead>
<tbody>
<tr><td>No column information</td><td>64.5</td><td>1.093</td></tr>
<tr><td>Hybrid with hardware specifications</td><td>91.2</td><td>1.041</td></tr>
</tbody>
</table>
</div>
<p>
On our own grid the same construction reaches 87.0 per cent with no calibration run on the new
accelerator and 88.9 per cent with two, the latter being 99 per cent of what eight deliver.
Uncertainty-directed acquisition reaches a target accuracy at roughly half the budget of random
acquisition.
</p>
<h3>7.2 Placement regret under memory infeasibility</h3>
<p>
A grid in which one accelerator wins every cell cannot discriminate placement policies. Extending to
training workloads from 0.25 to 2 billion parameters, where optimiser state makes memory bind,
produces 19 infeasible cells and three distinct winners, and regret becomes informative.
</p>
<div class="tablewrap">
<table>
<caption><b>Table 5.</b> Placement regret on the training grid, 32 configurations, 19 infeasible cells.</caption>
<thead><tr><th>Policy</th><th>Regret</th><th>Top-1</th><th>Infeasible picks</th></tr></thead>
<tbody>
<tr><td>Oracle</td><td>1.000</td><td>100%</td><td>0</td></tr>
<tr><td>Bilinear, configuration features</td><td>1.012</td><td>87.1%</td><td>4</td></tr>
<tr><td>Fixed ranking (additive)</td><td>1.046</td><td>80.6%</td><td>4</td></tr>
<tr><td>Largest memory</td><td>1.662</td><td>19.4%</td><td>0</td></tr>
<tr><td>Lowest power limit</td><td>2.154</td><td>9.7%</td><td>7</td></tr>
</tbody>
</table>
</div>
<p>
The model beats a fixed ranking and both hardware heuristics badly. The final column records a
limitation rather than a result: the model recommends an infeasible machine in 4 of 31 configurations,
because it predicts energy but not memory. Peak memory is measured and simply not modelled, which
makes multi-output prediction the obvious remedy.
</p>

<h2>8. Fleet Composition Dominates</h2>
<p>
Idle power was measured rather than assumed: 40 per cent of the enforced limit on the T4 and 41 on the
L4, against 25 on the L40S and 16 on the A100. Larger parts are markedly more energy-proportional, and
instrumented measurement of an H100 reports 10.4 per cent [38]. At 5 per cent duty the T4 draws 86 per
cent of full-load power while the A100 draws 25, so a workload that cannot fill a device forfeits the
advantage of small parts entirely. Creating a CUDA context alone raises idle draw 2.8x on the T4 and
3.3x on the L40S, while resident weights add essentially nothing: the cost attaches to holding the
device.
</p>
<p>
Replaying arrivals against a finite pool with measured idle power gives placement 6.9 to 9.5 per cent
against performance-first in a five-type pool, and 0.0 per cent against a tuned fixed ranking in a
two-type pool, where no ranking can invert. The model sits within 0.1 to 1.2 per cent of oracle, so
prediction accuracy is not the binding constraint; opportunity is. Consolidation with power-down is
worth 13.2 per cent at a 300-second sleep threshold for 0.7 seconds of added delay, and becomes
harmful below 120 seconds as wake energy dominates.
</p>

FIG2

<h3>8.1 Three recipes for fleet design</h3>
<p>
<b>Recipe 1, the frontier.</b> Enumerating all 1001 compositions of ten slots gives a Pareto frontier
of 23 non-dominated fleets. Under a 60-second service constraint 162 are feasible, and the optimum is
<i>heterogeneous</i>: four L4 plus six L40S at 4,476 J per job, 34.3 per cent below the all-A100 fleet
a throughput-first buyer would purchase. The unconstrained optimum is a trap: all-L4 reaches 2,492 J
per job at 2,142 seconds of mean delay, which is unusable.
</p>
<p>
<b>Recipe 2, marginal value.</b> Treating accelerator types as players in a cooperative game, with the
value of a coalition being the best saving achievable by a fleet drawn from it, the Shapley value gives
each type's average marginal contribution. Value is concentrated and mix-dependent: under a uniform
mix the L40S captures 27.95 of 33.73 percentage points of achievable saving, while under an fp16-only
mix the L4 takes 15.76 of 36.65 and the L40S falls to 8.24. The efficiency axiom is satisfied exactly
in every case. Because the value depends on how many of each type rather than merely which are
present, this is a multi-unit variant of a coalitional game.
</p>
<p>
<b>Recipe 3, mix sensitivity.</b> The energy-optimal fleet differs for every workload mix tested: seven
mixes yield seven distinct optimal fleets. Provisioning for the wrong mix is costly, and asymmetrically
so. A fleet chosen for a legacy single-precision mix costs <b>100.4 per cent</b> more than optimal when
it actually runs a transformer-serving mix. A transformer-only mix, notably, admits no feasible fleet
at all among 1001 compositions under a 60-second constraint, which is a capacity result rather than a
composition one.
</p>
<p>
These recipes correspond to three ways a facility can obtain the mix that drives them: state an
expected composition at design time, observe the composition of current production allocations, or
actively probe by sampling workloads across configurations. Only the first is implemented here.
</p>

<h2>9. Threats to Validity</h2>
<p>
<b>A single efficiency-optimised part anchors Section 4.</b> The L4 is energy-optimal in all 120
inference cells, so performance-first and energy-first could never coincide. Extending to training
breaks this, but the inference result should be read as scoped to hardware sets containing an
efficiency-oriented part.
</p>
<p>
<b>Embodied carbon shifts the optimum without overturning it.</b> One extra HGX H100 baseboard costs
1,312 kg CO2e embodied [53] against roughly 98,000 kg operational over five years at 400 gCO2e per
kWh, so a fleet 50 per cent larger in device count needs only about 0.7 per cent operational saving to
break even. At 20 gCO2e per kWh that breakeven rises to roughly 13 per cent, and further at low
utilisation. The defensible statement is that heterogeneity remains optimal but its degree should
shrink as grid carbon intensity and utilisation fall. Total device count and aggregate embodied carbon
should be reported alongside operational energy.
</p>
<p>
<b>Corpus limitations.</b> MLPerf missingness is not at random, since vendors submit where their
hardware performs well, so only between-method comparisons on identical cells are drawn from it. Only
two accelerators appear in both our grid and MLPerf's datacenter set, so cross-corpus transfer cannot
be tested. Measurements come from vendor counters on one vendor and provider, excluding host
processor, memory and cooling. On Jetson-class hardware time efficiency has been reported to imply
energy efficiency [14], so these claims are scoped to heterogeneous datacenter fleets.
</p>

<h2>10. Conclusion</h2>
<p>
Energy-optimal accelerator selection is recoverable neither from vendor specifications nor from
measured throughput, because the governing signal is a small fraction of the variance that additive
models cannot represent by construction, and it is produced by execution configuration rather than
hardware identity. Three implications follow. Evaluation should report ranking accuracy and placement
regret rather than absolute error. The descriptor that matters most, numerical precision, is the one
schedulers do not receive. And benchmarks that report each cell at its tuned optimum conceal the
effect that placement research needs, so corpora supporting such research must vary configuration and
publish non-optimal cells.
</p>
<p>
The practical conclusion is simpler. A facility's energy bill is set primarily by what it buys, second
by whether it lets idle machines sleep, and only third by where it puts each job. The right fleet is
heterogeneous, it depends on the expected workload mix, and getting that mix wrong costs more than any
scheduling policy can recover.
</p>
"""

REFS = """<h2>References</h2>
<ol class="refs">
<li>Israel Electricity Authority. Suspension of processing of grid-connection applications for server farms, 140 days. Reported 22 July 2026.</li>
<li>Lin, L., Wijayawardana, R., Rao, V., Nguyen, H., Gnibga, W. E., Chien, A. A. Exploding AI power use. <i>ACM e-Energy</i>, 2024. arXiv:2311.11645.</li>
<li>International Energy Agency. <i>Energy and AI: energy demand from AI</i>, 2025.</li>
<li>Tschand, A., Rajan, A. T. R., Idgunji, S., Ghosh, A., Holleman, J., Kiraly, C., et al. MLPerf Power: benchmarking the energy efficiency of machine learning systems from microwatts to megawatts for sustainable AI. arXiv:2410.12032, 2024.</li>
<li>Chung, J.-W., Ma, J. J., Wu, R., Liu, J., Kweon, O. J., Xia, Y., Wu, Z., Chowdhury, M. The ML.ENERGY benchmark. <i>NeurIPS Datasets and Benchmarks</i>, 2025. arXiv:2505.06371.</li>
<li>Chung, J.-W., Wu, R., Ma, J. J., Chowdhury, M. Where do the joules go? arXiv:2601.22076, 2026.</li>
<li>Tripp, C. E., Perr-Sauer, J., Gafur, J., et al. Measuring the energy consumption and efficiency of deep neural networks. arXiv:2403.08151, 2024.</li>
<li>Optimum-Benchmark. llm-perf-leaderboard. Hugging Face Datasets.</li>
<li>Lee, S., Phanishayee, A., Mahajan, D. Forecasting GPU performance for deep learning training and inference. <i>ASPLOS</i>, 2025. arXiv:2407.13853.</li>
<li>Fadel Argerich, M., Furst, J., Patino-Martinez, M. WattGPU. <i>IJCAI Workshop on Sustainability and Resource-Efficiency of AI</i>, 2026. arXiv:2607.02391.</li>
<li>Zoubeirou a Mayaki, M. The energy consumption of transformer fine-tuning. arXiv:2606.23546, 2026.</li>
<li>Barba Roque, E., Cruz, L. FLOPs vs real work. arXiv:2608.14550, 2026.</li>
<li>Desislavov, R., Martinez-Plumed, F., Hernandez-Orallo, J. Compute and energy consumption trends in deep learning inference. <i>Sustainable Computing</i> 38, 2023. arXiv:2109.05472.</li>
<li>Prashanthi, S. K., Sahoo, K. K., Saikia, A. R., et al. Pagoda. arXiv:2509.20189, 2025.</li>
<li>Golden, A., Wu, C.-J., Wei, G.-Y., Brooks, D. The xPU-athalon. <i>ISPASS</i>, 2026. arXiv:2604.10852.</li>
<li>Maliakel, P. J., Ilager, S., Brandic, I. Characterizing LLM inference energy-performance tradeoffs. arXiv:2501.08219, 2025.</li>
<li>Delimitrou, C., Kozyrakis, C. Paragon. <i>ASPLOS</i>, 2013, pp. 77-88.</li>
<li>Delimitrou, C., Kozyrakis, C. Quasar. <i>ASPLOS</i>, 2014, pp. 127-144.</li>
<li>Klimovic, A., Litz, H., Kozyrakis, C. Selecta. <i>USENIX ATC</i>, 2018, pp. 759-773.</li>
<li>Yadwadkar, N. J., Hariharan, B., Gonzalez, J. E., Smith, B., Katz, R. H. Selecting the best VM across multiple public clouds. <i>SoCC</i>, 2017.</li>
<li>Alipourfard, O., Liu, H. H., Chen, J., Venkataraman, S., Yu, M., Zhang, M. CherryPick. <i>NSDI</i>, 2017, pp. 469-482.</li>
<li>Zhang, M., Chen, Y. Inductive matrix completion based on graph neural networks. arXiv:1904.12058, 2019.</li>
<li>Ledent, A., Alves, R., Kloft, M. Orthogonal inductive matrix completion. arXiv:2004.01653, 2020.</li>
<li>Wang, R., Shivanna, R., Cheng, D. Z., Jain, S., Lin, D., Hong, L., Chi, E. H. DCN V2. <i>WWW</i>, 2021. DOI 10.1145/3442381.3450078.</li>
<li>Apartsin, A., Meshulam, Y., Aperstein, Y. Acting on the unseen. arXiv:2605.25584, 2026.</li>
<li>You, J., Chung, J.-W., Chowdhury, M. Zeus. <i>NSDI</i>, 2023. arXiv:2208.06102.</li>
<li>Chung, J.-W., Gu, Y., Jang, I., Meng, L., Bansal, N., Chowdhury, M. Reducing energy bloat in large model training. <i>SOSP</i>, 2024.</li>
<li>Patel, P., Choukse, E., Zhang, C., Goiri, I., Warrier, B., Mahalingam, N., Bianchini, R. Characterizing power management opportunities for LLMs in the cloud. <i>ASPLOS</i>, 2024.</li>
<li>Li, B., Patel, T., Samsi, S., Gadepally, V., Tiwari, D. MISO. <i>SoCC</i>, 2022, pp. 173-189.</li>
<li>Quach, R., Wang, Y., Jahanshahi, A., Wong, D., Kim, H. ECLIP. <i>ISLPED</i>, 2025. arXiv:2506.12598.</li>
<li>Han, B.-S., Parekh, K., Lin, W.-C., Paul, T., Gandhi, A., Liu, Z. Energy-efficient GPU SM allocation. <i>ACM SIGMETRICS Performance Evaluation Review</i> 53(2), pp. 33-38, 2025.</li>
<li>Lipe, E., Karia, N., Espenshade, C., Stein, C., Tantawi, A., Tardieu, O. Energy efficient scheduling of AI/ML workloads on multi-instance GPUs with dynamic repartitioning. <i>CCGrid</i>, 2025. arXiv:2606.25082.</li>
<li>Lu, T., Reda, S. Agentic CPU-GPU scheduling for heterogeneous AI workloads. arXiv:2607.22242, 2026.</li>
<li>Griggs, T., Liu, X., Yu, J., Kim, D., Chiang, W.-L., Cheung, A., Stoica, I. Melange: cost efficient large language model serving by exploiting GPU heterogeneity. arXiv:2404.14527, 2024.</li>
<li>Jiang, Y., Fu, F., Yao, X., He, G., Miao, X., Klimovic, A., Cui, B., Yuan, B., Yoneki, E. Demystifying cost-efficiency in LLM serving over heterogeneous GPUs. arXiv:2502.00722, 2025.</li>
<li>Ke, L., Gupta, U., Hempstead, M., Wu, C.-J., Lee, H.-H. S., Zhang, X. Hercules. arXiv:2203.07424, 2022.</li>
<li>Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, I., Maleki, S., Bianchini, R. Splitwise. <i>ISCA</i>, 2024. arXiv:2311.18677.</li>
<li>Vercellino, R., Willard, J., Campos, G., et al. Measurement of generative AI workload power profiles. arXiv:2604.07345, 2026.</li>
<li>Lei, Y., Fernandez, J., Kypriotis, V., Skarlatos, D., Strubell, E., Sherry, J., Vosler, D. The energy cost of execution-idle in GPU clusters. arXiv:2604.04745, 2026.</li>
<li>Wang, J., Berger, D. S., Kazhamiaka, F., Irvene, C., Zhang, C., Choukse, E., Frost, K., Fonseca, R., Warrier, B., Bansal, C., Stern, J., Bianchini, R., Sriraman, A. Designing cloud servers for lower carbon. <i>ISCA</i>, 2024.</li>
<li>Uwizeyimana, I., Enright Jerger, N. Carbon-aware server replacement. <i>ISPASS</i>, 2025.</li>
<li>Nikolaou, P., Gabbay, F., Haj-Yahya, J., Sazeides, Y. Optimizing the server's upgrade cycle. <i>DCEE Workshop</i>, 2025. arXiv:2510.05787.</li>
<li>Fan, X., Weber, W.-D., Barroso, L. A. Power provisioning for a warehouse-sized computer. <i>ISCA</i>, 2007.</li>
<li>Barroso, L. A., Holzle, U. The case for energy-proportional computing. <i>IEEE Computer</i> 40(12), pp. 33-37, 2007.</li>
<li>Gerkey, B. P., Mataric, M. J. A formal analysis and taxonomy of task allocation in multi-robot systems. <i>IJRR</i> 23(9), pp. 939-954, 2004.</li>
<li>Korsah, G. A., Stentz, A., Dias, M. B. A comprehensive taxonomy for multi-robot task allocation. <i>IJRR</i> 32(12), pp. 1495-1512, 2013.</li>
<li>Wilde, N., Alonso-Mora, J. Designing heterogeneous robot fleets for task allocation and sequencing. arXiv:2312.07234, 2023.</li>
<li>Stralz, M., Alharbi, M., Huang, Y., Zardini, G. Task-driven co-design of heterogeneous multi-robot systems. arXiv:2604.21894, 2026.</li>
<li>Notomista, G., Mayya, S., Emam, Y., Kroninger, C., Bohannon, A., Hutchinson, S., Egerstedt, M. A resilient and energy-aware task allocation framework for heterogeneous multi-robot systems. arXiv:2105.05586, 2021.</li>
<li>Service, T. C., Adams, J. A. Coalition formation for task allocation: theory and algorithms. <i>Autonomous Agents and Multi-Agent Systems</i> 22(2), pp. 225-248, 2011.</li>
<li>Vig, L., Adams, J. A. Issues in multi-robot coalition formation. In <i>Multi-Robot Systems: From Swarms to Intelligent Automata III</i>, Springer, 2005, pp. 15-26.</li>
<li>Rahwan, T., Michalak, T. P., Wooldridge, M., Jennings, N. R. Coalition structure generation: a survey. <i>Artificial Intelligence</i> 229, pp. 139-174, 2015.</li>
<li>NVIDIA. HGX H100 product carbon footprint summary. ISO 14067, third-party reviewed.</li>
<li>Li, Y., Hu, Y., Choukse, E., Fonseca, R., Suh, G. E., Gupta, U. EcoServe. arXiv:2502.05043, 2025.</li>
<li>Hans, R., Zhao, Y., Lee, B. CarbonSim. <i>IGSC</i>, 2026. arXiv:2606.06438.</li>
<li>Frantar, E., Castro, R. L., Chen, J., Hoefler, T., Alistarh, D. MARLIN: mixed-precision auto-regressive parallel inference on large language models. <i>PPoPP</i>, 2025. arXiv:2408.11743.</li>
</ol>
"""

FOOTER = """<p class="footer">
Draft. Pending: workload-level execution-idle measurements; multi-output prediction to close the
feasibility gap of Section 7.2; implementation of the observed-allocation and active-probing routes to
the workload mix. The four-level dependency names in [46] were taken from citing literature rather
than the paywalled original.
</p>

</body>
</html>
"""

body = BODY.replace("FIG1", figure(fig1, 1,
    "Vendor specifications and measured performance both fail to predict energy. "
    "(a) Peak fp16 throughput against median measured energy per sample: a 5.6x range in advertised "
    "performance yields r = -0.016, and the L4 at 121 TFLOP/s is more efficient than the L40S at 362. "
    "(b) Measured throughput against energy at fp16, batch 32, with the five accelerators connected "
    "within each workload; the lowest-energy machine is circled and the fastest is rightmost. If "
    "throughput determined energy the curves would be monotone and the circled point always rightmost."))
body = body.replace("FIG2", figure(fig2, 2,
    "The fleet design space. Every composition of ten slots over five accelerator types, plotted as "
    "facility energy per job against delivered service level, shaded by how many accelerator types the "
    "fleet contains. The line is the Pareto frontier of 23 non-dominated fleets; everything above it "
    "is dominated and should never be purchased. The all-A100 fleet a throughput-first buyer would "
    "choose is marked in red; the energy-optimal fleet meeting a 60-second constraint is marked in "
    "green and is heterogeneous, at 34.3 per cent lower energy per job."))

out = head + body + REFS + FOOTER
io.open("paper/greenmatch-paper.html", "w", encoding="utf-8", newline="\n").write(out)
print(f"consolidated: {len(out)} chars, {out.count('<svg')} figures, "
      f"{out.count('<li>', out.index('ol class=\"refs\"'))} references")
