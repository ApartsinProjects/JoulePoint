# -*- coding: utf-8 -*-
"""Rebuild the paper with the fleet-property headline, a full related-work survey,
and a consistent reference numbering across all sections."""
import io, re

head = io.open("experiments/_paper_head.html", encoding="utf-8").read()
tail = io.open("experiments/_paper_tail.html", encoding="utf-8").read()
fig = io.open("experiments/_paper_fig.html", encoding="utf-8").read()

head = head.replace("<title>Performance Is Not a Proxy for Energy</title>",
                    "<title>Energy Efficiency Is a Fleet Property</title>")

FRONT = """<body>

<div class="titleblock">
  <h1>Energy Efficiency Is a Fleet Property:<br>Specification and Throughput Both Mispredict Data-Center AI Energy</h1>
  <p class="authors">Alexander Apartsin</p>
  <p class="affil">Department of Computer Science, Holon Institute of Technology</p>
  <p class="date">19 August 2026</p>
</div>

<div class="abstract">
  <span class="lbl">Abstract</span>
  <p>
  Data centers purchase accelerators on peak throughput per watt and place work on whichever device is
  fastest or first available. We show that both practices rest on a premise that measurement does not
  support. On a complete grid of 120 workload–hardware–configuration cells spanning five accelerator
  families and three architecture generations, vendor peak throughput correlates with measured energy
  per sample at r = −0.016: a 5.6x range in advertised performance produces no trend in energy
  whatsoever. Measured throughput fares little better, a perfect performance oracle identifying the
  energy-optimal accelerator in 59.2 per cent of pairwise comparisons against 50 per cent for chance,
  because speed and power are positively correlated and largely cancel. The reason is structural.
  Logarithmic energy decomposes almost entirely into additive workload and machine terms, and we prove
  that any additive model induces a machine ordering invariant across workloads, making it formally
  equivalent to a single fixed ranking. The decision therefore rests wholly on an interaction residual
  carrying 2.5 per cent of the variance, which nonetheless corresponds to differences up to 1.40x in
  energy and inverts 45 per cent of accelerator pairs. That residual is produced by execution
  configuration rather than hardware identity: across three independently collected corpora its
  magnitude tracks the configuration variation each admits, rising from 0.3 per cent where cells are
  vendor-tuned, to 2.5 per cent under controlled precision and batch variation, to 9.0 per cent where
  quantisation scheme varies. Modelling it as a function of configuration features raises ranking
  accuracy from a provable additive ceiling of 81.7 per cent to 87.9 per cent, and reduces the energy
  regret of onboarding a previously unmeasured accelerator from 3.81x to 1.04x from published
  specifications alone. Quantifying three levers against measured idle power then inverts the field's
  implicit ordering: fleet composition is worth 34 per cent, consolidation 13 per cent, and placement
  7 to 10 per cent, falling to zero on a homogeneous fleet. Energy efficiency is a property of the
  fleet, not of the device.
  </p>
  <p class="keywords"><b>Keywords:</b> energy-aware scheduling; heterogeneous accelerators; matrix
  completion; cold start; data-center energy; fleet procurement; workload placement.</p>
</div>

<h2>1. Introduction</h2>

<p>
Electricity has become a binding constraint on artificial-intelligence capacity rather than a line in
its operating budget. In July 2026 the Israel Electricity Authority suspended processing of new
server-farm grid-connection applications for 140 days, after pending requests reached approximately
27,000 MW, roughly three times national average consumption [1]. The same conclusion is reached
elsewhere by a different route: EirGrid and Dominion cannot host projected artificial-intelligence
load under firm-reliability assumptions, and relaxing reliability guarantees for new facilities raises
hostable capacity by factors of 1.6 to 4.1 [2]. Global projections place data-center electricity demand
near 945 TWh by 2030, with accelerated servers driving most of the increase [3]. Where connection
capacity is rationed rather than merely expensive, the operative quantity becomes useful computation
per connected megawatt.
</p>

<p>
Two decisions determine that quantity. The first is which accelerators a facility buys, ordinarily
settled by comparing advertised throughput per watt. The second is which accelerator each arriving
workload runs on, ordinarily settled by availability or by preferring the fastest free device. Both
rest on the same unstated premise: that a device advertising more computation per watt, or completing
work sooner, will consume less energy for the work actually submitted.
</p>

<p>
<b>We find that premise fails at both decision points, and that its failure reorders the levers
available for saving energy.</b> Vendor peak throughput carries essentially no information about
measured energy efficiency, at r = −0.016 across a 5.6x range of advertised performance. Measured
throughput carries little more, a perfect performance oracle ranking accelerators correctly for energy
in 59.2 per cent of pairwise comparisons. The reason is that energy is not a property of a device at
all. It is a property of the match between a workload, its execution configuration, and a device, and
that match is expressed in a component of the data so small that conventional model fitting discards
it: 2.5 per cent of the variance in logarithmic energy, which nonetheless decides every placement.
</p>

<p>
The practical consequence is the paper's headline. If energy efficiency is a property of the match
rather than the device, then it can only be exploited where a facility holds devices that differ, and
the size of the opportunity is fixed by the fleet before any scheduler runs. Quantifying three levers
against measured idle power on replayed arrivals bears this out: choosing the fleet composition is
worth 34 per cent, allowing idle machines to sleep is worth 13 per cent, and placing each job on the
predicted best available device is worth 7 to 10 per cent, falling to exactly zero on a homogeneous
fleet and on a two-type fleet where no ranking can invert. Most published work on data-center
artificial-intelligence energy optimises the last and smallest of these three. <b>Energy efficiency is
a property of the fleet, not of the device, and a facility cannot schedule its way out of a homogeneous
one.</b>
</p>

<h3>1.1 Contributions</h3>
<ul>
<li><b>A measured refutation of the performance premise.</b> On 120 fully measured cells, the fastest
accelerator was never the lowest-energy one, at a median penalty of 41.6 per cent, and vendor peak
throughput predicts measured energy at r = −0.016 (Section 4, Figure 1).</li>
<li><b>A structural explanation.</b> We prove that additive models are exactly equivalent to a fixed
accelerator ranking, so 97.5 per cent of the variance in logarithmic energy is decision-irrelevant by
construction and the additive ceiling of 81.7 per cent ranking accuracy cannot be raised by more data
(Section 5).</li>
<li><b>Identification of configuration as the source of the interaction</b>, corroborated across three
independently collected corpora whose interaction magnitudes order exactly as their configuration
variation does, with a physical mechanism for the quantisation case (Section 6).</li>
<li><b>A prediction method for unmeasured pairs and unseen hardware</b>, reaching 87.9 per cent ranking
accuracy and cutting cold-start energy regret from 3.81x to 1.04x on a public 33-accelerator matrix
using specification sheets alone (Section 7).</li>
<li><b>A facility-scale accounting</b> that orders procurement, consolidation and placement by measured
worth, and identifies the fleet compositions for which placement is worth nothing (Section 8).</li>
</ul>

<h2>2. Related Work</h2>

<h3>2.1 Measurement infrastructure and public corpora</h3>
<p>
Energy measurement for machine-learning systems has matured rapidly. MLPerf Power [4] established
certified wall-power submission across many vendor systems; the ML.ENERGY benchmark [5] provides
automated inference energy measurement over dozens of models and tasks, and its diagnostic successor
[6] attributes energy to latent system-side metrics rather than to latency. BUTTER-E [7] contributes
63,527 measured runs over 30,582 configurations with node-level watt-meters, together with per-node
hardware and idle-power tables. The llm-perf leaderboard [8] supplies per-phase energy for many
language models across a small hardware set with quantisation and dtype as explicit axes. Adjacent
corpora include production cluster telemetry [9] and clock-and-power-cap sweeps on datacenter parts for
non-machine-learning workloads [10]. These corpora are individually dense on one axis and thin on the
others, a structural property that Section 6 exploits.
</p>

<h3>2.2 Predicting performance, power and energy</h3>
<p>
NeuSight [11] forecasts training and inference latency on unseen accelerators by decomposing kernels
into tiles, reducing GPT-3 latency error on H100 from over 100 per cent to a few per cent, but does not
address energy. WattGPU [12] predicts inference power and latency for unseen GPU and model
combinations from published metadata, and reports that its power model transfers substantially better
than its latency model, an asymmetry consistent with our finding that the two are distinct prediction
problems. Roofline-inspired scaling models for fine-tuning state explicitly that neither
floating-point operation counts nor runtime is a sufficient proxy for energy [13], and replication
studies find that operation-count corrections systematically underestimate execution time [14, 15].
Pagoda [16] constructs calibrated energy and time rooflines for edge accelerators, and reports the
boundary condition that on Jetson-class hardware time efficiency does imply energy efficiency across
all power modes, which our Section 9 treats as a scope limit. Characterisations of inference
energy-performance tradeoffs across workloads and GPU scaling [17, 18] and cross-vendor accelerator
comparisons [19] report that the optimal platform varies with batch size, sequence length and model
size, and that rankings invert under latency constraints. The present work differs in taking energy
ordering itself as the estimand and in isolating the component of the data responsible for it.
</p>

<h3>2.3 Recommendation machinery for resource allocation</h3>
<p>
Applying recommender-system methods to cluster management originates with Paragon [20], which uses
singular value decomposition with stochastic-gradient reconstruction to classify unseen applications
for heterogeneity and interference, and Quasar [21], which extends the approach from placement to
allocation quantity, determining scale-up and scale-out amounts from a small number of profiling runs.
Selecta [22] performs latent-factor completion over a sparse application-by-instance-type runtime
matrix; PARIS [23] and CherryPick [24] address cloud configuration selection through hybrid
performance modelling and Bayesian optimisation respectively, and Micky [25] formulates instance
selection as a multi-armed bandit. This line is directly ancestral to our method, and the distinction
is the estimand: every system named optimises quality of service, runtime or monetary cost, and none
targets energy. Paragon references energy only as motivation for consolidation. On the modelling side
we draw on inductive matrix completion [26, 27], which admits prediction for rows and columns unseen
at training time, and on explicit low-rank feature crossing [28]. Our own prior result on
communication-free collaborative filtering under a hidden low-rank compatibility structure [29]
supplies the sample-complexity argument that motivates sparse profiling, and the present work carries
that structure from an abstract scalar signal into metered physical energy.
</p>

<h3>2.4 Energy-aware scheduling, partitioning and power management</h3>
<p>
Zeus [30] establishes an energy-time Pareto frontier for training via batch size and power-limit
search, reporting improvements between 15 and 76 per cent; Perseus [31] reduces training energy by up
to 30 per cent without throughput loss. PowerFlow [32] improves job completion time at equal energy,
and POLCA [33] treats the utility power contract as the binding constraint and oversubscribes it,
fitting 30 per cent more servers into a fixed power envelope, which is the closest existing work to
the connection-capacity framing of Section 1. On spatial partitioning, MISO [34] predicts favourable
multi-instance GPU partitions using multi-process service as a proxy, ECLIP [35] reports 25 per cent
better energy efficiency through kernel-wise compute-unit partitioning, and recent work predicts the
throughput-per-watt-optimal streaming-multiprocessor split for co-located pairs, exceeding equal
partitioning by 35 per cent and approaching the offline optimum [36]. Energy-efficient scheduling on
multi-instance GPUs with dynamic repartitioning [37] and agentic CPU-GPU assignment [38] both report
that performance-first placement is suboptimal across heterogeneous workloads. Carbon Explorer [39]
performs capacity planning against a carbon objective. To our knowledge no published work optimises
accelerator procurement mix against an energy budget or a grid interconnection cap, which Section 8
addresses.
</p>

<h3>2.5 Idle power and energy proportionality</h3>
<p>
Paragon's premise that servers "are not energy-proportional and consume a large fraction of peak power
even at low utilization" [20] has been substantially revised for modern accelerators. Instrumented
whole-facility measurement reports an H100 SXM idling at 72.5 W against 696 W measured peak,
approximately 10.4 per cent [40]. The stranded energy has instead migrated inside allocated jobs:
execution-idle, where a device is allocated and nominally executing but nearly inactive, accounts for
19.7 per cent of in-execution time and 10.7 per cent of energy across 756 GPUs over 31 days, rising to
48 per cent of energy for serving workloads [41]. Server sleep-state characterisation [42] and analyses
of realised idle residency [43] quantify the wake-latency tradeoffs that Section 8 parameterises. Our
own measurements in Section 8 add the observation that energy proportionality is markedly worse on
small accelerators than on large ones.
</p>

<h3>2.6 Quantisation and its execution cost</h3>
<p>
Post-training quantisation methods [44, 45] reduce weight precision to four bits, and their runtime
behaviour is central to Section 6. Standard pipelines dequantise weights in kernels separate from the
matrix multiplication, with dequantisation accounting for the majority of time within quantised
matrix multiplication and fused alternatives substantially reducing that overhead [46]. This supplies
the physical mechanism by which a configuration choice alters which hardware property is limiting, and
therefore which accelerator is energy-optimal.
</p>

<h2>3. Methodology</h2>
"""

REFS = """<h2>References</h2>
<ol class="refs">
<li>Israel Electricity Authority. Suspension of processing of grid-connection applications for server farms, 140 days. Reported 22 July 2026.</li>
<li>Lin, L., Wijayawardana, R., Rao, V., Nguyen, H., Gnibga, W. E., and Chien, A. A. Exploding AI power use: an opportunity to rethink grid planning and management. <i>ACM e-Energy</i>, 2024. arXiv:2311.11645.</li>
<li>International Energy Agency. <i>Energy and AI: energy demand from AI</i>, 2025.</li>
<li>Tschand, A., Rajan, A. T., et al. MLPerf Power: benchmarking the energy efficiency of machine learning systems from microwatts to megawatts. 2024. arXiv:2410.12032.</li>
<li>Chung, J.-W., Ma, J. J., Wu, R., et al. The ML.ENERGY benchmark: toward automated inference energy measurement and optimization. <i>NeurIPS Datasets and Benchmarks</i>, 2025. arXiv:2505.06371.</li>
<li>Chung, J.-W., Wu, R., Ma, J. J., and Chowdhury, M. Where do the joules go? Diagnosing inference energy consumption. 2026. arXiv:2601.22076.</li>
<li>Tripp, C. E., Perr-Sauer, J., Gafur, J., et al. Measuring the energy consumption and efficiency of deep neural networks: an empirical analysis and design recommendations. 2024. arXiv:2403.08151.</li>
<li>Optimum-Benchmark. llm-perf-leaderboard. Hugging Face Datasets.</li>
<li>MIT AI Accelerator. MIT SuperCloud dataset: labelled deep-learning jobs with node power telemetry.</li>
<li>Afzal, A., et al. GPU energy sweet-spot study: clock and power-cap sweeps on A40, A100, H100 and H200. 2026. arXiv:2607.00819.</li>
<li>Lee, S., Phanishayee, A., and Mahajan, D. Forecasting GPU performance for deep learning training and inference. <i>ASPLOS</i>, 2025. arXiv:2407.13853.</li>
<li>Fadel Argerich, M., Fürst, J., and Patiño-Martínez, M. WattGPU: predicting inference power and latency on unseen GPUs and LLMs. <i>IJCAI Workshop on Sustainability and Resource-Efficiency of AI</i>, 2026. arXiv:2607.02391.</li>
<li>Zoubeirou a Mayaki, M. The energy consumption of transformer fine-tuning: a roofline-inspired scaling model. 2026. arXiv:2606.23546.</li>
<li>Barba Roque, E. and Cruz, L. FLOPs vs real work: the importance of replication in AI efficiency assessment. 2026. arXiv:2608.14550.</li>
<li>Desislavov, R., Martínez-Plumed, F., and Hernández-Orallo, J. Compute and energy consumption trends in deep learning inference. <i>Sustainable Computing</i>, 38, 2023. arXiv:2109.05472.</li>
<li>Prashanthi, S. K., Sahoo, K. K., Saikia, A. R., et al. Pagoda: an energy and time roofline study for DNN workloads on edge accelerators. 2025. arXiv:2509.20189.</li>
<li>Maliakel, P. J., Ilager, S., and Brandic, I. Characterizing LLM inference energy-performance tradeoffs across workloads and GPU scaling. 2025. arXiv:2501.08219.</li>
<li>Mayr, M., Wind, S., Schröder, L., et al. AI application benchmarking: power-aware performance analysis for vision and language models. 2026. arXiv:2603.16164.</li>
<li>Golden, A., Wu, C.-J., Wei, G.-Y., and Brooks, D. The xPU-athalon: quantifying the competition of AI acceleration. <i>ISPASS</i>, 2026. arXiv:2604.10852.</li>
<li>Delimitrou, C. and Kozyrakis, C. Paragon: QoS-aware scheduling for heterogeneous datacenters. <i>ASPLOS</i>, 2013, pp. 77–88.</li>
<li>Delimitrou, C. and Kozyrakis, C. Quasar: resource-efficient and QoS-aware cluster management. <i>ASPLOS</i>, 2014, pp. 127–144.</li>
<li>Klimovic, A., Litz, H., and Kozyrakis, C. Selecta: heterogeneous cloud storage configuration for data analytics. <i>USENIX ATC</i>, 2018, pp. 759–773.</li>
<li>Yadwadkar, N. J., Hariharan, B., Gonzalez, J. E., Smith, B., and Katz, R. H. Selecting the best VM across multiple public clouds. <i>SoCC</i>, 2017.</li>
<li>Alipourfard, O., Liu, H. H., Chen, J., Venkataraman, S., Yu, M., and Zhang, M. CherryPick: adaptively unearthing the best cloud configurations for big data analytics. <i>NSDI</i>, 2017, pp. 469–482.</li>
<li>Hsu, C.-J., Nair, V., Menzies, T., and Freeh, V. Micky: a cheaper alternative for selecting cloud instances. 2018. arXiv:1803.05587.</li>
<li>Zhang, M. and Chen, Y. Inductive matrix completion based on graph neural networks. 2019. arXiv:1904.12058.</li>
<li>Ledent, A., Alves, R., and Kloft, M. Orthogonal inductive matrix completion. 2020. arXiv:2004.01653.</li>
<li>Wang, R., Shivanna, R., Cheng, D. Z., et al. DCN V2: improved deep and cross network for web-scale learning to rank systems. <i>WWW</i>, 2021.</li>
<li>Apartsin, A., Meshulam, Y., and Aperstein, Y. Acting on the unseen: communication-free collaborative filtering for decentralized multi-robot task allocation. 2026. arXiv:2605.25584.</li>
<li>You, J., Chung, J.-W., and Chowdhury, M. Zeus: understanding and optimizing GPU energy consumption of DNN training. <i>NSDI</i>, 2023. arXiv:2208.06102.</li>
<li>Chung, J.-W., Gu, Y., Jang, I., Meng, L., Bansal, N., and Chowdhury, M. Reducing energy bloat in large model training. <i>SOSP</i>, 2024.</li>
<li>Gu, D., Xie, X., Huang, G., Jin, X., and Liu, X. Energy-efficient GPU clusters scheduling for deep learning. 2023. arXiv:2304.06381.</li>
<li>Patel, P., Choukse, E., Zhang, C., Goiri, Í., Warrier, B., Mahalingam, N., and Bianchini, R. Characterizing power management opportunities for LLMs in the cloud. <i>ASPLOS</i>, 2024.</li>
<li>Li, B., Patel, T., Samsi, S., Gadepally, V., and Tiwari, D. MISO: exploiting multi-instance GPU capability on multi-tenant GPU clusters. <i>SoCC</i>, 2022, pp. 173–189.</li>
<li>Quach, R., Wang, Y., Jahanshahi, A., Wong, D., and Kim, H. ECLIP: energy-efficient and practical co-location of ML inference on spatially partitioned GPUs. <i>ISLPED</i>, 2025. arXiv:2506.12598.</li>
<li>Han, B.-S., Parekh, K., Lin, W.-C., Paul, T., Gandhi, A., and Liu, Z. Energy-efficient GPU SM allocation. <i>ACM SIGMETRICS Performance Evaluation Review</i>, 2025.</li>
<li>Lipe, E., Karia, N., Espenshade, C., Stein, C., Tantawi, A., and Tardieu, O. Energy efficient scheduling of AI/ML workloads on multi-instance GPUs with dynamic repartitioning. 2026. arXiv:2606.25082.</li>
<li>Lu, T. and Reda, S. Agentic CPU-GPU scheduling for heterogeneous AI workloads. 2026. arXiv:2607.22242.</li>
<li>Acun, B., Lee, B., Kazhamiaka, F., et al. Carbon Explorer: a holistic framework for designing carbon aware datacenters. <i>ASPLOS</i>, 2023, pp. 118–132.</li>
<li>Vercellino, R., Willard, J., Campos, G., et al. Measurement of generative AI workload power profiles for whole-facility data center infrastructure planning. 2026. arXiv:2604.07345.</li>
<li>Lei, Y., Fernandez, J., Kypriotis, V., Skarlatos, D., Strubell, E., Sherry, J., and Vosler, D. The energy cost of execution-idle in GPU clusters. 2026. arXiv:2604.04745.</li>
<li>Griffiths, A., Morsman, A., and Veitch, P. Understanding the performance and power saving tradeoffs of server sleep states. <i>IEEE CloudNet</i>, 2023.</li>
<li>Antoniou, G., Volos, H., Haj Yahya, J., and Sazeides, Y. How long can you sleep? Idle time system inefficiencies and opportunities. <i>DCEE Workshop</i>, 2025. arXiv:2510.07449.</li>
<li>Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. GPTQ: accurate post-training quantization for generative pre-trained transformers. <i>ICLR</i>, 2023. arXiv:2210.17323.</li>
<li>Lin, J., Tang, J., Tang, H., et al. AWQ: activation-aware weight quantization for LLM compression and acceleration. <i>MLSys</i>, 2024. arXiv:2306.00978.</li>
<li>Frantar, E., Castro, R. L., Chen, J., Hoefler, T., and Alistarh, D. MARLIN: mixed-precision auto-regressive parallel inference on large language models. 2024.</li>
</ol>
"""

FOOTER = """<p class="footer">
Draft. Pending inclusion: the large-model training grid that produces memory-infeasible cells and
restores placement regret as a discriminating metric; workload-level characterisation of execution-idle
energy; figures 2 and 3. Three secondary citations await verification against full text.
</p>

</body>
</html>
"""

# renumber citations in the preserved tail to the new scheme
REMAP = {"1": "1", "2": "2", "3": "20", "4": "21", "5": "22", "6": "23", "7": "24",
         "8": "30", "9": "5", "10": "11", "11": "12", "12": "13", "13": "6",
         "14": "19", "15": "4", "16": "8", "17": "46", "18": "40", "19": "16", "20": "29"}
def fix(m):
    return "[" + ", ".join(REMAP.get(x.strip(), x.strip()) for x in m.group(1).split(",")) + "]"
tail_fixed = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", fix, tail)

out = head + FRONT + tail_fixed + REFS + FOOTER
# re-insert the figure before section 5 if the split dropped it
if "<svg" not in out:
    out = out.replace("<h2>5. The Decision Resides in a Low-Variance Residual</h2>",
                      '<figure style="margin:16px 0;text-align:center">' + fig +
                      "</figure>\n\n<h2>5. The Decision Resides in a Low-Variance Residual</h2>")
io.open("paper/greenmatch-paper.html", "w", encoding="utf-8", newline="\n").write(out)
print("rebuilt:", len(out), "chars")
print("svg present:", "<svg" in out)
print("refs:", out.count('<li>', out.index('<ol class="refs">')))
