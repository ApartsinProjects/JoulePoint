Second opinion
Executive conclusion

Cut collaborative filtering as a positive contribution of this paper. Do not build the paper around “CF improves energy-aware placement.” Your own results now contradict that claim too thoroughly: the main grid has zero decision headroom, most other corpora have only ~2%, free-embedding CF produces negligible gains, descriptor-based CF fails when rows are scarce, and exact assignment does not improve greedy placement. 

cf_problem_brief +1

I would not remove the experiments completely. A shortened negative CF experiment belongs in the main paper because it strengthens the more interesting result:

Interaction is necessary for workload-dependent placement, but interaction variance and ranking accuracy do not tell you whether useful placement opportunity exists.

That is substantially stronger than “we tried CF and it did not work.”

The paper should become primarily a measurement/decision-analysis paper, not a recommender-system paper.

1. Is the CF thread rescuable?
Straight answer: not in its current formulation

You currently have three separate facts:

Additive effects explain 97.5% of variance but mathematically cannot perform workload-specific placement.

The small interaction component causes many ranking inversions.

Those inversions often have almost no economic/energy value, because one fixed machine is already close to or exactly the oracle. On the main grid L4 wins every row, so your nominal 87.9% versus 81.7% ranking improvement has exactly zero energy value. 

cf_problem_brief +1

Point 3 kills a conventional CF story.

The particularly damaging results are not the negative cross-validation results. Those could conceivably be repaired with more data. The damaging result is the headroom geometry:

Corpus	Headroom
main grid	0.00%
extended	1.97%
training	2.00%
llm-perf	1.81%
Grid'5000	8.33%

And your free-embedding experiment shows that even when CF gets the observations it needs, the obtainable energy improvement is tiny. 

cf_problem_brief

That means there are two independent barriers:

useful learned placement=
headroom
decision opportunity
	​

	​

×
learnability
ability to infer interactions
	​

	​


Your experiments show failure on both axes in different datasets.

What I would retain

Keep perhaps one main-text subsection:

“Can interaction-aware learning exploit the residual?”

It should contain:

additive ceiling;

rank-1 inductive model as a deliberately simple interaction learner;

leave-family-out results;

oracle-loading experiment showing that the rank-1 representation is adequate but estimating the loading from few rows is not;

headroom analysis showing that better ranking frequently cannot translate into significant energy savings;

partial-observability result as the one important exception.

Move the sparsity sweeps, extra CF variants, detailed ablations and most recommender machinery to supplementary material.

The partial-observability experiment is different

Your strongest result—recovering 32%/56% of the unknown-precision gap from partial observations—is defensible, but it answers a different problem:

infer an unobserved workload/configuration state from a few measured executions.

That is much closer to latent-context adaptation / active profiling / meta-learning than “collaborative filtering for placement.” 

cf_problem_brief

There is a possible research thread there. But using it to rescue the current CF claim would make the paper less coherent.

Verdict

Current static job × hardware CF: cut as a claimed solution.

CF as a negative control demonstrating why interaction prediction is insufficient: keep.

2. Public datasets: many workloads × many hardware × measured energy × actual headroom

I searched specifically for datasets that could fill your missing quadrant. The result is not good.

2.1 Watt Counts is now the strongest candidate — but there is a release-status complication

The Watt Counts study is exactly the geometry you need: 50 LLMs, 10 NVIDIA GPUs, >5,000 experiments, 372 observed model–GPU combinations, with energy per token in batch mode and GPU energy obtained by integrating NVML measurements. It explicitly reports that the optimal GPU changes across models: H100 is best on 45/50 models in batch, H200 on 2, L4 on 3. 
arXiv

Paper:

https://arxiv.org/abs/2604.09048

The measurement is genuinely energy, not TDP: they sample GPU power through NVML and integrate it to joules; their 10 Hz choice gives <1% integrated-energy error relative to their 200 Hz comparison. 
arXiv

Important complication: version 1 of the paper is internally inconsistent. It calls the dataset “open” but also says the dataset/code will be released on GitHub “upon acceptance.” 
arXiv

However, the later WattGPU work has a genuinely public repository containing a Watt Counts subset used for 42 dense LLMs × 8 server-grade GPUs:

https://github.com/maufadel/wattgpu

Paper:

https://arxiv.org/abs/2607.02391

WattGPU explicitly states that its experiments use 42 LLMs and 8 GPUs and that its data/code are public. 
arXiv

Does it have real HEADROOM?

Yes in the qualitative decision-theoretic sense; exact aggregate HEADROOM: NOT VERIFIED.

Because different GPUs actually minimize measured energy for different rows, there is nonzero oracle-vs-fixed opportunity unless those differences are exact ties. Watt Counts also reports substantial hardware-dependent energy differences. 
arXiv
+1

But I would not quote a HEADROOM number until you calculate your exact

E
best fixed
	​

/E
row oracle
	​

−1

on the released rows using precisely your weighting convention.

Also note the warning sign: in batch, H100 wins 45/50 models. That could still mean aggregate headroom is modest. The more interesting regime may be server load, where their reported hardware-dependent savings are much larger. 
arXiv
+1

Confidence: HIGH that this is currently your best public AI candidate.
Confidence: LOW about its exact HEADROOM until calculated.

2.2 A real 18-hardware × 9-workload HPC energy corpus

This one is worth knowing about even though it is not AI.

“Power, performance and system measures of HPC benchmarks on multiple hardware.”

Dataset:

https://zenodo.org/records/14914799

Associated paper:

https://doi.org/10.1016/j.suscom.2025.101106

It contains public raw/processed data associated with experiments across heterogeneous HPC hardware; the associated work uses 9 HPC benchmarks over 18 clusters, with power/performance/system measurements and frequency variation. The Zenodo record is public and provides data.csv, augmented data, raw data and scripts. 
Zenodo

This is actually quite attractive as a methodological external validation corpus:

9 workloads is not large, but materially better than your 5-row Grid'5000 case;

18 targets is excellent;

targets are genuinely heterogeneous;

energy/performance measurements are available;

it is not an LLM/GPU corpus.

HEADROOM: NOT VERIFIED. I found no author-reported equivalent of your fixed-vs-oracle metric. Do not cite it as having 10%, 20%, etc. until computed.

Confidence: HIGH on existence and matrix structure; LOW/UNKNOWN on your exact headroom.

2.3 SWEAT: 35 workload configurations × 5 infrastructures

A very recent dataset:

https://zenodo.org/records/20181490

SWEAT provides controlled benchmark execution across five heterogeneous infrastructures and includes energy-consumption measurements. Its workload tree contains standardized CPU, memory and disk configurations; the authors describe 35 configurations with repeated runs across the infrastructures. 
Zenodo

Advantages:

controlled execution rather than scheduler logs;

same workload definitions across targets;

energy measured in the dataset;

enough rows to test your row-scarcity argument.

Problems:

only five hardware targets;

workloads are systems/microbenchmark workloads, not AI;

Zenodo currently lists the creator as Anonymous, which lowers provenance confidence;

HEADROOM NOT VERIFIED.

So I would treat SWEAT as an interesting robustness dataset, not as the dataset that solves your paper.

2.4 HPC accounting archives: generally the wrong data-generating process

MIT Supercloud is enormous and tempting:

https://arxiv.org/abs/2108.02037

https://dcc.mit.edu

It contains hundreds of thousands of jobs and GPU monitoring. But it does not give you the counterfactual matrix you need: normally each production job is executed on the machine that actually received it, rather than replayed across 5–20 alternative machines.

That distinction is fatal.

You observe

E(i,j
i
	​

)

rather than

{E(i,1),E(i,2),…,E(i,m)}.

Consequently the unobserved cells are not ordinary MCAR matrix-completion missingness. They are policy-selected counterfactual outcomes.

Using CF naïvely on such traces risks learning the historical scheduler. Proper treatment belongs to logged-bandit/off-policy learning and requires overlap/propensities or another credible identification assumption. Counterfactual Risk Minimization is directly relevant to that distinction. 
Proceedings of Machine Learning Research

So I would not pursue generic Slurm accounting archives as the missing CF corpus unless you can identify repeated/replayed applications across hardware.

2.5 Compiler/kernel autotuning

There are papers with the right experimental idea but I could not verify an open matrix that satisfies your requirements.

For example:

https://arxiv.org/abs/2211.07260

“Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning” explores GEMM parameter configurations across several GPUs with energy measurements. It is conceptually excellent for your problem because workload/configuration × hardware interactions should be large.

But I did not find a public raw result corpus that I am confident exists. Therefore I would not list it as a usable dataset.

2.6 Mobile/NPU device farms

A recent example, PowerBench:

https://arxiv.org/abs/2607.05475

Repository advertised at:

https://github.com/open-compass/PowerBench

The paper considers energy across CPU/GPU/NPU backends. But I could not verify a populated public result matrix comparable to Watt Counts. I would not count it yet.

Dataset bottom line

After looking specifically for this geometry, my conservative result is:

Dataset	Rows × hardware	Measured energy	AI?	Counterfactual matrix?	Verified useful headroom?
Watt Counts / public WattGPU subset	50×10 study; public subset 42×8	Yes	Yes	Mostly	Positive; exact H not computed
HPC 18-cluster corpus	9×18	Yes	No	Yes	Unknown
SWEAT	~35×5	Yes	No	Yes	Unknown
MIT Supercloud	huge	telemetry	partly	No	Not identifiable directly
GPU autotuning paper	large config space ×5	Yes	kernels	Yes	raw public corpus not verified
PowerBench	several models/backends	Yes in paper	Yes	potentially	public matrix not verified

I did not find a second public AI dataset that I can confidently say satisfies all four of your criteria and also has verified substantial aggregate HEADROOM.

That scarcity is itself relevant: the available benchmarking ecosystem is badly shaped for evaluating workload-aware hardware placement.

3. What recommender formulation would make CF genuinely necessary?

There is no recommender trick that fixes your present static problem.

Changing MF to BPR, NCF, factorization machines, graph CF, higher rank, Bayesian MF, etc. does not alter the fundamental bound

ΔE≤E
fixed
	​

−E
oracle
	​

.

If that quantity is 0–2%, a more

You've reached the maximum length for this conversation, but you can keep talking by starting a new chat.

Start new chat