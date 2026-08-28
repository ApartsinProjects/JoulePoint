# 1. PUBLIC DATASETS

I used a strict interpretation of your criterion:

> **Same identifiable workload/configuration must occur on ≥3 distinct hardware platforms, and the released data must contain measured energy or energy obtained by integrating measured electrical telemetry—not merely TDP, nominal power, runtime, or a model-based estimate.**

I found only a few convincing additions. Many apparently relevant datasets fail once this criterion is enforced.

## A. Datasets that meet, or substantially meet, the criterion

| Dataset | Hardware coverage | Energy | Licence | Fit / comments |
|---|---:|---|---|---|
| **Energy-Inference / Wilkins et al.** | **3 systems:** Apple M1 Pro; NVIDIA A100 system; NVIDIA V100 system | **Measured** | **⚠️ Not stated in repository** | **Strong fit.** Raw `stats/` directory explicitly contains the project's energy data, with scripts for energy monitoring on Apple and NVIDIA hardware. Same LLM workloads, notably Llama-2-7B and Mistral-7B, are run across the three systems. The associated paper also uses these measurements to demonstrate workload-aware heterogeneous routing.  |
| **DeepEn2023** | **8 edge devices/SoCs** | **Measured externally**, reported in mJ | **⚠️ Not stated/verified** | **Strong at kernel level.** Thousands of generated kernels are deployed on different devices and energy is measured during execution. The paper uses external Monsoon instrumentation; the dataset explicitly reports CPU/GPU energy in mJ. The repository says, however, that **currently only the kernel-level dataset is uploaded**; model/application data are described but not yet released.  |
| **Grid'5000 multi-hardware HPC benchmark measurements** | **18 clusters / processor platforms** | Electrical measurements; energy-to-solution from telemetry | **⚠️ Dataset licence not re-verified** | **Strong matrix structure, but CPU/HPC rather than AI accelerators.** Repeated NAS Parallel Benchmark workloads were executed across a large collection of Grid'5000 platforms and DVFS settings. Grid'5000's monitoring infrastructure includes external wattmeters/BMC/PDU/RAPL sources. **⚠️ I could not re-open the Zenodo schema to establish whether joules are stored directly or must be integrated from the measured power series.**  |
| **del Rey et al. replication package** | **3 GPUs:** GTX 750 Ti, RTX 3070, RTX 3090 | GPU energy from measured GPU telemetry; RAM contribution is estimated | **⚠️ Dataset licence not verified** | **Partial strict fit.** At least a subset of architectures occurs on all three GPUs; the low-memory GTX 750 Ti cannot run most of the larger architectures. Thus this is **not a dense 5-model × 3-GPU matrix**. The experiment is nevertheless highly relevant statistically.  |

### URLs

**Energy-Inference**
- Data/code: https://github.com/grantwilkins/energy-inference
- Paper: https://arxiv.org/abs/2407.00010

The repository explicitly says that `stats/` contains the energy data and distinguishes inference from “energy monitoring inference.” 

**DeepEn2023**
- Repository: https://github.com/amai-gsu/DeepEn2023
- Project/data page: https://amai-gsu.github.io/DeepEn2023/
- Paper: https://arxiv.org/abs/2310.18329
- ACM DOI: https://doi.org/10.1145/3583740.3628442

DeepEn is particularly interesting for your configuration argument: it contains generated kernel configurations and measured energy across heterogeneous SoCs. Its README explicitly distinguishes kernel-, model-, and application-level energy datasets and reports the measurements in mJ. 

**Grid'5000**
- Zenodo DOI: https://doi.org/10.5281/zenodo.10982238

This is probably the best additional dataset for stress-testing your **matrix/statistical methodology** because the same benchmark family is repeated across many machines, even though it is not an AI-accelerator corpus. 

**del Rey et al.**
- Replication package: https://doi.org/10.5281/zenodo.17041051
- Earlier archive cited from the preprint: https://zenodo.org/doi/10.5281/zenodo.11505891
- Paper: https://arxiv.org/abs/2307.05520
- Journal DOI: https://doi.org/10.1016/j.csi.2026.104170

**Correction to my interim note above:** I initially described this as a full 5×3 factorial. That was too strong. The low-end GTX 750 Ti could execute only a subset of the architectures because of memory limitations; the principal five-architecture comparison is effectively dominated by the RTX 3070 and RTX 3090 environments. 

---

## B. Public datasets/repositories that are relevant but **do not satisfy your strict criterion**

These are still worth mentioning because a reviewer may find them and ask why you did not use them.

| Dataset | Why it looks relevant | Why I would exclude it from your main matrix |
|---|---|---|
| **EcoCompute** | RTX 5090, RTX 4090D, A800; 100+ configurations; CC BY 4.0 | The visible source datasets use largely **different models on the three GPUs**, so I could not establish a common workload across all three. More importantly, energy is calculated from sampled average power and throughput rather than being a directly released energy measurement.  |
| **Bench360** | Paper evaluates four tasks on **3 hardware platforms**, with engines, quantization and serving configurations | Repository appears to expose benchmark software/configuration rather than a clearly identifiable raw three-hardware energy-results corpus. **⚠️ Raw public measurement dataset not verified.**  |
| **High-resolution AI Data Center Training Workloads Dataset** | H100, B200 and RTX 3060 measurements; 1.8M telemetry samples | Excellent **power telemetry**, but I could not establish a released joule/job-energy variable or identical workloads repeated across all three hardware classes.  |
| **SURF/Lisa — Generic and ML Workloads in an HPC Datacenter** | Production HPC job/node traces with 30-s power telemetry; CPU and GPU nodes | Real production energy analysis, but not a controlled same-workload × ≥3-hardware factorial; GPU population is principally TITAN RTX / GTX 1080 Ti.  |
| **Cross-Hardware Energy Benchmarking of DL Vision Architectures** | Public Figshare replication package | Only **two hardware classes** in the reported comparison, A100 GPU and AMD EPYC CPU.  |
| **SPECpower public results** | Hundreds of systems running a standard workload | Reports power and performance-per-watt, rather than your desired per-workload measured energy-to-completion matrix.  |

URLs:

- EcoCompute HF: https://huggingface.co/datasets/hongpingzhang/ecocompute-energy-efficiency  
- EcoCompute Zenodo: https://doi.org/10.5281/zenodo.18900289  
- EcoCompute GitHub: https://github.com/hongping-zh/ecocompute-ai
- Bench360: https://github.com/slinusc/bench360  
- Bench360 paper: https://arxiv.org/abs/2511.16682
- AI data-center dataset: https://doi.org/10.6084/m9.figshare.31654879  
- Associated GitHub: https://github.com/Ahmed-Elsayed95/High-resolution-AI-Data-Center-Training-Workloads-Dataset
- SURF/Lisa paper: https://arxiv.org/abs/2409.08949
- SURF/Lisa dataset: https://doi.org/10.5281/zenodo.13685426
- SURF/Lisa code: https://github.com/atlarge-research/2024-icpads-hpc-workload-characterization
- Cross-Hardware Figshare: https://figshare.com/articles/dataset/Cross-Hardware_Energy_Benchmarking_of_Deep_Learning_Vision_Architectures_Replication_Package/32410590

### Bottom line on datasets

Outside the corpora you already have, the **surprisingly sparse result is real**. I do **not** find another large, clean, public AI dataset analogous to MLPerf Power containing a dense:

\[
\text{workload}\times\text{accelerator}\times\text{configuration}\rightarrow
(\text{energy},\text{latency},\text{throughput})
\]

tensor over many accelerator models.

The strongest additional AI cases I can substantiate are **Energy-Inference** and **DeepEn2023**; Grid'5000 is excellent for validating the general statistical phenomenon on a much wider hardware population.

---

# 2. RELATED WORK THAT COULD THREATEN NOVELTY

## A. Fleet composition / provisioning + carbon/energy + SLA

### 1. EcoServe — **serious novelty threat**

**Li et al., “EcoServe: Designing Carbon-Aware AI Inference Systems.”**

- https://arxiv.org/abs/2502.05043
- Microsoft Research: https://www.microsoft.com/en-us/research/publication/ecoserve-designing-carbon-aware-ai-inference-systems/

This is the closest paper I found to your claim (c).

EcoServe explicitly describes itself as a **carbon-aware resource provision and scheduling framework** and maintains performance targets/SLOs.  It does more than runtime placement: its capacity-planning/right-sizing stage decides heterogeneous resources, considering GPU types including L4/A6000/A100/H100, followed by runtime scheduling/load balancing. Its reported right-sizing component produces substantial carbon reduction, and the full framework reports up to 47% while preserving SLOs. 

Therefore:

> **“Previous work optimizes placement, whereas we are the first to optimize fleet composition” is not safe.**

What still appears distinguishable is your **quantitative decomposition of the attainable benefit into composition versus placement**, particularly under the *same empirical workload×accelerator energy matrix*.

---

### 2. DynamoLLM — close, but not composition in your sense

**“DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency.”**

- https://arxiv.org/abs/2408.00741

It optimizes energy subject to inference latency/SLO considerations by selecting runtime/system parameters such as GPU frequency, tensor/model parallelism and the number of instances. The reported experimental cluster is homogeneous H100 hardware. 

So this is:

**energy objective + SLA + resource provisioning/scaling: yes**  
**choice of accelerator type / heterogeneous purchase mix: no**

It should be cited, but it does not invalidate a carefully stated composition contribution.

---

### 3. Splitwise — heterogeneous provisioning but not primarily an energy objective

**“Splitwise: Efficient Generative LLM Inference Using Phase Splitting.”**

- https://arxiv.org/abs/2311.18677

Splitwise separately provisions prefill/decode resources and evaluates heterogeneous machine configurations under performance constraints. Its main optimization framing is efficiency/cost/throughput rather than “choose accelerator fleet to minimize measured energy/carbon.” 

Adjacent, but not a direct anticipation of your formulation.

---

### 4. Mélange — accelerator mix + SLO, but **cost**, not energy

**“Mélange: Cost Efficient LLM Serving by Exploiting GPU Heterogeneity.”**

- https://arxiv.org/abs/2404.14527

This is particularly relevant structurally: it determines a heterogeneous GPU allocation/mix while satisfying serving SLOs. Its objective, however, is monetary cost rather than measured energy/carbon. 

Thus it is excellent related work for your **optimization formulation**, but not a direct energy-composition predecessor.

---

### 5. FleetOpt — almost exactly the procurement variable, wrong objective

**“FleetOpt”**
- https://arxiv.org/abs/2603.16514

It explicitly optimizes GPU types/counts for an inference fleet under tail-latency constraints, but the objective is fleet/cost efficiency rather than energy/carbon. 

Again, reviewers may view your work as an energy analogue of this family, so differentiation should be explicit.

---

## B. Placement-only work that matters for claim (c)

### Wilkins et al.: direct empirical placement precedent

The same **Energy-Inference** work above studies workload-aware routing over heterogeneous machines. It reports roughly a **7.5% energy reduction** from using workload information for routing rather than a workload-unaware allocation. 

That means:

> **The claim “the energy value of heterogeneous placement has not been measured” is unsafe.**

But this paper does **not** solve your more specific question:

\[
\text{How much attainable energy improvement comes from choosing the fleet}
\quad\text{versus}\quad
\text{routing intelligently inside that fixed fleet?}
\]

I did not find that exact decomposition there.

Other recent placement/runtime papers include:

- **CEDAR**, heterogeneous/carbon-aware agentic LLM routing with latency considerations; no hardware-purchase decision. 
- **CARBONDIS**, carbon-aware DNN inference scheduling subject to latency; placement rather than procurement. 
- **throttLL'eM**, energy-aware GPU frequency/resource adaptation while meeting inference SLOs. 
- **Festina**, shared-GPU scheduling/resource adaptation under TTFT/TBT SLOs; runtime rather than procurement. 
- **DEFT**, joint heterogeneous placement/DVFS optimization, but again assumes an available hardware pool rather than deciding what to buy. 

---

## C. Your additive-model argument

For

\[
E_{wh}=\mu+\alpha_w+\beta_h,
\]

comparison of accelerators \(h_1,h_2\) gives

\[
E_{w,h_1}-E_{w,h_2}
  =\beta_{h_1}-\beta_{h_2}.
\]

The workload term cancels. Consequently,

\[
\operatorname*{argmin}_{h} E_{wh}
=
\operatorname*{argmin}_{h}\beta_h
\]

for every workload \(w\).

So yes: **an additive workload+hardware model is mathematically incapable of workload-dependent accelerator selection. It is exactly a global/fixed accelerator ranking.** Interactions are the only mechanism by which the ranking can change.

However, I would **not present this algebraic equivalence itself as a novel theorem**. It is an elementary consequence of the standard “no interaction” two-factor model: additive effects imply parallel responses/no change in factor differences. I did **not** find an AI-energy paper explicitly spelling out the consequence as “additive model = fixed accelerator ranking,” but the underlying statistical fact is standard. 

A safer novelty statement is therefore:

> *We exploit the decision-theoretic consequence of the additive model: despite explaining almost all variance, it cannot make workload-specific hardware decisions, because it induces the same hardware ordering for every workload.*

That is considerably stronger and safer than saying you discovered a new statistical theorem.

---

## D. The ~2.5% interaction residual: **there is a close predecessor**

This is the literature result I would take most seriously.

**del Rey et al., “Estimating Deep Learning energy consumption based on model architecture and training environment.”**

- https://arxiv.org/abs/2307.05520
- https://doi.org/10.1016/j.csi.2026.104170

They explicitly question FLOPs/TDP-style energy estimation across different execution environments and report that architecture/environment/main factors explain nearly all variance while **interaction effects account for less than roughly 3% of total energy variance**. 

That is uncomfortably close to your empirical **~2.5% interaction residual**.

The crucial distinction is that they do **not**, as far as I could verify:

1. derive the **fixed-ranking consequence** of the additive model;
2. show that a small interaction component can nevertheless contain essentially **all information required for choosing the best accelerator**;
3. interpret the interaction through precision/batch/quantization configuration;
4. connect it to procurement versus placement value;
5. demonstrate that high \(R^2\) energy prediction is therefore potentially useless for the actual hardware-selection decision.

That combination looks much more defensible as your contribution.

---

## E. Configuration interactions themselves are no longer novel

A very recent paper is particularly relevant:

**Zine et al., “Attention to Detail: Evaluating Energy, Performance, and Accuracy Trade-offs Across vLLM Configurations.”**

- https://arxiv.org/abs/2607.09172

It studies energy/performance/accuracy over a large controlled set of vLLM configurations and explicitly analyzes configuration/task main and interaction effects. The paper was submitted in July 2026. 

It therefore makes a broad statement such as

> “Existing work ignores workload × execution-configuration interactions”

unsafe.

Your narrower result can still be distinct:

> The interactions explain only a tiny portion of **energy variance**, yet determine a disproportionately large fraction of the **hardware-selection decisions**, and those reversals can be traced to execution configuration.

That “variance importance ≠ decision importance” point is, based on what I found, much less occupied.

---

## F. How safe are your three claims?

| Claim | Novelty assessment | Why |
|---|---|---|
| **(a) Peak vendor specs fail to predict least-energy accelerator** | **Not novel if stated broadly** | del Rey already shows that TDP/FLOPs and similar simple hardware-centric proxies are unreliable across environments.  |
| **(a) Measured throughput also fails to identify the least-energy accelerator** | **Potentially novel / relatively safe** | I found efficiency trade-off studies, but no paper in this search that cleanly establishes your exact claim across a multi-accelerator energy matrix. Avoid claiming absolute firstness. |
| **(b) Additive workload+machine model = fixed hardware ranking** | **Correct, but not a novel mathematical theorem** | Standard consequence of a no-interaction additive model. Your contribution is applying it to accelerator choice and exposing its decision failure. |
| **(b) Only ~2.5% residual variance matters for accelerator choice** | **Interesting but needs careful positioning** | del Rey already reports <3% interaction variance. Novelty should be that this tiny residual contains the *decision-changing* information, not merely that interaction variance is small.  |
| **Configuration produces the ranking reversals** | **Only narrowly novel** | Recent vLLM work already studies task×configuration interactions. Precision/batch/quantization as the explanation of cross-accelerator ranking reversals is more specific.  |
| **(c) Fleet composition is an energy/carbon optimization variable under SLA** | **Definitely not novel broadly** | EcoServe already performs heterogeneous resource provision/right-sizing plus scheduling while maintaining SLOs.  |
| **Composition dominates placement quantitatively** | **Looks substantially safer** | I found individual composition and placement improvements, but no paper cleanly decomposing the two opportunities under a common workload×hardware energy matrix. |
| **Placement benefit becomes exactly 0 on homogeneous hardware** | **Mathematically true only under assumptions; don't oversell as empirical novelty** | It requires exchangeable machines/configurations. Queueing, DVFS, thermal state, consolidation/idle energy, locality, networking, etc. can make physical placement matter even among nominally identical accelerators. |

For claim (c), I would state the limiting result explicitly as conditional:

\[
E(w,h_i,c)=E(w,h_j,c)
\quad\forall i,j\text{ of the same hardware/configuration}
\]

implies that accelerator-level placement has zero optimization value. This does **not** claim that all data-center scheduling becomes irrelevant on homogeneous fleets.

---

# Most useful three

**1. Energy-Inference — most useful new AI dataset.**  
https://github.com/grantwilkins/energy-inference

It gives you precisely what most datasets lack: same LLM inference workloads, three genuinely different systems, empirical energy data, and an existing workload-aware routing result. It is particularly useful for independently checking both claims (a) and (c). 

**2. del Rey et al. — most important paper to cite defensively.**  
https://arxiv.org/abs/2307.05520

Its **<3% interaction variance** result and criticism of TDP/FLOPs estimation are close enough that omitting it would expose you to a reviewer objection. Your distinction should be **decision importance of the residual**, not merely its magnitude. 

**3. EcoServe — most important threat to the fleet-composition claim.**  
https://arxiv.org/abs/2502.05043

It rules out a broad “first to consider heterogeneous fleet composition under carbon and SLA constraints” claim. Your safer contribution is a **formal/empirical decomposition of composition opportunity versus placement opportunity**, rather than composition optimization itself. 

If choosing the **three datasets only**, I would use **Energy-Inference + DeepEn2023 + Grid'5000**.

# Items I could NOT verify

- **Grid'5000:** I verified the multi-hardware experiments and energy/power monitoring infrastructure, but **could not conclusively verify whether the Zenodo files contain a direct joules field versus power traces that must be integrated**. I also did not independently verify the dataset licence. 
- **Energy-Inference:** the GitHub repository is public and explicitly contains energy data, but **I found no explicit repository/data licence**. Publicly readable does not itself grant reuse rights. 
- **DeepEn2023:** **dataset licence not verified**. The repository currently says only the kernel-level dataset has been uploaded and access is obtained after a survey. I did not inspect the downloaded corpus row-by-row to prove that every individual kernel configuration occurs on ≥3 devices. 
- **del Rey replication package:** exact archive licence not verified. Also, the data are **not** a complete 5×3 matrix; only some models fit the GTX 750 Ti. RAM energy is estimated rather than instrumented, although GPU energy comes from telemetry. 
- **Bench360:** paper and benchmark repository definitely exist, but **I could not verify a released raw three-hardware measurement corpus**, so I would not call it a public dataset yet. 
- **Yao et al., CNN inference on M40/P4/V100:** I verified reports of the three-GPU experiment, but **could not find a public raw-data archive**. Paper DOI: https://doi.org/10.1002/cpe.6064. 
- **ECBA-MLI:** experiments span CPU/VPU/TPU/GPU classes, but **I could not verify a reusable public measurement corpus**. Paper DOI: https://doi.org/10.1109/EDGE55608.2022.00016. 
- **InferenceX / SemiAnalysis:** https://inferencex.semianalysis.com/ clearly covers many current accelerator families, but I could not verify a downloadable raw dataset containing measured joules, its exact measurement methodology, or a data licence. 
- **nsfcac/ai-inference-energy:** https://github.com/nsfcac/ai-inference-energy supports heterogeneous GPU energy experiments, but I could not verify that it contains a sufficiently complete released cross-GPU results corpus rather than primarily the measurement framework.
- **High-resolution AI Data Center Training Workloads Dataset:** I could not prove identical workload/configuration overlap across RTX 3060, H100 and B200, nor an explicit per-job joules variable, so I would treat it as a telemetry dataset rather than one of your core energy matrices. 
- **Prior additive=fixed-ranking statement:** I found no AI-energy paper explicitly making your exact argument. This is **not proof that none exists**; more importantly, the underlying algebra is standard enough that I would not make a priority/first-proof claim.
- **Procurement-versus-placement decomposition:** I found EcoServe-style provisioning and several placement studies, but **no paper I could verify that computes the attainable energy/carbon benefit of fleet composition and placement separately under the same workload×accelerator matrix**, then demonstrates the collapse of the placement term for an exchangeable homogeneous fleet. Of your proposed contributions, this appears to be the safest novelty territory.