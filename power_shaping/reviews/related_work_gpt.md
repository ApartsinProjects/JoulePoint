```markdown
# Related-work review plan

The related work should be organized around the paper's actual novelty claims, not simply around "green AI." The manuscript claims: (i) a dense measured `(GPU, power cap, batch)` response surface, ELF; (ii) an empirical operating-point law \(P(R)=P_0+aR^\beta\); (iii) the energy-minimizing "Joule point"; (iv) SLO-aware allocation under a fleet power budget; and (v) an economic split incentive under time-based GPU pricing. 

A particularly important positioning point is that the paper already correctly says that **the existence of a sub-TDP energy optimum is not its novelty**.  That disclaimer should be strengthened, because several additional literatures come very close to individual pieces of the paper.

## 1. Convex power–performance laws and speed-scaling theory — **MISSING; HIGH PRIORITY**

This is the largest conceptual omission. Long before GPU DVFS work, scheduling theory explicitly modeled power as a convex/superlinear function of processing speed and optimized energy subject to job deadlines. Because your paper derives \(P(R)=P_0+aR^\beta\), a closed-form minimum of \(P/R\), and deadline-aware power allocation, reviewers familiar with speed scaling may immediately make this connection. Your empirical contribution is still different, but Section 2 needs to acknowledge the lineage.

- **ADD — Yao, Demers & Shenker, "A Scheduling Model for Reduced CPU Energy," FOCS 1995. DOI: 10.1109/SFCS.1995.492493.**  
  They formulate minimum-energy scheduling with convex power \(P(s)\) versus processor speed and explicitly analyze \(P(s)=s^p\); your distinction is that ELF empirically establishes a GPU-inference operating-point law with a nonzero floor \(P_0\), and shows that inference rate alone does **not** identify power once batch/configuration varies. 

- **ADD — Miyoshi et al., "Critical Power Slope: Understanding the Runtime Effects of Frequency Scaling," ICS 2002. DOI: 10.1145/514191.514200.**  
  This is an especially important precursor to the Joule point: it explains why slowing a processor does not necessarily reduce total energy and derives a criterion for an energy-efficient operating point; your contribution is a measured GPU power-cap surface, the \(P_0+aR^\beta\) fit, and a per-inference stationary point exposed through a deployable power-cap knob. 

- **ADD — Gandhi, Harchol-Balter, Das & Lefurgy, "Optimal Power Allocation in Server Farms," SIGMETRICS 2009. DOI: 10.1145/1555349.1555368.**  
  They show theoretically and experimentally that maximum server power is not always optimal under a farm-wide power budget; your scheduler works at per-model GPU inference operating points and optimizes energy/SLO behavior using measured response curves rather than abstract server speed curves. 

**Positioning consequence:** do not present superlinear power-versus-rate itself as an unprecedented functional form. A safer and stronger claim is: **ELF empirically establishes this form for GPU inference along fixed operating-point families, quantifies \(P_0,\beta\), and demonstrates that pooling different batch operating points destroys a rate-only model.** That last property is genuinely central to your coordinate-system argument. The paper reports the pooled fit collapsing to roughly \(R^2=0.04\). 

---

## 2. GPU power capping, DVFS, and frequency control — **PRESENT, but now TOO THIN on inference**

Current Section 2 already cites Tang et al., the Mei survey, Zeus, BatchDVFS, Perseus, and μ-Serve, so the historical/training side is reasonably covered.  The problem is that several **recent inference systems are closer to your paper than most of those citations**.

- **KEEP — Zeus, You, Chung & Chowdhury, NSDI 2023. arXiv:2208.06102.**  
  Zeus jointly optimizes batch size and GPU power limit for DNN training; ELF instead turns dense inference measurements into a reusable response surface and derives the Joule point rather than operating an online training optimizer.

- **KEEP — POLCA, Patel et al., ASPLOS 2024. DOI: 10.1145/3620666.3651329.**  
  POLCA studies power-management opportunities and power oversubscription for cloud LLMs; your focus is energy per useful inference and the structure of the complete cap-response surface rather than primarily safe cluster power oversubscription.

- **KEEP — DynamoLLM, Stojkovic et al., HPCA 2025. arXiv:2408.00741.**  
  DynamoLLM co-manages cluster frequency/parallelism for LLM inference; your distinction is the response law, closed-form energy optimum, ELF dataset, and economic interpretation.

- **ADD — Kakolyris et al., "SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving" / throttLL'eM. arXiv:2408.05235.**  
  throttLL'eM changes GPU frequency and instance configuration at iteration granularity to minimize LLM inference energy while satisfying SLOs; your work manipulates exposed power caps and develops an operating-point calculus/dataset rather than a predictive serving controller. 

- **ADD — Liu et al., "GreenLLM: SLO-Aware Dynamic Frequency Scaling for Energy-Efficient LLM Serving." arXiv:2508.16449; DAC 2026.**  
  GreenLLM separately controls prefill and decode frequencies around latency SLOs; your response surface spans heterogeneous inference models and seeks a general energy-space characterization rather than phase-specific LLM DVFS. 

- **MUST ADD — Hankendi et al., "PALS: Power-Aware LLM Serving for Mixture-of-Experts Models." arXiv:2605.21427.**  
  PALS explicitly says that it treats **GPU power caps as a first-class control knob** and jointly tunes them with batch size inside vLLM, making it extremely close to your current wording; your defensible distinction is ELF + the empirical law + derivatives/Joule point + cross-model characterization + economics, not being first to make the cap a control variable. 

**Required wording change:** the present statement that other systems operate in hardware space whereas "we make the power cap itself a first-class decision variable" is now too strong.  PALS in particular makes essentially that claim. Reframe yours as **a measured coordinate system and calculus over power-cap operating points**, rather than first-class power control itself.

---

## 3. LLM inference serving and resource scheduling — **BROAD COVERAGE GOOD; POWER-AWARE EDGE THIN**

Orca, vLLM, Sarathi-Serve, DistServe, Splitwise, and Gavel give you a solid conventional-serving baseline.  You should not add many more ordinary throughput-oriented serving papers. Add the systems that directly combine serving SLOs with power/energy.

- **ADD — Stojkovic et al., "TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms," ASPLOS 2025. DOI: 10.1145/3676641.3716025.**  
  TAPAS performs placement, request routing, and VM reconfiguration under thermal/power constraints; your scheduler instead reasons about each job's internal `(GPU, cap)` energy operating point and energy-per-work optimum. 

- **MUST ADD — Wang et al., "Energy-Aware Scheduling for Serverless LLM Serving on Shared GPUs" (Festina). arXiv:2606.30391.**  
  Festina jointly controls placement, SM partitioning, batching, consolidation, and GPU operating points under TTFT/TBT SLOs; unlike Festina's production-oriented control plane, your core contribution is the empirical response geometry/law and Joule-point analysis, with the current fleet result still trace-driven rather than a live serving deployment. 

That distinction is especially important because your own limitations say ELF uses standalone inference passes rather than continuous vLLM/TensorRT-LLM serving, and Section 8 is simulation rather than a live deployment. 

---

## 4. Accelerator power modeling and inference-energy datasets — **TOO THIN**

The current section jumps from modern measurement benchmarks directly to ELF. It needs the older GPU power-modeling lineage and the newer open energy datasets against which ELF's novelty can be stated precisely. Current coverage of MLPerf Power, ML.ENERGY, and From Words to Watts should remain. 

### Power/performance models

- **ADD — Hong & Kim, "An Integrated GPU Power and Performance Model," ISCA 2010. DOI: 10.1145/1815961.1815998.**  
  This early GPU model predicts performance and power to locate efficient active-core configurations; your law is deliberately much lower-dimensional and measurement-driven, relating achieved inference rate to board power along cap sweeps rather than predicting microarchitectural events. 

- **ADD — Nagasaka et al., "Statistical Power Modeling of GPU Kernels Using Performance Counters," IGCC 2010. DOI: 10.1109/GREENCOMP.2010.5598315.**  
  They predict kernel power from GPU performance counters; your result deliberately avoids counter-based prediction and instead identifies an empirical response law and operating-point surface directly from cap/throughput measurements. 

- **ADD — Leng et al., "GPUWattch: Enabling Energy Optimizations in GPGPUs," ISCA 2013. DOI: 10.1145/2508148.2485964.**  
  GPUWattch is an architectural GPU power-modeling framework for energy optimization; ELF is measured on deployed NVIDIA cards and is intended for scheduler-level decisions rather than architectural simulation. 

### Measurement datasets/benchmarks

- **KEEP — MLPerf Power. arXiv:2410.12032.**  
  It standardizes comparable ML energy measurements across systems; ELF is narrower in hardware/model breadth but deliberately much denser along the power-cap axis, enabling derivatives and stationary-point analysis. 

- **KEEP — ML.ENERGY, Chung et al. arXiv:2505.06371; NeurIPS Datasets & Benchmarks 2025.**  
  ML.ENERGY measures realistic generative-AI inference across many architectures/tasks and supports automatic optimization recommendations; ELF contributes dense cap-response surfaces rather than a broad leaderboard. 

- **KEEP — Samsi et al., "From Words to Watts." arXiv:2310.03003.**  
  It benchmarks LLM inference energy across GPU configurations and scale; ELF differs by systematically sweeping the **same workload/device through many power caps and batch operating points**, which is what lets you estimate elasticity and the Joule point. 

- **ADD — Tripp et al., "Measuring the Energy Consumption and Efficiency of Deep Neural Networks..." / BUTTER-E. arXiv:2403.08151.**  
  BUTTER-E releases tens of thousands of measured CPU/GPU DNN runs and explores nonlinear energy effects across network configurations; it is primarily a training/model-design dataset, whereas ELF resolves the within-device power-cap response of inference workloads. 

- **ADD — Husom et al., "The Price of Prompting: Profiling Energy Use in Large Language Models Inference" / MELODI. arXiv:2407.16893.**  
  MELODI releases energy measurements across prompts, models, and deployment frameworks; ELF's distinctive axis is controlled power-cap × batch variation rather than prompt/model heterogeneity. 

- **MUST ADD — Argerich, Fürst & Patiño-Martínez, "Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures." arXiv:2604.09048.**  
  Watt Counts contains >5,000 experiments for 50 LLMs across 10 NVIDIA GPUs and explicitly studies energy-optimal hardware selection; ELF should therefore claim **greater operating-point density and cap sensitivity**, not superior breadth or scale. 

**Best ELF positioning:** "Existing datasets are broad across models, prompts, or hardware; ELF is designed to be dense *within* each workload–GPU pair over power cap and batch, so local slopes, elasticity, and stationary energy points are observable." That is much safer than implying that inference-energy datasets are generally sparse.

---

## 5. Energy-optimal operating points, EDP, race-to-idle, and undervolting — **TOO THIN / SHOULD BE ITS OWN PARAGRAPH**

Right now EDP appears as one sentence inside GPU DVFS. Because the Joule point is one of the named contributions, reviewers need to see explicitly how it differs from the long history of energy-optimal frequency/voltage points.

- **KEEP — Gonzalez & Horowitz, "Energy Dissipation in General Purpose Microprocessors," IEEE JSSC 1996. DOI: 10.1109/4.535411.**  
  This is the EDP lineage; EDP balances energy and delay as a chosen composite objective, whereas your Joule point minimizes \(E=P/R\) itself and reports latency separately as the explicit cost.

- **ADD — Miyoshi et al., Critical Power Slope, DOI above.**  
  This is arguably the closest conceptual ancestor of the Joule point and should be explicitly discussed here even if also cited in the speed-scaling paragraph.

- **ADD — Leng et al., "Safe Limits on Voltage Reduction Efficiency in GPUs: A Direct Measurement Approach," MICRO 2015. DOI: 10.1145/2830772.2830811.**  
  They find substantial GPU voltage guardband and energy savings by moving toward workload-specific \(V_{\min}\); your Joule point uses an ordinary exposed power-cap interface, does not undervolt below validated operating conditions, and accepts a throughput/latency trade rather than pursuing voltage savings at fixed frequency. 

The distinction should be stated mathematically: your Joule point minimizes

\[
e(R)=P_0/R+aR^{\beta-1},
\]

giving

\[
R^*=\left[\frac{P_0}{a(\beta-1)}\right]^{1/\beta},
\]

so its novelty is the **specific measured GPU-inference response model and deployable mapping from that stationary rate to a power cap**, not the general discovery that hardware may have an energy-optimal point below peak performance. 

---

## 6. Carbon-, electricity-, and grid-aware datacenter scheduling — **CURRENTLY FAIRLY GOOD, but historical roots are missing**

The current paragraph already contains recent grid shedding, carbon-aware scheduling, renewable/battery co-design, Ecovisor, CarbonScaler, and demand response.  Keep these, but add foundational electricity/renewable-aware scheduling so the connection from an external energy signal to workload scheduling is historically complete.

- **ADD — Qureshi et al., "Cutting the Electric Bill for Internet-Scale Systems," SIGCOMM 2009. DOI: 10.1145/1592568.1592584.**  
  This work exploits spatial and temporal electricity-price variation by moving computation across datacenters; your mechanism operates inside a GPU/fleet and changes the watts consumed by a running inference operating point rather than primarily moving work geographically. 

- **ADD — Rao et al., "Minimizing Electricity Cost: Optimization of Distributed Internet Data Centers in a Multi-Electricity-Market Environment," INFOCOM 2010. DOI: 10.1109/INFCOM.2010.5461933.**  
  They optimize workload routing across electricity markets; your response surface supplies a finer-grained local actuator that could sit underneath such a market-aware controller. 

- **ADD — Goiri et al., "GreenSlot: Scheduling Energy Consumption in Green Datacenters," SC 2011. DOI: 10.1145/2063384.2063411.**  
  GreenSlot shifts delay-tolerant batch jobs toward renewable-energy availability; your controller can alter per-job GPU power immediately while preserving a specified SLO rather than relying only on temporal deferral. 

- **ADD — Goiri et al., "Parasol and GreenSwitch: Managing Datacenters Powered by Renewable Energy," ASPLOS 2013. DOI: 10.1145/2451116.2451123.**  
  GreenSwitch jointly schedules workload and energy supply in a solar/battery/grid datacenter; your work supplies a device-level power-performance control surface that such facility-level schemes generally abstract away. 

- **OPTIONAL ADD — Wiesner et al., "Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud." arXiv:2110.13234.**  
  It quantifies temporal carbon shifting for delay-tolerant computation; your method changes the energy consumed at a given execution time and therefore complements rather than substitutes for temporal shifting. 

- **ADD — Li et al., "EcoServe: Designing Carbon-Aware AI Inference Systems." arXiv:2502.05043.**  
  EcoServe explicitly optimizes operational and embodied carbon for AI inference while maintaining SLOs; your work is narrower in objective—joules/power operating points—but provides a measured actuator that a carbon-aware policy could price according to time-varying carbon intensity. 

This supports your abstract's statement that the same operating-point mechanism can respond to a power budget, electricity price, or carbon signal without claiming that carbon-aware scheduling itself is new. 

---

## 7. Adaptive/selective inference and computation reduction — **PRESENT BUT THIN**

This is not central to the current experiments, but Section 11 makes a fairly ambitious claim that energy space can expand to an "energy-to-entropy" space. Current Section 2 cites BranchyNet, conditional computation, a dynamic-network survey, speculative decoding, and CalexNet.  Add a few canonical systems demonstrating that computation itself is an input-dependent control variable.

- **ADD — Huang et al., "Multi-Scale Dense Networks for Resource Efficient Image Classification," ICLR 2018. arXiv:1703.09844.**  
  MSDNet allocates unequal computation across examples using early exits and explicitly studies budgeted inference; your present paper fixes the model/computation and varies the hardware operating point, with Section 11 proposing their eventual combination. 

- **ADD — Wang et al., "SkipNet: Learning Dynamic Routing in Convolutional Networks," ECCV 2018. arXiv:1711.09485.**  
  SkipNet conditionally skips blocks for easy inputs; this changes useful computation \(C\), whereas your measured sections hold \(C\) fixed and change the joules used to execute it. 

- **ADD — Xin et al., "DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference," ACL 2020. DOI: 10.18653/v1/2020.acl-main.204; arXiv:2004.12993.**  
  DeeBERT extends early exiting to transformers; it is a useful bridge from your current BranchyNet citation to modern language-model inference. 

- **ADD — Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding." arXiv:2404.16710.**  
  LayerSkip combines transformer early exits with self-speculative decoding and is especially relevant to your proposed energy-to-entropy extension; unlike your current work, it changes the amount/path of model computation rather than GPU power allocation. 

- **OPTIONAL ADD — Song et al., "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU." arXiv:2312.12456.**  
  Despite the name, PowerInfer is not a GPU power-management paper: it exploits activation sparsity and CPU/GPU placement of hot/cold neurons, making it a useful example of selective computation rather than a competitor to the Joule-point mechanism. 

---

## 8. Energy-aware cloud pricing, split incentives, and demand-response incentives — **MISSING; HIGH PRIORITY**

This theme is required because economic incentives are one of the paper's five stated contributions, not merely discussion. Section 9 argues something more specific than "per-joule pricing is good": under current GPU economics, hourly GPU rental dominates electricity cost so strongly that **a naive per-joule surcharge still leaves uncapped operation cost-optimal**. 

There is already a literature on energy-aware cloud pricing, so do **not** present energy-sensitive pricing itself as new.

- **ADD — Zhan et al., "Extending Demand Response to Tenants in Cloud Data Centers via Non-intrusive Workload Flexibility Pricing." arXiv:1603.05746.**  
  This paper explicitly identifies the split incentive between cloud operators seeking demand response and tenants charged by conventional usage pricing, then rewards tenant flexibility; your contribution is to expose and quantify the same incentive conflict specifically at the GPU power-cap/Joule-point level. 

- **ADD — Aldossary et al., "Energy-Aware Cost Prediction and Pricing of Virtual Machines in Cloud Computing Environments," Future Generation Computer Systems 2019. DOI: 10.1016/j.future.2018.10.027.**  
  They model VM-level energy attribution and propose energy-based cloud pricing schemes; your distinctive result is that merely passing through ordinary electricity cost per joule is insufficient because GPU rental cost dominates the energy term. 

- **ADD — Liu, Rocca & Guitart, "Energy-Aware Dynamic Pricing Model for Cloud Environments," GECON 2019. DOI: 10.1007/978-3-030-36027-6_7.**  
  They explicitly criticize fixed time-based cloud pricing for being oblivious to actual energy cost and propose dynamic energy-aware prices; your analysis is GPU-inference-specific and ties the resulting distortion directly to the measured energy/throughput response curve. 

- **OPTIONAL ADD — Paul, Zhong & Bose, "Energy-aware pricing for cloud services," ICICS 2015. DOI: 10.1109/ICICS.2015.7459946.**  
  Another direct precedent for energy-sensitive cloud pricing; useful mainly to demonstrate that "per-energy pricing" itself is established and should not carry your novelty claim. 

**Recommended novelty statement for Section 9:**  
"The novelty is not the proposal of energy-aware cloud pricing. It is the measured demonstration that time-based GPU rental creates an operating-point-level split incentive: the renter's throughput-maximizing point and the provider/grid's energy-minimizing Joule point diverge sharply, and at current electricity-to-GPU-rental price ratios a simple joule pass-through is too weak to close that gap."

That is both more defensible and more interesting than "we propose per-joule pricing."

---

# Recommended Section 2 architecture

1. **Energy proportionality and speed scaling**  
   Barroso/Hölzle → Yao/Demers/Shenker → Critical Power Slope → establish that convex power/performance and sub-maximal energy optima are old; explain what is new about the GPU-inference operating-point law.

2. **GPU DVFS, power capping, and energy-optimal operating points**  
   Tang/Mei → Zeus/BatchDVFS/Perseus → undervolting → distinguish Joule point from EDP, DVFS optima, and voltage guardband work.

3. **Power-aware LLM serving**  
   Orca/vLLM/Sarathi/DistServe/Splitwise only briefly as context; emphasize POLCA, DynamoLLM, TAPAS, throttLL'eM, GreenLLM, PALS, Festina. This is now the most important competitive paragraph.

4. **GPU power modeling and energy measurement datasets**  
   Hong/Kim → Nagasaka → GPUWattch → MLPerf Power/ML.ENERGY/Words-to-Watts → BUTTER-E/MELODI/Watt Counts → state ELF's *density-of-operating-points* novelty precisely.

5. **Grid-, carbon-, and electricity-aware scheduling**  
   Qureshi/Rao → GreenSlot/GreenSwitch → modern carbon/grid-aware work → explain ELF as a device/job-level actuator underneath facility/grid policies.

6. **Adaptive/selective inference**  
   Keep short because it is future work rather than evaluated contribution; BranchyNet/MSDNet/SkipNet/DeeBERT/LayerSkip/speculative decoding are enough.

7. **Pricing and incentives**  
   Add a new paragraph. Energy-aware cloud pricing and tenant demand-response incentives exist; your contribution is the GPU-operating-point split incentive and the quantitative finding that ordinary per-joule pass-through is economically too small.

# Highest-priority additions if citation budget is tight

1. **Yao, Demers & Shenker 1995** — because it directly anticipates convex power-speed deadline scheduling.  
2. **Miyoshi et al. 2002 / Critical Power Slope** — because it directly anticipates an energy-optimal operating point.  
3. **PALS, arXiv:2605.21427** — because it explicitly makes GPU power caps a first-class LLM-serving knob.  
4. **Festina, arXiv:2606.30391** — because it jointly schedules LLM serving and GPU operating points under SLOs.  
5. **TAPAS, ASPLOS 2025** — power-aware LLM cluster scheduling.  
6. **Watt Counts, arXiv:2604.09048** — the most important recent dataset comparison for ELF.  
7. **Hong & Kim 2010** — foundational GPU power/performance modeling.  
8. **Leng et al. MICRO 2015** — establishes the distinct undervolting/energy-optimal GPU lineage.  
9. **Zhan et al., arXiv:1603.05746** — directly addresses the cloud-tenant split incentive under demand response.  
10. **Aldossary et al. 2019** — directly establishes prior energy-based cloud pricing.

The resulting positioning is stronger if the paper claims **not** "we discovered that GPUs should sometimes run below TDP," "we are first to control GPU power in serving," or "we introduce energy-aware pricing." The harder-to-displace contribution is the combination of **dense controlled operating-point measurements → empirical response law → derivative/stationary-point calculus → per-job/fleet decisions → quantified incentive mismatch**. That chain is what Section 2 should be designed to protect.
```